"""Dependency-free semantic sound classifier.

This is AURA's default :class:`~aura.classification.base.SoundClassifier`. It
combines streaming DSP features into per-category evidence scores. It is not as
accurate as a trained AudioSet model, but it has three properties that matter
for the MVP:

* **It always runs.** No model download, no PyTorch, no ONNX. The engine is
  demonstrable on a clean machine, which is a hard requirement for a live demo.
* **It is fast.** Sub-millisecond per frame, so the latency budget is spent on
  audio, not inference.
* **It is multi-label and stream-aware.** It reports several concurrent sounds
  and tags each with the separated stream it came from, which is what lets the
  mixer actually change the audio rather than just label it.

Swap in :class:`~aura.classification.onnx_backend.ONNXAudioClassifier` for
YAMNet/PANNs/AST accuracy; the rest of the system is unchanged.

Each detector below is written as evidence accumulation: several weak cues are
combined rather than one threshold, because single thresholds break the moment
the acoustic environment shifts.
"""

from __future__ import annotations

import numpy as np

from ..types import AudioFrame, SeparatedAudio, SoundEvent, StreamKind
from .base import SoundClassifier
from .features import FeatureExtractor, FrameFeatures

# Below this level the scene is treated as silence.
SILENCE_FLOOR_DBFS = -68.0


# --------------------------------------------------------------------------
# Fuzzy membership helpers
# --------------------------------------------------------------------------


def ramp(x: float, lo: float, hi: float) -> float:
    """0 below ``lo``, 1 above ``hi``, linear between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def inv_ramp(x: float, lo: float, hi: float) -> float:
    """1 below ``lo``, 0 above ``hi``."""
    return 1.0 - ramp(x, lo, hi)


def window(x: float, lo: float, plateau_lo: float, plateau_hi: float, hi: float) -> float:
    """Trapezoidal membership: 1 on ``[plateau_lo, plateau_hi]``, tapering to 0."""
    if x < plateau_lo:
        return ramp(x, lo, plateau_lo)
    if x > plateau_hi:
        return inv_ramp(x, plateau_hi, hi)
    return 1.0


def combine(*evidence: tuple[float, float]) -> float:
    """Weighted geometric-ish mean of ``(score, weight)`` pairs.

    A geometric blend is used deliberately: one strongly contradicting cue
    should be able to veto a detection, which an arithmetic mean cannot do.
    """
    total_w = sum(w for _, w in evidence)
    if total_w <= 0:
        return 0.0
    acc = 0.0
    for score, w in evidence:
        acc += w * np.log(max(score, 1e-3))
    return float(np.clip(np.exp(acc / total_w), 0.0, 1.0))


# --------------------------------------------------------------------------
# Detectors -- one per category
# --------------------------------------------------------------------------


def detect_human_speech(f: FrameFeatures) -> float:
    """A single nearby talker.

    Keyed on syllabic modulation plus voicing. The low-band requirement is what
    separates a person in the room from a PA announcement, which arrives
    band-limited with no chest resonance.
    """
    return combine(
        (ramp(f.modulation_depth, 0.18, 0.45), 2.2),
        (ramp(f.voicing, 0.18, 0.45), 2.0),
        (window(f.modulation_rate_hz, 1.2, 2.2, 8.0, 11.0), 1.0),
        (ramp(f.band_energy.get("low_total", 0.0), 0.06, 0.22), 1.4),
        (window(f.entropy, 0.28, 0.48, 0.86, 0.95), 1.2),
        (ramp(f.flux_variance * 1000.0, 8.0, 45.0), 1.2),
        (inv_ramp(f.tonality, 0.50, 0.85), 1.0),
    )


def detect_crowd_speech(f: FrameFeatures) -> float:
    """Babble: many talkers at once.

    Voicing stays high (it is still speech) but the syllabic modulation washes
    out once several talkers overlap. Low modulation *with* voicing present is
    the signature; that combination is also what rules out traffic noise.
    """
    return combine(
        (ramp(f.voicing, 0.14, 0.38), 2.2),
        (window(f.modulation_depth, 0.0, 0.04, 0.30, 0.52), 1.8),
        (window(f.entropy, 0.32, 0.48, 0.78, 0.92), 1.2),
        (inv_ramp(f.flatness, 0.08, 0.28), 1.4),
        (inv_ramp(f.band_energy.get("low_total", 0.0), 0.10, 0.30), 1.6),
        (inv_ramp(f.tonality, 0.45, 0.80), 1.0),
    )


def detect_vehicle_horn(f: FrameFeatures) -> float:
    """Loud, tonal, mid-band -- and pitch-stable.

    Pitch stability is the whole distinction from a siren, which is why
    ``peak_range_hz`` carries the heaviest veto weight here.
    """
    return combine(
        (ramp(f.tonality, 0.45, 0.72), 2.2),
        (window(f.tonal_peak_hz, 180.0, 280.0, 2000.0, 2800.0), 1.8),
        (inv_ramp(f.peak_range_hz, 130.0, 400.0), 2.0),
        (inv_ramp(f.entropy, 0.38, 0.72), 1.4),
        (ramp(f.dbfs, -44.0, -22.0), 1.0),
        (inv_ramp(f.modulation_depth, 0.18, 0.45), 1.6),
    )


def detect_emergency_siren(f: FrameFeatures) -> float:
    """Tonal like a horn, but the pitch sweeps.

    ``peak_range_hz`` dominates: a wailing siren moves its dominant peak by
    hundreds of Hz within a second, which nothing else in the taxonomy does.
    """
    return combine(
        (ramp(f.peak_range_hz, 140.0, 430.0), 2.6),
        (ramp(f.tonality, 0.45, 0.78), 1.8),
        (window(f.tonal_peak_hz, 330.0, 520.0, 3200.0, 4500.0), 1.4),
        (inv_ramp(f.entropy, 0.32, 0.68), 1.2),
        (ramp(f.dbfs, -50.0, -28.0), 0.8),
    )


def detect_engine_noise(f: FrameFeatures) -> float:
    """Low-frequency rumble, spectrally static."""
    return combine(
        (ramp(f.band_energy.get("low_total", 0.0), 0.25, 0.55), 2.5),
        (inv_ramp(f.centroid_hz, 180.0, 700.0), 2.0),
        (inv_ramp(f.modulation_depth, 0.14, 0.45), 1.2),
        (ramp(f.stationarity, 0.58, 0.92), 1.0),
        (inv_ramp(f.zcr, 0.10, 0.34), 1.0),
        (inv_ramp(f.tonality, 0.55, 0.85), 0.8),
    )


def detect_general_traffic(f: FrameFeatures) -> float:
    """Broadband road ambience: noisy, unvoiced, mid-centred."""
    return combine(
        (ramp(f.flatness, 0.10, 0.32), 1.8),
        (ramp(f.entropy, 0.58, 0.84), 1.6),
        (window(f.centroid_hz, 380.0, 850.0, 3600.0, 6200.0), 1.4),
        (inv_ramp(f.voicing, 0.12, 0.38), 1.6),
        (inv_ramp(f.tonality, 0.35, 0.72), 1.0),
        (ramp(f.dbfs, -46.0, -26.0), 1.0),
        (inv_ramp(f.band_energy.get("hf_total", 0.0), 0.18, 0.42), 0.8),
    )


def detect_alarm(f: FrameFeatures) -> float:
    """High beeping tone: tonal, switching on and off a few times a second."""
    return combine(
        (window(f.tonal_peak_hz, 1700.0, 2500.0, 6200.0, 8200.0), 2.2),
        (ramp(f.tonality, 0.42, 0.70), 1.6),
        (ramp(f.modulation_depth, 0.14, 0.45), 1.8),
        (inv_ramp(f.entropy, 0.28, 0.62), 1.0),
        (inv_ramp(f.peak_range_hz, 140.0, 420.0), 1.0),
    )


def detect_warning_signal(f: FrameFeatures) -> float:
    """Reversing beeper / level-crossing tone: like an alarm, an octave lower."""
    return combine(
        (window(f.tonal_peak_hz, 560.0, 780.0, 1750.0, 2500.0), 2.2),
        (ramp(f.tonality, 0.42, 0.72), 1.6),
        (ramp(f.modulation_depth, 0.12, 0.42), 1.6),
        (inv_ramp(f.entropy, 0.28, 0.62), 1.0),
        (inv_ramp(f.peak_range_hz, 140.0, 420.0), 1.2),
    )


def detect_bicycle_bell(f: FrameFeatures) -> float:
    """Struck metal: extremely tonal, high, decaying -- and *not* beeping."""
    return combine(
        (ramp(f.tonality, 0.58, 0.88), 2.0),
        (window(f.tonal_peak_hz, 1400.0, 1900.0, 7000.0, 9200.0), 1.8),
        (inv_ramp(f.entropy, 0.20, 0.58), 1.4),
        (inv_ramp(f.modulation_depth, 0.22, 0.52), 1.2),
        (inv_ramp(f.peak_range_hz, 130.0, 400.0), 1.0),
    )


def detect_music(f: FrameFeatures) -> float:
    """Harmonic, sustained and rhythmically steady rather than syllabic.

    The flux-variance veto is what stops speech being read as music: speech
    starts and stops constantly, music holds.
    """
    return combine(
        (inv_ramp(f.flux_variance * 1000.0, 10.0, 45.0), 1.8),
        (ramp(f.tonality, 0.22, 0.52), 1.4),
        (window(f.modulation_depth, 0.0, 0.03, 0.32, 0.58), 1.4),
        (ramp(f.stationarity, 0.72, 0.95), 1.2),
        (window(f.entropy, 0.30, 0.45, 0.76, 0.90), 1.2),
        (ramp(f.band_energy.get("low_total", 0.0), 0.06, 0.24), 1.0),
        (ramp(f.voicing, 0.10, 0.35), 0.8),
    )


def detect_mechanical_noise(f: FrameFeatures) -> float:
    """Machinery: broadband, high-centred, with fluctuating impacts."""
    return combine(
        (ramp(f.band_energy.get("hf_total", 0.0), 0.10, 0.34), 2.0),
        (window(f.centroid_hz, 1100.0, 2100.0, 6800.0, 9500.0), 1.6),
        (ramp(f.flux_variance * 1000.0, 4.0, 26.0), 1.4),
        (ramp(f.entropy, 0.52, 0.80), 1.2),
        (inv_ramp(f.voicing, 0.12, 0.38), 1.0),
        (ramp(f.dbfs, -52.0, -30.0), 0.8),
    )


def detect_announcement(f: FrameFeatures) -> float:
    """PA speech: speech modulation, but band-limited by a cheap loudspeaker."""
    return combine(
        (ramp(f.modulation_depth, 0.26, 0.50), 2.8),
        (ramp(f.voicing, 0.22, 0.48), 1.6),
        (inv_ramp(f.band_energy.get("low_total", 0.0), 0.05, 0.20), 2.2),
        (inv_ramp(f.band_energy.get("hf_total", 0.0), 0.04, 0.20), 1.2),
        (window(f.entropy, 0.18, 0.30, 0.64, 0.86), 1.0),
        (ramp(f.flux_variance * 1000.0, 8.0, 45.0), 1.0),
    )


def detect_approaching_vehicle(f: FrameFeatures) -> float:
    """Defined by trajectory, not timbre: a vehicle spectrum getting louder.

    Treated as safety-critical, so the evidence bar is deliberately low.
    """
    return combine(
        (ramp(f.level_slope_db_s, 2.0, 12.0), 2.6),
        (ramp(f.band_energy.get("low_total", 0.0), 0.12, 0.42), 1.6),
        (ramp(f.dbfs, -52.0, -32.0), 1.0),
        (inv_ramp(f.modulation_depth, 0.30, 0.65), 0.6),
        (inv_ramp(f.voicing, 0.15, 0.40), 2.0),
    )


def detect_ambient(f: FrameFeatures) -> float:
    """The "nothing in particular" hypothesis."""
    return combine(
        (inv_ramp(f.dbfs, -52.0, -28.0), 1.4),
        (ramp(f.flatness, 0.14, 0.40), 1.2),
        (inv_ramp(f.tonality, 0.25, 0.62), 1.0),
        (inv_ramp(f.voicing, 0.12, 0.35), 1.0),
        (inv_ramp(f.modulation_depth, 0.18, 0.45), 0.8),
    )


#: ``category -> (detector, stream_to_analyse, stream_to_tag)``
#:
#: The stream a category is *analysed* on can differ from the one it is
#: *tagged* with. Transient events are detected on the full mix (separation
#: would smear a 200 ms horn) but tagged as TRANSIENT so the mixer routes them
#: through the protected path.
DETECTORS: dict[str, tuple] = {
    "human_speech": (detect_human_speech, StreamKind.PRIMARY_SPEECH, StreamKind.PRIMARY_SPEECH),
    "announcement": (detect_announcement, StreamKind.PRIMARY_SPEECH, StreamKind.PRIMARY_SPEECH),
    "crowd_speech": (detect_crowd_speech, StreamKind.SECONDARY_SPEECH, StreamKind.SECONDARY_SPEECH),
    # Narrowband events are analysed on the full mix, not on the separator's
    # transient stream. Analysing the isolated stream was tried and was worse:
    # residual speech harmonics and music partials survive into that stream and
    # the siren detector fires on them, because a sweeping partial looks like a
    # sweeping siren once the context around it has been stripped away. The mix
    # keeps that context, at the cost of missing quiet bells in loud traffic.
    "vehicle_horn": (detect_vehicle_horn, None, StreamKind.TRANSIENT),
    "emergency_siren": (detect_emergency_siren, None, StreamKind.TRANSIENT),
    "alarm": (detect_alarm, None, StreamKind.TRANSIENT),
    "bicycle_bell": (detect_bicycle_bell, None, StreamKind.TRANSIENT),
    "warning_signal": (detect_warning_signal, None, StreamKind.TRANSIENT),
    "engine_noise": (detect_engine_noise, StreamKind.ENVIRONMENT, StreamKind.ENVIRONMENT),
    "general_traffic": (detect_general_traffic, StreamKind.ENVIRONMENT, StreamKind.ENVIRONMENT),
    "mechanical_noise": (detect_mechanical_noise, StreamKind.ENVIRONMENT, StreamKind.ENVIRONMENT),
    "music": (detect_music, StreamKind.ENVIRONMENT, StreamKind.ENVIRONMENT),
    "approaching_vehicle": (detect_approaching_vehicle, StreamKind.ENVIRONMENT, StreamKind.ENVIRONMENT),
    "ambient": (detect_ambient, None, StreamKind.ENVIRONMENT),
}


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------


class DSPSoundClassifier(SoundClassifier):
    """Streaming multi-label classifier built from DSP evidence.

    Parameters
    ----------
    sample_rate:
        Rate of incoming frames.
    threshold:
        Minimum smoothed confidence for an event to be reported.
    attack / release:
        Exponential smoothing coefficients. Attack is fast so safety sounds
        appear immediately; release is slow so cards do not flicker.
    """

    name = "dsp-heuristic-v1"

    def __init__(
        self,
        sample_rate: int,
        *,
        threshold: float = 0.42,
        attack: float = 0.55,
        release: float = 0.12,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.attack = attack
        self.release = release
        self._mix_fx = FeatureExtractor(sample_rate)
        self._stream_fx: dict[StreamKind, FeatureExtractor] = {}
        self._smoothed: dict[str, float] = {}
        self.last_mix_features: FrameFeatures | None = None

    # -- SoundClassifier ---------------------------------------------------

    def reset(self) -> None:
        self._mix_fx.reset()
        for fx in self._stream_fx.values():
            fx.reset()
        self._smoothed.clear()

    def classify(
        self,
        frame: AudioFrame,
        separated: SeparatedAudio | None = None,
    ) -> list[SoundEvent]:
        mix_f = self._mix_fx.extract(frame.samples)
        self.last_mix_features = mix_f

        # Per-stream features, so speech cues are measured on the speech signal
        # rather than on a mixture that is 90% engine rumble.
        stream_f: dict[StreamKind, FrameFeatures] = {}
        if separated is not None:
            for kind, sig in separated.streams.items():
                fx = self._stream_fx.get(kind)
                if fx is None:
                    fx = FeatureExtractor(separated.sample_rate)
                    self._stream_fx[kind] = fx
                stream_f[kind] = fx.extract(sig)

        silent = mix_f.dbfs < SILENCE_FLOOR_DBFS

        raw: dict[str, tuple[float, FrameFeatures, StreamKind]] = {}
        for key, (fn, analyse_on, tag_as) in DETECTORS.items():
            feats = stream_f.get(analyse_on, mix_f) if analyse_on else mix_f
            if silent and key != "ambient":
                score = 0.0
            else:
                score = float(fn(feats))
                # A stream that is essentially empty cannot host a detection.
                if analyse_on is not None and feats.dbfs < SILENCE_FLOOR_DBFS:
                    score = 0.0
            raw[key] = (score, feats, tag_as)

        raw = self._apply_exclusions(raw)

        events: list[SoundEvent] = []
        for key, (score, feats, tag_as) in raw.items():
            prev = self._smoothed.get(key, 0.0)
            coeff = self.attack if score > prev else self.release
            smoothed = prev + coeff * (score - prev)
            self._smoothed[key] = smoothed
            if smoothed >= self.threshold:
                events.append(
                    SoundEvent(
                        category=key,
                        confidence=smoothed,
                        stream=tag_as,
                        level_db=feats.dbfs,
                        detected_at=frame.captured_at,
                        metadata={
                            "raw_confidence": round(score, 4),
                            "backend": self.name,
                        },
                    )
                )

        if not events:
            events.append(
                SoundEvent(
                    category="ambient",
                    confidence=max(self._smoothed.get("ambient", 0.0), 0.3),
                    stream=StreamKind.ENVIRONMENT,
                    level_db=mix_f.dbfs,
                    detected_at=frame.captured_at,
                    metadata={"backend": self.name, "fallback": True},
                )
            )

        events.sort(key=lambda e: e.confidence, reverse=True)
        return events

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _apply_exclusions(
        raw: dict[str, tuple[float, FrameFeatures, StreamKind]],
    ) -> dict[str, tuple[float, FrameFeatures, StreamKind]]:
        """Resolve categories that compete for the same acoustic evidence.

        Without this a siren also lights up ``vehicle_horn``, and a harmonic
        horn reads as ``crowd_speech``, because all of them are "harmonic and
        loud". Two mechanisms are used:

        *Competition groups* -- within a group the categories are alternative
        explanations of one signal, so runners-up are scaled by how far behind
        the leader they are. This is smooth (no threshold cliffs) and keeps a
        genuine tie showing both, which matters when two things really are
        happening at once.

        *Tonal veto* -- a confident narrowband event is strong evidence that
        the broadband hypotheses are just describing that same tone.

        This matters beyond display: the mixer applies gain *per category*, so
        a double detection would attenuate one physical sound twice.
        """

        def score(k: str) -> float:
            return raw[k][0]

        def scale(k: str, factor: float) -> None:
            s_, f_, st_ = raw[k]
            raw[k] = (s_ * float(np.clip(factor, 0.0, 1.0)), f_, st_)

        def compete(group: list[str], gamma: float) -> None:
            """Scale each member by ``(score / best) ** gamma``."""
            present = [k for k in group if k in raw]
            if len(present) < 2:
                return
            best_key = max(present, key=score)
            best = score(best_key)
            if best <= 1e-3:
                return
            for k in present:
                if k == best_key:
                    continue
                scale(k, (score(k) / best) ** gamma)

        # Alternative explanations of a single narrowband event.
        TONAL = ["emergency_siren", "vehicle_horn", "alarm", "warning_signal", "bicycle_bell"]
        compete(TONAL, gamma=2.0)

        # Alternative explanations of a single voiced signal.
        compete(["human_speech", "crowd_speech", "announcement"], gamma=1.6)

        # Alternative explanations of one broadband bed.
        compete(["general_traffic", "mechanical_noise", "ambient"], gamma=1.3)

        # Music and babble are both "harmonic, unsyllabic, mid-band"; and a
        # low-centred harmonic bed is either music or an engine, not both.
        compete(["music", "crowd_speech"], gamma=1.5)
        compete(["music", "engine_noise"], gamma=1.5)

        # A confident tone means the broadband hypotheses are describing it.
        tonal_strength = max((score(k) for k in TONAL if k in raw), default=0.0)
        if tonal_strength > 0.45:
            veto = 1.0 - 0.85 * tonal_strength
            for k in ("crowd_speech", "music", "human_speech", "announcement",
                      "general_traffic", "ambient"):
                if k in raw:
                    scale(k, veto)

        # An approaching vehicle *is* engine and tyre noise -- the two labels
        # describe one physical source. The safety-critical label takes
        # ownership so the mixer does not both boost and suppress it.
        approaching = score("approaching_vehicle")
        if approaching > 0.45:
            claim = 1.0 - 0.85 * approaching
            for k in ("engine_noise", "general_traffic"):
                if k in raw:
                    scale(k, claim)

        # Ambient is the "nothing interesting" hypothesis: any confident
        # foreground detection should push it down.
        foreground = max((score(k) for k in raw if k != "ambient"), default=0.0)
        scale("ambient", (1.0 - foreground) ** 1.6)

        return raw

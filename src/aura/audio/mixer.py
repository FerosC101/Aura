"""The Audio Mixer: where decisions become sound.

Takes the separated streams and the policy engine's gain decisions and builds
the signal the user actually hears. This is the component that makes AURA an
audio product rather than a classifier with a nice dashboard.

Two kinds of gain are applied:

**Per-stream scalar gain** for speech and transients. These streams are already
isolated, so a single multiplier is the right tool, applied as a per-sample
ramp from the previous block's value to avoid zipper noise.

**Per-band spectral gain** for the environment stream. Engine rumble, road
noise and machinery all share that stream but occupy different parts of the
spectrum, so one scalar cannot express "drop the rumble, keep the rest". Each
category's decision is projected onto the shaper's bands in proportion to its
spectral footprint, and the resulting curve is applied with zero added latency.

A safety-critical decision always wins its stream: within a stream the mixer
takes the most audible critical gain rather than an average, so a siren is
never diluted by a suppression decision sharing the same signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..classification.taxonomy import get_category
from ..types import GainDecision, Priority, SeparatedAudio, StreamKind
from .biquad import SHAPER_BANDS, MultibandShaper, SoftLimiter

#: Streams that receive a single scalar gain.
SCALAR_STREAMS = (
    StreamKind.PRIMARY_SPEECH,
    StreamKind.SECONDARY_SPEECH,
    StreamKind.TRANSIENT,
)


@dataclass(slots=True)
class MixReport:
    """What the mixer actually did, for metrics and the UI."""

    stream_gain_db: dict[str, float] = field(default_factory=dict)
    environment_band_db: list[float] = field(default_factory=list)
    input_level_db: float = -90.0
    output_level_db: float = -90.0
    limiter_active: bool = False
    applied: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "stream_gain_db": {k: round(v, 2) for k, v in self.stream_gain_db.items()},
            "environment_band_db": [round(v, 2) for v in self.environment_band_db],
            "environment_bands_hz": list(SHAPER_BANDS),
            "input_level_db": round(self.input_level_db, 1),
            "output_level_db": round(self.output_level_db, 1),
            "limiter_active": self.limiter_active,
            "applied_gain_db": {k: round(v, 2) for k, v in self.applied.items()},
        }


def _band_weights(category: str, bands: tuple[float, ...] = SHAPER_BANDS) -> np.ndarray:
    """How strongly a category's spectrum overlaps each shaper band.

    Uses the taxonomy's declared ``band_hz``, evaluated on a log-frequency
    axis because that is how the bands are spaced and how hearing works.
    """
    lo, hi = get_category(category).band_hz
    lo = max(lo, 20.0)
    hi = max(hi, lo * 1.05)
    w = np.zeros(len(bands), dtype=np.float64)
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    for i, f in enumerate(bands):
        # Each shaper band covers roughly a third of a decade around itself.
        b_lo, b_hi = np.log10(f / 1.6), np.log10(f * 1.6)
        overlap = min(log_hi, b_hi) - max(log_lo, b_lo)
        w[i] = max(overlap, 0.0) / (b_hi - b_lo)
    return np.clip(w, 0.0, 1.0)


class AudioMixer:
    """Applies gain decisions to separated streams and reconstructs the output."""

    def __init__(
        self,
        sample_rate: int,
        *,
        slew_ms: float = 25.0,
        band_slew: float = 0.25,
        output_trim_db: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.slew_ms = slew_ms
        self.band_slew = band_slew
        self.output_trim = 10.0 ** (output_trim_db / 20.0)

        self._shaper = MultibandShaper(sample_rate)
        self._limiter = SoftLimiter(sample_rate)
        self._stream_gain: dict[StreamKind, float] = dict.fromkeys(StreamKind, 1.0)
        self._band_db = np.zeros(len(SHAPER_BANDS), dtype=np.float64)
        self._weight_cache: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._shaper.reset()
        self._limiter.reset()
        self._stream_gain = dict.fromkeys(StreamKind, 1.0)
        self._band_db[:] = 0.0

    # -- main entry point --------------------------------------------------

    def mix(
        self,
        separated: SeparatedAudio,
        decisions: list[GainDecision],
        *,
        bypass: bool = False,
    ) -> tuple[np.ndarray, MixReport]:
        """Build the output block from separated streams and decisions."""
        length = max((s.size for s in separated.streams.values()), default=0)
        if length == 0:
            return np.zeros(0, dtype=np.float32), MixReport()

        dry = np.zeros(length, dtype=np.float64)
        for s in separated.streams.values():
            dry[: s.size] += s
        report = MixReport(input_level_db=_dbfs(dry))

        if bypass:
            out = self._limiter.process((dry * self.output_trim).astype(np.float32))
            report.output_level_db = _dbfs(out)
            report.limiter_active = self._limiter.active
            return out, report

        by_stream = self._group(decisions)
        out = np.zeros(length, dtype=np.float64)

        # --- scalar-gain streams ---
        for kind in SCALAR_STREAMS:
            sig = separated.streams.get(kind)
            if sig is None or sig.size == 0:
                continue
            target_db = by_stream.get(kind, 0.0)
            target = 10.0 ** (target_db / 20.0)
            out[: sig.size] += sig * self._ramp(kind, target, sig.size)
            report.stream_gain_db[kind.value] = target_db

        # --- spectrally shaped environment ---
        env = separated.streams.get(StreamKind.ENVIRONMENT)
        if env is not None and env.size:
            band_db = self._environment_bands(decisions)
            # Asymmetric slew: attenuate quickly, return to flat slowly.
            #
            # A category dropping below the detection threshold for one frame
            # is not evidence that the background went away, but a symmetric
            # slew would treat it as such and let the suppressed rumble surge
            # back for 100 ms. That pumping is far more noticeable than the
            # suppression itself, so recovery is deliberately ~4x slower than
            # onset.
            rising = band_db > self._band_db
            rate = np.where(rising, self.band_slew * 0.25, self.band_slew)
            self._band_db += rate * (band_db - self._band_db)
            self._shaper.set_gains(self._band_db)
            out[: env.size] += self._shaper.process(env.astype(np.float64))
            report.environment_band_db = [float(v) for v in self._band_db]
            report.stream_gain_db[StreamKind.ENVIRONMENT.value] = float(self._band_db.mean())

        out *= self.output_trim
        mixed = self._limiter.process(out.astype(np.float32))

        report.output_level_db = _dbfs(mixed)
        report.limiter_active = self._limiter.active
        report.applied = {d.category: d.gain_db for d in decisions}
        return mixed, report

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _group(decisions: list[GainDecision]) -> dict[StreamKind, float]:
        """Collapse decisions to one gain per scalar stream.

        A safety-critical decision takes the stream outright at its most
        audible value. Otherwise the gains are averaged by confidence, so an
        unsure detection cannot drag the whole stream down.
        """
        buckets: dict[StreamKind, list[GainDecision]] = {}
        for d in decisions:
            buckets.setdefault(d.stream, []).append(d)

        result: dict[StreamKind, float] = {}
        for kind, group in buckets.items():
            critical = [d for d in group if d.always_audible or d.safety_override]
            if critical:
                result[kind] = max(d.gain_db for d in critical)
                continue
            weights = np.array([max(d.confidence, 1e-3) for d in group])
            gains = np.array([d.gain_db for d in group])
            result[kind] = float(np.sum(gains * weights) / np.sum(weights))

        # The separator splits speech by pitch comb, which is not the same
        # distinction the classifier makes between "one talker" and "babble":
        # measured routing puts roughly half of a crowd into the primary
        # stream. Applying only the primary-labelled decisions to that stream
        # would leave most of a room's conversation untouched while the UI
        # reported it as suppressed.
        #
        # So the speech path is governed by every speech decision, blended by
        # confidence -- unless one has been explicitly promoted. Person Lock
        # raises its speaker to HIGH priority, and that decision takes the
        # primary stream outright, which is what keeps "lift this voice, duck
        # the rest" working.
        speech = [d for d in decisions
                  if d.stream in (StreamKind.PRIMARY_SPEECH, StreamKind.SECONDARY_SPEECH)]
        if speech:
            promoted = [d for d in speech
                        if d.stream is StreamKind.PRIMARY_SPEECH
                        and d.priority.rank >= Priority.HIGH.rank]
            if promoted:
                result[StreamKind.PRIMARY_SPEECH] = max(d.gain_db for d in promoted)
            else:
                w = np.array([max(d.confidence, 1e-3) for d in speech])
                g = np.array([d.gain_db for d in speech])
                result[StreamKind.PRIMARY_SPEECH] = float(np.sum(g * w) / np.sum(w))
        return result

    def _environment_bands(self, decisions: list[GainDecision]) -> np.ndarray:
        """Project environment-stream decisions onto the shaper's bands."""
        n = len(SHAPER_BANDS)
        num = np.zeros(n)
        den = np.zeros(n)
        floor = np.full(n, -np.inf)

        for d in decisions:
            if d.stream is not StreamKind.ENVIRONMENT:
                continue
            w = self._weight_cache.get(d.category)
            if w is None:
                w = _band_weights(d.category)
                self._weight_cache[d.category] = w
            weight = w * max(d.confidence, 1e-3)
            num += weight * d.gain_db
            den += weight
            if d.always_audible or d.safety_override:
                # A protected sound sets a floor on every band it occupies, so
                # suppressing the rumble around it cannot bury it.
                floor = np.maximum(floor, np.where(w > 0.05, d.gain_db, -np.inf))

        if not np.any(den > 1e-6):
            # Nothing was detected in the environment stream this frame. Hold
            # the existing curve: absence of a detection is not a detection of
            # absence, and snapping to flat would be heard as the background
            # rushing back in.
            return self._band_db.copy()

        band_db = np.divide(num, den, out=np.zeros(n), where=den > 1e-6)
        band_db = np.where(np.isfinite(floor), np.maximum(band_db, floor), band_db)
        return np.clip(band_db, -30.0, 12.0)

    def _ramp(self, kind: StreamKind, target: float, length: int) -> np.ndarray:
        """Per-sample gain ramp from the previous block's value."""
        prev = self._stream_gain.get(kind, 1.0)
        if abs(target - prev) < 1e-4:
            self._stream_gain[kind] = target
            return np.full(length, target, dtype=np.float64)
        # Ramp over the slew time or the block, whichever is shorter.
        n_slew = min(length, max(1, int(self.sample_rate * self.slew_ms / 1000.0)))
        ramp = np.empty(length, dtype=np.float64)
        ramp[:n_slew] = np.linspace(prev, target, n_slew, endpoint=False)
        ramp[n_slew:] = target
        self._stream_gain[kind] = target
        return ramp

    def describe(self) -> dict:
        return {
            "bands_hz": list(SHAPER_BANDS),
            "slew_ms": self.slew_ms,
            "band_slew": self.band_slew,
        }


def _dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -90.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * float(np.log10(max(rms, 1e-9)))

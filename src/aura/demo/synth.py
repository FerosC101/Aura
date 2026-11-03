"""Procedural sound-source synthesis for PSC Demo Mode and tests.

A live demo cannot depend on a room containing an ambulance. These generators
produce deterministic, physically-motivated approximations of each taxonomy
category so that:

* **PSC Demo Mode** can play a repeatable multi-source scene where the correct
  answer is known in advance, and
* **the test suite** can assert that the classifier finds the right sounds
  without shipping audio fixtures.

Everything is written with NumPy only and returns float32 in [-1, 1].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_SR = 16000


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _t(duration: float, sr: int) -> np.ndarray:
    return np.arange(int(duration * sr), dtype=np.float64) / sr


def _normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-9:
        return x.astype(np.float32)
    return (x * (peak / m)).astype(np.float32)


def _onepole_lowpass(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """Cheap one-pole IIR. Uses ``lfilter`` so the recursion runs in C."""
    from scipy.signal import lfilter

    a = float(np.exp(-2.0 * np.pi * cutoff / sr))
    return lfilter([1.0 - a], [1.0, -a], x)


def _resonator(x: np.ndarray, freq: float, q: float, sr: int) -> np.ndarray:
    """Two-pole resonant band-pass; the formant building block."""
    from scipy.signal import lfilter

    w = 2.0 * np.pi * freq / sr
    r = float(np.exp(-w / (2.0 * q)))
    b = [1.0 - r]
    a = [1.0, -2.0 * r * np.cos(w), r * r]
    return lfilter(b, a, x)


def _bandpass(x: np.ndarray, lo: float, hi: float, sr: int, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    nyq = sr / 2.0
    lo_n = max(1e-4, min(lo / nyq, 0.99))
    hi_n = max(lo_n + 1e-4, min(hi / nyq, 0.995))
    sos = butter(order, [lo_n, hi_n], btype="band", output="sos")
    return sosfilt(sos, x)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def speech(
    duration: float = 3.0,
    sr: int = DEFAULT_SR,
    *,
    f0: float = 125.0,
    syllable_rate: float = 4.2,
    formants: tuple[float, ...] = (520.0, 1500.0, 2560.0, 3400.0),
    seed: int | None = 7,
) -> np.ndarray:
    """Source-filter speech approximation.

    A jittered glottal pulse train through four formant resonators, gated by a
    syllabic envelope with pauses. Produces the ~4 Hz amplitude modulation and
    harmonic structure the classifier keys on for ``human_speech``.
    """
    rng = _rng(seed)
    t = _t(duration, sr)
    n = t.size

    # Glottal excitation: pulse train with jitter and a shimmer envelope.
    jitter = 1.0 + 0.02 * _onepole_lowpass(rng.standard_normal(n), 12.0, sr)
    phase = np.cumsum(2.0 * np.pi * f0 * jitter / sr)
    # Band-limited sawtooth-ish glottal pulse.
    excitation = np.zeros(n)
    for k in range(1, 26):
        if k * f0 > sr / 2.2:
            break
        excitation += np.sin(k * phase) / (k ** 1.15)
    excitation += 0.06 * rng.standard_normal(n)  # aspiration

    voiced = np.zeros(n)
    for i, fhz in enumerate(formants):
        gain = 1.0 / (1.0 + 0.5 * i)
        voiced += gain * _resonator(excitation, fhz, q=9.0 + 2.0 * i, sr=sr)

    # Syllabic envelope: raised-cosine syllables separated by short pauses,
    # plus longer inter-phrase silences so the modulation is realistic.
    env_phase = 2.0 * np.pi * syllable_rate * t
    syl = 0.5 * (1.0 + np.sin(env_phase))
    syl = syl ** 1.6
    phrase = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.35 * t + 1.0))
    gate = np.clip(phrase * 1.6 - 0.25, 0.0, 1.0)
    env = _onepole_lowpass(syl * gate, 25.0, sr)

    return _normalize(voiced * env, 0.85)


def crowd_speech(
    duration: float = 3.0,
    sr: int = DEFAULT_SR,
    *,
    voices: int = 9,
    seed: int | None = 11,
) -> np.ndarray:
    """Babble: many overlapping talkers, so syllabic modulation averages out."""
    rng = _rng(seed)
    acc = np.zeros(int(duration * sr))
    for _ in range(voices):
        v = speech(
            duration,
            sr,
            f0=float(rng.uniform(95.0, 215.0)),
            syllable_rate=float(rng.uniform(3.0, 6.0)),
            formants=(
                float(rng.uniform(430.0, 700.0)),
                float(rng.uniform(1150.0, 1900.0)),
                float(rng.uniform(2300.0, 2900.0)),
            ),
            seed=int(rng.integers(0, 10_000)),
        ).astype(np.float64)
        shift = int(rng.integers(0, max(1, v.size)))
        acc += np.roll(v, shift) * float(rng.uniform(0.4, 1.0))
    # Distance: babble reaches you filtered and reverberant.
    acc = _bandpass(acc, 220.0, 3400.0, sr)
    return _normalize(acc, 0.7)


def engine_noise(
    duration: float = 3.0,
    sr: int = DEFAULT_SR,
    *,
    rpm_hz: float = 47.0,
    seed: int | None = 3,
) -> np.ndarray:
    """Diesel-ish rumble: low harmonic stack plus low-passed combustion noise."""
    rng = _rng(seed)
    t = _t(duration, sr)
    # Slow RPM drift, as in a jeepney idling in traffic.
    drift = 1.0 + 0.05 * np.sin(2.0 * np.pi * 0.18 * t) + 0.02 * np.sin(2.0 * np.pi * 0.05 * t)
    phase = np.cumsum(2.0 * np.pi * rpm_hz * drift / sr)
    sig = np.zeros(t.size)
    for k in range(1, 13):
        sig += (0.9 / (k ** 1.5)) * np.sin(k * phase + 0.3 * k)
    combustion = _onepole_lowpass(rng.standard_normal(t.size), 260.0, sr)
    sig += 0.55 * combustion
    sig = _onepole_lowpass(sig, 420.0, sr)
    return _normalize(sig, 0.8)


def general_traffic(
    duration: float = 3.0,
    sr: int = DEFAULT_SR,
    *,
    seed: int | None = 5,
) -> np.ndarray:
    """Broadband road ambience with slow swells from passing vehicles."""
    rng = _rng(seed)
    t = _t(duration, sr)
    noise = rng.standard_normal(t.size)
    # Pink-ish tilt via cascaded one-poles.
    pink = _onepole_lowpass(noise, 2400.0, sr) + 0.5 * _onepole_lowpass(noise, 600.0, sr)
    band = _bandpass(pink, 140.0, 5200.0, sr)
    swell = 1.0 + 0.45 * np.sin(2.0 * np.pi * 0.23 * t) + 0.25 * np.sin(2.0 * np.pi * 0.11 * t + 2.0)
    return _normalize(band * swell, 0.75)


def vehicle_horn(
    duration: float = 1.2,
    sr: int = DEFAULT_SR,
    *,
    fundamental: float = 420.0,
    seed: int | None = 2,
) -> np.ndarray:
    """Two-tone horn with a hard onset and a stable pitch."""
    t = _t(duration, sr)
    # Real horns are two slightly detuned tones plus strong harmonics.
    sig = np.zeros(t.size)
    for base, amp in ((fundamental, 1.0), (fundamental * 1.26, 0.85)):
        for k, kamp in ((1, 1.0), (2, 0.55), (3, 0.33), (4, 0.18), (5, 0.1)):
            sig += amp * kamp * np.sin(2.0 * np.pi * base * k * t)
    attack = np.clip(t / 0.012, 0.0, 1.0)
    release = np.clip((duration - t) / 0.08, 0.0, 1.0)
    return _normalize(sig * attack * release, 0.95)


def emergency_siren(
    duration: float = 3.0,
    sr: int = DEFAULT_SR,
    *,
    low: float = 720.0,
    high: float = 1500.0,
    sweep_hz: float = 0.55,
) -> np.ndarray:
    """Wailing siren: a tone whose pitch sweeps continuously."""
    t = _t(duration, sr)
    # Triangular sweep between low and high -- the pitch variation is the cue.
    tri = np.arcsin(np.sin(2.0 * np.pi * sweep_hz * t)) * (2.0 / np.pi)
    freq = low + (high - low) * (0.5 * (tri + 1.0))
    phase = np.cumsum(2.0 * np.pi * freq / sr)
    sig = np.sin(phase) + 0.45 * np.sin(2.0 * phase) + 0.18 * np.sin(3.0 * phase)
    return _normalize(sig, 0.95)


def alarm(
    duration: float = 3.0,
    sr: int = DEFAULT_SR,
    *,
    tone_hz: float = 3100.0,
    beep_rate: float = 2.4,
) -> np.ndarray:
    """Smoke-alarm style high beeping."""
    t = _t(duration, sr)
    carrier = np.sin(2.0 * np.pi * tone_hz * t) + 0.3 * np.sin(2.0 * np.pi * 2 * tone_hz * t)
    duty = (np.sin(2.0 * np.pi * beep_rate * t) > 0.25).astype(np.float64)
    duty = _onepole_lowpass(duty, 60.0, sr)
    return _normalize(carrier * duty, 0.9)


def warning_signal(
    duration: float = 3.0,
    sr: int = DEFAULT_SR,
    *,
    tone_hz: float = 1050.0,
    beep_rate: float = 1.4,
) -> np.ndarray:
    """Reversing beeper / level-crossing tone."""
    t = _t(duration, sr)
    carrier = np.sin(2.0 * np.pi * tone_hz * t) + 0.25 * np.sin(2.0 * np.pi * 3 * tone_hz * t)
    duty = (np.sin(2.0 * np.pi * beep_rate * t) > 0.0).astype(np.float64)
    duty = _onepole_lowpass(duty, 70.0, sr)
    return _normalize(carrier * duty, 0.9)


def bicycle_bell(duration: float = 1.5, sr: int = DEFAULT_SR) -> np.ndarray:
    """Struck metal: high inharmonic partials with exponential decay."""
    t = _t(duration, sr)
    sig = np.zeros(t.size)
    for f, a, d in ((2340.0, 1.0, 3.2), (3610.0, 0.6, 4.1), (5180.0, 0.35, 5.5), (6900.0, 0.2, 7.0)):
        sig += a * np.sin(2.0 * np.pi * f * t) * np.exp(-d * t)
    # Two strikes.
    second = np.roll(sig, int(0.45 * sr))
    second[: int(0.45 * sr)] = 0.0
    return _normalize(sig + 0.8 * second, 0.9)


def music(duration: float = 3.0, sr: int = DEFAULT_SR, *, seed: int | None = 13) -> np.ndarray:
    """A simple chord progression with rhythmic envelope."""
    t = _t(duration, sr)
    chords = [(220.0, 277.2, 329.6), (196.0, 246.9, 293.7), (174.6, 220.0, 261.6), (196.0, 261.6, 311.1)]
    sig = np.zeros(t.size)
    seg = t.size // len(chords)
    for i, chord in enumerate(chords):
        s, e = i * seg, min((i + 1) * seg, t.size)
        tt = t[s:e] - t[s]
        acc = np.zeros(tt.size)
        for f in chord:
            for k, a in ((1, 1.0), (2, 0.45), (3, 0.28), (4, 0.15), (5, 0.08)):
                acc += a * np.sin(2.0 * np.pi * f * k * tt)
        acc *= np.exp(-0.6 * tt) + 0.4
        sig[s:e] = acc
    beat = 0.6 + 0.4 * (0.5 * (1.0 + np.sin(2.0 * np.pi * 2.0 * t)))
    return _normalize(sig * beat, 0.8)


def mechanical_noise(duration: float = 3.0, sr: int = DEFAULT_SR, *, seed: int | None = 17) -> np.ndarray:
    """Construction: broadband machinery plus periodic impacts."""
    rng = _rng(seed)
    t = _t(duration, sr)
    machine = _bandpass(rng.standard_normal(t.size), 400.0, 6500.0, sr)
    impacts = np.zeros(t.size)
    period = int(0.22 * sr)
    for start in range(0, t.size, period):
        length = min(int(0.09 * sr), t.size - start)
        if length <= 0:
            break
        env = np.exp(-28.0 * np.arange(length) / sr)
        impacts[start:start + length] += env * rng.standard_normal(length) * 3.0
    impacts = _bandpass(impacts, 250.0, 7000.0, sr)
    return _normalize(0.6 * machine + impacts, 0.85)


def announcement(duration: float = 3.0, sr: int = DEFAULT_SR, *, seed: int | None = 23) -> np.ndarray:
    """PA speech: band-limited and lightly distorted."""
    raw = speech(duration, sr, f0=142.0, syllable_rate=3.6, seed=seed).astype(np.float64)
    band = _bandpass(raw, 340.0, 3300.0, sr)
    driven = np.tanh(2.6 * band)
    return _normalize(driven, 0.85)


def approaching_vehicle(duration: float = 3.0, sr: int = DEFAULT_SR, *, seed: int | None = 29) -> np.ndarray:
    """A vehicle closing distance: rising level plus a slight Doppler rise."""
    rng = _rng(seed)
    t = _t(duration, sr)
    base = engine_noise(duration, sr, rpm_hz=62.0, seed=seed).astype(np.float64)
    tyre = _bandpass(rng.standard_normal(t.size), 500.0, 4000.0, sr) * 0.4
    approach_db = -26.0 + 26.0 * (t / max(t[-1], 1e-6))
    gain = 10.0 ** (approach_db / 20.0)
    return _normalize((base + tyre) * gain, 0.9)


def ambient(duration: float = 3.0, sr: int = DEFAULT_SR, *, seed: int | None = 31) -> np.ndarray:
    """Quiet room tone."""
    rng = _rng(seed)
    n = int(duration * sr)
    hum = _onepole_lowpass(rng.standard_normal(n), 700.0, sr)
    return _normalize(hum, 0.12)


#: Registry so tests and the demo scene can look up a generator by category key.
#: Each entry takes ``(duration, sr)`` as keywords and returns float32 mono.
GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    "human_speech": speech,
    "crowd_speech": crowd_speech,
    "engine_noise": engine_noise,
    "general_traffic": general_traffic,
    "vehicle_horn": vehicle_horn,
    "emergency_siren": emergency_siren,
    "alarm": alarm,
    "warning_signal": warning_signal,
    "bicycle_bell": bicycle_bell,
    "music": music,
    "mechanical_noise": mechanical_noise,
    "announcement": announcement,
    "approaching_vehicle": approaching_vehicle,
    "ambient": ambient,
}


def generate(category: str, duration: float = 3.0, sr: int = DEFAULT_SR) -> np.ndarray:
    """Synthesize one isolated source by taxonomy key."""
    fn = GENERATORS.get(category)
    if fn is None:
        raise KeyError(f"no generator for category {category!r}")
    return fn(duration=duration, sr=sr)


# --------------------------------------------------------------------------
# Multi-source scenes
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SceneLayer:
    """One source inside a scene, with its schedule and level."""

    category: str
    gain_db: float = -12.0
    start: float = 0.0
    end: float | None = None
    fade: float = 0.15

    def envelope(self, duration: float, sr: int) -> np.ndarray:
        n = int(duration * sr)
        t = np.arange(n) / sr
        end = duration if self.end is None else self.end
        env = ((t >= self.start) & (t < end)).astype(np.float64)
        if self.fade > 0:
            f = max(1, int(self.fade * sr))
            kernel = np.hanning(2 * f + 1)
            kernel /= kernel.sum()
            env = np.convolve(env, kernel, mode="same")
        return env


def render_scene(
    layers: list[SceneLayer],
    duration: float,
    sr: int = DEFAULT_SR,
    *,
    return_stems: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Render a multi-source scene, optionally returning per-source stems.

    The stems are what make the demo honest: the UI can show, and the tests can
    assert, exactly how much of each true source survived the mix.
    """
    n = int(duration * sr)
    mix = np.zeros(n, dtype=np.float64)
    stems: dict[str, np.ndarray] = {}

    for layer in layers:
        src = generate(layer.category, duration=duration, sr=sr).astype(np.float64)
        if src.size < n:
            src = np.tile(src, int(np.ceil(n / src.size)))
        src = src[:n]
        signal = src * layer.envelope(duration, sr) * (10.0 ** (layer.gain_db / 20.0))
        stems.setdefault(layer.category, np.zeros(n))
        stems[layer.category] += signal
        mix += signal

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        scale = 0.98 / peak
        mix *= scale
        for k in stems:
            stems[k] *= scale

    mix_f = mix.astype(np.float32)
    if return_stems:
        return mix_f, {k: v.astype(np.float32) for k, v in stems.items()}
    return mix_f


# --------------------------------------------------------------------------
# Spoken words (formant trajectories)
# --------------------------------------------------------------------------

#: Vowel formant targets (F1, F2, F3) in Hz, from standard measurements.
#: A word is a path through this space -- which is exactly what a keyword
#: spotter matches on, and what fixed-formant synthesis cannot provide.
VOWELS: dict[str, tuple[float, float, float]] = {
    "iy": (270.0, 2290.0, 3010.0),   # beet
    "ih": (390.0, 1990.0, 2550.0),   # bit
    "eh": (530.0, 1840.0, 2480.0),   # bet
    "ae": (660.0, 1720.0, 2410.0),   # bat
    "aa": (730.0, 1090.0, 2440.0),   # father
    "ao": (570.0, 840.0, 2410.0),    # bought
    "ow": (450.0, 1030.0, 2380.0),   # boat
    "uh": (440.0, 1020.0, 2240.0),   # book
    "uw": (300.0, 870.0, 2240.0),    # boot
    "er": (490.0, 1350.0, 1690.0),   # bird
    "ah": (640.0, 1190.0, 2390.0),   # but
}

#: Rough vowel skeletons for the names used in demos and tests.
WORD_VOWELS: dict[str, tuple[str, ...]] = {
    "vince": ("ih", "eh"),
    "maria": ("ah", "iy", "ah"),
    "ben": ("eh", "ah"),
    "ana": ("aa", "ah"),
    "hello": ("eh", "ow"),
    "stop": ("aa", "ah"),
    "next-station": ("eh", "ih", "ey" if "ey" in VOWELS else "eh", "ah"),
}


def spoken_word(
    vowels: tuple[str, ...] | str,
    duration: float = 0.7,
    sr: int = DEFAULT_SR,
    *,
    f0: float = 120.0,
    seed: int | None = 5,
    rate_jitter: float = 0.0,
) -> np.ndarray:
    """Synthesize a word as a trajectory through vowel space.

    Unlike :func:`speech`, whose formants are fixed for the whole clip, this
    glides between vowel targets. That trajectory is the thing a query-by-
    example keyword spotter actually matches, so it is what makes Name
    Awareness testable without shipping recorded audio.
    """
    if isinstance(vowels, str):
        vowels = WORD_VOWELS.get(vowels.lower(), ("ah",))
    targets: list[tuple[float, float, float]] = [VOWELS.get(v, VOWELS["ah"]) for v in vowels]
    if not targets:
        targets = [VOWELS["ah"]]

    rng = _rng(seed)
    t = _t(duration, sr)
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Glottal excitation with jitter and a gentle falling intonation.
    contour = f0 * (1.0 + 0.06 * np.cos(np.pi * t / max(t[-1], 1e-6)))
    jitter = 1.0 + 0.015 * _onepole_lowpass(rng.standard_normal(n), 10.0, sr)
    phase = np.cumsum(2.0 * np.pi * contour * jitter / sr)
    excitation = np.zeros(n)
    for k in range(1, 26):
        if k * f0 > sr / 2.2:
            break
        excitation += np.sin(k * phase) / (k ** 1.15)
    excitation += 0.05 * rng.standard_normal(n)

    # Anchor each vowel at its own point in time, then glide between them.
    anchors = np.linspace(0.0, 1.0, len(targets)) if len(targets) > 1 else np.array([0.5])
    if rate_jitter:
        anchors = np.clip(anchors + rng.uniform(-rate_jitter, rate_jitter, anchors.size), 0.0, 1.0)
        anchors = np.sort(anchors)
    pos = t / max(t[-1], 1e-6)

    voiced = np.zeros(n)
    for fi in range(3):
        track = np.interp(pos, anchors, [tg[fi] for tg in targets])
        # Piecewise-constant resonators over short segments approximate a
        # time-varying filter at a fraction of the cost.
        seg = max(1, int(0.012 * sr))
        out = np.zeros(n)
        for start in range(0, n, seg):
            stop = min(start + seg, n)
            fhz = float(track[start:stop].mean())
            out[start:stop] = _resonator(excitation[start:stop], fhz, q=10.0 + 2.0 * fi, sr=sr)
        voiced += out / (1.0 + 0.5 * fi)

    # Syllable envelope: one raised-cosine lobe per vowel.
    env = np.zeros(n)
    for a in anchors:
        env += np.exp(-0.5 * ((pos - a) / (0.42 / len(targets))) ** 2)
    env = np.clip(env, 0.0, 1.5)
    # Fade the edges so the word does not click.
    edge = max(1, int(0.02 * sr))
    env[:edge] *= np.linspace(0.0, 1.0, edge)
    env[-edge:] *= np.linspace(1.0, 0.0, edge)

    return _normalize(voiced * env, 0.85)


def utterance(
    words: list[str],
    sr: int = DEFAULT_SR,
    *,
    f0: float = 120.0,
    gap: float = 0.18,
    seed: int | None = 5,
) -> np.ndarray:
    """Concatenate spoken words with pauses, for Name Awareness tests."""
    rng = _rng(seed)
    parts: list[np.ndarray] = []
    for w in words:
        parts.append(
            spoken_word(w, duration=float(rng.uniform(0.55, 0.8)), sr=sr,
                        f0=f0, seed=int(rng.integers(0, 10_000)), rate_jitter=0.05)
        )
        parts.append(np.zeros(int(gap * sr), dtype=np.float32))
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

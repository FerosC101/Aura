"""Streaming acoustic feature extraction.

Everything here is designed for *streaming*: each call handles one short
overlapping frame and updates a small amount of recurrent state (spectral flux,
amplitude-modulation history, tonal-peak track). No feature requires seeing a
long file, which is what keeps the classifier usable inside a real-time
callback.

The features are deliberately model-agnostic. They feed the built-in DSP
classifier, but they are also useful context for neural backends (level, onset,
modulation depth), which is why they live in their own module.

Two design notes worth keeping in mind when extending this file:

* **Flatness is computed over log-spaced band power densities, not raw FFT
  bins.** Bin-wise spectral flatness saturates near zero for virtually every
  real signal because of the spectrum's dynamic range; per-band PSD restores a
  usable 0-1 spread where white noise ~ 1 and a pure tone ~ 0.
* **Pitch movement is tracked on the dominant spectral peak, not on the
  autocorrelation f0.** Autocorrelation octave-errors badly on the loud tonal
  events (horns, sirens) that matter most, and a siren is defined precisely by
  its peak frequency sweeping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-10

#: Number of log-spaced analysis bands used for flatness / band-power output.
N_LOG_BANDS = 32
LOG_BAND_LO = 50.0
LOG_BAND_HI = 8000.0


@dataclass(slots=True)
class FrameFeatures:
    """Acoustic descriptors for one analysis frame."""

    # Level
    rms: float = 0.0
    dbfs: float = -90.0
    peak: float = 0.0
    crest_factor: float = 1.0

    # Spectral shape
    centroid_hz: float = 0.0
    rolloff_hz: float = 0.0
    bandwidth_hz: float = 0.0
    flatness: float = 1.0  # 1.0 = noise-like, ~0 = tonal (log-band PSD based)
    entropy: float = 1.0  # normalised spectral entropy over log bands
    slope: float = 0.0
    zcr: float = 0.0

    # Band energy ratios (fractions of total energy)
    band_energy: dict[str, float] = field(default_factory=dict)

    # Tonality / harmonicity
    tonality: float = 0.0  # peak prominence over the notched local floor
    harmonicity: float = 0.0  # autocorrelation periodicity strength
    f0_hz: float = 0.0
    voicing: float = 0.0  # harmonicity gated to a plausible human f0 range
    n_tonal_peaks: int = 0
    tonal_peak_hz: float = 0.0
    tonal_persistence: float = 0.0  # fraction of recent frames that were tonal

    # Dynamics (recurrent)
    flux: float = 0.0  # positive spectral flux (onset strength)
    flux_variance: float = 0.0
    level_slope_db_s: float = 0.0  # rising level => something approaching
    modulation_depth: float = 0.0  # 2-10 Hz envelope modulation (speech-like)
    modulation_rate_hz: float = 0.0
    peak_drift_hz: float = 0.0  # std-dev of the dominant tonal peak
    peak_range_hz: float = 0.0  # full sweep of the dominant tonal peak
    pitch_variation_hz: float = 0.0  # std-dev of tracked f0 (voiced frames)
    stationarity: float = 0.0  # 1.0 = unchanging spectrum

    # Raw spectrum + band powers, retained for classifiers and the UI.
    magnitude: np.ndarray | None = None
    freqs: np.ndarray | None = None
    log_band_db: np.ndarray | None = None


# Named analysis bands (Hz). Chosen to line up with the taxonomy's band_hz.
BANDS: dict[str, tuple[float, float]] = {
    "sub": (20.0, 80.0),
    "low": (80.0, 250.0),
    "low_mid": (250.0, 800.0),
    "mid": (800.0, 2000.0),
    "high_mid": (2000.0, 4000.0),
    "high": (4000.0, 8000.0),
    "air": (8000.0, 20000.0),
}

# Speech intelligibility band, used repeatedly.
SPEECH_BAND = (200.0, 3500.0)


class FeatureExtractor:
    """Stateful, streaming feature extractor.

    Parameters
    ----------
    sample_rate:
        Input rate of the frames that will be passed to :meth:`extract`.
    history:
        Number of past frames kept for modulation / flux / peak-track
        statistics. With a 20 ms hop, 48 frames is roughly one second -- long
        enough to see speech syllable rate (~4 Hz) and siren warble (~0.5 Hz)
        without adding latency, because features are always reported for the
        *current* frame.
    """

    def __init__(self, sample_rate: int, history: int = 48) -> None:
        self.sample_rate = sample_rate
        self.history = history
        self._prev_mag: np.ndarray | None = None
        self._env_hist: deque[float] = deque(maxlen=history)
        self._flux_hist: deque[float] = deque(maxlen=history)
        self._f0_hist: deque[float] = deque(maxlen=history)
        self._db_hist: deque[float] = deque(maxlen=history)
        self._peak_hist: deque[float] = deque(maxlen=history)
        self._tonal_hist: deque[float] = deque(maxlen=history)
        self._window: np.ndarray | None = None
        self._band_slices: list[tuple[int, int]] | None = None
        self._band_index_n: int = -1
        self._hop_seconds: float = 0.02

    def reset(self) -> None:
        self._prev_mag = None
        for d in (
            self._env_hist, self._flux_hist, self._f0_hist,
            self._db_hist, self._peak_hist, self._tonal_hist,
        ):
            d.clear()

    # -- main entry point --------------------------------------------------

    def extract(self, samples: np.ndarray) -> FrameFeatures:
        n = samples.size
        if n == 0:
            return FrameFeatures()

        self._hop_seconds = n / self.sample_rate
        f = FrameFeatures()

        # ---- level ----
        x = samples.astype(np.float32, copy=False)
        f.rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + EPS))
        f.peak = float(np.max(np.abs(x)))
        f.dbfs = 20.0 * float(np.log10(max(f.rms, 1e-9)))
        f.crest_factor = float(f.peak / max(f.rms, EPS))
        f.zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))

        # ---- spectrum (Hann-windowed rFFT) ----
        if self._window is None or self._window.size != n:
            self._window = np.hanning(n).astype(np.float32)
        spec = np.fft.rfft(x * self._window)
        mag = np.abs(spec).astype(np.float32)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate).astype(np.float32)
        f.magnitude = mag
        f.freqs = freqs

        power = np.square(mag, dtype=np.float64)
        total = float(power.sum()) + EPS

        # ---- spectral shape ----
        f.centroid_hz = float((freqs * power).sum() / total)
        f.bandwidth_hz = float(np.sqrt(((freqs - f.centroid_hz) ** 2 * power).sum() / total))
        cumulative = np.cumsum(power)
        idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
        f.rolloff_hz = float(freqs[min(idx, freqs.size - 1)])

        mask = (freqs > 60.0) & (freqs < 10000.0)
        if mask.sum() > 4:
            lf = np.log10(freqs[mask])
            lp = np.log10(power[mask] + EPS)
            f.slope = float(np.polyfit(lf, lp, 1)[0])

        # ---- log-band PSD, flatness and entropy ----
        psd = self._log_band_psd(power, freqs, n)
        f.log_band_db = (10.0 * np.log10(psd + 1e-18)).astype(np.float32)
        gmean = float(np.exp(np.mean(np.log(psd + 1e-18))))
        amean = float(np.mean(psd)) + 1e-18
        f.flatness = float(np.clip(gmean / amean, 0.0, 1.0))
        p = psd / (psd.sum() + 1e-18)
        f.entropy = float(-np.sum(p * np.log(p + 1e-18)) / np.log(len(p)))

        # ---- band energies ----
        for name, (lo, hi) in BANDS.items():
            bmask = (freqs >= lo) & (freqs < hi)
            f.band_energy[name] = float(power[bmask].sum() / total) if bmask.any() else 0.0
        smask = (freqs >= SPEECH_BAND[0]) & (freqs < SPEECH_BAND[1])
        f.band_energy["speech"] = float(power[smask].sum() / total) if smask.any() else 0.0
        f.band_energy["low_total"] = f.band_energy["sub"] + f.band_energy["low"]
        f.band_energy["hf_total"] = f.band_energy["high"] + f.band_energy["air"]

        # ---- tonality ----
        f.tonality, f.n_tonal_peaks, f.tonal_peak_hz = self._tonality(mag, freqs)

        # ---- harmonicity + f0 ----
        f.harmonicity, f.f0_hz = self._harmonicity(x)
        # Voicing = periodicity that is actually in the human pitch range.
        in_range = 1.0 if 70.0 <= f.f0_hz <= 350.0 else 0.0
        f.voicing = f.harmonicity * in_range

        # ---- recurrent dynamics ----
        if self._prev_mag is not None and self._prev_mag.size == mag.size:
            diff = mag - self._prev_mag
            f.flux = float(np.sum(np.maximum(diff, 0.0)) / (mag.size * (np.mean(mag) + EPS)))
            denom = float(np.linalg.norm(mag) * np.linalg.norm(self._prev_mag)) + EPS
            f.stationarity = float(np.dot(mag, self._prev_mag) / denom)
        self._prev_mag = mag

        self._flux_hist.append(f.flux)
        self._env_hist.append(f.rms)
        self._db_hist.append(f.dbfs)
        self._f0_hist.append(f.f0_hz)
        self._tonal_hist.append(f.tonality)
        # Only track the peak when there genuinely is a tonal peak to track,
        # otherwise noise would look like a wildly sweeping siren.
        self._peak_hist.append(f.tonal_peak_hz if f.tonality > 0.45 else float("nan"))

        if len(self._flux_hist) > 4:
            f.flux_variance = float(np.var(np.asarray(self._flux_hist, dtype=np.float64)))
        f.modulation_depth, f.modulation_rate_hz = self._modulation()
        f.level_slope_db_s = self._level_slope()
        f.pitch_variation_hz = self._pitch_variation()
        f.peak_drift_hz, f.peak_range_hz = self._peak_track()
        if self._tonal_hist:
            arr = np.asarray(self._tonal_hist, dtype=np.float64)
            f.tonal_persistence = float(np.mean(arr > 0.5))

        return f

    # -- internals ---------------------------------------------------------

    def _log_band_psd(self, power: np.ndarray, freqs: np.ndarray, n: int) -> np.ndarray:
        """Power density per log-spaced band.

        Dividing by the bin count per band turns this into a density, so white
        noise -- whose per-band *energy* grows with bandwidth -- comes out flat.
        """
        if self._band_slices is None or self._band_index_n != n:
            edges = np.geomspace(LOG_BAND_LO, min(LOG_BAND_HI, self.sample_rate / 2 - 1),
                                 N_LOG_BANDS + 1)
            slices = []
            for i in range(N_LOG_BANDS):
                lo = int(np.searchsorted(freqs, edges[i], side="left"))
                hi = int(np.searchsorted(freqs, edges[i + 1], side="left"))
                if hi <= lo:
                    hi = min(lo + 1, freqs.size)
                slices.append((lo, hi))
            self._band_slices = slices
            self._band_index_n = n

        out = np.empty(N_LOG_BANDS, dtype=np.float64)
        for i, (lo, hi) in enumerate(self._band_slices):
            seg = power[lo:hi]
            out[i] = float(seg.mean()) if seg.size else 0.0
        return out

    def _tonality(self, mag: np.ndarray, freqs: np.ndarray) -> tuple[float, int, float]:
        """Strength and count of narrowband peaks above the local noise floor.

        The background estimate uses a *notched* moving average -- the kernel
        has a hole at its centre -- so a bin never contributes to its own
        floor. Without the notch a strong tone drags its own reference up and
        scores as noise, which is exactly the horn/siren case we must not miss.
        """
        if mag.size < 16:
            return 0.0, 0, 0.0

        bin_hz = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
        k = max(9, int(round(600.0 / max(bin_hz, 1e-6)))) | 1
        k = min(k, (mag.size // 2) | 1)
        if k < 5:
            return 0.0, 0, 0.0
        kernel = np.ones(k, dtype=np.float32)
        centre = k // 2
        notch = max(1, int(round(120.0 / max(bin_hz, 1e-6))))
        kernel[max(0, centre - notch): centre + notch + 1] = 0.0
        ksum = float(kernel.sum())
        if ksum <= 0:
            return 0.0, 0, 0.0
        kernel /= ksum

        floor = np.convolve(mag, kernel, mode="same") + EPS
        ratio = mag / floor

        band = (freqs >= 150.0) & (freqs <= 8000.0)
        if not band.any():
            return 0.0, 0, 0.0
        r = ratio[band]
        fb = freqs[band]
        mb = mag[band]

        energy_gate = mb > (0.2 * float(mb.max()) + EPS)
        peaks = (r > 8.0) & energy_gate
        n_peaks = int(np.count_nonzero(peaks))
        max_ratio = float(np.max(r[energy_gate])) if energy_gate.any() else 1.0
        strength = float(np.clip(np.log10(max(max_ratio, 1.0)) / 2.0, 0.0, 1.0))
        peak_hz = float(fb[int(np.argmax(mb))]) if mb.size else 0.0
        return strength, n_peaks, peak_hz

    def _harmonicity(self, x: np.ndarray) -> tuple[float, float]:
        """Normalised autocorrelation peak in the 60-500 Hz f0 range."""
        n = x.size
        if n < 256:
            return 0.0, 0.0
        sr = self.sample_rate
        min_lag = max(2, int(sr / 500.0))
        max_lag = min(n - 1, int(sr / 60.0))
        if max_lag <= min_lag:
            return 0.0, 0.0

        xm = x - float(np.mean(x))
        energy = float(np.dot(xm, xm)) + EPS
        size = 1 << int(np.ceil(np.log2(2 * n)))
        spec = np.fft.rfft(xm, size)
        ac = np.fft.irfft(spec * np.conj(spec), size)[: max_lag + 1]
        ac /= energy

        seg = ac[min_lag: max_lag + 1]
        if seg.size == 0:
            return 0.0, 0.0
        best = int(np.argmax(seg))
        strength = float(np.clip(seg[best], 0.0, 1.0))
        lag = min_lag + best
        # Reject lags pinned to the search boundary: they are artefacts of the
        # window length, not real pitch.
        if lag >= max_lag - 1 or lag <= min_lag + 1:
            return strength * 0.4, 0.0
        f0 = float(sr / lag)
        return strength, f0 if strength > 0.25 else 0.0

    def _modulation(self) -> tuple[float, float]:
        """Depth and rate of amplitude modulation in the 1.5-10 Hz band.

        Speech has a strong ~4 Hz syllable rate; steady noise does not. This is
        the single most discriminative cue separating speech from traffic.
        """
        if len(self._env_hist) < 12:
            return 0.0, 0.0
        env = np.asarray(self._env_hist, dtype=np.float64)
        mean = float(env.mean())
        if mean < 1e-6:
            return 0.0, 0.0
        env = env - mean
        frame_rate = 1.0 / max(self._hop_seconds, 1e-4)

        spec = np.abs(np.fft.rfft(env * np.hanning(env.size)))
        mfreqs = np.fft.rfftfreq(env.size, 1.0 / frame_rate)
        band = (mfreqs >= 1.5) & (mfreqs <= 10.0)
        if not band.any() or spec.sum() < EPS:
            return 0.0, 0.0
        bspec = spec[band]
        bfreq = mfreqs[band]
        peak = int(np.argmax(bspec))
        depth = float(np.clip(2.0 * bspec[peak] / (env.size * mean + EPS), 0.0, 1.0))
        return depth, float(bfreq[peak])

    def _level_slope(self) -> float:
        """dB per second trend of the recent level -- positive = approaching."""
        if len(self._db_hist) < 8:
            return 0.0
        db = np.asarray(self._db_hist, dtype=np.float64)
        t = np.arange(db.size) * self._hop_seconds
        if t[-1] <= 0:
            return 0.0
        slope = float(np.polyfit(t, db, 1)[0])
        return float(np.clip(slope, -200.0, 200.0))

    def _pitch_variation(self) -> float:
        """Std-dev of tracked f0 over voiced frames."""
        voiced = [v for v in self._f0_hist if v > 0.0]
        if len(voiced) < 6:
            return 0.0
        return float(np.std(np.asarray(voiced, dtype=np.float64)))

    def _peak_track(self) -> tuple[float, float]:
        """Drift and total sweep of the dominant tonal peak.

        A siren sweeps hundreds of Hz; a horn holds one pitch. This pair is the
        cue that actually separates them, and it works where autocorrelation
        pitch tracking octave-errors.
        """
        arr = np.asarray(self._peak_hist, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if valid.size < 5:
            return 0.0, 0.0
        return float(np.std(valid)), float(np.max(valid) - np.min(valid))


def band_energy(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """Fraction of spectral energy inside ``[lo, hi)``."""
    power = np.square(mag, dtype=np.float64)
    total = float(power.sum()) + EPS
    mask = (freqs >= lo) & (freqs < hi)
    return float(power[mask].sum() / total)

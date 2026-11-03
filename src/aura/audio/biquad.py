"""Biquad filters for zero-latency spectral shaping.

The mixer has to attenuate engine rumble while leaving a horn untouched, and
both live in the same separated stream. That means frequency-dependent gain.

A second STFT stage would be the easy way to get it and would cost another hop
of delay on top of the separator's. Cascaded RBJ peaking filters give the same
shaping with *no* added latency, at the price of non-linear phase -- an
acceptable trade for a live monitoring path, and the one real audio products
make.

Coefficients are recomputed whenever a band's gain moves, and the gains
themselves are slewed by the caller, so the response never jumps.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt

#: Centre frequencies of the shaping bands, in Hz. Chosen to line up with the
#: taxonomy's ``band_hz``: sub/engine, low body, speech fundamentals, speech
#: intelligibility, horn/siren range, and air.
SHAPER_BANDS: tuple[float, ...] = (60.0, 160.0, 400.0, 1000.0, 2500.0, 6000.0)
SHAPER_Q = 0.9


def peaking_sos(freq: float, q: float, gain_db: float, sample_rate: int) -> np.ndarray:
    """RBJ peaking-EQ biquad as a single second-order section."""
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * min(freq, sample_rate * 0.45) / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * a_
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_
    a0 = 1.0 + alpha / a_
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0], dtype=np.float64)


class MultibandShaper:
    """A cascade of peaking filters with independently settable band gains.

    Filter state is carried across blocks so there are no discontinuities at
    block boundaries, and coefficients are only rebuilt when a gain actually
    moves by an audible amount.
    """

    #: Rebuild threshold, in dB. Below this a gain change is inaudible and not
    #: worth recomputing six sets of coefficients for.
    REBUILD_EPS = 0.05

    def __init__(
        self,
        sample_rate: int,
        bands: tuple[float, ...] = SHAPER_BANDS,
        q: float = SHAPER_Q,
    ) -> None:
        self.sample_rate = sample_rate
        self.bands = bands
        self.q = q
        self._gains = np.zeros(len(bands), dtype=np.float64)
        self._sos = self._build(self._gains)
        self._zi = np.zeros((len(bands), 2), dtype=np.float64)
        self._target = np.zeros(len(bands), dtype=np.float64)
        self._decouple = self._interaction_inverse()

    def _interaction_inverse(self) -> np.ndarray:
        """Inverse of the band-interaction matrix.

        Neighbouring peaking filters overlap, so setting six bands to -18 dB
        yields about -23 dB at each centre. The response is very nearly linear
        in dB for moderate gains, so measuring the cross-coupling once and
        inverting it lets :meth:`set_gains` hit its targets: ask for -18 dB and
        measure -18 dB.
        """
        from scipy.signal import sosfreqz

        n = len(self.bands)
        probe = 6.0  # dB; large enough to measure cleanly, small enough to stay linear
        w = 2.0 * np.pi * np.asarray(self.bands, dtype=np.float64) / self.sample_rate
        m = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            sos = np.stack([
                peaking_sos(f, self.q, probe if k == j else 0.0, self.sample_rate)
                for k, f in enumerate(self.bands)
            ])
            _, h = sosfreqz(sos, worN=w)
            m[:, j] = 20.0 * np.log10(np.abs(h) + 1e-12) / probe
        try:
            return np.linalg.inv(m)
        except np.linalg.LinAlgError:
            return np.eye(n)

    def _build(self, gains: np.ndarray) -> np.ndarray:
        return np.stack([
            peaking_sos(f, self.q, float(g), self.sample_rate)
            for f, g in zip(self.bands, gains, strict=True)
        ])

    def reset(self) -> None:
        self._zi[:] = 0.0

    @property
    def gains_db(self) -> np.ndarray:
        """The requested response at each band centre, in dB."""
        return self._target.copy()

    def set_gains(self, gains_db: np.ndarray) -> None:
        g = np.clip(np.asarray(gains_db, dtype=np.float64), -40.0, 12.0)
        if g.size != len(self.bands):
            raise ValueError(f"expected {len(self.bands)} band gains, got {g.size}")
        if np.max(np.abs(g - self._target)) < self.REBUILD_EPS:
            return
        self._target = g
        # Solve for the raw filter gains that produce the requested response.
        raw = np.clip(self._decouple @ g, -48.0, 18.0)
        self._gains = raw
        self._sos = self._build(raw)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if np.max(np.abs(self._target)) < self.REBUILD_EPS:
            return x  # Flat response: skip the filtering entirely.
        y, self._zi = sosfilt(self._sos, x.astype(np.float64), zi=self._zi)
        return y

    def response_db(self, freqs: np.ndarray) -> np.ndarray:
        """Magnitude response at ``freqs``, for the UI's shaping curve."""
        from scipy.signal import sosfreqz

        w = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64) / self.sample_rate
        _, h = sosfreqz(self._sos, worN=w)
        return 20.0 * np.log10(np.abs(h) + 1e-12)


class SoftLimiter:
    """Look-ahead-free soft limiter protecting the output from boosts.

    Person Lock, Name Awareness and the safety lift can all add gain at once.
    Without a limiter those stack into clipping, which would be both unpleasant
    and, for a safety sound, actively counterproductive.
    """

    def __init__(self, sample_rate: int, threshold_db: float = -1.0,
                 release_ms: float = 80.0) -> None:
        self.threshold = 10.0 ** (threshold_db / 20.0)
        self.release = float(np.exp(-1.0 / (sample_rate * release_ms / 1000.0)))
        self._gain = 1.0
        self.active = False

    def reset(self) -> None:
        self._gain = 1.0
        self.active = False

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        self.active = peak > self.threshold
        target = self.threshold / peak if self.active else 1.0
        # Attack instantly, release smoothly.
        if target < self._gain:
            self._gain = target
        else:
            self._gain = target + (self._gain - target) * self.release
        return (x * self._gain).astype(np.float32)

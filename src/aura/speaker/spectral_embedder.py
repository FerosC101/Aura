"""Dependency-free speaker embedding from spectral statistics.

Computes MFCC-style features with a NumPy mel filterbank, then summarises an
utterance by the mean and standard deviation of its cepstra and deltas. Cepstral
mean normalisation is applied so that the microphone and room contribute less
than the speaker does.

This will not match a trained x-vector network, and the docs say so plainly. It
does reliably separate a small enrolled set, which is what Person Lock needs for
the MVP, and it keeps the default install free of PyTorch.
"""

from __future__ import annotations

import numpy as np

from .base import SpeakerIdentifier

EPS = 1e-10


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_filters: int, n_fft: int, sample_rate: int,
                   fmin: float = 60.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filterbank, shape ``(n_filters, n_fft // 2 + 1)``."""
    fmax = fmax or sample_rate / 2.0
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz_points = np.asarray(mel_to_hz(mel_points), dtype=np.float64)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    fb = np.zeros((n_filters, n_bins), dtype=np.float64)
    for i in range(n_filters):
        left, centre, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        if centre == left:
            centre = min(left + 1, n_bins - 1)
        if right == centre:
            right = min(centre + 1, n_bins - 1)
        for b in range(left, centre):
            fb[i, b] = (b - left) / max(centre - left, 1)
        for b in range(centre, right):
            fb[i, b] = (right - b) / max(right - centre, 1)
    # Area-normalise so wide high-frequency filters do not dominate.
    areas = fb.sum(axis=1, keepdims=True)
    return fb / np.maximum(areas, EPS)


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II matrix."""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    m = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    m[0] *= 1.0 / np.sqrt(2.0)
    return m * np.sqrt(2.0 / n_in)


class SpectralSpeakerEmbedder(SpeakerIdentifier):
    """MFCC statistics embedder."""

    name = "mfcc-stats-v1"
    min_audio_seconds = 1.0
    # Measured on held-out synthetic and recorded speech; see
    # scripts/eval_speaker.py, which regenerates these numbers.
    impostor_prior = (0.953, 0.025)

    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        n_mels: int = 40,
        n_cepstra: int = 20,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_cepstra = n_cepstra
        self.frame_len = int(sample_rate * frame_ms / 1000.0)
        self.hop_len = int(sample_rate * hop_ms / 1000.0)
        self.n_fft = 1 << int(np.ceil(np.log2(self.frame_len)))
        self._fb = mel_filterbank(n_mels, self.n_fft, sample_rate)
        self._dct = dct_matrix(n_cepstra, n_mels)
        self._window = np.hamming(self.frame_len)
        # cepstral mean + spread + delta spread (c0 excluded), plus 3 pitch stats.
        self.dim = 3 * (n_cepstra - 1) + 3

    # -- SpeakerIdentifier -------------------------------------------------

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """Summarise a voice as a standardised statistics vector.

        The vector concatenates three complementary views of the speaker:

        * **Cepstral means** -- average vocal-tract shape. This is the single
          most speaker-specific part, so unlike a channel-robust ASR front-end
          we deliberately keep it rather than normalising it away.
        * **Cepstral and delta spreads** -- how much the speaker moves their
          articulators, which captures speaking style.
        * **Pitch statistics** -- log-f0 mean and spread. Pitch separates voices
          that happen to share a similar vocal-tract length, and it is cheap.

        The result is standardised (zero mean, unit variance across its own
        dimensions) before normalising. Without that step every embedding is a
        vector of similar-magnitude positive numbers and cosine similarity
        saturates near 1.0 for everyone.
        """
        x = np.asarray(samples, dtype=np.float64).ravel()
        if sample_rate != self.sample_rate:
            x = self._resample(x, sample_rate, self.sample_rate)
        if x.size < int(self.min_audio_seconds * self.sample_rate * 0.5):
            return None

        cepstra = self._mfcc(x)
        if cepstra is None or cepstra.shape[0] < 4:
            return None

        # Drop low-energy frames: silence carries no speaker identity and its
        # statistics would otherwise swamp a short enrolment clip.
        energy = cepstra[:, 0]
        keep = energy > (np.max(energy) - 25.0)
        if keep.sum() >= 4:
            cepstra = cepstra[keep]

        # c0 is frame loudness, not identity -- drop it from the mean term so
        # the embedding does not change when the speaker moves closer.
        shape = cepstra[:, 1:]
        deltas = np.diff(cepstra, axis=0) if cepstra.shape[0] > 1 else np.zeros_like(cepstra)

        f0_mean, f0_std, voiced = self._pitch_stats(x)

        vec = np.concatenate([
            shape.mean(axis=0),
            shape.std(axis=0),
            deltas[:, 1:].std(axis=0),
            [f0_mean, f0_std, voiced],
        ])
        if not np.all(np.isfinite(vec)):
            return None

        # Standardise within the vector, then L2-normalise. This turns cosine
        # similarity into a correlation, which actually discriminates.
        vec = vec - vec.mean()
        spread = float(vec.std())
        if spread < 1e-9:
            return None
        vec = vec / spread
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return (vec / norm).astype(np.float32)

    def _pitch_stats(self, x: np.ndarray) -> tuple[float, float, float]:
        """Log-f0 mean, log-f0 spread and voiced fraction over the clip."""
        sr = self.sample_rate
        win = int(0.04 * sr)
        hop = int(0.02 * sr)
        if x.size < win:
            return 0.0, 0.0, 0.0
        min_lag, max_lag = max(2, int(sr / 350.0)), min(win - 1, int(sr / 70.0))
        if max_lag <= min_lag + 1:
            return 0.0, 0.0, 0.0

        f0s = []
        n = 1 + (x.size - win) // hop
        for i in range(n):
            seg = x[i * hop: i * hop + win]
            seg = seg - seg.mean()
            e = float(np.dot(seg, seg))
            if e < 1e-9:
                continue
            size = 1 << int(np.ceil(np.log2(2 * win)))
            sp = np.fft.rfft(seg, size)
            ac = np.fft.irfft(sp * np.conj(sp), size)[: max_lag + 1] / e
            band = ac[min_lag: max_lag + 1]
            b = int(np.argmax(band))
            if band[b] > 0.35:
                f0s.append(sr / (min_lag + b))

        if not f0s:
            return 0.0, 0.0, 0.0
        logf = np.log(np.asarray(f0s))
        # Scaled so pitch is comparable in magnitude to the cepstral terms
        # rather than being washed out by them.
        return float(logf.mean() * 8.0), float(logf.std() * 8.0), float(len(f0s) / max(n, 1) * 8.0)

    # -- internals ---------------------------------------------------------

    def _mfcc(self, x: np.ndarray) -> np.ndarray | None:
        if x.size < self.frame_len:
            return None
        # Pre-emphasis lifts the higher formants that carry speaker identity.
        x = np.append(x[0], x[1:] - 0.97 * x[:-1])
        n_frames = 1 + (x.size - self.frame_len) // self.hop_len
        if n_frames < 1:
            return None
        idx = (np.arange(self.frame_len)[None, :]
               + self.hop_len * np.arange(n_frames)[:, None])
        frames = x[idx] * self._window
        spec = np.abs(np.fft.rfft(frames, self.n_fft, axis=1)) ** 2
        mel = spec @ self._fb.T
        log_mel = np.log(mel + EPS)
        return log_mel @ self._dct.T

    @staticmethod
    def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
        if src == dst or x.size == 0:
            return x
        n_out = int(round(x.size * dst / src))
        return np.interp(
            np.linspace(0.0, x.size - 1, n_out),
            np.arange(x.size),
            x,
        )

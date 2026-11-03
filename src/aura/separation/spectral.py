"""Real-time spectral source separation.

A weighted-overlap-add (WOLA) separator that splits a mono stream into four
signals that sum back to the input:

``primary_speech``
    The dominant harmonic voice -- bins lying on the strongest pitch comb.
``secondary_speech``
    Remaining speech-band energy: competing talkers and babble.
``transient``
    Prominent narrowband events -- horns, sirens, alarms, bells. Extracted
    separately so the mixer can hold them up while pushing everything around
    them down. This is the stream that makes "preserve the horn, suppress the
    engine" physically possible rather than a display label.
``environment``
    Everything else: engine rumble, road noise, room tone.

Design choices, and why
-----------------------
*Latency over separation quality*, as the MVP brief requires. There is no
lookahead: the analysis window only ever reaches backwards, so the added
algorithmic delay is exactly one hop.

*Square-root Hann analysis and synthesis windows at 50% overlap* satisfy the
COLA condition, so with unity masks the output is a bit-accurate delayed copy
of the input. The test suite asserts this; without it, "AURA off" and "AURA on
with everything preserved" would not sound the same, and every gain the mixer
applies would be measured against a distorted baseline.

*Noise tracking by recursive minimum statistics* rather than a trained VAD, so
the separator inherits no model dependency.
"""

from __future__ import annotations

import numpy as np

from ..types import SeparatedAudio, StreamKind
from .base import SourceSeparator

EPS = 1e-12


class WOLAProcessor:
    """Streaming STFT/ISTFT with square-root Hann windows at 50% overlap.

    Handles one hop at a time and can overlap-add an arbitrary number of masked
    copies of the same spectrum into independent output streams.
    """

    def __init__(self, hop: int, n_streams: int) -> None:
        self.hop = hop
        self.nfft = hop * 2
        # Periodic Hann; its square-root gives COLA-compliant analysis and
        # synthesis windows at 50% overlap.
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.nfft) / self.nfft)
        self.window = np.sqrt(hann).astype(np.float64)
        self._input = np.zeros(self.nfft, dtype=np.float64)
        self._tails = [np.zeros(self.hop, dtype=np.float64) for _ in range(n_streams)]

    def reset(self) -> None:
        self._input[:] = 0.0
        for t in self._tails:
            t[:] = 0.0

    def analyse(self, block: np.ndarray) -> np.ndarray:
        """Shift ``block`` into the window and return its spectrum."""
        self._input[: self.hop] = self._input[self.hop:]
        self._input[self.hop:] = block
        return np.fft.rfft(self._input * self.window)

    def synthesise(self, spectra: list[np.ndarray]) -> list[np.ndarray]:
        """Inverse-transform each masked spectrum and overlap-add it."""
        out: list[np.ndarray] = []
        for i, spec in enumerate(spectra):
            frame = np.fft.irfft(spec, self.nfft) * self.window
            y = frame[: self.hop] + self._tails[i]
            self._tails[i] = frame[self.hop:].copy()
            out.append(y)
        return out


class SpectralSourceSeparator(SourceSeparator):
    """Mask-based speech / transient / environment separator.

    Parameters
    ----------
    sample_rate, hop:
        Must match the engine's frame size.
    noise_adapt:
        Smoothing for the recursive noise-floor estimate. Lower adapts slower
        and is more robust to a talker who never pauses.
    """

    name = "spectral-wola-v1"

    # Speech is masked in this band; outside it everything is environment.
    SPEECH_LO_HZ = 150.0
    SPEECH_HI_HZ = 5500.0

    def __init__(
        self,
        sample_rate: int,
        hop: int,
        *,
        noise_adapt: float = 0.012,
        mask_floor: float = 0.06,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop = hop
        self.mask_floor = mask_floor
        self.noise_adapt = noise_adapt

        self._wola = WOLAProcessor(hop, n_streams=4)
        self.nfft = self._wola.nfft
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / sample_rate)
        n_bins = self.freqs.size

        self._noise_psd = np.full(n_bins, 1e-7, dtype=np.float64)
        self._prev_gain = np.zeros(n_bins, dtype=np.float64)
        self._prev_speech_psd = np.zeros(n_bins, dtype=np.float64)

        self._speech_band = (self.freqs >= self.SPEECH_LO_HZ) & (self.freqs <= self.SPEECH_HI_HZ)
        # Smooth band edges so the mask does not click at the boundary.
        self._band_weight = self._raised_band_weight()

        self._f0 = 0.0
        self._voicing = 0.0
        self.last_masks: dict[StreamKind, np.ndarray] = {}

    # -- SourceSeparator ---------------------------------------------------

    @property
    def streams(self) -> tuple[StreamKind, ...]:
        return (
            StreamKind.PRIMARY_SPEECH,
            StreamKind.SECONDARY_SPEECH,
            StreamKind.TRANSIENT,
            StreamKind.ENVIRONMENT,
        )

    @property
    def algorithmic_delay_samples(self) -> int:
        # One hop: the WOLA tail must be completed by the next block.
        return self.hop

    def reset(self) -> None:
        self._wola.reset()
        self._noise_psd[:] = 1e-7
        self._prev_gain[:] = 0.0
        self._prev_speech_psd[:] = 0.0
        self._f0 = 0.0
        self._voicing = 0.0

    def separate(self, block: np.ndarray) -> SeparatedAudio:
        x = np.asarray(block, dtype=np.float64)
        spec = self._wola.analyse(x)
        power = np.abs(spec) ** 2

        # Order matters. Narrowband events are extracted *first*, from the full
        # spectrum. Running the speech mask first routed sirens into the speech
        # stream -- a siren is a loud tone well above the noise floor, which is
        # exactly what a Wiener mask is built to keep -- and a siren in the
        # speech stream gets attenuated the moment the user reduces speech.
        f0, comb, voicing = self._pitch_analysis(x, power)
        transient_mask = self._transient_selector(
            power, voiced_comb=comb if voicing > 0.45 else None
        )

        # Everything the transient stream did not claim is available to speech.
        rest = 1.0 - transient_mask
        speech_gain = self._speech_mask(power) * rest * self._voicing_factor(voicing)

        primary_mask = speech_gain * comb
        secondary_mask = speech_gain * (1.0 - comb)
        environment_mask = rest - speech_gain

        # Retained for diagnostics: applying these same masks to known stems is
        # how the separation benchmark measures interference rejection.
        self.last_masks = {
            StreamKind.PRIMARY_SPEECH: primary_mask,
            StreamKind.SECONDARY_SPEECH: secondary_mask,
            StreamKind.TRANSIENT: transient_mask,
            StreamKind.ENVIRONMENT: environment_mask,
        }

        streams = self._wola.synthesise([
            spec * primary_mask,
            spec * secondary_mask,
            spec * transient_mask,
            spec * environment_mask,
        ])

        return SeparatedAudio(
            streams={
                StreamKind.PRIMARY_SPEECH: streams[0].astype(np.float32),
                StreamKind.SECONDARY_SPEECH: streams[1].astype(np.float32),
                StreamKind.TRANSIENT: streams[2].astype(np.float32),
                StreamKind.ENVIRONMENT: streams[3].astype(np.float32),
            },
            sample_rate=self.sample_rate,
        )

    # -- mask estimation ---------------------------------------------------

    def _raised_band_weight(self) -> np.ndarray:
        """Cosine-tapered speech-band weighting."""
        w = np.zeros_like(self.freqs, dtype=np.float64)
        lo, hi = self.SPEECH_LO_HZ, self.SPEECH_HI_HZ
        taper = 250.0
        for i, f in enumerate(self.freqs):
            if f < lo - taper or f > hi + taper:
                w[i] = 0.0
            elif f < lo:
                w[i] = 0.5 * (1.0 - np.cos(np.pi * (f - (lo - taper)) / taper))
            elif f > hi:
                w[i] = 0.5 * (1.0 + np.cos(np.pi * (f - hi) / taper))
            else:
                w[i] = 1.0
        return w

    def _speech_mask(self, power: np.ndarray) -> np.ndarray:
        """Decision-directed Wiener gain over a tracked noise floor.

        The noise floor follows the signal downwards quickly and upwards
        slowly, so a steady engine is learned as noise while a voice on top of
        it is not.
        """
        # Recursive minimum tracking.
        below = power < self._noise_psd
        self._noise_psd = np.where(
            below,
            0.7 * self._noise_psd + 0.3 * power,
            self._noise_psd * (1.0 + self.noise_adapt) + self.noise_adapt * power,
        )
        np.maximum(self._noise_psd, 1e-9, out=self._noise_psd)

        post_snr = power / (self._noise_psd + EPS)
        # Decision-directed a priori SNR (Ephraim-Malah style).
        prior = 0.92 * (self._prev_speech_psd / (self._noise_psd + EPS)) + 0.08 * np.maximum(
            post_snr - 1.0, 0.0
        )
        gain = prior / (1.0 + prior)
        gain *= self._band_weight

        # Temporal smoothing kills musical noise, the classic artefact of
        # per-bin masking.
        gain = 0.65 * gain + 0.35 * self._prev_gain
        gain = np.clip(gain, self.mask_floor * self._band_weight, 1.0)

        self._prev_gain = gain
        self._prev_speech_psd = (gain ** 2) * power
        return gain

    def _voicing_factor(self, voicing: float) -> float:
        """How much of the speech mask to allow through, based on voicing.

        An SNR-based mask cannot tell a voice from a lorry: both are simply
        "above the noise floor", so relaxing the noise tracker enough to keep
        speech also pulls road noise into the speech stream. Voicing is the
        discriminator -- speech is periodic, traffic is not.

        The factor is smoothed with a fast attack and a slow release so that
        the unvoiced consonants inside an utterance are not dropped: once a
        voice is established the gate stays open across the gaps in it.
        """
        target = float(np.clip((voicing - 0.15) / 0.45, 0.0, 1.0))
        coeff = 0.5 if target > self._voicing else 0.04
        self._voicing += coeff * (target - self._voicing)
        # Never fully closed: a whispered or distant talker still deserves
        # some share of the speech stream.
        return 0.12 + 0.88 * self._voicing

    def _pitch_analysis(
        self, x: np.ndarray, power: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        """Track the dominant pitch and decide whether it is really a voice.

        Returns the smoothed f0, a Gaussian comb over its harmonics, and a
        continuous voicing strength in [0, 1].

        The voiced test exists because autocorrelation happily reports a
        subharmonic for a pure tone: a 1100 Hz siren looks "periodic at 275 Hz"
        and its comb would then shield it from transient extraction. A real
        voice puts substantial energy at its own fundamental and second
        harmonic; a siren puts none there at all, so that is what we check.
        """
        f0 = self._estimate_f0(x)
        if f0 > 0:
            self._f0 = f0 if self._f0 == 0 else 0.7 * self._f0 + 0.3 * f0
        else:
            self._f0 *= 0.85
        f0 = self._f0

        if f0 < 60.0:
            # No pitch: treat all speech-band energy as primary so an unvoiced
            # or whispered talker is not discarded, but report no voicing so
            # the gate above decides how much of it to keep.
            return 0.0, np.ones_like(self.freqs, dtype=np.float64), 0.0

        harmonic_index = np.maximum(np.round(self.freqs / f0), 1.0)
        dist = np.abs(self.freqs - harmonic_index * f0)
        width = max(f0 * 0.35, float(self.freqs[1] - self.freqs[0]) * 1.5)
        comb = np.clip(np.exp(-0.5 * (dist / width) ** 2), 0.0, 1.0)

        total = float(power.sum()) + EPS
        low_harmonics = 0.0
        for k in (1, 2):
            sel = np.abs(self.freqs - k * f0) < width
            if sel.any():
                low_harmonics += float(power[sel].sum())
        # Voicing needs both a plausible human pitch and real energy at the
        # first two harmonics. A pure tone reports a subharmonic and would
        # otherwise masquerade as a voice.
        in_range = 1.0 if 70.0 <= f0 <= 350.0 else 0.0
        harmonic_share = float(np.clip((low_harmonics / total) / 0.05, 0.0, 1.0))
        voicing = in_range * harmonic_share
        return f0, comb, voicing

    def _estimate_f0(self, x: np.ndarray) -> float:
        n = x.size
        if n < 128:
            return 0.0
        sr = self.sample_rate
        min_lag = max(2, int(sr / 350.0))
        max_lag = min(n - 1, int(sr / 70.0))
        if max_lag <= min_lag + 1:
            return 0.0
        xm = x - x.mean()
        energy = float(np.dot(xm, xm)) + EPS
        size = 1 << int(np.ceil(np.log2(2 * n)))
        sp = np.fft.rfft(xm, size)
        ac = np.fft.irfft(sp * np.conj(sp), size)[: max_lag + 1] / energy
        seg = ac[min_lag: max_lag + 1]
        best = int(np.argmax(seg))
        if seg[best] < 0.30:
            return 0.0
        return float(sr / (min_lag + best))

    def _transient_selector(
        self, power: np.ndarray, voiced_comb: np.ndarray | None
    ) -> np.ndarray:
        """Fraction of each bin belonging to a prominent narrowband event.

        Uses the same notched-floor prominence idea as the classifier: a horn
        or siren is a handful of bins towering over their neighbours. When a
        genuine voice is present its harmonic comb is shielded, so speech
        partials are not stripped out into the protected stream.
        """
        mag = np.sqrt(power)
        if mag.size < 16:
            return np.zeros_like(mag)

        bin_hz = float(self.freqs[1] - self.freqs[0]) if self.freqs.size > 1 else 1.0
        k = max(9, int(round(700.0 / max(bin_hz, 1e-6)))) | 1
        k = min(k, (mag.size // 2) | 1)
        if k < 5:
            return np.zeros_like(mag)
        kernel = np.ones(k)
        centre = k // 2
        notch = max(1, int(round(140.0 / max(bin_hz, 1e-6))))
        kernel[max(0, centre - notch): centre + notch + 1] = 0.0
        ksum = kernel.sum()
        if ksum <= 0:
            return np.zeros_like(mag)
        kernel /= ksum

        floor = np.convolve(mag, kernel, mode="same") + EPS
        prominence = mag / floor
        # Soft selector: fully claims bins ~10x above their surroundings.
        sel = np.clip((prominence - 3.0) / 7.0, 0.0, 1.0)

        # Below 200 Hz "peaks" are engine harmonics. Routing those into the
        # protected stream would defeat the entire point of suppressing an
        # engine, so the transient stream starts above them.
        sel[self.freqs < 200.0] = 0.0

        if voiced_comb is not None:
            sel *= 1.0 - voiced_comb

        return sel

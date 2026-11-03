"""Input conditioning before analysis.

Deliberately minimal. Every operation here happens on the signal the user will
eventually hear, so anything aggressive would colour the output and make the
"AURA off" comparison unfair. The preprocessor therefore only removes things
that are not part of the acoustic scene:

* **DC offset**, which some USB interfaces introduce and which corrupts RMS and
  zero-crossing measurements.
* **Channel downmix**, because the engine is mono.

Automatic gain control is available but **off by default** and never applied to
the audio path -- only to a copy used for analysis. Normalising the signal the
user hears would destroy the loudness relationships the whole product is about.
"""

from __future__ import annotations

import numpy as np


class AudioPreprocessor:
    """Conditions incoming blocks and measures their level."""

    def __init__(
        self,
        sample_rate: int,
        *,
        remove_dc: bool = True,
        analysis_agc: bool = False,
        agc_target_dbfs: float = -24.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.remove_dc = remove_dc
        self.analysis_agc = analysis_agc
        self.agc_target = 10.0 ** (agc_target_dbfs / 20.0)
        self._dc = 0.0
        self._agc_gain = 1.0

    def reset(self) -> None:
        self._dc = 0.0
        self._agc_gain = 1.0

    @staticmethod
    def downmix(block: np.ndarray) -> np.ndarray:
        """Collapse any channel layout to mono float32."""
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        return arr.mean(axis=1).astype(np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        """Return the conditioned audio-path signal."""
        x = self.downmix(block)
        if x.size == 0:
            return x
        if self.remove_dc:
            # Slow-moving estimate; a per-block mean would remove real
            # low-frequency content from the engine rumble we want to measure.
            self._dc += 0.002 * (float(np.mean(x)) - self._dc)
            x = x - np.float32(self._dc)
        return x

    def analysis_copy(self, block: np.ndarray) -> np.ndarray:
        """A gain-normalised copy for the classifier, if AGC is enabled.

        Kept separate from :meth:`process` on purpose: the classifier benefits
        from a consistent level, the listener must not have theirs rewritten.
        """
        if not self.analysis_agc or block.size == 0:
            return block
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        if rms < 1e-6:
            return block
        target = float(np.clip(self.agc_target / rms, 0.1, 20.0))
        self._agc_gain += 0.1 * (target - self._agc_gain)
        return (block * self._agc_gain).astype(np.float32)

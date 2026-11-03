"""The :class:`SourceSeparator` abstraction.

Separation is what makes AURA more than an equaliser: to raise a voice while
lowering the bus it is sitting in, the two have to become independent signals
first.

The MVP ships a real-time spectral separator. Heavier models (Conv-TasNet,
SepFormer, Demucs) implement the same interface and can be dropped in behind
it; the engine only ever sees :class:`~aura.types.SeparatedAudio`.

Contract
--------
* ``separate`` is called once per hop from the audio thread and must return
  signals of exactly ``hop`` samples, time-aligned with each other.
* The returned streams SHOULD sum back to (a delayed copy of) the input. The
  mixer relies on this: if the streams do not sum to the input, unity gain on
  everything would not reproduce the original scene.
* ``algorithmic_delay_samples`` must report any delay the implementation adds,
  so :class:`~aura.metrics.monitor.MetricsMonitor` can include it in the
  end-to-end latency figure rather than under-reporting it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import SeparatedAudio, StreamKind


class SourceSeparator(ABC):
    """Abstract real-time source separator."""

    name: str = "unnamed"

    @abstractmethod
    def separate(self, block: np.ndarray) -> SeparatedAudio:
        """Split one hop of mono audio into independently addressable streams."""

    @property
    def streams(self) -> tuple[StreamKind, ...]:
        """Which streams this separator can emit."""
        return (StreamKind.PRIMARY_SPEECH, StreamKind.ENVIRONMENT)

    @property
    def algorithmic_delay_samples(self) -> int:
        """Delay this separator adds, in samples."""
        return 0

    def reset(self) -> None:
        """Clear internal buffers."""

    def close(self) -> None:
        """Release resources."""

    @property
    def is_available(self) -> bool:
        return True

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.is_available,
            "streams": [s.value for s in self.streams],
            "algorithmic_delay_samples": self.algorithmic_delay_samples,
        }


class PassthroughSeparator(SourceSeparator):
    """Null separator: everything is environment. Useful as a latency baseline."""

    name = "passthrough"

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def separate(self, block: np.ndarray) -> SeparatedAudio:
        return SeparatedAudio(
            streams={StreamKind.ENVIRONMENT: block.astype(np.float32, copy=True)},
            sample_rate=self.sample_rate,
        )

    @property
    def streams(self) -> tuple[StreamKind, ...]:
        return (StreamKind.ENVIRONMENT,)

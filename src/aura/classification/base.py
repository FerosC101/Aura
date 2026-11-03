"""The :class:`SoundClassifier` abstraction.

Every sound-event backend implements this interface. AURA ships a dependency-free
DSP classifier so the engine runs anywhere, and an ONNX backend that hosts
YAMNet / PANNs / AST. The pipeline only ever talks to this interface, so the
inference backend can be replaced without touching policy, mixing or UI code.

Contract
--------
``classify`` is called once per audio frame from the real-time path. It MUST:

* be non-blocking and bounded in time (the engine enforces a budget),
* return *multi-label* results -- real scenes contain several sounds at once,
* tag each event with the :class:`~aura.types.StreamKind` it belongs to so the
  mixer knows which separated signal the gain applies to,
* never raise; return an empty list instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import AudioFrame, SeparatedAudio, SoundEvent
from .taxonomy import all_categories


class SoundClassifier(ABC):
    """Abstract semantic sound-event detector."""

    #: Human-readable backend name, surfaced in the UI status panel.
    name: str = "unnamed"

    #: Whether this backend needs an external model file to be present.
    requires_model: bool = False

    @abstractmethod
    def classify(
        self,
        frame: AudioFrame,
        separated: SeparatedAudio | None = None,
    ) -> list[SoundEvent]:
        """Return every sound category believed present in this frame."""

    @property
    def categories(self) -> list[str]:
        """Category keys this backend can emit."""
        return [c.key for c in all_categories()]

    def warmup(self) -> None:
        """Optional: run a dummy inference so the first real frame is not slow."""

    def reset(self) -> None:
        """Clear any recurrent state (called on mode switch / stream restart)."""

    def close(self) -> None:
        """Release model resources."""

    @property
    def is_available(self) -> bool:
        """Whether this backend can actually run right now."""
        return True

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.is_available,
            "requires_model": self.requires_model,
            "categories": self.categories,
        }

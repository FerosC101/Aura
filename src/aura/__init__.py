"""AURA - AI-Powered Selective Awareness.

Hear what matters. Silence what doesn't.
"""

from .config import EngineConfig
from .pipeline.engine import AuraEngine
from .types import (
    AcousticScene,
    Action,
    AudioFrame,
    ContextState,
    GainDecision,
    ListeningMode,
    Priority,
    SoundEvent,
    StreamKind,
)

__version__ = "0.1.0"
__all__ = [
    "AuraEngine", "EngineConfig", "AcousticScene", "Action", "AudioFrame",
    "ContextState", "GainDecision", "ListeningMode", "Priority", "SoundEvent",
    "StreamKind", "__version__",
]

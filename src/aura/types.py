"""Core value types shared across every AURA Engine component.

These types are the contract between modules. A component may be swapped for a
different implementation (a neural classifier for the DSP one, Demucs for the
spectral separator) as long as it keeps speaking in these types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class ListeningMode(str, Enum):
    """The high-level intent the user is in right now."""

    CARE = "care"
    FOCUS = "focus"
    COMMUTE = "commute"
    WALK = "walk"
    SOCIAL = "social"
    CUSTOM = "custom"


class Action(str, Enum):
    """What the policy engine decided to do with a sound.

    Ordered from loudest to quietest so that ``max()`` over a set of actions
    picks the most audible one -- this is what makes safety overrides composable.
    """

    ENHANCE = "enhance"
    PRESERVE = "preserve"
    REDUCE = "reduce"
    SUPPRESS = "suppress"

    @property
    def rank(self) -> int:
        return _ACTION_RANK[self]


_ACTION_RANK = {
    Action.ENHANCE: 3,
    Action.PRESERVE: 2,
    Action.REDUCE: 1,
    Action.SUPPRESS: 0,
}


class Priority(str, Enum):
    """How strongly a decision resists being overridden."""

    CRITICAL = "critical"  # Safety. Never fully silenced.
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

    @property
    def rank(self) -> int:
        return _PRIORITY_RANK[self]


_PRIORITY_RANK = {
    Priority.CRITICAL: 3,
    Priority.HIGH: 2,
    Priority.NORMAL: 1,
    Priority.LOW: 0,
}


class ContextState(str, Enum):
    """Physical / social situation, detected or declared."""

    STATIONARY = "stationary"
    WALKING = "walking"
    COMMUTING = "commuting"
    STUDYING = "studying"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class StreamKind(str, Enum):
    """The separated streams the mixer can address independently."""

    PRIMARY_SPEECH = "primary_speech"
    SECONDARY_SPEECH = "secondary_speech"
    ENVIRONMENT = "environment"
    TRANSIENT = "transient"  # Impulsive events: horns, alarms, bells.


# --------------------------------------------------------------------------
# Audio frames and separated streams
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AudioFrame:
    """One block of mono audio moving through the pipeline.

    ``samples`` is always float32 in [-1, 1]. ``captured_at`` is a
    ``time.perf_counter()`` stamp taken as close to the driver callback as
    possible; every latency figure AURA reports is derived from it.
    """

    samples: np.ndarray
    sample_rate: int
    captured_at: float = field(default_factory=time.perf_counter)
    sequence: int = 0

    @property
    def duration_ms(self) -> float:
        return 1000.0 * len(self.samples) / self.sample_rate

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples), dtype=np.float64)))

    @property
    def dbfs(self) -> float:
        return 20.0 * np.log10(max(self.rms, 1e-9))


@dataclass(slots=True)
class SeparatedAudio:
    """Output of a :class:`~aura.separation.base.SourceSeparator`.

    A mapping of :class:`StreamKind` to time-domain signals that sum
    (approximately) back to the input frame. Implementations are free to emit
    only a subset; the mixer treats missing streams as silence.
    """

    streams: dict[StreamKind, np.ndarray]
    sample_rate: int
    residual: np.ndarray | None = None

    def get(self, kind: StreamKind, length: int) -> np.ndarray:
        s = self.streams.get(kind)
        if s is None:
            return np.zeros(length, dtype=np.float32)
        return s

    @property
    def kinds(self) -> list[StreamKind]:
        return list(self.streams.keys())


# --------------------------------------------------------------------------
# Detection and decisions
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SoundEvent:
    """A classified sound present in the current acoustic scene."""

    category: str
    confidence: float
    stream: StreamKind = StreamKind.ENVIRONMENT
    level_db: float = -60.0
    detected_at: float = field(default_factory=time.perf_counter)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


@dataclass(slots=True)
class GainDecision:
    """What the :class:`~aura.policy.engine.AcousticPolicyEngine` decided.

    ``gain_db`` is relative to the sound's natural level: 0 dB means "pass
    through untouched", positive means enhance, negative means attenuate.
    """

    category: str
    action: Action
    gain_db: float
    priority: Priority
    always_audible: bool = False
    safety_override: bool = False
    reason: str = ""
    stream: StreamKind = StreamKind.ENVIRONMENT
    confidence: float = 1.0

    @property
    def linear_gain(self) -> float:
        return float(10.0 ** (self.gain_db / 20.0))


@dataclass(slots=True)
class AcousticScene:
    """The complete, UI-ready snapshot of one processed frame."""

    events: list[SoundEvent]
    decisions: list[GainDecision]
    mode: ListeningMode
    context: ContextState
    person_lock_active: bool = False
    person_lock_name: str | None = None
    person_lock_similarity: float = 0.0
    name_detected: bool = False
    safety_override_active: bool = False
    input_level_db: float = -60.0
    output_level_db: float = -60.0
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Latency accounting
# --------------------------------------------------------------------------


@dataclass(slots=True)
class LatencyBreakdown:
    """Per-stage latency in milliseconds for a single frame."""

    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    separation_ms: float = 0.0
    classification_ms: float = 0.0
    speaker_ms: float = 0.0
    policy_ms: float = 0.0
    mixing_ms: float = 0.0
    output_ms: float = 0.0

    @property
    def inference_ms(self) -> float:
        """Everything that constitutes "the AI part"."""
        return self.classification_ms + self.speaker_ms

    @property
    def processing_ms(self) -> float:
        """DSP + inference, excluding driver-side capture and output buffering."""
        return (
            self.preprocess_ms
            + self.separation_ms
            + self.classification_ms
            + self.speaker_ms
            + self.policy_ms
            + self.mixing_ms
        )

    @property
    def total_ms(self) -> float:
        return self.capture_ms + self.processing_ms + self.output_ms

    def as_dict(self) -> dict[str, float]:
        return {
            "capture_ms": round(self.capture_ms, 3),
            "preprocess_ms": round(self.preprocess_ms, 3),
            "separation_ms": round(self.separation_ms, 3),
            "classification_ms": round(self.classification_ms, 3),
            "speaker_ms": round(self.speaker_ms, 3),
            "policy_ms": round(self.policy_ms, 3),
            "mixing_ms": round(self.mixing_ms, 3),
            "output_ms": round(self.output_ms, 3),
            "inference_ms": round(self.inference_ms, 3),
            "processing_ms": round(self.processing_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }

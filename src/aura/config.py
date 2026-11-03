"""Engine configuration.

Defaults are chosen for a live demo on a laptop: native device sample rate (no
resampling stage), and a 20 ms block, which is the knee of the latency/stability
curve for Python audio callbacks. Smaller blocks are selectable and genuinely
lower latency, but leave less headroom before dropouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import ListeningMode


@dataclass(slots=True)
class EngineConfig:
    """Everything needed to construct an :class:`~aura.pipeline.engine.AuraEngine`."""

    sample_rate: int = 48000
    #: Frames per block. 960 @ 48 kHz = 20 ms.
    block_size: int = 960

    input_device: int | str | None = None
    output_device: int | str | None = None
    #: "low", "high", or an explicit latency in seconds for PortAudio.
    device_latency: str | float = "low"
    #: Use one full-duplex stream (lowest latency) rather than two streams.
    duplex: bool = True

    mode: ListeningMode = ListeningMode.COMMUTE
    enabled: bool = True
    suppression_strength: float = 1.0
    #: Master output trim, in dB.
    output_trim_db: float = 0.0

    classifier_backend: str = "dsp"
    separator_backend: str = "spectral"
    speaker_backend: str = "spectral"
    #: Optional path to an ONNX model when ``classifier_backend`` is "onnx".
    classifier_model_path: str | None = None

    classifier_threshold: float = 0.42
    #: Cap on how many sound cards the UI receives per frame.
    max_reported_events: int = 6

    person_lock_enabled: bool = True
    name_awareness_enabled: bool = False
    auto_context: bool = False
    allow_motion_context: bool = False

    #: How often the scene is published to subscribers, in Hz.
    publish_rate_hz: float = 12.0
    profile_dir: str | None = None
    safety_categories: list[str] | None = field(default=None)

    @property
    def block_ms(self) -> float:
        return 1000.0 * self.block_size / self.sample_rate

    def as_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "block_ms": round(self.block_ms, 2),
            "input_device": self.input_device,
            "output_device": self.output_device,
            "duplex": self.duplex,
            "mode": self.mode.value,
            "enabled": self.enabled,
            "suppression_strength": self.suppression_strength,
            "output_trim_db": self.output_trim_db,
            "classifier_backend": self.classifier_backend,
            "separator_backend": self.separator_backend,
            "speaker_backend": self.speaker_backend,
            "classifier_threshold": self.classifier_threshold,
            "person_lock_enabled": self.person_lock_enabled,
            "name_awareness_enabled": self.name_awareness_enabled,
            "auto_context": self.auto_context,
            "publish_rate_hz": self.publish_rate_hz,
        }

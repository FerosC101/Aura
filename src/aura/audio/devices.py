"""Audio device discovery.

Thin wrapper over PortAudio (via ``sounddevice``) that degrades gracefully: if
no audio backend is present the rest of AURA still runs in file and demo modes,
which is what keeps the test suite and CI usable on machines with no sound card.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - environment dependent
    import sounddevice as sd

    AUDIO_AVAILABLE = True
    AUDIO_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    sd = None  # type: ignore[assignment]
    AUDIO_AVAILABLE = False
    AUDIO_IMPORT_ERROR = str(exc)


@dataclass(slots=True)
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: str
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def can_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def can_output(self) -> bool:
        return self.max_output_channels > 0

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "max_input_channels": self.max_input_channels,
            "max_output_channels": self.max_output_channels,
            "default_samplerate": self.default_samplerate,
            "hostapi": self.hostapi,
            "is_default_input": self.is_default_input,
            "is_default_output": self.is_default_output,
        }


def list_devices() -> list[DeviceInfo]:
    if not AUDIO_AVAILABLE:
        return []
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_in, default_out = sd.default.device
    except Exception:  # pragma: no cover
        return []

    out: list[DeviceInfo] = []
    for i, d in enumerate(devices):
        out.append(
            DeviceInfo(
                index=i,
                name=str(d.get("name", f"device {i}")),
                max_input_channels=int(d.get("max_input_channels", 0)),
                max_output_channels=int(d.get("max_output_channels", 0)),
                default_samplerate=float(d.get("default_samplerate", 48000.0)),
                hostapi=str(hostapis[d["hostapi"]]["name"]) if "hostapi" in d else "",
                is_default_input=(i == default_in),
                is_default_output=(i == default_out),
            )
        )
    return out


def default_input() -> DeviceInfo | None:
    return next((d for d in list_devices() if d.is_default_input), None)


def default_output() -> DeviceInfo | None:
    return next((d for d in list_devices() if d.is_default_output), None)


def resolve(spec: int | str | None, *, want_input: bool) -> DeviceInfo | None:
    """Resolve an index or a (partial, case-insensitive) name to a device."""
    devices = list_devices()
    if not devices:
        return None
    if spec is None:
        return default_input() if want_input else default_output()
    if isinstance(spec, int):
        return next((d for d in devices if d.index == spec), None)
    needle = spec.strip().lower()
    usable = [d for d in devices if (d.can_input if want_input else d.can_output)]
    exact = next((d for d in usable if d.name.lower() == needle), None)
    return exact or next((d for d in usable if needle in d.name.lower()), None)

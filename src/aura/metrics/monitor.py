"""Latency and load accounting.

AURA's central claim is that it works in real time, so the latency figures have
to be honest. Three things this module is careful about:

* **Driver latency is included.** PortAudio reports the input and output buffer
  latencies it negotiated with CoreAudio; those dominate the total and omitting
  them would flatter the numbers considerably.
* **Algorithmic delay is included.** The separator's overlap-add tail is real
  delay the listener experiences even though it costs no CPU time.
* **Percentiles, not just means.** A pipeline that averages 3 ms but spikes to
  40 ms sounds broken. p95 and max are what determine whether audio glitches,
  so they are what the UI shows.

"Processing load" is wall-clock processing time over the block's duration: at
100% the CPU cannot keep up and dropouts begin.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from ..types import LatencyBreakdown

STAGES = (
    "capture", "preprocess", "separation", "classification",
    "speaker", "policy", "mixing", "output",
)


@dataclass(slots=True)
class StageStats:
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass(slots=True)
class MetricsSnapshot:
    latency: dict[str, float] = field(default_factory=dict)
    stages: dict[str, dict[str, float]] = field(default_factory=dict)
    total: dict[str, float] = field(default_factory=dict)
    processing_load: float = 0.0
    frames: int = 0
    frames_per_second: float = 0.0
    xruns: int = 0
    block_ms: float = 0.0
    device_latency_ms: dict[str, float] = field(default_factory=dict)
    algorithmic_delay_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "latency": self.latency,
            "stages": self.stages,
            "total": self.total,
            "processing_load": round(self.processing_load, 4),
            "frames": self.frames,
            "frames_per_second": round(self.frames_per_second, 1),
            "xruns": self.xruns,
            "block_ms": round(self.block_ms, 3),
            "device_latency_ms": self.device_latency_ms,
            "algorithmic_delay_ms": round(self.algorithmic_delay_ms, 3),
        }


class MetricsMonitor:
    """Rolling per-stage timing over the last ``window`` frames."""

    def __init__(self, sample_rate: int, block_size: int, window: int = 200) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.block_ms = 1000.0 * block_size / sample_rate
        self.window = window

        self._stage: dict[str, deque[float]] = {s: deque(maxlen=window) for s in STAGES}
        self._total: deque[float] = deque(maxlen=window)
        self._wall: deque[float] = deque(maxlen=window)
        self._current = LatencyBreakdown()

        self.frames = 0
        self.xruns = 0
        self.device_input_latency_ms = 0.0
        self.device_output_latency_ms = 0.0
        self.algorithmic_delay_ms = 0.0
        self._started = time.perf_counter()
        self._last_frame_time = self._started

    # -- recording ---------------------------------------------------------

    def reset(self) -> None:
        for d in self._stage.values():
            d.clear()
        self._total.clear()
        self._wall.clear()
        self.frames = 0
        self.xruns = 0
        self._started = time.perf_counter()
        self._last_frame_time = self._started

    def begin_frame(self) -> LatencyBreakdown:
        self._current = LatencyBreakdown()
        return self._current

    @contextmanager
    def stage(self, name: str):
        """Time one pipeline stage.

        ``with monitor.stage("classification"): ...``
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            setattr(self._current, f"{name}_ms", dt)

    def set_device_latency(self, input_ms: float, output_ms: float) -> None:
        self.device_input_latency_ms = float(input_ms)
        self.device_output_latency_ms = float(output_ms)

    def set_algorithmic_delay(self, samples: int) -> None:
        self.algorithmic_delay_ms = 1000.0 * samples / self.sample_rate

    def record_xrun(self, n: int = 1) -> None:
        self.xruns += n

    def end_frame(self, wall_ms: float | None = None) -> LatencyBreakdown:
        """Close the frame, folding in device and algorithmic delay."""
        b = self._current
        # Capture latency is what the driver already spent before we saw the
        # samples, plus the time to fill the block itself.
        b.capture_ms = self.device_input_latency_ms + self.block_ms
        b.output_ms = self.device_output_latency_ms + self.algorithmic_delay_ms

        for s in STAGES:
            self._stage[s].append(getattr(b, f"{s}_ms"))
        self._total.append(b.total_ms)
        if wall_ms is not None:
            self._wall.append(wall_ms)

        now = time.perf_counter()
        self._last_frame_time = now
        self.frames += 1
        return b

    # -- reporting ---------------------------------------------------------

    @staticmethod
    def _stats(values: deque[float]) -> StageStats:
        if not values:
            return StageStats()
        arr = np.asarray(values, dtype=np.float64)
        return StageStats(
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            max_ms=float(arr.max()),
        )

    @property
    def processing_load(self) -> float:
        """Fraction of each block's duration spent computing. >1.0 drops audio."""
        if not self._wall:
            return 0.0
        return float(np.mean(self._wall) / max(self.block_ms, 1e-6))

    def snapshot(self) -> MetricsSnapshot:
        elapsed = max(time.perf_counter() - self._started, 1e-6)
        return MetricsSnapshot(
            latency=self._current.as_dict(),
            stages={s: self._stats(self._stage[s]).as_dict() for s in STAGES},
            total=self._stats(self._total).as_dict(),
            processing_load=self.processing_load,
            frames=self.frames,
            frames_per_second=self.frames / elapsed,
            xruns=self.xruns,
            block_ms=self.block_ms,
            device_latency_ms={
                "input": round(self.device_input_latency_ms, 3),
                "output": round(self.device_output_latency_ms, 3),
            },
            algorithmic_delay_ms=self.algorithmic_delay_ms,
        )

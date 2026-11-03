"""Audio capture and output engines.

Three transports, one interface. The pipeline does not care which is running,
which is what lets the same engine serve a live demo, a file benchmark and the
test suite.

``DuplexAudioEngine``
    One PortAudio stream carrying input and output together. Processing happens
    inside the driver callback. This is the lowest-latency arrangement
    available from Python: handing blocks to a worker thread would add at least
    one more block of buffering plus scheduling jitter, and on a 20 ms block
    that is a 50% latency increase for no benefit. The processing budget is
    ~2% of real time, so the callback has ample headroom.

``SplitAudioEngine``
    Separate input and output streams, for when capture and playback are on
    different devices that cannot share a clock.

``OfflineAudioEngine``
    Drives the pipeline from an array as fast as (or in step with) real time.
    Used by PSC Demo Mode, the benchmarks and the tests.

The processing callback signature is ``(in_block, out_block) -> None``: it
receives mono input and writes mono output in place.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from .devices import AUDIO_AVAILABLE, AUDIO_IMPORT_ERROR, DeviceInfo, resolve

ProcessCallback = Callable[[np.ndarray, np.ndarray], None]


class AudioEngineError(RuntimeError):
    pass


class BaseAudioEngine(ABC):
    """Common surface for every transport."""

    def __init__(self, sample_rate: int, block_size: int, process: ProcessCallback) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.process = process
        self.running = False
        self.xruns = 0
        self.input_latency_ms = 0.0
        self.output_latency_ms = 0.0

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def describe(self) -> dict:
        return {
            "transport": type(self).__name__,
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "running": self.running,
            "xruns": self.xruns,
            "input_latency_ms": round(self.input_latency_ms, 3),
            "output_latency_ms": round(self.output_latency_ms, 3),
        }


class DuplexAudioEngine(BaseAudioEngine):
    """Single full-duplex PortAudio stream. The lowest-latency path."""

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        process: ProcessCallback,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        latency: str | float = "low",
    ) -> None:
        super().__init__(sample_rate, block_size, process)
        if not AUDIO_AVAILABLE:
            raise AudioEngineError(f"no audio backend available: {AUDIO_IMPORT_ERROR}")
        self.input_info: DeviceInfo | None = resolve(input_device, want_input=True)
        self.output_info: DeviceInfo | None = resolve(output_device, want_input=False)
        self.latency = latency
        self._stream: Any | None = None

    def start(self) -> None:
        import sounddevice as sd

        if self.running:
            return

        def callback(indata, outdata, frames, time_info, status):
            if status:
                # input_overflow / output_underflow both mean we missed audio.
                self.xruns += 1
            mono = indata[:, 0] if indata.ndim > 1 else indata
            out = np.zeros(frames, dtype=np.float32)
            try:
                self.process(np.asarray(mono, dtype=np.float32), out)
            except Exception:
                # Never let an exception escape into PortAudio: it would kill
                # the stream. Output silence for this block and keep going.
                out[:] = 0.0
            if outdata.ndim > 1:
                outdata[:] = out[:, None]
            else:
                outdata[:] = out

        self._stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            channels=(1, 1 if (self.output_info and self.output_info.max_output_channels < 2)
                      else 2),
            device=(
                self.input_info.index if self.input_info else None,
                self.output_info.index if self.output_info else None,
            ),
            latency=self.latency,
            callback=callback,
        )
        self._stream.start()
        self.running = True
        self.input_latency_ms = float(self._stream.latency[0]) * 1000.0
        self.output_latency_ms = float(self._stream.latency[1]) * 1000.0

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.running = False

    def describe(self) -> dict:
        d = super().describe()
        d["input_device"] = self.input_info.name if self.input_info else None
        d["output_device"] = self.output_info.name if self.output_info else None
        return d


class SplitAudioEngine(BaseAudioEngine):
    """Independent input and output streams joined by a queue.

    Needed when capture and playback devices cannot share a clock. Costs one
    extra block of latency versus duplex, which is reported honestly.
    """

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        process: ProcessCallback,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ) -> None:
        super().__init__(sample_rate, block_size, process)
        if not AUDIO_AVAILABLE:
            raise AudioEngineError(f"no audio backend available: {AUDIO_IMPORT_ERROR}")
        self.input_info = resolve(input_device, want_input=True)
        self.output_info = resolve(output_device, want_input=False)
        self._pending: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._in_stream: Any | None = None
        self._out_stream: Any | None = None

    def start(self) -> None:
        import sounddevice as sd

        if self.running:
            return

        def on_input(indata, frames, time_info, status):
            if status:
                self.xruns += 1
            mono = np.asarray(indata[:, 0] if indata.ndim > 1 else indata, dtype=np.float32)
            out = np.zeros(frames, dtype=np.float32)
            try:
                self.process(mono, out)
            except Exception:
                out[:] = 0.0
            with self._lock:
                self._pending.append(out)
                # Bound the queue: a backlog is latency, not safety.
                if len(self._pending) > 8:
                    del self._pending[:-4]
                    self.xruns += 1

        def on_output(outdata, frames, time_info, status):
            if status:
                self.xruns += 1
            with self._lock:
                block = self._pending.pop(0) if self._pending else None
            if block is None or block.size != frames:
                outdata[:] = 0.0
                return
            outdata[:] = block[:, None] if outdata.ndim > 1 else block

        self._in_stream = sd.InputStream(
            samplerate=self.sample_rate, blocksize=self.block_size, dtype="float32",
            channels=1, device=self.input_info.index if self.input_info else None,
            latency="low", callback=on_input,
        )
        self._out_stream = sd.OutputStream(
            samplerate=self.sample_rate, blocksize=self.block_size, dtype="float32",
            channels=2 if (self.output_info and self.output_info.max_output_channels >= 2) else 1,
            device=self.output_info.index if self.output_info else None,
            latency="low", callback=on_output,
        )
        self._in_stream.start()
        self._out_stream.start()
        self.running = True
        self.input_latency_ms = float(self._in_stream.latency) * 1000.0
        # The hand-off queue costs one additional block.
        self.output_latency_ms = (
            float(self._out_stream.latency) * 1000.0
            + 1000.0 * self.block_size / self.sample_rate
        )

    def stop(self) -> None:
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._in_stream = self._out_stream = None
        self.running = False
        with self._lock:
            self._pending.clear()


class OfflineAudioEngine(BaseAudioEngine):
    """Drives the pipeline from an in-memory signal.

    ``realtime=True`` paces blocks at wall-clock speed so PSC Demo Mode behaves
    like a live stream; ``False`` runs flat out for benchmarks and tests.
    """

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        process: ProcessCallback,
        signal: np.ndarray,
        *,
        realtime: bool = True,
        loop: bool = True,
        on_block: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        super().__init__(sample_rate, block_size, process)
        self.signal = np.asarray(signal, dtype=np.float32)
        self.realtime = realtime
        self.loop = loop
        self.on_block = on_block
        self.position = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Nominal figures for a synthetic source, so the reported total latency
        # stays comparable with the live path.
        self.input_latency_ms = 0.0
        self.output_latency_ms = 0.0

    def set_signal(self, signal: np.ndarray, *, reset_position: bool = True) -> None:
        self.signal = np.asarray(signal, dtype=np.float32)
        if reset_position:
            self.position = 0

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run, name="aura-offline", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = self.block_size / self.sample_rate
        next_at = time.perf_counter()
        while not self._stop.is_set():
            block = self._next_block()
            if block is None:
                break
            out = np.zeros(self.block_size, dtype=np.float32)
            try:
                self.process(block, out)
            except Exception:
                out[:] = 0.0
            if self.on_block is not None:
                self.on_block(out)
            if self.realtime:
                next_at += period
                delay = next_at - time.perf_counter()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    # Fell behind: resynchronise rather than accumulate drift.
                    next_at = time.perf_counter()
                    self.xruns += 1
        self.running = False

    def _next_block(self) -> np.ndarray | None:
        n = self.block_size
        if self.signal.size == 0:
            return np.zeros(n, dtype=np.float32)
        if self.position + n > self.signal.size:
            if not self.loop:
                return None
            self.position = 0
        block = self.signal[self.position: self.position + n]
        self.position += n
        if block.size < n:
            block = np.pad(block, (0, n - block.size))
        return block

    def run_sync(self) -> None:
        """Process the whole signal on the calling thread (benchmarks/tests)."""
        n = self.block_size
        for start in range(0, self.signal.size - n + 1, n):
            out = np.zeros(n, dtype=np.float32)
            self.process(self.signal[start:start + n], out)
            if self.on_block is not None:
                self.on_block(out)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self.running = False


# Named aliases matching the architecture in the brief. The capture and output
# responsibilities are two faces of one stream in duplex mode, so they share an
# implementation rather than pretending to be independent.
AudioCaptureEngine = DuplexAudioEngine
AudioOutputEngine = DuplexAudioEngine

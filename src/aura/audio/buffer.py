"""Lock-free-ish ring buffering between the audio callback and everything else.

The real-time path never allocates and never blocks: the callback writes into a
preallocated ring and returns. Consumers that are allowed to be late -- the
WebSocket broadcaster, the waveform display -- read from the ring and simply
miss data if they fall behind, which is the correct failure mode for a
monitoring view. Dropping a UI frame is fine; stalling the audio thread is not.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Single-producer / single-consumer float32 ring buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._write = 0
        self._available = 0
        self._lock = threading.Lock()
        self.overruns = 0

    def __len__(self) -> int:
        with self._lock:
            return self._available

    def clear(self) -> None:
        with self._lock:
            self._write = 0
            self._available = 0

    def write(self, block: np.ndarray) -> None:
        n = block.size
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
                self._data[:] = block[-self.capacity:]
                self._write = 0
                self._available = self.capacity
                self.overruns += 1
                return
            end = self._write + n
            if end <= self.capacity:
                self._data[self._write:end] = block
            else:
                split = self.capacity - self._write
                self._data[self._write:] = block[:split]
                self._data[: end - self.capacity] = block[split:]
            self._write = end % self.capacity
            new_avail = self._available + n
            if new_avail > self.capacity:
                self.overruns += 1
                new_avail = self.capacity
            self._available = new_avail

    def latest(self, n: int) -> np.ndarray:
        """Most recent ``n`` samples, zero-padded at the front if short."""
        with self._lock:
            n = min(n, self.capacity)
            avail = min(n, self._available)
            out = np.zeros(n, dtype=np.float32)
            if avail == 0:
                return out
            start = (self._write - avail) % self.capacity
            end = start + avail
            if end <= self.capacity:
                out[n - avail:] = self._data[start:end]
            else:
                split = self.capacity - start
                out[n - avail: n - avail + split] = self._data[start:]
                out[n - avail + split:] = self._data[: end - self.capacity]
            return out


class AudioBufferManager:
    """Owns the input and output rings plus the frame counter.

    Also produces the decimated waveform the UI draws, downsampled here rather
    than in the browser so the WebSocket carries tens of floats per frame
    instead of thousands.
    """

    def __init__(self, sample_rate: int, seconds: float = 4.0) -> None:
        self.sample_rate = sample_rate
        capacity = max(1024, int(sample_rate * seconds))
        self.input = RingBuffer(capacity)
        self.output = RingBuffer(capacity)
        self.frames = 0

    def push(self, in_block: np.ndarray, out_block: np.ndarray) -> None:
        self.input.write(in_block)
        self.output.write(out_block)
        self.frames += 1

    def reset(self) -> None:
        self.input.clear()
        self.output.clear()
        self.frames = 0

    def waveform(self, points: int = 96, seconds: float = 0.75) -> dict[str, list[float]]:
        """Peak-decimated input and output envelopes for the UI."""
        n = max(points, int(self.sample_rate * seconds))
        return {
            "input": _decimate_peak(self.input.latest(n), points),
            "output": _decimate_peak(self.output.latest(n), points),
        }


def _decimate_peak(x: np.ndarray, points: int) -> list[float]:
    """Peak-preserving decimation -- a transient must not vanish from the view."""
    if x.size == 0 or points <= 0:
        return [0.0] * max(points, 0)
    step = max(1, x.size // points)
    usable = step * points
    if usable > x.size:
        x = np.pad(x, (usable - x.size, 0))
    blocks = np.abs(x[-usable:]).reshape(points, step)
    return [round(float(v), 4) for v in blocks.max(axis=1)]

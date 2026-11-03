"""Test helpers shared across modules.

Kept out of ``conftest.py`` so they can be imported directly; conftest is for
fixtures, and pytest does not expose it as an importable module.
"""

from __future__ import annotations

import threading

import numpy as np

from aura.pipeline.engine import AuraEngine

SR = 16000
HOP = 320


def drive(engine: AuraEngine, signal: np.ndarray, *, with_worker: bool = False) -> np.ndarray:
    """Push ``signal`` through the engine block by block on this thread.

    Driving the pipeline synchronously makes tests deterministic: the real
    engine runs its DSP on an audio callback and its speaker/name analysis on a
    worker, and asserting against thread timing would make every test flaky.
    The semantic worker is started only when a test actually needs it.
    """
    hop = engine.config.block_size
    worker = None
    if with_worker:
        engine._running = True
        engine._worker_stop.clear()
        worker = threading.Thread(target=engine._semantic_loop, daemon=True)
        worker.start()

    out_blocks = []
    try:
        for i in range(0, signal.size - hop + 1, hop):
            out = np.zeros(hop, dtype=np.float32)
            engine._process(signal[i:i + hop], out)
            out_blocks.append(out.copy())
    finally:
        if worker is not None:
            engine._worker_stop.set()
            worker.join(timeout=1.0)
            engine._running = False

    return np.concatenate(out_blocks) if out_blocks else np.zeros(0, dtype=np.float32)


def band_energy_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Energy in a frequency band, in dB."""
    if x.size < 64:
        return -120.0
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return 10.0 * np.log10(float(spec[mask].sum()) + 1e-18)

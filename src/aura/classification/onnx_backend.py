"""Neural sound-event classification via ONNX Runtime.

Hosts YAMNet, PANNs, AST or any AudioSet-compatible model exported to ONNX.
The model's own class names are mapped onto the AURA taxonomy through
:func:`~aura.classification.taxonomy.resolve_alias`, so the policy engine,
mixer, safety guard and UI never learn anything about AudioSet.

This backend is optional. It is not the default because the default has to run
on a clean machine with no model download, and because a 0.4 ms DSP pass leaves
far more of the latency budget for audio than a neural forward pass does.
When accuracy matters more than either, switch to it:

    aura live --classifier onnx --classifier-model ~/.cache/aura/models/yamnet.onnx

or ``EngineConfig(classifier_backend="onnx", classifier_model_path=...)``. On
ESC-50 this roughly triples recall and cuts safety-critical false alarms by a
factor of forty, at no cost in mean per-hop time; see `docs/BENCHMARKS.md`
§1b. If the model cannot be loaded the engine falls back to the DSP classifier
and records why in ``fallback_notes()``, which the CLI prints at startup.

Expected model contract
-----------------------
* Input: mono float32 waveform, shape ``(n,)`` or ``(1, n)``.
* Output: per-class scores, shape ``(classes,)`` or ``(frames, classes)``.
* A sibling ``<model>.labels.txt`` or ``<model>.csv`` giving one class name per
  line, in output order. YAMNet's ``yamnet_class_map.csv`` is understood.

Because most AudioSet models expect 16 kHz, input is resampled when the engine
runs at a different rate.
"""

from __future__ import annotations

import contextlib
import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ..types import AudioFrame, SeparatedAudio, SoundEvent, StreamKind
from .base import SoundClassifier
from .taxonomy import get_category, resolve_alias

#: Most AudioSet-derived models are trained at this rate.
MODEL_SAMPLE_RATE = 16000
#: Seconds of audio each inference sees. YAMNet uses ~0.96 s.
WINDOW_SECONDS = 0.96


class ONNXAudioClassifier(SoundClassifier):
    """AudioSet-compatible classifier behind the AURA interface."""

    name = "onnx"
    requires_model = True

    def __init__(
        self,
        sample_rate: int,
        *,
        model_path: str | None = None,
        threshold: float = 0.42,
        window_seconds: float = WINDOW_SECONDS,
        inference_stride: int = 3,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model_path = model_path
        self.unavailable_reason: str | None = None
        self._session: Any | None = None
        self._labels: list[str] = []
        self._mapping: list[str | None] = []
        self._input_name = ""

        # Inference runs on a rolling window every ``inference_stride`` hops;
        # the cached result is reported in between. A neural pass is far too
        # slow for every 20 ms block, and sound events do not change that fast.
        self._window_len = int(window_seconds * sample_rate)
        self._window: deque[float] = deque(maxlen=self._window_len)
        self._stride = max(1, inference_stride)
        self._counter = 0
        self._cached: list[SoundEvent] = []

        self._load(model_path)

    # -- setup -------------------------------------------------------------

    def _load(self, model_path: str | None) -> None:
        if not model_path:
            self.unavailable_reason = "no model path configured"
            return
        path = Path(model_path).expanduser()
        if not path.is_file():
            self.unavailable_reason = f"model not found at {path}"
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            self.unavailable_reason = f"onnxruntime not installed ({exc})"
            return

        try:
            opts = ort.SessionOptions()
            # One thread: the audio callback is already the real-time thread and
            # an inference pool fighting it for cores causes dropouts.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:
            self.unavailable_reason = f"could not open model: {exc}"
            self._session = None
            return

        self._labels = self._load_labels(path)
        if not self._labels:
            self.unavailable_reason = f"no label file beside {path.name}"
            self._session = None
            return
        # Precompute the AudioSet -> AURA mapping once.
        self._mapping = [resolve_alias(label) for label in self._labels]
        self.name = f"onnx:{path.stem}"

    @staticmethod
    def _load_labels(model_path: Path) -> list[str]:
        for candidate in (
            model_path.with_suffix(".labels.txt"),
            model_path.with_suffix(".csv"),
            model_path.parent / "yamnet_class_map.csv",
            model_path.parent / "labels.txt",
        ):
            if not candidate.is_file():
                continue
            if candidate.suffix == ".csv":
                with candidate.open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.reader(fh))
                if not rows:
                    continue
                # yamnet_class_map.csv is index,mid,display_name
                start = 1 if rows[0] and not rows[0][0].isdigit() else 0
                return [r[-1] for r in rows[start:] if r]
            return [
                line.strip() for line in candidate.read_text("utf-8").splitlines() if line.strip()
            ]
        return []

    @property
    def is_available(self) -> bool:
        return self._session is not None

    @property
    def categories(self) -> list[str]:
        mapped = {m for m in self._mapping if m}
        return sorted(mapped) if mapped else super().categories

    # -- inference ---------------------------------------------------------

    def warmup(self) -> None:
        if self._session is None:
            return
        silence = np.zeros(int(WINDOW_SECONDS * MODEL_SAMPLE_RATE), dtype=np.float32)
        try:
            self._session.run(None, {self._input_name: silence})
        except Exception:
            # Some exports want a batch dimension; try that shape instead.
            with contextlib.suppress(Exception):
                self._session.run(None, {self._input_name: silence[None, :]})

    def reset(self) -> None:
        self._window.clear()
        self._cached = []
        self._counter = 0

    def close(self) -> None:
        self._session = None

    def classify(
        self,
        frame: AudioFrame,
        separated: SeparatedAudio | None = None,
    ) -> list[SoundEvent]:
        if self._session is None:
            return []

        self._window.extend(frame.samples.tolist())
        self._counter += 1
        if self._counter % self._stride != 0 or len(self._window) < self._window_len:
            return self._cached

        audio = np.asarray(self._window, dtype=np.float32)
        if self.sample_rate != MODEL_SAMPLE_RATE:
            audio = _resample(audio, self.sample_rate, MODEL_SAMPLE_RATE)

        try:
            scores = self._run(audio)
        except Exception:
            # An inference failure must never take the audio path down.
            return self._cached

        # Several AudioSet classes can map to one AURA category; take the
        # strongest rather than summing, which would inflate confidence.
        best: dict[str, float] = {}
        for score, key in zip(scores, self._mapping, strict=False):
            if key and score > best.get(key, 0.0):
                best[key] = float(score)

        events = [
            SoundEvent(
                category=key,
                confidence=score,
                stream=get_category(key).stream,
                level_db=frame.dbfs,
                detected_at=frame.captured_at,
                metadata={"backend": self.name},
            )
            for key, score in best.items()
            if score >= self.threshold
        ]
        if not events:
            events = [SoundEvent("ambient", 0.4, StreamKind.ENVIRONMENT, frame.dbfs,
                                 metadata={"backend": self.name, "fallback": True})]
        events.sort(key=lambda e: e.confidence, reverse=True)
        self._cached = events
        return events

    def _run(self, audio: np.ndarray) -> np.ndarray:
        session = self._session
        if session is None:
            return np.zeros(0, dtype=np.float64)
        try:
            outputs = session.run(None, {self._input_name: audio})
        except Exception:
            # Some exports expect a batch dimension.
            outputs = session.run(None, {self._input_name: audio[None, :]})
        scores = np.asarray(outputs[0], dtype=np.float64)
        if scores.ndim > 1:
            # Frame-wise output: take the strongest frame per class.
            scores = scores.reshape(-1, scores.shape[-1]).max(axis=0)
        if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
            scores = 1.0 / (1.0 + np.exp(-scores))  # logits
        return scores


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n_out = int(round(x.size * dst / src))
    return np.interp(
        np.linspace(0.0, x.size - 1, n_out), np.arange(x.size), x
    ).astype(np.float32)

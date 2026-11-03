"""ECAPA-TDNN speaker embeddings via SpeechBrain.

A trained x-vector network, markedly better at Person Lock than the built-in
MFCC-statistics embedder, at the cost of a PyTorch dependency and a model
download. Install with the ``speaker`` extra and select it with
``EngineConfig(speaker_backend="ecapa")``.

The two backends are not interchangeable at rest: embeddings from one cannot be
compared with embeddings from the other, and their similarity scales differ
entirely (ECAPA puts strangers near 0.0, the MFCC embedder near 0.95). Each
therefore declares its own ``impostor_prior``, and stored profiles record which
backend produced them so the engine can refuse to mix them.
"""

from __future__ import annotations

import numpy as np

from .base import SpeakerIdentifier

MODEL_SAMPLE_RATE = 16000


class ECAPASpeakerEmbedder(SpeakerIdentifier):
    """SpeechBrain ECAPA-TDNN embeddings."""

    name = "ecapa-tdnn"
    dim = 192
    min_audio_seconds = 1.0
    # Cosine similarity between different speakers clusters near zero for a
    # trained x-vector network, unlike the correlation-style built-in embedder.
    impostor_prior = (0.05, 0.14)

    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.unavailable_reason: str | None = None
        self._model = None
        try:
            import torch  # noqa: F401
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            self.unavailable_reason = f"speechbrain/torch not installed ({exc})"
            return
        try:
            self._model = EncoderClassifier.from_hparams(
                source=source,
                savedir=savedir or f"/tmp/aura-{source.split('/')[-1]}",
                run_opts={"device": "cpu"},
            )
        except Exception as exc:  # pragma: no cover - network / disk dependent
            self.unavailable_reason = f"could not load {source}: {exc}"
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        if self._model is None:
            return None
        import torch

        x = np.asarray(samples, dtype=np.float32).ravel()
        if sample_rate != MODEL_SAMPLE_RATE:
            n_out = int(round(x.size * MODEL_SAMPLE_RATE / sample_rate))
            x = np.interp(
                np.linspace(0.0, x.size - 1, n_out), np.arange(x.size), x
            ).astype(np.float32)
        if x.size < int(self.min_audio_seconds * MODEL_SAMPLE_RATE * 0.5):
            return None

        try:
            with torch.no_grad():
                vec = self._model.encode_batch(torch.from_numpy(x).unsqueeze(0))
            out = vec.squeeze().cpu().numpy().astype(np.float32)
        except Exception:
            return None

        norm = float(np.linalg.norm(out))
        if norm < 1e-9 or not np.isfinite(norm):
            return None
        return out / norm

    def close(self) -> None:
        self._model = None

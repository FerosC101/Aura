"""Backend factories.

One place that knows how to build each pluggable component, so swapping a
backend is a configuration change rather than a code change. Unavailable
backends fall back to the dependency-free default with a recorded reason, which
is surfaced in the UI's status panel -- a silent fallback would make it look
like a neural model was running when it was not.
"""

from __future__ import annotations

from ..separation.base import PassthroughSeparator, SourceSeparator
from ..separation.spectral import SpectralSourceSeparator
from ..speaker.base import SpeakerIdentifier
from ..speaker.spectral_embedder import SpectralSpeakerEmbedder
from .base import SoundClassifier
from .dsp_classifier import DSPSoundClassifier

#: Populated when a requested backend could not be constructed.
FALLBACK_NOTES: dict[str, str] = {}


def build_classifier(
    backend: str,
    sample_rate: int,
    *,
    threshold: float = 0.42,
    model_path: str | None = None,
) -> SoundClassifier:
    backend = (backend or "dsp").lower()
    if backend in ("onnx", "yamnet", "panns", "ast"):
        try:
            from .onnx_backend import ONNXAudioClassifier

            clf = ONNXAudioClassifier(sample_rate, model_path=model_path, threshold=threshold)
            if clf.is_available:
                FALLBACK_NOTES.pop("classifier", None)
                return clf
            reason = clf.unavailable_reason or "model unavailable"
        except Exception as exc:
            reason = str(exc)
        FALLBACK_NOTES["classifier"] = (
            f"'{backend}' unavailable ({reason}); using the built-in DSP classifier"
        )
    return DSPSoundClassifier(sample_rate, threshold=threshold)


def build_separator(backend: str, sample_rate: int, hop: int) -> SourceSeparator:
    backend = (backend or "spectral").lower()
    if backend in ("none", "passthrough"):
        return PassthroughSeparator(sample_rate)
    return SpectralSourceSeparator(sample_rate, hop)


def build_speaker_identifier(backend: str, sample_rate: int) -> SpeakerIdentifier:
    backend = (backend or "spectral").lower()
    if backend in ("ecapa", "speechbrain"):
        try:
            from ..speaker.ecapa import ECAPASpeakerEmbedder

            emb = ECAPASpeakerEmbedder(sample_rate)
            if emb.is_available:
                FALLBACK_NOTES.pop("speaker", None)
                return emb
            reason = emb.unavailable_reason or "model unavailable"
        except Exception as exc:
            reason = str(exc)
        FALLBACK_NOTES["speaker"] = (
            f"'{backend}' unavailable ({reason}); using MFCC-statistics embeddings"
        )
    return SpectralSpeakerEmbedder(sample_rate)


def fallback_notes() -> dict[str, str]:
    return dict(FALLBACK_NOTES)

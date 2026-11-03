"""Speaker embedding and identification abstractions.

Person Lock needs one capability: turn a few seconds of a voice into a compact
vector, and later decide whether a live speech segment came from that same
voice. Everything else -- enrolment UI, gain boosts, privacy controls -- is
built on top of that.

The default backend is a dependency-free spectral-statistics embedder. It is
markedly weaker than a trained x-vector network, but it distinguishes a handful
of enrolled speakers, which is the MVP bar. ECAPA-TDNN slots in behind the same
interface when ``speechbrain`` is installed.

Privacy contract
----------------
Implementations receive raw audio and must not retain it. Only the returned
embedding is persisted, and only with explicit consent -- see
:class:`~aura.speaker.person_lock.PersonLockManager`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SpeakerIdentifier(ABC):
    """Abstract speaker-embedding backend."""

    name: str = "unnamed"

    #: Dimensionality of the embeddings this backend produces.
    dim: int = 0

    #: Shortest audio, in seconds, that yields a usable embedding.
    min_audio_seconds: float = 1.0

    #: (mean, std) similarity this backend gives for *different* speakers.
    #:
    #: Every backend has its own similarity scale -- correlation-style
    #: embeddings cluster near 0.94 for strangers while an x-vector network
    #: sits near 0.0 -- so Person Lock cannot use one absolute threshold. This
    #: prior seeds the cohort normalisation before enough live scores exist,
    #: and must be measured per backend, not guessed.
    impostor_prior: tuple[float, float] = (0.5, 0.15)

    @abstractmethod
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """Return an L2-normalised embedding, or ``None`` if the audio is unusable."""

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity clipped to [0, 1].

        Embeddings are expected to be standardised, so this behaves as a
        correlation: unrelated voices land near zero rather than near one.
        """
        if a is None or b is None or a.size != b.size:
            return 0.0
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        cos = float(np.dot(a, b) / (na * nb))
        return float(np.clip(cos, 0.0, 1.0))

    def reset(self) -> None:
        """Clear recurrent state."""

    def close(self) -> None:
        """Release resources."""

    @property
    def is_available(self) -> bool:
        return True

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "dim": self.dim, "available": self.is_available}

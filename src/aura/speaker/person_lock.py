"""Person Lock: prioritise one trusted speaker's voice.

Enrolment takes a few seconds of a voice, produces an embedding, and discards
the audio. Detection compares the live speech stream against the enrolled
centroid and, when it matches, tells the policy engine to boost that voice and
push competing speech down.

Why cohort normalisation
------------------------
A fixed similarity threshold does not survive contact with real microphones:
absolute cosine scores drift with room, distance and backend. Instead the live
score is compared against a running estimate of what *non-matching* speech
scores in this same session, and the lock fires on the standardised margin.
That makes the decision relative to current conditions, and it is the same
z-norm idea used in production speaker verification.

Privacy
-------
Raw enrolment audio is held only inside :meth:`PersonLockManager.enrol` and is
never written to disk. Only embeddings are persisted, and only when the caller
passes ``consent=True``.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .base import SpeakerIdentifier


@dataclass(slots=True)
class SpeakerProfile:
    """An enrolled voice. Embeddings only -- never audio."""

    id: str
    name: str
    centroid: np.ndarray
    n_samples: int = 1
    within_spread: float = 0.0
    backend: str = ""
    created_at: float = field(default_factory=time.time)
    consented: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "centroid": [float(v) for v in self.centroid],
            "n_samples": self.n_samples,
            "within_spread": self.within_spread,
            "backend": self.backend,
            "created_at": self.created_at,
            "consented": self.consented,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpeakerProfile:
        return cls(
            id=d["id"],
            name=d["name"],
            centroid=np.asarray(d["centroid"], dtype=np.float32),
            n_samples=int(d.get("n_samples", 1)),
            within_spread=float(d.get("within_spread", 0.0)),
            backend=d.get("backend", ""),
            created_at=float(d.get("created_at", time.time())),
            consented=bool(d.get("consented", True)),
        )

    def public(self) -> dict:
        """UI-safe view: identity and provenance, no embedding vector."""
        return {
            "id": self.id,
            "name": self.name,
            "n_samples": self.n_samples,
            "backend": self.backend,
            "created_at": self.created_at,
            "consented": self.consented,
            "embedding_dim": int(self.centroid.size),
        }


@dataclass(slots=True)
class LockResult:
    """Outcome of one Person Lock evaluation."""

    active: bool = False
    profile_id: str | None = None
    name: str | None = None
    similarity: float = 0.0
    margin: float = 0.0
    speech_present: bool = False


class PersonLockManager:
    """Enrolment, storage and live matching for trusted speakers."""

    #: Standardised margin above the impostor distribution needed to lock.
    #:
    #: Calibrated in scripts/eval_speaker.py: the enrolled speaker scores ~1.7
    #: and the nearest impostor ~0.4, so 1.5 sits in the gap with headroom on
    #: both sides. It is paired with the ``best_other`` test below rather than
    #: used alone, because a single z-threshold is fragile when the impostor
    #: distribution is narrow.
    Z_THRESHOLD = 1.5
    #: Weight given to the backend's prior until live cohort scores accumulate.
    PRIOR_STRENGTH = 12.0
    #: Seconds of speech accumulated before a live match is attempted.
    WINDOW_SECONDS = 1.5

    def __init__(
        self,
        identifier: SpeakerIdentifier,
        sample_rate: int,
        *,
        window_seconds: float | None = None,
        z_threshold: float | None = None,
    ) -> None:
        self.identifier = identifier
        self.z_threshold = self.Z_THRESHOLD if z_threshold is None else z_threshold
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds or self.WINDOW_SECONDS
        self.profiles: dict[str, SpeakerProfile] = {}
        self.selected_id: str | None = None

        self._buffer: deque[np.ndarray] = deque()
        self._buffered = 0
        self._max_samples = int(self.window_seconds * sample_rate)
        self._impostor: deque[float] = deque(maxlen=120)
        self._last = LockResult()
        self._hold_until = 0.0

    # -- enrolment ---------------------------------------------------------

    def enrol(
        self,
        name: str,
        clips: list[np.ndarray],
        sample_rate: int,
        *,
        consent: bool = True,
        profile_id: str | None = None,
    ) -> SpeakerProfile:
        """Create a profile from one or more voice clips.

        The clips are consumed here and never retained: only the resulting
        centroid leaves this method.
        """
        if not consent:
            raise PermissionError("speaker enrolment requires explicit consent")

        embeddings = []
        for clip in clips:
            vec = self.identifier.embed(np.asarray(clip, dtype=np.float32), sample_rate)
            if vec is not None:
                embeddings.append(vec)
        if not embeddings:
            raise ValueError(
                "no usable speech found in the enrolment audio "
                f"(need at least {self.identifier.min_audio_seconds:g}s of voice)"
            )

        stack = np.stack(embeddings)
        centroid = stack.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        spread = (
            float(np.mean([1.0 - self.identifier.similarity(e, centroid) for e in embeddings]))
            if len(embeddings) > 1 else 0.0
        )

        profile = SpeakerProfile(
            id=profile_id or uuid.uuid4().hex[:12],
            name=name,
            centroid=centroid.astype(np.float32),
            n_samples=len(embeddings),
            within_spread=spread,
            backend=self.identifier.name,
            consented=True,
        )
        self.profiles[profile.id] = profile
        if self.selected_id is None:
            self.selected_id = profile.id
        return profile

    def delete(self, profile_id: str) -> bool:
        removed = self.profiles.pop(profile_id, None) is not None
        if self.selected_id == profile_id:
            self.selected_id = next(iter(self.profiles), None)
        return removed

    def clear(self) -> int:
        n = len(self.profiles)
        self.profiles.clear()
        self.selected_id = None
        self.reset()
        return n

    def select(self, profile_id: str | None) -> bool:
        if profile_id is None:
            self.selected_id = None
            self.reset()
            return True
        if profile_id not in self.profiles:
            return False
        self.selected_id = profile_id
        self.reset()
        return True

    @property
    def selected(self) -> SpeakerProfile | None:
        return self.profiles.get(self.selected_id) if self.selected_id else None

    # -- live matching -----------------------------------------------------

    def reset(self) -> None:
        self._buffer.clear()
        self._buffered = 0
        self._impostor.clear()
        self._last = LockResult()
        self._hold_until = 0.0

    def update(self, speech_block: np.ndarray, speech_present: bool) -> LockResult:
        """Feed one hop of the primary-speech stream and get the lock state.

        Returns the previous result unchanged until a full window of speech has
        accumulated, so the lock indicator does not flicker between windows.
        """
        target = self.selected
        if target is None:
            self._last = LockResult(speech_present=speech_present)
            return self._last

        if speech_present:
            self._buffer.append(np.asarray(speech_block, dtype=np.float32))
            self._buffered += speech_block.size
            while self._buffered > self._max_samples and len(self._buffer) > 1:
                self._buffered -= self._buffer.popleft().size

        now = time.perf_counter()
        if self._buffered < self._max_samples:
            # Hold the last decision briefly so short pauses do not drop the
            # lock mid-sentence.
            if now > self._hold_until:
                # Keep the last measured score visible: the UI shows a live
                # similarity readout, and zeroing it between windows would make
                # the indicator flicker even while the speaker is still talking.
                self._last = LockResult(
                    active=False,
                    profile_id=target.id,
                    name=target.name,
                    similarity=self._last.similarity,
                    margin=self._last.margin,
                    speech_present=speech_present,
                )
            return self._last

        window = np.concatenate(list(self._buffer))
        vec = self.identifier.embed(window, self.sample_rate)
        # Slide by half a window so decisions update ~2x per window.
        while self._buffered > self._max_samples // 2 and len(self._buffer) > 1:
            self._buffered -= self._buffer.popleft().size

        if vec is None:
            return self._last

        score = self.identifier.similarity(vec, target.centroid)

        # Cohort: every other enrolled profile is a known impostor, which gives
        # the normalisation something real to sit against.
        for pid, prof in self.profiles.items():
            if pid != target.id:
                self._impostor.append(self.identifier.similarity(vec, prof.centroid))

        # A directly competing profile scoring higher is decisive: whoever is
        # talking, it is not the locked speaker.
        best_other = max(
            (self.identifier.similarity(vec, p.centroid)
             for pid, p in self.profiles.items() if pid != target.id),
            default=0.0,
        )

        margin = self._margin(score)
        active = (
            margin >= self.z_threshold
            and score >= self.identifier.impostor_prior[0]
            and score >= best_other
        )

        if active:
            self._hold_until = now + 0.8
        else:
            # Non-matching speech becomes cohort evidence for future frames.
            self._impostor.append(score)

        self._last = LockResult(
            active=active,
            profile_id=target.id,
            name=target.name,
            similarity=score,
            margin=margin,
            speech_present=speech_present,
        )
        return self._last

    def _margin(self, score: float) -> float:
        """Standardised distance of ``score`` above the impostor distribution.

        The backend's prior and the live cohort are blended by shrinkage, so
        the lock is usable from the first window and becomes better calibrated
        to the actual room as evidence accumulates.
        """
        prior_mu, prior_sigma = self.identifier.impostor_prior
        if not self._impostor:
            return (score - prior_mu) / max(prior_sigma, 1e-3)

        arr = np.asarray(self._impostor, dtype=np.float64)
        n = float(arr.size)
        w = n / (n + self.PRIOR_STRENGTH)
        mu = w * float(arr.mean()) + (1.0 - w) * prior_mu
        sigma = w * float(arr.std()) + (1.0 - w) * prior_sigma
        return (score - mu) / max(sigma, 1e-3)

    @property
    def last(self) -> LockResult:
        return self._last

    def snapshot(self) -> dict:
        return {
            "selected_id": self.selected_id,
            "profiles": [p.public() for p in self.profiles.values()],
            "active": self._last.active,
            "name": self._last.name,
            "similarity": round(self._last.similarity, 4),
            "margin": round(self._last.margin, 3),
        }

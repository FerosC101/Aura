"""Name Awareness.

Optional, opt-in, and off by default. When enabled, AURA watches the speech
stream for the listener's name or other trigger words and briefly lifts the
speech that contains them.

Two backends behind one interface:

``TemplateKeywordSpotter`` (default)
    Query-by-example. The user records their name once or twice; AURA stores
    the MFCC trajectory and matches incoming speech against it with dynamic
    time warping. No model download, no transcription, and -- importantly for
    the privacy story -- **nothing is ever transcribed**. The system knows only
    "that sounded like the stored pattern", never what else was said.

``WhisperKeywordSpotter`` (optional)
    Real ASR via ``faster-whisper`` when the extra is installed. More accurate
    and handles trigger words that were never recorded, at the cost of
    transcribing speech locally. Requires its own explicit opt-in because
    transcription is a meaningfully larger privacy step than template matching.

The template backend is the default precisely because it is the more private
option, not merely the lighter one.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..speaker.spectral_embedder import SpectralSpeakerEmbedder


@dataclass(slots=True)
class NameHit:
    """A trigger-word detection."""

    word: str
    score: float
    detected_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "word": self.word,
            "score": round(self.score, 3),
            "detected_at": self.detected_at,
        }


class KeywordSpotter(ABC):
    """Abstract trigger-word detector."""

    name: str = "unnamed"
    #: Whether this backend transcribes speech (drives the consent copy).
    transcribes: bool = False

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int) -> NameHit | None:
        """Scan one window of speech for any configured trigger word."""

    def enrol(self, word: str, clips: list[np.ndarray], sample_rate: int) -> bool:
        """Teach the spotter a word from examples. Not all backends need this."""
        return False

    def forget(self, word: str | None = None) -> int:
        """Erase enrolled words. Part of the interface because the Privacy
        Center's delete-all must work whichever backend is active."""
        return 0

    @property
    def words(self) -> list[str]:
        return []

    def reset(self) -> None: ...

    @property
    def is_available(self) -> bool:
        return True

    def describe(self) -> dict:
        return {
            "name": self.name,
            "available": self.is_available,
            "transcribes": self.transcribes,
            "words": self.words,
        }


def _mfcc_sequence(embedder: SpectralSpeakerEmbedder, x: np.ndarray) -> np.ndarray | None:
    """Per-frame cepstra, silence-trimmed and mean-normalised.

    Energy gating removes the dead air around a word so that two recordings of
    the same name line up on their speech, not on their pauses. Cepstral mean
    subtraction then removes the microphone and room, leaving phonetic shape --
    which is what the matcher should be comparing.
    """
    c = embedder._mfcc(np.asarray(x, dtype=np.float64))
    if c is None or c.shape[0] < 3:
        return None
    energy = c[:, 0]
    keep = energy > (np.max(energy) - 25.0)
    if keep.sum() >= 3:
        c = c[keep]
    c = c[:, 1:13]  # c1..c12: phonetic shape, without loudness
    if c.shape[0] < 3:
        return None
    return c - c.mean(axis=0, keepdims=True)


#: Windows whose length ratio to the template falls outside this range are
#: not compared at all -- a name is short and should not match a sentence.
MIN_LENGTH_RATIO, MAX_LENGTH_RATIO = 0.4, 3.0


def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.34) -> float:
    """Normalised DTW distance between two cepstral sequences.

    A Sakoe-Chiba band constrains the warping path. Beyond the usual speed
    benefit, the band is what stops a long stretch of unrelated speech from
    warping itself into a match for a short name.

    Normalising by ``n + m`` keeps distances comparable across the different
    clip lengths that arise when the same word is said at different speeds.
    """
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    cost = 1.0 - (an @ bn.T)

    width = max(2, int(band * max(n, m)))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, int(i * m / n) - width)
        hi = min(m, int(i * m / n) + width)
        for j in range(lo, hi + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    result = acc[n, m]
    return float(result / (n + m)) if np.isfinite(result) else float("inf")


class TemplateKeywordSpotter(KeywordSpotter):
    """Query-by-example spotting with DTW. No transcription, no model."""

    name = "dtw-template-v1"
    transcribes = False

    #: DTW distance below which a window counts as a match; lower is stricter.
    #:
    #: Calibrated in scripts/eval_name.py: the enrolled name scores 0.16-0.21
    #: and other words 0.33-0.48, so 0.28 sits in a wide gap and gives 6/6
    #: detections with no false positives.
    #:
    #: Scope, stated plainly: this backend matches a name spoken as its own
    #: short utterance -- "Vince?", "Vince, over here" -- which is how a name
    #: is normally used to get someone's attention. It is *not* reliable at
    #: finding a name buried mid-sentence in continuous speech; that needs the
    #: :class:`WhisperKeywordSpotter` backend (the ``asr`` extra).
    DEFAULT_THRESHOLD = 0.28

    def __init__(self, sample_rate: int = 16000, *, threshold: float | None = None) -> None:
        self.sample_rate = sample_rate
        self.threshold = self.DEFAULT_THRESHOLD if threshold is None else threshold
        self._embedder = SpectralSpeakerEmbedder(sample_rate)
        self._templates: dict[str, list[np.ndarray]] = {}

    @property
    def words(self) -> list[str]:
        return list(self._templates)

    def enrol(self, word: str, clips: list[np.ndarray], sample_rate: int) -> bool:
        seqs = []
        for clip in clips:
            x = np.asarray(clip, dtype=np.float64)
            if sample_rate != self.sample_rate:
                x = self._embedder._resample(x, sample_rate, self.sample_rate)
            seq = _mfcc_sequence(self._embedder, x)
            if seq is not None:
                seqs.append(seq)
        if not seqs:
            return False
        self._templates[word] = seqs
        return True

    def forget(self, word: str | None = None) -> int:
        if word is None:
            n = len(self._templates)
            self._templates.clear()
            return n
        return 1 if self._templates.pop(word, None) is not None else 0

    def detect(self, audio: np.ndarray, sample_rate: int) -> NameHit | None:
        if not self._templates:
            return None
        x = np.asarray(audio, dtype=np.float64)
        if sample_rate != self.sample_rate:
            x = self._embedder._resample(x, sample_rate, self.sample_rate)
        seq = _mfcc_sequence(self._embedder, x)
        if seq is None:
            return None

        best_word, best_dist = None, float("inf")
        for word, templates in self._templates.items():
            for tpl in templates:
                ratio = seq.shape[0] / max(tpl.shape[0], 1)
                if not (MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO):
                    continue
                d = dtw_distance(tpl, seq)
                if d < best_dist:
                    best_word, best_dist = word, d

        if best_word is None or best_dist > self.threshold:
            return None
        score = float(np.clip(1.0 - best_dist / max(self.threshold, 1e-6), 0.0, 1.0))
        return NameHit(word=best_word, score=score)


class WhisperKeywordSpotter(KeywordSpotter):
    """Local ASR trigger-word spotting. Requires the ``asr`` extra."""

    name = "faster-whisper"
    transcribes = True

    def __init__(self, words: list[str], *, model_size: str = "tiny.en") -> None:
        self._words = [w.strip().lower() for w in words if w.strip()]
        self._model = None
        self._error: str | None = None
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as exc:  # pragma: no cover - optional dependency
            self._error = str(exc)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def words(self) -> list[str]:
        return list(self._words)

    def detect(self, audio: np.ndarray, sample_rate: int) -> NameHit | None:  # pragma: no cover
        if self._model is None or not self._words:
            return None
        try:
            segments, _ = self._model.transcribe(
                np.asarray(audio, dtype=np.float32), language="en", beam_size=1
            )
            text = " ".join(s.text for s in segments).lower()
        except Exception:
            return None
        for w in self._words:
            if w in text:
                return NameHit(word=w, score=1.0)
        return None

    def describe(self) -> dict:
        d = super().describe()
        if self._error:
            d["error"] = self._error
        return d


class NameAwareness:
    """Buffers speech and reports trigger-word hits with a hold time.

    The hold exists so that a detection produces an audible, sustained lift
    rather than a single-frame blip the listener would never notice.
    """

    #: Seconds of speech examined per detection attempt.
    #:
    #: Kept close to the length of a spoken name. The matcher compares whole
    #: sequences, so a much longer buffer dilutes the name among everything
    #: else said around it.
    WINDOW_SECONDS = 1.0
    #: How long a hit keeps boosting speech afterwards.
    HOLD_SECONDS = 2.5
    #: Minimum gap between detections, so one utterance fires once.
    COOLDOWN_SECONDS = 1.5

    def __init__(
        self,
        spotter: KeywordSpotter,
        sample_rate: int,
        *,
        enabled: bool = False,
    ) -> None:
        self.spotter = spotter
        self.sample_rate = sample_rate
        self.enabled = enabled
        self._buffer: deque[np.ndarray] = deque()
        self._buffered = 0
        self._window = int(self.WINDOW_SECONDS * sample_rate)
        self._hold_until = 0.0
        self._last_attempt = 0.0
        self.last_hit: NameHit | None = None

    def reset(self) -> None:
        self._buffer.clear()
        self._buffered = 0
        self._hold_until = 0.0
        self._last_attempt = 0.0
        self.last_hit = None

    @property
    def active(self) -> bool:
        """Whether a recent hit is still lifting the speech stream."""
        return self.enabled and time.time() < self._hold_until

    def update(self, speech_block: np.ndarray, speech_present: bool) -> bool:
        """Feed one hop of speech. Returns whether Name Awareness is active."""
        if not self.enabled or not self.spotter.words:
            return False

        if speech_present:
            self._buffer.append(np.asarray(speech_block, dtype=np.float32))
            self._buffered += speech_block.size
            while self._buffered > self._window and len(self._buffer) > 1:
                self._buffered -= self._buffer.popleft().size

        now = time.time()
        if now < self._hold_until:
            return True
        if self._buffered < self._window or (now - self._last_attempt) < self.COOLDOWN_SECONDS:
            return False

        self._last_attempt = now
        window = np.concatenate(list(self._buffer))
        hit = self.spotter.detect(window, self.sample_rate)
        if hit is not None:
            self.last_hit = hit
            self._hold_until = now + self.HOLD_SECONDS
            self._buffer.clear()
            self._buffered = 0
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "active": self.active,
            "backend": self.spotter.describe(),
            "words": self.spotter.words,
            "last_hit": self.last_hit.as_dict() if self.last_hit else None,
        }

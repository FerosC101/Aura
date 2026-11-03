"""The personalised Acoustic Profile.

Everything AURA remembers about one listener, and the only thing it writes to
disk. The privacy contract is enforced by what this class *can* hold: there is
no field for audio, and there is no code path that puts a waveform here.

Stored:

* preferred and most-used listening modes,
* explicit mixer positions per mode (the user's own overrides),
* which categories they habitually preserve or suppress,
* suppression intensity,
* speaker embeddings for Person Lock (vectors only, consent-gated),
* trigger words for Name Awareness,
* learned preference records.

Not stored, ever: raw or encoded microphone audio, transcripts, or timestamped
logs of what was heard.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..types import ListeningMode
from .learning import PreferenceLearningEngine

SCHEMA_VERSION = 1


def default_profile_dir() -> Path:
    """Local, per-user storage location. Overridable with ``AURA_HOME``."""
    env = os.environ.get("AURA_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".aura"


@dataclass(slots=True)
class PrivacySettings:
    """Every privacy-relevant switch, defaulted to the conservative choice."""

    store_speaker_embeddings: bool = False
    allow_motion_context: bool = False
    allow_cloud_inference: bool = False
    allow_name_awareness: bool = False
    auto_context_detection: bool = False
    retain_scene_history: bool = False

    def as_dict(self) -> dict:
        return {
            "store_speaker_embeddings": self.store_speaker_embeddings,
            "allow_motion_context": self.allow_motion_context,
            "allow_cloud_inference": self.allow_cloud_inference,
            "allow_name_awareness": self.allow_name_awareness,
            "auto_context_detection": self.auto_context_detection,
            "retain_scene_history": self.retain_scene_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PrivacySettings:
        return cls(**{k: bool(v) for k, v in d.items() if k in cls.__slots__})


@dataclass(slots=True)
class AcousticProfile:
    """One listener's stored preferences."""

    user_name: str = ""
    onboarded: bool = False
    preferred_mode: ListeningMode = ListeningMode.COMMUTE
    primary_uses: list[str] = field(default_factory=list)

    #: mode -> {category: gain_db}. Explicit mixer positions.
    mode_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Global 0..1 scaling of suppression depth.
    suppression_strength: float = 1.0
    #: Safety categories the user has chosen to keep protected.
    safety_categories: list[str] = field(default_factory=list)
    #: Name Awareness trigger words.
    trigger_words: list[str] = field(default_factory=list)

    #: Usage counters, used to surface a preferred mode over time.
    mode_usage: dict[str, float] = field(default_factory=dict)
    #: Rolling counts of categories the user pushes up / pulls down.
    preserve_counts: dict[str, int] = field(default_factory=dict)
    suppress_counts: dict[str, int] = field(default_factory=dict)

    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    #: Serialized SpeakerProfile dicts. Only written with consent.
    speaker_profiles: list[dict] = field(default_factory=list)
    selected_speaker_id: str | None = None

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # -- overrides ---------------------------------------------------------

    def overrides_for(self, mode: ListeningMode | str) -> dict[str, float]:
        key = mode.value if isinstance(mode, ListeningMode) else str(mode)
        return dict(self.mode_overrides.get(key, {}))

    def set_override(
        self, mode: ListeningMode | str, category: str, gain_db: float | None
    ) -> None:
        """Set or clear one mixer position. ``None`` returns it to the mode default."""
        key = mode.value if isinstance(mode, ListeningMode) else str(mode)
        bucket = self.mode_overrides.setdefault(key, {})
        if gain_db is None:
            bucket.pop(category, None)
        else:
            bucket[category] = float(gain_db)
            if gain_db > 0:
                self.preserve_counts[category] = self.preserve_counts.get(category, 0) + 1
            elif gain_db < -6:
                self.suppress_counts[category] = self.suppress_counts.get(category, 0) + 1
        self.updated_at = time.time()

    def note_mode_use(self, mode: ListeningMode | str, seconds: float) -> None:
        key = mode.value if isinstance(mode, ListeningMode) else str(mode)
        self.mode_usage[key] = self.mode_usage.get(key, 0.0) + max(seconds, 0.0)
        self.updated_at = time.time()

    @property
    def most_used_mode(self) -> ListeningMode:
        if not self.mode_usage:
            return self.preferred_mode
        return ListeningMode(max(self.mode_usage, key=lambda k: self.mode_usage[k]))

    def frequently_preserved(self, top: int = 5) -> list[str]:
        return [k for k, _ in sorted(self.preserve_counts.items(), key=lambda kv: -kv[1])[:top]]

    def frequently_suppressed(self, top: int = 5) -> list[str]:
        return [k for k, _ in sorted(self.suppress_counts.items(), key=lambda kv: -kv[1])[:top]]

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "user_name": self.user_name,
            "onboarded": self.onboarded,
            "preferred_mode": self.preferred_mode.value,
            "primary_uses": list(self.primary_uses),
            "mode_overrides": {k: dict(v) for k, v in self.mode_overrides.items()},
            "suppression_strength": self.suppression_strength,
            "safety_categories": list(self.safety_categories),
            "trigger_words": list(self.trigger_words),
            "mode_usage": dict(self.mode_usage),
            "preserve_counts": dict(self.preserve_counts),
            "suppress_counts": dict(self.suppress_counts),
            "privacy": self.privacy.as_dict(),
            "speaker_profiles": list(self.speaker_profiles),
            "selected_speaker_id": self.selected_speaker_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AcousticProfile:
        return cls(
            user_name=d.get("user_name", ""),
            onboarded=bool(d.get("onboarded", False)),
            preferred_mode=ListeningMode(d.get("preferred_mode", "commute")),
            primary_uses=list(d.get("primary_uses", [])),
            mode_overrides={k: dict(v) for k, v in d.get("mode_overrides", {}).items()},
            suppression_strength=float(d.get("suppression_strength", 1.0)),
            safety_categories=list(d.get("safety_categories", [])),
            trigger_words=list(d.get("trigger_words", [])),
            mode_usage=dict(d.get("mode_usage", {})),
            preserve_counts={k: int(v) for k, v in d.get("preserve_counts", {}).items()},
            suppress_counts={k: int(v) for k, v in d.get("suppress_counts", {}).items()},
            privacy=PrivacySettings.from_dict(d.get("privacy", {})),
            speaker_profiles=list(d.get("speaker_profiles", [])),
            selected_speaker_id=d.get("selected_speaker_id"),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )

    def public(self) -> dict:
        """UI-facing view. Speaker embeddings are summarised, never sent."""
        d = self.to_dict()
        d["speaker_profiles"] = [
            {k: v for k, v in p.items() if k != "centroid"} | {
                "embedding_dim": len(p.get("centroid", []))
            }
            for p in self.speaker_profiles
        ]
        d["most_used_mode"] = self.most_used_mode.value
        d["frequently_preserved"] = self.frequently_preserved()
        d["frequently_suppressed"] = self.frequently_suppressed()
        return d


class ProfileStore:
    """Atomic JSON persistence for a profile plus its learned preferences.

    Writes go to a temporary file in the same directory and are then renamed,
    so an interrupted write cannot leave a half-written profile behind.
    """

    def __init__(self, directory: Path | str | None = None, *, filename: str = "profile.json") -> None:
        self.directory = Path(directory) if directory else default_profile_dir()
        self.path = self.directory / filename

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> tuple[AcousticProfile, PreferenceLearningEngine]:
        learner = PreferenceLearningEngine()
        if not self.exists():
            return AcousticProfile(), learner
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt profile must not stop the engine from starting.
            return AcousticProfile(), learner
        profile = AcousticProfile.from_dict(raw.get("profile", {}))
        learner.load_dict(raw.get("learning", {}))
        return profile, learner

    def save(self, profile: AcousticProfile, learner: PreferenceLearningEngine) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "profile": profile.to_dict(),
            "learning": learner.to_dict(),
        }
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return self.path

    def delete(self) -> bool:
        """Erase everything on disk. Backs the Privacy Center's delete-all."""
        if self.path.exists():
            self.path.unlink()
            return True
        return False

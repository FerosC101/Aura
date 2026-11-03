"""Preference learning.

AURA watches what the listener corrects. If someone keeps pushing announcements
up during Commute Mode, announcements should start louder in Commute Mode.

The MVP implements this as **weighted preference scores**: a per-(mode,
category) bias in dB, updated from manual adjustments by an exponential moving
average with a confidence weight that grows as evidence accumulates.

Why this shape
--------------
The update rule is deliberately written as a bandit-style interface --
:meth:`observe_adjustment` is an *reward signal keyed by context and arm*, and
:meth:`bias_for` is *the current policy for that context*. Replacing the EMA
with a contextual bandit or a continual-learning model means reimplementing
those two methods; nothing else in the system needs to change.

Three safeguards, because a learner in the audio path can do real harm:

* Bias is clamped to +/-6 dB, so learning tunes the mix and never takes it over.
* Safety-critical categories are excluded outright -- no amount of habitual
  dismissal teaches AURA to quieten a siren.
* Learning decays towards neutral when a preference stops being reinforced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..classification.taxonomy import is_safety_critical
from ..types import ListeningMode

#: Hard limit on learned influence, in dB.
MAX_BIAS_DB = 6.0
#: Adjustments smaller than this are treated as noise, not intent.
MIN_ADJUSTMENT_DB = 1.0


@dataclass(slots=True)
class PreferenceRecord:
    """Accumulated evidence for one (mode, category) pair."""

    mode: str
    category: str
    bias_db: float = 0.0
    weight: float = 0.0
    observations: int = 0
    last_seen: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        """Full-precision view, used for persistence.

        Deliberately unrounded: this is the stored form, and rounding it would
        make a save/load cycle lossy for no benefit. The display rounding lives
        in :meth:`PreferenceLearningEngine.summary` instead.
        """
        return {
            "mode": self.mode,
            "category": self.category,
            "bias_db": self.bias_db,
            "weight": self.weight,
            "observations": self.observations,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PreferenceRecord:
        return cls(
            mode=d["mode"],
            category=d["category"],
            bias_db=float(d.get("bias_db", 0.0)),
            weight=float(d.get("weight", 0.0)),
            observations=int(d.get("observations", 0)),
            last_seen=float(d.get("last_seen", time.time())),
        )


class PreferenceLearningEngine:
    """Learns per-mode, per-category gain biases from manual corrections."""

    #: How fast a new observation moves the stored bias.
    LEARNING_RATE = 0.28
    #: Observations needed before the bias is applied at full strength.
    CONFIDENCE_SATURATION = 4.0
    #: Half-life of an unreinforced preference, in days.
    DECAY_HALF_LIFE_DAYS = 21.0

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.records: dict[tuple[str, str], PreferenceRecord] = {}

    # -- learning ----------------------------------------------------------

    def observe_adjustment(
        self,
        mode: ListeningMode | str,
        category: str,
        delta_db: float,
    ) -> PreferenceRecord | None:
        """Record that the user moved ``category`` by ``delta_db`` in ``mode``.

        ``delta_db`` is the difference between what the user set and what the
        policy engine had chosen -- the correction, not the absolute value.
        Learning from the correction is what makes the signal meaningful: it
        says "you were 4 dB too quiet", which stays true if the mode's baseline
        later changes.
        """
        if not self.enabled:
            return None
        if abs(delta_db) < MIN_ADJUSTMENT_DB:
            return None
        if is_safety_critical(category) and delta_db < 0:
            # Never learn to suppress a safety sound, however often asked.
            return None

        mode_key = mode.value if isinstance(mode, ListeningMode) else str(mode)
        key = (mode_key, category)
        rec = self.records.get(key) or PreferenceRecord(mode=mode_key, category=category)

        target = float(np.clip(delta_db, -MAX_BIAS_DB, MAX_BIAS_DB))
        rec.bias_db = float(
            np.clip(
                rec.bias_db + self.LEARNING_RATE * (target - rec.bias_db),
                -MAX_BIAS_DB,
                MAX_BIAS_DB,
            )
        )
        rec.observations += 1
        rec.weight = min(1.0, rec.observations / self.CONFIDENCE_SATURATION)
        rec.last_seen = time.time()
        self.records[key] = rec
        return rec

    # -- policy ------------------------------------------------------------

    def bias_for(self, mode: ListeningMode | str, category: str) -> float:
        """Current learned bias in dB, scaled by confidence and recency."""
        if not self.enabled:
            return 0.0
        mode_key = mode.value if isinstance(mode, ListeningMode) else str(mode)
        rec = self.records.get((mode_key, category))
        if rec is None:
            return 0.0
        return rec.bias_db * rec.weight * self._recency(rec)

    def biases_for_mode(self, mode: ListeningMode | str) -> dict[str, float]:
        """Every learned bias applying to ``mode``, ready for PolicyContext."""
        mode_key = mode.value if isinstance(mode, ListeningMode) else str(mode)
        out: dict[str, float] = {}
        for (m, category), rec in self.records.items():
            if m != mode_key:
                continue
            value = rec.bias_db * rec.weight * self._recency(rec)
            if abs(value) > 0.1:
                out[category] = float(value)
        return out

    def _recency(self, rec: PreferenceRecord) -> float:
        """Exponential decay, so stale habits fade instead of persisting forever."""
        days = max(0.0, (time.time() - rec.last_seen) / 86400.0)
        return float(0.5 ** (days / self.DECAY_HALF_LIFE_DAYS))

    # -- management --------------------------------------------------------

    def forget(self, mode: str | None = None, category: str | None = None) -> int:
        """Delete learned preferences. Backs the Privacy Center's reset."""
        victims = [
            k for k in self.records
            if (mode is None or k[0] == mode) and (category is None or k[1] == category)
        ]
        for k in victims:
            del self.records[k]
        return len(victims)

    def summary(self) -> list[dict]:
        """Human-readable view of what has been learned, strongest first."""
        rows = []
        for rec in self.records.values():
            effective = rec.bias_db * rec.weight * self._recency(rec)
            if abs(effective) < 0.1:
                continue
            rows.append({
                **rec.as_dict(),
                "bias_db": round(rec.bias_db, 2),
                "weight": round(rec.weight, 3),
                "effective_db": round(effective, 2),
                "direction": "louder" if effective > 0 else "quieter",
            })
        rows.sort(key=lambda r: -abs(r["effective_db"]))
        return rows

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "records": [r.as_dict() for r in self.records.values()],
        }

    def load_dict(self, d: dict) -> None:
        self.enabled = bool(d.get("enabled", True))
        self.records = {}
        for raw in d.get("records", []):
            rec = PreferenceRecord.from_dict(raw)
            self.records[(rec.mode, rec.category)] = rec

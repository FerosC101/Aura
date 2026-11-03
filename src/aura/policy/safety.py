"""The safety layer.

:class:`SafetyGuard` is the last component to touch a gain decision and the
only one that cannot be overridden -- not by a mode, not by the user's mixer,
not by learned preferences. Its job is narrow and absolute: a sound that could
warn you about physical danger must stay audible.

Three rules, in order of application:

1. **Floor.** A safety-critical category can never be attenuated below
   :data:`MIN_SAFETY_GAIN_DB`, whatever the mode or the user asked for.
2. **Lift.** A quiet safety sound in a loud scene is boosted so it clears the
   rest of the mix rather than merely surviving at its own level.
3. **Doubt favours audibility.** When classification confidence sits in the
   uncertain band, a *possibly* safety-critical sound is preserved instead of
   suppressed. A false alarm costs the user a moment of unwanted noise; a
   missed siren costs considerably more.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..classification.taxonomy import get_category, is_safety_critical
from ..types import Action, GainDecision, Priority, SoundEvent

#: No safety-critical sound is ever attenuated past this, in dB.
MIN_SAFETY_GAIN_DB = -3.0

#: Below this confidence a detection is too weak to act on at all.
UNCERTAIN_LOW = 0.30
#: Between UNCERTAIN_LOW and this, a safety category gets the benefit of doubt.
UNCERTAIN_HIGH = 0.72

#: Extra gain applied to a safety sound that is quiet relative to the scene.
MAX_SAFETY_LIFT_DB = 8.0


@dataclass(slots=True)
class SafetyVerdict:
    """Result of applying the guard to one decision."""

    decision: GainDecision
    overridden: bool
    reason: str = ""


class SafetyGuard:
    """Enforces minimum audibility for safety-critical sounds."""

    def __init__(
        self,
        *,
        enabled_categories: set[str] | None = None,
        min_gain_db: float = MIN_SAFETY_GAIN_DB,
        lift_enabled: bool = True,
    ) -> None:
        #: Categories the user has kept switched on in onboarding / Privacy
        #: Center. Defaults to every safety-critical category in the taxonomy.
        self.enabled_categories = enabled_categories
        self.min_gain_db = min_gain_db
        self.lift_enabled = lift_enabled
        self.last_override_count = 0

    # -- queries -----------------------------------------------------------

    def is_protected(self, category: str) -> bool:
        if not is_safety_critical(category):
            return False
        if self.enabled_categories is None:
            return True
        return category in self.enabled_categories

    def protected_categories(self) -> list[str]:
        from ..classification.taxonomy import all_categories

        return [c.key for c in all_categories() if self.is_protected(c.key)]

    # -- enforcement -------------------------------------------------------

    def apply(
        self,
        decision: GainDecision,
        event: SoundEvent,
        *,
        scene_level_db: float = -30.0,
    ) -> SafetyVerdict:
        """Clamp one decision so a protected sound stays audible."""
        category = decision.category
        if not self.is_protected(category):
            return SafetyVerdict(decision, False)

        overridden = False
        reasons: list[str] = []
        gain = decision.gain_db
        action = decision.action

        # Rule 1: hard floor.
        if gain < self.min_gain_db:
            gain = self.min_gain_db
            action = Action.PRESERVE
            overridden = True
            reasons.append(f"floor at {self.min_gain_db:+.0f} dB")

        # Rule 2: lift a quiet warning above a loud scene.
        if self.lift_enabled:
            deficit = scene_level_db - event.level_db
            if deficit > 6.0:
                lift = min(MAX_SAFETY_LIFT_DB, deficit - 6.0)
                if lift > 0.5:
                    gain = max(gain, lift)
                    overridden = True
                    reasons.append(f"lifted {lift:+.0f} dB over scene")

        # Rule 3: uncertainty resolves towards audibility.
        if UNCERTAIN_LOW <= event.confidence < UNCERTAIN_HIGH and gain < 0.0:
            gain = 0.0
            action = Action.PRESERVE
            overridden = True
            reasons.append(f"uncertain ({event.confidence:.0%}) -- preserved")

        if not overridden:
            # Still mark it critical so downstream components know not to
            # trim it during headroom management.
            decision.always_audible = True
            decision.priority = Priority.CRITICAL
            return SafetyVerdict(decision, False)

        self.last_override_count += 1
        return SafetyVerdict(
            GainDecision(
                category=category,
                action=action,
                gain_db=gain,
                priority=Priority.CRITICAL,
                always_audible=True,
                safety_override=True,
                reason="; ".join(reasons),
                stream=decision.stream,
                confidence=decision.confidence,
            ),
            True,
            "; ".join(reasons),
        )

    def describe(self) -> dict:
        return {
            "min_gain_db": self.min_gain_db,
            "lift_enabled": self.lift_enabled,
            "protected": self.protected_categories(),
            "uncertain_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
            "labels": {c: get_category(c).label for c in self.protected_categories()},
        }

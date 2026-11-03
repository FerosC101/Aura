"""The Acoustic Policy Engine.

This is where AURA decides what the world should sound like. It takes the
acoustic scene -- what is out there and how sure we are -- plus everything we
know about the listener, and produces one :class:`~aura.types.GainDecision`
per detected sound.

The decision is built in layers, each of which can only be overridden by the
layer beneath it:

1. **Mode intent** -- the declarative table for the active listening mode.
2. **User override** -- an explicit mixer position always wins over the mode.
3. **Learned preference** -- a nudge from repeated manual corrections.
4. **Person Lock** -- promote the locked voice, demote what competes with it.
5. **Name Awareness** -- a temporary lift when the user's name is heard.
6. **Confidence scaling** -- act tentatively on tentative detections.
7. **SafetyGuard** -- absolute, and last. Nothing above it can silence a siren.

Ordering matters and is the point: learned behaviour can shape the mix, but it
can never talk the system out of a safety floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..classification.taxonomy import get_category
from ..types import (
    Action,
    ContextState,
    GainDecision,
    ListeningMode,
    Priority,
    SoundEvent,
    StreamKind,
)
from .modes import ACTION_GAIN_DB, ModeProfile, get_mode
from .safety import UNCERTAIN_LOW, SafetyGuard

#: Gain applied to the locked speaker's stream, in dB.
PERSON_LOCK_BOOST_DB = 6.0
#: Extra attenuation applied to speech competing with the locked speaker.
PERSON_LOCK_DUCK_DB = -7.0
#: Temporary lift when the user's name is detected.
NAME_AWARENESS_BOOST_DB = 5.0


@dataclass(slots=True)
class PolicyContext:
    """Everything the engine needs to know beyond the sounds themselves."""

    mode: ListeningMode = ListeningMode.COMMUTE
    context: ContextState = ContextState.UNKNOWN
    person_lock_active: bool = False
    person_lock_name: str | None = None
    name_detected: bool = False
    #: Explicit user mixer positions, category -> gain in dB.
    user_overrides: dict[str, float] = field(default_factory=dict)
    #: Learned nudges, category -> gain delta in dB.
    learned_bias: dict[str, float] = field(default_factory=dict)
    #: Global 0..1 scaling of how hard suppression is applied.
    suppression_strength: float = 1.0
    #: Master toggle. When off, everything passes through at unity.
    enabled: bool = True
    scene_level_db: float = -30.0


class AcousticPolicyEngine:
    """Turns detections plus context into per-category gain decisions."""

    def __init__(self, safety_guard: SafetyGuard | None = None) -> None:
        self.safety = safety_guard or SafetyGuard()
        self.last_safety_override = False

    # -- main entry point --------------------------------------------------

    def decide(
        self,
        events: list[SoundEvent],
        ctx: PolicyContext,
    ) -> list[GainDecision]:
        """Produce one decision per detected event."""
        if not ctx.enabled:
            return [
                GainDecision(
                    category=e.category,
                    action=Action.PRESERVE,
                    gain_db=0.0,
                    priority=Priority.NORMAL,
                    reason="AURA bypassed",
                    stream=e.stream,
                    confidence=e.confidence,
                )
                for e in events
            ]

        profile = get_mode(ctx.mode)
        decisions: list[GainDecision] = []
        any_override = False

        for event in events:
            decision = self._decide_one(event, profile, ctx)
            verdict = self.safety.apply(decision, event, scene_level_db=ctx.scene_level_db)
            any_override |= verdict.overridden
            decisions.append(verdict.decision)

        self.last_safety_override = any_override
        decisions.sort(key=lambda d: (-d.priority.rank, -d.confidence))
        return decisions

    # -- layers ------------------------------------------------------------

    def _decide_one(
        self,
        event: SoundEvent,
        profile: ModeProfile,
        ctx: PolicyContext,
    ) -> GainDecision:
        category = event.category
        meta = get_category(category)
        reasons: list[str] = []

        # 1. Mode intent.
        action = profile.intent_for(category)
        gain = ACTION_GAIN_DB[action]
        priority = profile.priority_for(category)
        reasons.append(f"{profile.label}: {action.value}")

        # Modes scale how hard they suppress; Walk deliberately stays gentle.
        if gain < 0:
            gain *= profile.suppression_scale * ctx.suppression_strength

        # 2. Explicit user override beats the mode outright.
        if category in ctx.user_overrides:
            gain = float(ctx.user_overrides[category])
            action = _action_for_gain(gain)
            priority = max(priority, Priority.NORMAL, key=lambda p: p.rank)
            reasons.append(f"you set {gain:+.0f} dB")

        # 3. Learned nudge from repeated manual corrections.
        bias = float(ctx.learned_bias.get(category, 0.0))
        if abs(bias) > 0.25:
            gain += bias
            reasons.append(f"learned {bias:+.1f} dB")

        # 4. Person Lock.
        if ctx.person_lock_active:
            if event.stream is StreamKind.PRIMARY_SPEECH:
                gain += PERSON_LOCK_BOOST_DB
                priority = Priority.HIGH
                action = _action_for_gain(gain)
                reasons.append(f"Person Lock: {ctx.person_lock_name}")
            elif event.stream is StreamKind.SECONDARY_SPEECH:
                gain += PERSON_LOCK_DUCK_DB * ctx.suppression_strength
                action = _action_for_gain(gain)
                reasons.append("ducked behind locked speaker")

        # 5. Name Awareness.
        if ctx.name_detected and event.stream in (
            StreamKind.PRIMARY_SPEECH,
            StreamKind.SECONDARY_SPEECH,
        ):
            gain = max(gain, 0.0) + NAME_AWARENESS_BOOST_DB
            priority = max(priority, Priority.HIGH, key=lambda p: p.rank)
            action = _action_for_gain(gain)
            reasons.append("your name was heard")

        # 6. Confidence scaling.
        #
        # A tentative detection should produce a tentative action. Scaling the
        # gain towards 0 dB means an unsure "that might be an engine" trims a
        # few dB instead of gouging 16 dB out of something that may be a voice.
        confidence_weight = _confidence_weight(event.confidence)
        if confidence_weight < 0.999:
            gain *= confidence_weight
            reasons.append(f"scaled to {event.confidence:.0%} confidence")

        gain = float(np.clip(gain, -40.0, 12.0))

        return GainDecision(
            category=category,
            action=_action_for_gain(gain),
            gain_db=gain,
            priority=priority,
            always_audible=meta.safety_critical,
            reason="; ".join(reasons),
            stream=event.stream,
            confidence=event.confidence,
        )

    # -- explainability ----------------------------------------------------

    def explain(self, category: str, ctx: PolicyContext) -> dict:
        """Describe what would happen to ``category`` right now, and why.

        Used by the UI's per-category inspector and by the test suite; it runs
        the same code path as a live decision with a nominal confidence.
        """
        event = SoundEvent(
            category=category,
            confidence=0.9,
            stream=get_category(category).stream,
            level_db=ctx.scene_level_db,
        )
        decision = self._decide_one(event, get_mode(ctx.mode), ctx)
        verdict = self.safety.apply(decision, event, scene_level_db=ctx.scene_level_db)
        return {
            "category": category,
            "label": get_category(category).label,
            "action": verdict.decision.action.value,
            "gain_db": round(verdict.decision.gain_db, 1),
            "priority": verdict.decision.priority.value,
            "safety_override": verdict.decision.safety_override,
            "always_audible": verdict.decision.always_audible,
            "reason": verdict.decision.reason,
        }


def _confidence_weight(confidence: float) -> float:
    """How much of the intended gain change to actually apply.

    Ramps from 0 at :data:`~aura.policy.safety.UNCERTAIN_LOW` to 1 at 0.85, so
    weak detections barely move the mix and strong ones act in full.
    """
    if confidence <= UNCERTAIN_LOW:
        return 0.0
    return float(np.clip((confidence - UNCERTAIN_LOW) / (0.85 - UNCERTAIN_LOW), 0.0, 1.0))


def _action_for_gain(gain_db: float) -> Action:
    """Label a gain value, so the UI and the engine never disagree."""
    if gain_db >= 2.0:
        return Action.ENHANCE
    if gain_db > -3.0:
        return Action.PRESERVE
    if gain_db > -12.0:
        return Action.REDUCE
    return Action.SUPPRESS

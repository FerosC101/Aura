"""Context detection.

Decides -- or accepts -- which situation the listener is in, so the right
listening mode can be suggested. Manual selection always wins; automatic
detection is a suggestion the user can accept, and it never switches modes on
its own.

Evidence sources, in the order they are trusted:

1. **Manual declaration.** The user said what they are doing. Done.
2. **Motion and location hints.** Supplied by the caller (a phone client, a
   watch) and only when the user has opted in. AURA never reaches for these
   itself.
3. **Acoustic signature.** Always available and privacy-neutral, because it
   uses the classifier output that already exists rather than any new data.

The acoustic evidence is accumulated over ~10 seconds. Context is a property of
a situation, not of a frame, and reacting to single frames would have the mode
flickering between Walk and Commute at every traffic light.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..types import ContextState, ListeningMode, SoundEvent

#: Which mode each context suggests.
CONTEXT_TO_MODE: dict[ContextState, ListeningMode] = {
    ContextState.STATIONARY: ListeningMode.FOCUS,
    ContextState.STUDYING: ListeningMode.FOCUS,
    ContextState.COMMUTING: ListeningMode.COMMUTE,
    ContextState.WALKING: ListeningMode.WALK,
    ContextState.SOCIAL: ListeningMode.SOCIAL,
    ContextState.UNKNOWN: ListeningMode.COMMUTE,
}


@dataclass(slots=True)
class MotionHint:
    """Optional client-supplied motion/location signal. Opt-in only."""

    speed_mps: float | None = None
    step_rate_hz: float | None = None
    place_category: str | None = None  # "transit", "campus", "outdoors", "cafe", ...
    received_at: float = field(default_factory=time.time)

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.received_at) < 30.0


@dataclass(slots=True)
class ContextReading:
    state: ContextState
    confidence: float
    suggested_mode: ListeningMode
    source: str
    evidence: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "confidence": round(self.confidence, 3),
            "suggested_mode": self.suggested_mode.value,
            "source": self.source,
            "evidence": {k: round(v, 3) for k, v in self.evidence.items()},
        }


class ContextEngine:
    """Accumulates evidence and reports the listener's situation."""

    #: Seconds of history used for the acoustic vote.
    WINDOW_SECONDS = 10.0

    def __init__(self, *, auto_enabled: bool = False, allow_motion: bool = False) -> None:
        #: Experimental automatic detection. Off by default -- the MVP's
        #: contract is that the user picks the mode and AURA may suggest.
        self.auto_enabled = auto_enabled
        #: Motion/GPS consumption. Opt-in, per the privacy requirements.
        self.allow_motion = allow_motion

        self.manual_state: ContextState | None = None
        self.motion: MotionHint | None = None
        self._history: deque[tuple[float, dict[str, float], float]] = deque(maxlen=600)
        self._last = ContextReading(
            ContextState.UNKNOWN, 0.0, ListeningMode.COMMUTE, "default"
        )

    # -- inputs ------------------------------------------------------------

    def declare(self, state: ContextState | str | None) -> None:
        """Manual context declaration; overrides detection entirely."""
        if state is None:
            self.manual_state = None
            return
        self.manual_state = ContextState(state) if isinstance(state, str) else state

    def submit_motion(self, hint: MotionHint) -> bool:
        """Accept a motion hint. Returns False if the user has not opted in."""
        if not self.allow_motion:
            return False
        self.motion = hint
        return True

    def observe(self, events: list[SoundEvent], level_db: float) -> None:
        """Feed one frame's classifier output into the evidence window."""
        weights = {e.category: e.confidence for e in events}
        self._history.append((time.time(), weights, level_db))

    # -- detection ---------------------------------------------------------

    def read(self) -> ContextReading:
        if self.manual_state is not None:
            self._last = ContextReading(
                state=self.manual_state,
                confidence=1.0,
                suggested_mode=CONTEXT_TO_MODE[self.manual_state],
                source="manual",
            )
            return self._last

        if not self.auto_enabled:
            return self._last

        evidence = self._acoustic_evidence()
        source = "acoustic"

        if self.allow_motion and self.motion is not None and self.motion.is_fresh:
            evidence = self._apply_motion(evidence)
            source = "acoustic+motion"

        if not evidence:
            return self._last

        state = max(evidence, key=lambda k: evidence[k])
        total = sum(evidence.values()) or 1.0
        confidence = evidence[state] / total

        self._last = ContextReading(
            state=state,
            confidence=float(confidence),
            suggested_mode=CONTEXT_TO_MODE[state],
            source=source,
            evidence={k.value: v for k, v in evidence.items()},
        )
        return self._last

    def _acoustic_evidence(self) -> dict[ContextState, float]:
        """Vote for each context from the recent acoustic scene."""
        cutoff = time.time() - self.WINDOW_SECONDS
        recent = [(w, lv) for t, w, lv in self._history if t >= cutoff]
        if len(recent) < 10:
            return {}

        def mean_of(cat: str) -> float:
            return float(np.mean([w.get(cat, 0.0) for w, _ in recent]))

        levels = np.asarray([lv for _, lv in recent], dtype=np.float64)
        loudness = float(np.mean(levels))
        # How much the level moves: a street varies far more than a library.
        variability = float(np.std(levels))

        engine = mean_of("engine_noise")
        traffic = mean_of("general_traffic")
        approaching = mean_of("approaching_vehicle")
        horn = mean_of("vehicle_horn")
        speech = mean_of("human_speech")
        crowd = mean_of("crowd_speech")
        announcement = mean_of("announcement")
        ambient = mean_of("ambient")
        mechanical = mean_of("mechanical_noise")

        # Commuting: you are *inside* something running. Steady engine, PA
        # announcements, and a level that does not swing much.
        commuting = engine * 1.6 + announcement * 1.4 + crowd * 0.4
        commuting *= 1.0 + max(0.0, 0.35 - variability / 20.0)

        # Walking: outdoors, traffic that comes and goes, vehicles approaching.
        walking = approaching * 2.0 + traffic * 1.2 + horn * 0.8
        walking *= 1.0 + min(variability / 12.0, 0.8)

        # Social: people, plural, at conversational level.
        social = speech * 1.3 + crowd * 1.5
        social *= 1.0 + max(0.0, (loudness + 40.0) / 30.0)

        # Studying / stationary: quiet and unchanging. Distinguished only by
        # whether any speech is around at all.
        quiet = max(0.0, (-28.0 - loudness) / 25.0) + ambient * 0.8
        studying = quiet * (1.0 + 0.5 * speech) + mechanical * 0.2
        stationary = quiet * 1.1

        evidence = {
            ContextState.COMMUTING: commuting,
            ContextState.WALKING: walking,
            ContextState.SOCIAL: social,
            ContextState.STUDYING: studying,
            ContextState.STATIONARY: stationary,
        }
        return {k: float(max(v, 0.0)) for k, v in evidence.items() if v > 0.01}

    def _apply_motion(self, evidence: dict[ContextState, float]) -> dict[ContextState, float]:
        """Weight the acoustic vote by motion, which is far less ambiguous.

        A jeepney and a pavement can sound similar; 9 m/s versus 1.3 m/s does
        not. Motion multiplies rather than replaces, so a wrong or stale sensor
        reading cannot by itself decide the context.
        """
        m = self.motion
        if m is None:
            return evidence
        out = dict(evidence)

        if m.speed_mps is not None:
            if m.speed_mps > 6.0:
                out[ContextState.COMMUTING] = out.get(ContextState.COMMUTING, 0.1) * 2.5
            elif 0.6 <= m.speed_mps <= 2.2:
                out[ContextState.WALKING] = out.get(ContextState.WALKING, 0.1) * 2.5
            elif m.speed_mps < 0.3:
                for k in (ContextState.STATIONARY, ContextState.STUDYING, ContextState.SOCIAL):
                    out[k] = out.get(k, 0.1) * 1.6

        if m.step_rate_hz is not None and 1.2 <= m.step_rate_hz <= 2.8:
            out[ContextState.WALKING] = out.get(ContextState.WALKING, 0.1) * 2.0

        place_boost = {
            "transit": ContextState.COMMUTING,
            "campus": ContextState.STUDYING,
            "library": ContextState.STUDYING,
            "outdoors": ContextState.WALKING,
            "cafe": ContextState.SOCIAL,
            "restaurant": ContextState.SOCIAL,
        }
        target = place_boost.get((m.place_category or "").lower())
        if target is not None:
            out[target] = out.get(target, 0.1) * 1.8

        return out

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict:
        reading = self._last
        return {
            **reading.as_dict(),
            "auto_enabled": self.auto_enabled,
            "allow_motion": self.allow_motion,
            "manual": self.manual_state.value if self.manual_state else None,
            "motion_available": self.motion is not None and self.motion.is_fresh,
        }

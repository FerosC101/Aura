"""Listening-mode definitions.

A mode is a declarative table: for each sound category, what should happen to
it. Keeping modes as data rather than code means a new mode is a dictionary,
and the UI can render any mode without knowing which ones exist.

Categories a mode does not mention fall through to its ``default`` intent, so
adding a category to the taxonomy never silently breaks a mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Action, ListeningMode, Priority

#: Default gain, in dB, applied for each action. The policy engine adjusts
#: these by confidence, user preference and suppression strength.
ACTION_GAIN_DB: dict[Action, float] = {
    Action.ENHANCE: 5.0,
    Action.PRESERVE: 0.0,
    Action.REDUCE: -9.0,
    Action.SUPPRESS: -16.0,
}


@dataclass(frozen=True, slots=True)
class ModeProfile:
    """How one listening mode treats the taxonomy."""

    mode: ListeningMode
    label: str
    tagline: str
    description: str
    intents: dict[str, Action]
    priorities: dict[str, Priority] = field(default_factory=dict)
    default: Action = Action.PRESERVE
    #: Multiplier on the attenuation of REDUCE/SUPPRESS decisions. Walk mode
    #: stays deliberately gentle so the user keeps situational awareness.
    suppression_scale: float = 1.0

    def intent_for(self, category: str) -> Action:
        return self.intents.get(category, self.default)

    def priority_for(self, category: str) -> Priority:
        if category in self.priorities:
            return self.priorities[category]
        action = self.intent_for(category)
        if action is Action.ENHANCE:
            return Priority.HIGH
        if action is Action.PRESERVE:
            return Priority.NORMAL
        return Priority.LOW

    def public(self) -> dict:
        return {
            "mode": self.mode.value,
            "label": self.label,
            "tagline": self.tagline,
            "description": self.description,
            "default": self.default.value,
            "suppression_scale": self.suppression_scale,
            "intents": {k: v.value for k, v in self.intents.items()},
        }


FOCUS = ModeProfile(
    mode=ListeningMode.FOCUS,
    label="Focus",
    tagline="Study and deep work",
    description=(
        "Holds the room quiet. Your locked speaker, your name and anything "
        "safety-critical still get through."
    ),
    intents={
        # A single nearby talker is preserved: in a lecture or a study room the
        # person addressing you is usually the reason you can hear at all.
        # "Surrounding conversations", which the brief asks Focus to reduce,
        # are crowd_speech -- babble with no dominant talker -- and those are
        # suppressed below. Person Lock then lifts your chosen speaker further.
        "human_speech": Action.PRESERVE,
        "crowd_speech": Action.SUPPRESS,
        "announcement": Action.PRESERVE,
        "general_traffic": Action.SUPPRESS,
        "engine_noise": Action.SUPPRESS,
        "mechanical_noise": Action.SUPPRESS,
        "music": Action.REDUCE,
        "ambient": Action.SUPPRESS,
        "alarm": Action.PRESERVE,
        "emergency_siren": Action.PRESERVE,
        "vehicle_horn": Action.PRESERVE,
        "warning_signal": Action.PRESERVE,
        "bicycle_bell": Action.REDUCE,
        "approaching_vehicle": Action.REDUCE,
    },
    default=Action.REDUCE,
    suppression_scale=1.15,
)

COMMUTE = ModeProfile(
    mode=ListeningMode.COMMUTE,
    label="Commute",
    tagline="Jeepney, bus, train",
    description=(
        "Keeps the conductor, announcements and the road audible while the "
        "engine and the crowd drop away."
    ),
    intents={
        "human_speech": Action.PRESERVE,
        "announcement": Action.ENHANCE,
        "crowd_speech": Action.REDUCE,
        "vehicle_horn": Action.PRESERVE,
        "emergency_siren": Action.PRESERVE,
        "warning_signal": Action.PRESERVE,
        "alarm": Action.PRESERVE,
        "engine_noise": Action.SUPPRESS,
        "general_traffic": Action.REDUCE,
        "mechanical_noise": Action.REDUCE,
        "music": Action.REDUCE,
        "ambient": Action.REDUCE,
        "approaching_vehicle": Action.PRESERVE,
        "bicycle_bell": Action.PRESERVE,
    },
    priorities={"announcement": Priority.HIGH},
    default=Action.REDUCE,
)

WALK = ModeProfile(
    mode=ListeningMode.WALK,
    label="Walk",
    tagline="Streets and crossings",
    description=(
        "Everything that can hit you stays loud. Repetitive city noise is "
        "trimmed, never removed."
    ),
    intents={
        "approaching_vehicle": Action.ENHANCE,
        "vehicle_horn": Action.ENHANCE,
        "emergency_siren": Action.ENHANCE,
        "bicycle_bell": Action.ENHANCE,
        "warning_signal": Action.ENHANCE,
        "alarm": Action.PRESERVE,
        "human_speech": Action.PRESERVE,
        "announcement": Action.PRESERVE,
        "crowd_speech": Action.REDUCE,
        "general_traffic": Action.REDUCE,
        "engine_noise": Action.REDUCE,
        "mechanical_noise": Action.REDUCE,
        "music": Action.REDUCE,
        "ambient": Action.REDUCE,
    },
    priorities={
        "approaching_vehicle": Priority.CRITICAL,
        "vehicle_horn": Priority.CRITICAL,
        "emergency_siren": Priority.CRITICAL,
        "bicycle_bell": Priority.CRITICAL,
        "warning_signal": Priority.CRITICAL,
    },
    default=Action.PRESERVE,
    # Deliberately gentle. Walking is the mode where over-suppression is
    # dangerous, so even "reduce" stays mild here.
    suppression_scale=0.7,
)

SOCIAL = ModeProfile(
    mode=ListeningMode.SOCIAL,
    label="Social",
    tagline="Conversation in a loud room",
    description="Lifts the person in front of you above the room behind them.",
    intents={
        "human_speech": Action.ENHANCE,
        "crowd_speech": Action.SUPPRESS,
        "announcement": Action.PRESERVE,
        "music": Action.REDUCE,
        "general_traffic": Action.SUPPRESS,
        "engine_noise": Action.SUPPRESS,
        "mechanical_noise": Action.SUPPRESS,
        "ambient": Action.SUPPRESS,
        "alarm": Action.PRESERVE,
        "emergency_siren": Action.PRESERVE,
        "vehicle_horn": Action.PRESERVE,
        "warning_signal": Action.PRESERVE,
        "bicycle_bell": Action.PRESERVE,
        "approaching_vehicle": Action.PRESERVE,
    },
    default=Action.REDUCE,
)

CARE = ModeProfile(
    mode=ListeningMode.CARE,
    label="Care",
    tagline="Clearer conversations when they matter most",
    description=(
        "For the conversations you cannot afford to mishear: a consultation, "
        "pharmacy instructions, a caregiver explaining a dose. Lifts the person "
        "in front of you and pushes down the room they are standing in."
    ),
    intents={
        "human_speech": Action.ENHANCE,
        "announcement": Action.PRESERVE,
        # The defining difference from Social: in a clinic the competing sound
        # is usually *other speech* -- a waiting room, a television, a radio at
        # the counter -- and that is what makes the words in front of you hard
        # to follow. It is pushed further down here than anywhere else.
        "crowd_speech": Action.SUPPRESS,
        # Reduced rather than suppressed: a television and an air-conditioner
        # mask speech but carry no risk, and `suppression_scale` already makes
        # a Care-mode "reduce" deeper than any other mode's. Gouging 19 dB out
        # of a 60%-confident detection is how a voice gets mistaken for a fan.
        "music": Action.REDUCE,
        "mechanical_noise": Action.REDUCE,     # air-conditioning, fans
        "ambient": Action.REDUCE,
        "general_traffic": Action.REDUCE,
        "engine_noise": Action.REDUCE,
        # Safety is never traded for clarity, however quiet the room needs to be.
        "alarm": Action.PRESERVE,
        "emergency_siren": Action.PRESERVE,
        "vehicle_horn": Action.PRESERVE,
        "warning_signal": Action.PRESERVE,
        "bicycle_bell": Action.PRESERVE,
        "approaching_vehicle": Action.PRESERVE,
    },
    default=Action.REDUCE,
    # Indoors, with the talker close by, there is room to push harder than the
    # mobile modes dare: nothing here is a moving hazard.
    suppression_scale=1.2,
)

CUSTOM = ModeProfile(
    mode=ListeningMode.CUSTOM,
    label="Custom",
    tagline="Your own mix",
    description="Every category is yours to set. Safety floors still apply.",
    intents={},
    default=Action.PRESERVE,
)

MODE_PROFILES: dict[ListeningMode, ModeProfile] = {
    ListeningMode.CARE: CARE,
    ListeningMode.FOCUS: FOCUS,
    ListeningMode.COMMUTE: COMMUTE,
    ListeningMode.WALK: WALK,
    ListeningMode.SOCIAL: SOCIAL,
    ListeningMode.CUSTOM: CUSTOM,
}


def get_mode(mode: ListeningMode | str) -> ModeProfile:
    if isinstance(mode, str):
        mode = ListeningMode(mode)
    return MODE_PROFILES[mode]


def all_modes() -> list[ModeProfile]:
    return list(MODE_PROFILES.values())

"""The AURA sound taxonomy.

Categories are *registered*, not hard-coded into the policy engine or the UI.
Adding a new sound class means appending one :class:`SoundCategory` here (or
calling :func:`register_category` from a plug-in); the policy engine, mixer,
safety guard and UI all pick it up automatically because they look categories
up through this registry.

Each category carries the metadata every downstream component needs:

* ``safety_critical`` -- the SafetyGuard enforces a floor on it.
* ``stream`` -- which separated stream it is expected to live in, so the mixer
  knows which signal to apply the gain to.
* ``band_hz`` -- rough spectral home, used by the DSP classifier and by the
  mixer's per-category band shaping.
* ``aliases`` -- names used by external models (AudioSet/YAMNet class names),
  so a neural backend can be mapped onto the AURA taxonomy without changing
  any policy code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..types import StreamKind


@dataclass(frozen=True, slots=True)
class SoundCategory:
    key: str
    label: str
    description: str
    stream: StreamKind = StreamKind.ENVIRONMENT
    safety_critical: bool = False
    transient: bool = False
    band_hz: tuple[float, float] = (20.0, 20000.0)
    aliases: tuple[str, ...] = field(default_factory=tuple)
    #: Word-boundary regexes for broad families where listing every external
    #: class name is impractical -- AudioSet alone has ~150 music classes, one
    #: per genre and instrument. Consulted only after exact matching fails, and
    #: **ignored for safety-critical categories**, where a guess is a false
    #: alert and the explicit ``aliases`` list is the whole contract.
    alias_patterns: tuple[str, ...] = field(default_factory=tuple)
    icon: str = "waveform"


# --------------------------------------------------------------------------
# The initial AURA taxonomy (MVP requirement: >= 9 categories)
# --------------------------------------------------------------------------

_BUILTIN: tuple[SoundCategory, ...] = (
    SoundCategory(
        key="human_speech",
        label="Human Speech",
        description="A nearby person speaking, close enough to be intelligible.",
        stream=StreamKind.PRIMARY_SPEECH,
        band_hz=(85.0, 4000.0),
        aliases=("Speech", "Male speech, man speaking", "Female speech, woman speaking",
                 "Conversation", "Narration, monologue", "Child speech, kid speaking"),
        icon="speech",
    ),
    SoundCategory(
        key="crowd_speech",
        label="Crowd Conversation",
        description="Overlapping background voices; babble with no single dominant talker.",
        stream=StreamKind.SECONDARY_SPEECH,
        band_hz=(150.0, 3500.0),
        aliases=("Babbling", "Crowd", "Hubbub, speech noise, speech babble",
                 "Chatter", "Children playing"),
        icon="crowd",
    ),
    SoundCategory(
        key="vehicle_horn",
        label="Vehicle Horn",
        description="Car, jeepney, bus or truck horn. Safety-critical.",
        stream=StreamKind.TRANSIENT,
        safety_critical=True,
        transient=True,
        band_hz=(300.0, 4000.0),
        aliases=("Vehicle horn, car horn, honking", "Honk", "Toot", "Air horn, truck horn"),
        icon="horn",
    ),
    SoundCategory(
        key="emergency_siren",
        label="Emergency Siren",
        description="Ambulance, fire truck or police siren. Safety-critical.",
        stream=StreamKind.TRANSIENT,
        safety_critical=True,
        band_hz=(500.0, 3000.0),
        aliases=("Siren", "Emergency vehicle", "Ambulance (siren)", "Fire engine, fire truck (siren)",
                 "Police car (siren)", "Civil defense siren"),
        icon="siren",
    ),
    SoundCategory(
        key="engine_noise",
        label="Engine Noise",
        description="Low-frequency engine rumble from the vehicle you are inside or beside.",
        band_hz=(20.0, 300.0),
        aliases=("Engine", "Idling", "Medium engine (mid frequency)", "Engine starting",
                 "Vehicle", "Motor vehicle (road)", "Diesel engine",
                 "Light engine (high frequency)", "Heavy engine (low frequency)",
                 "Engine knocking", "Accelerating, revving, vroom"),
        icon="engine",
    ),
    SoundCategory(
        key="general_traffic",
        label="General Traffic",
        description="Broadband road ambience: tyre noise, passing cars, distant highway.",
        band_hz=(100.0, 6000.0),
        aliases=("Traffic noise, roadway noise", "Road", "Car passing by", "Tire squeal",
                 "Car", "Truck", "Bus", "Motorcycle", "Race car, auto racing"),
        icon="traffic",
    ),
    SoundCategory(
        key="alarm",
        label="Alarm",
        description="Fire alarm, smoke detector, clock alarm or alert tone. Safety-critical.",
        stream=StreamKind.TRANSIENT,
        safety_critical=True,
        transient=True,
        band_hz=(800.0, 6000.0),
        # Not bare "Bell": AudioSet emits it for church bells, cowbells and
        # tubular bells alike, and it was this category's single largest source
        # of false alerts (38 of 40 church-bell recordings). The specific bells
        # that *are* alerts are listed individually.
        aliases=("Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Alarm clock",
                 "Buzzer", "Beep, bleep", "Car alarm", "Doorbell",
                 "Telephone bell ringing"),
        icon="alarm",
    ),
    SoundCategory(
        key="music",
        label="Music",
        description="Musical content: harmonic, rhythmic, sustained.",
        band_hz=(40.0, 12000.0),
        aliases=("Music", "Musical instrument", "Singing", "Pop music", "Background music"),
        # AudioSet devotes ~150 classes to genres and instruments. Whole words
        # only: "\bdrum\b" takes "Drum kit" and leaves "Drumbeat"-style
        # compounds alone, and nothing here can reach a safety category.
        alias_patterns=(
            r"\bmusic\b", r"\bmusical\b", r"\bsinging\b", r"\bsong\b",
            r"\bguitar\b", r"\bpiano\b", r"\bdrum\b", r"\bviolin\b",
            r"\btrumpet\b", r"\bflute\b", r"\bsaxophone\b", r"\bcello\b",
            r"\bbanjo\b", r"\bharp\b", r"\baccordion\b", r"\borgan\b",
            r"\bsynthesizer\b", r"\bchoir\b", r"\bchant\b", r"\bmelody\b",
            r"\bharmonica\b", r"\bukulele\b", r"\bmandolin\b", r"\bbass\b",
        ),
        icon="music",
    ),
    SoundCategory(
        key="mechanical_noise",
        label="Construction / Mechanical",
        description="Drills, jackhammers, fans, air-conditioning, machinery.",
        band_hz=(60.0, 8000.0),
        aliases=("Jackhammer", "Power tool", "Drill", "Sawing", "Mechanical fan",
                 "Air conditioning", "Machinery", "Hammer", "Chainsaw",
                 "Vacuum cleaner", "Sewing machine", "Tools",
                 "Dental drill, dentist's drill", "Helicopter",
                 "Propeller, airscrew"),
        icon="construction",
    ),
    # ---- Extensions beyond the required nine -----------------------------
    SoundCategory(
        key="announcement",
        label="Transport Announcement",
        description="Amplified public-address speech: terminal, bus, LRT/MRT announcements.",
        stream=StreamKind.PRIMARY_SPEECH,
        band_hz=(200.0, 4000.0),
        aliases=("Public address system", "Announcement", "Loudspeaker"),
        icon="announcement",
    ),
    SoundCategory(
        key="bicycle_bell",
        label="Bicycle Bell",
        description="Bicycle bell or similar warning chime. Safety-critical while walking.",
        stream=StreamKind.TRANSIENT,
        safety_critical=True,
        transient=True,
        band_hz=(1500.0, 8000.0),
        aliases=("Bicycle bell", "Ding", "Chime", "Jingle bell"),
        icon="bell",
    ),
    SoundCategory(
        key="approaching_vehicle",
        label="Approaching Vehicle",
        description="A vehicle whose level is rising quickly -- closing distance. Safety-critical.",
        safety_critical=True,
        band_hz=(50.0, 4000.0),
        # Deliberately empty. "Approaching" is a *level trajectory*, not a
        # timbre: no AudioSet class distinguishes a car coming towards you from
        # one going away. Mapping "Car" here made every parked taxi raise a
        # safety-critical alert under the ONNX backend. The DSP path detects
        # this from level rise, which is the only signal that carries it.
        aliases=(),
        icon="approach",
    ),
    SoundCategory(
        key="warning_signal",
        label="Warning Signal",
        description="Reversing beeper, level-crossing bell, generic hazard tone. Safety-critical.",
        stream=StreamKind.TRANSIENT,
        safety_critical=True,
        transient=True,
        band_hz=(700.0, 5000.0),
        aliases=("Reversing beeps", "Warning", "Train horn", "Train whistle"),
        icon="warning",
    ),
    SoundCategory(
        key="ambient",
        label="Ambient Noise",
        description="Unstructured low-information background: HVAC hum, room tone, wind.",
        band_hz=(20.0, 20000.0),
        aliases=("Inside, small room", "Inside, large room or hall", "Wind", "White noise",
                 "Environmental noise", "Silence", "Rain", "Raindrop",
                 "Rain on surface", "Stream", "Outside, urban or manmade"),
        icon="ambient",
    ),
)


_REGISTRY: dict[str, SoundCategory] = {c.key: c for c in _BUILTIN}
_ALIAS_INDEX: dict[str, str] = {}
#: Compiled (pattern, category) pairs from ``alias_patterns``.
_ALIAS_PATTERNS: list[tuple[re.Pattern[str], str]] = []


def _normalise(name: str) -> str:
    """Lower-case and collapse whitespace, so lookup is not spacing-sensitive."""
    return " ".join(name.strip().lower().split())


#: Aliases claimed by more than one category, newest registration winning.
#: Exposed rather than silently resolved: a collision means the taxonomy is
#: ambiguous about a real sound, which is a modelling decision, not a typo.
ALIAS_COLLISIONS: dict[str, tuple[str, str]] = {}


def _reindex_aliases() -> None:
    _ALIAS_INDEX.clear()
    ALIAS_COLLISIONS.clear()
    _ALIAS_PATTERNS.clear()
    for cat in _REGISTRY.values():
        if cat.alias_patterns and not cat.safety_critical:
            _ALIAS_PATTERNS.extend(
                (re.compile(p), cat.key) for p in cat.alias_patterns
            )
        for name in (cat.key, cat.label, *cat.aliases):
            alias = _normalise(name)
            previous = _ALIAS_INDEX.get(alias)
            if previous is not None and previous != cat.key:
                ALIAS_COLLISIONS[alias] = (previous, cat.key)
            _ALIAS_INDEX[alias] = cat.key


_reindex_aliases()


# --------------------------------------------------------------------------
# Public registry API
# --------------------------------------------------------------------------


def register_category(category: SoundCategory, *, overwrite: bool = False) -> SoundCategory:
    """Add a new sound class to the live taxonomy.

    This is the supported extension point: a plug-in classifier that can detect
    ``dog_bark`` registers the category once at import time and every other
    component -- policy, safety, mixer, UI -- handles it without modification.
    """
    if category.key in _REGISTRY and not overwrite:
        raise ValueError(f"category {category.key!r} already registered")
    _REGISTRY[category.key] = category
    _reindex_aliases()
    return category


def get_category(key: str) -> SoundCategory:
    """Look up a category, falling back to a synthesized generic entry."""
    cat = _REGISTRY.get(key)
    if cat is not None:
        return cat
    return SoundCategory(key=key, label=key.replace("_", " ").title(),
                         description="Unregistered sound category.")


def all_categories() -> list[SoundCategory]:
    return list(_REGISTRY.values())


def category_keys() -> list[str]:
    return list(_REGISTRY.keys())


def is_safety_critical(key: str) -> bool:
    return get_category(key).safety_critical


def stream_for(key: str) -> StreamKind:
    return get_category(key).stream


def resolve_alias(name: str) -> str | None:
    """Map an external model's class name onto an AURA category key.

    Used by neural backends so that, e.g., YAMNet's ``"Vehicle horn, car horn,
    honking"`` becomes AURA's ``vehicle_horn`` without the policy engine ever
    learning about AudioSet.

    Matching is **exact** on the whole name or on one comma-separated synonym,
    never on substrings. An earlier version fell back to ``alias in name or
    name in alias``, which silently produced:

    ===================  =========================  =========================
    AudioSet class       Mapped to                  Because
    ===================  =========================  =========================
    ``Toothbrush``       ``vehicle_horn``           ``"toot"`` in "toothbrush"
    ``Rain``             ``warning_signal``         "rain" in ``"train horn"``
    ``Shuffling cards``  ``approaching_vehicle``    ``"car"`` in "cards"
    ``Belly laugh``      ``bicycle_bell``           ``"bell"`` in "belly"
    ``Crow``             ``crowd_speech``           "crow" in ``"crowd"``
    ===================  =========================  =========================

    Three of those five are safety-critical categories, so a toothbrush raised
    a horn alert. Unmapped is the correct answer for a sound the taxonomy has
    no home for; guessing from a shared prefix is not.
    """
    lowered = _normalise(name)
    key = _ALIAS_INDEX.get(lowered)
    if key:
        return key
    # AudioSet names are comma-separated synonym lists -- "Vehicle horn, car
    # horn, honking", "Boat, Water vehicle" -- so try each synonym on its own.
    if "," in lowered:
        for part in lowered.split(","):
            key = _ALIAS_INDEX.get(_normalise(part))
            if key:
                return key
    for pattern, cat_key in _ALIAS_PATTERNS:
        if pattern.search(lowered):
            return cat_key
    return None


SAFETY_CRITICAL_KEYS: frozenset[str] = frozenset(
    c.key for c in _BUILTIN if c.safety_critical
)

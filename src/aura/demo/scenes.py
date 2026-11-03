"""Demo scenes: PSC Demo Mode and the Philippine presets.

A live demo cannot rely on the room containing an ambulance, a jeepney and a
cooperative stranger. These scenes render multi-source audio where the correct
answer is known in advance, so the semantic filtering can be shown reliably and
compared against a fixed baseline.

Each scene also declares ``expect`` -- what AURA should do with each source --
which the test suite asserts against. That keeps the demo honest: if a scene
stops behaving as advertised, a test fails rather than a judge noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import ListeningMode
from .synth import DEFAULT_SR, SceneLayer, render_scene


@dataclass(slots=True)
class DemoScene:
    """A scripted multi-source acoustic scene."""

    key: str
    title: str
    subtitle: str
    description: str
    layers: list[SceneLayer]
    duration: float = 12.0
    mode: ListeningMode = ListeningMode.COMMUTE
    #: category -> expected action, asserted by the tests.
    expect: dict[str, str] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)

    def render(self, sample_rate: int = DEFAULT_SR, *, stems: bool = False):
        return render_scene(self.layers, self.duration, sample_rate, return_stems=stems)

    def public(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "subtitle": self.subtitle,
            "description": self.description,
            "duration": self.duration,
            "mode": self.mode.value,
            "expect": dict(self.expect),
            "highlights": list(self.highlights),
            "layers": [
                {
                    "category": layer.category,
                    "gain_db": layer.gain_db,
                    "start": layer.start,
                    "end": layer.end if layer.end is not None else self.duration,
                }
                for layer in self.layers
            ],
        }


# --------------------------------------------------------------------------
# The headline PSC demo scene
# --------------------------------------------------------------------------

PSC_DEMO = DemoScene(
    key="psc_demo",
    title="PSC Demo Scene",
    subtitle="Five sources, one decision each",
    description=(
        "Engine noise and crowd conversation run throughout. A selected speaker "
        "talks over them. A horn sounds once; a siren passes later. AURA should "
        "hold the speaker, the horn and the siren while pushing the engine and "
        "the crowd down."
    ),
    mode=ListeningMode.COMMUTE,
    duration=12.0,
    layers=[
        SceneLayer("engine_noise", gain_db=-13.0),
        SceneLayer("crowd_speech", gain_db=-11.0),
        SceneLayer("human_speech", gain_db=-8.0, start=1.0, end=11.0),
        SceneLayer("vehicle_horn", gain_db=-7.0, start=4.0, end=5.2),
        SceneLayer("emergency_siren", gain_db=-9.0, start=7.5, end=11.0),
    ],
    expect={
        "human_speech": "preserve",
        "vehicle_horn": "preserve",
        "emergency_siren": "preserve",
        "engine_noise": "suppress",
        "crowd_speech": "reduce",
    },
    highlights=[
        "Engine and crowd drop away",
        "The speaker stays clear",
        "Horn and siren are never suppressed",
    ],
)


# --------------------------------------------------------------------------
# Philippine use cases
# --------------------------------------------------------------------------

JEEPNEY_COMMUTE = DemoScene(
    key="jeepney",
    title="Jeepney / Bus Commute",
    subtitle="EDSA, rush hour",
    description=(
        "Diesel rumble and road noise fill the cabin while passengers talk. The "
        "conductor calls out and announcements come over a tinny speaker. AURA "
        "keeps the conductor and the announcements, and the horns outside."
    ),
    mode=ListeningMode.COMMUTE,
    duration=14.0,
    layers=[
        SceneLayer("engine_noise", gain_db=-10.0),
        SceneLayer("general_traffic", gain_db=-11.0),
        SceneLayer("crowd_speech", gain_db=-14.0),
        SceneLayer("human_speech", gain_db=-9.0, start=2.0, end=6.0),
        SceneLayer("announcement", gain_db=-10.0, start=7.0, end=10.5),
        SceneLayer("vehicle_horn", gain_db=-8.0, start=11.5, end=12.6),
    ],
    expect={
        "human_speech": "preserve",
        "announcement": "enhance",
        "vehicle_horn": "preserve",
        "engine_noise": "suppress",
        "general_traffic": "reduce",
        "crowd_speech": "reduce",
    },
    highlights=[
        "Conductor and announcements stay up",
        "Diesel rumble suppressed",
        "Horns outside still reach you",
    ],
)

UNIVERSITY_CLASSROOM = DemoScene(
    key="classroom",
    title="University Classroom",
    subtitle="Lecture with hallway noise",
    description=(
        "An instructor speaks over ceiling fans and chatter drifting in from the "
        "corridor. AURA holds the instructor and keeps the fire alarm audible, "
        "while distant conversation and fan noise recede."
    ),
    mode=ListeningMode.FOCUS,
    duration=14.0,
    layers=[
        SceneLayer("mechanical_noise", gain_db=-15.0),
        SceneLayer("crowd_speech", gain_db=-14.0),
        SceneLayer("human_speech", gain_db=-9.0, start=0.5, end=9.0),
        SceneLayer("alarm", gain_db=-13.0, start=10.5, end=13.5),
    ],
    expect={
        "human_speech": "preserve",
        "alarm": "preserve",
        "crowd_speech": "suppress",
        "mechanical_noise": "suppress",
    },
    highlights=[
        "Instructor stays intelligible",
        "Hallway chatter and fans suppressed",
        "The fire alarm always comes through",
    ],
)

URBAN_WALKING = DemoScene(
    key="urban_walk",
    title="Urban Walking",
    subtitle="Crossing a busy street",
    description=(
        "Continuous traffic ambience with a vehicle closing in, a horn, and a "
        "siren. Everything that can hit you is preserved or enhanced; the "
        "repetitive city bed is trimmed but never removed."
    ),
    mode=ListeningMode.WALK,
    duration=14.0,
    layers=[
        SceneLayer("general_traffic", gain_db=-13.0),
        SceneLayer("crowd_speech", gain_db=-20.0),
        SceneLayer("approaching_vehicle", gain_db=-9.0, start=2.0, end=6.0),
        SceneLayer("vehicle_horn", gain_db=-7.0, start=6.2, end=7.2),
        SceneLayer("bicycle_bell", gain_db=-7.0, start=8.5, end=10.0),
        SceneLayer("emergency_siren", gain_db=-9.0, start=10.5, end=13.5),
    ],
    expect={
        "approaching_vehicle": "enhance",
        "vehicle_horn": "enhance",
        "emergency_siren": "enhance",
        "bicycle_bell": "enhance",
        "general_traffic": "reduce",
        "crowd_speech": "reduce",
    },
    highlights=[
        "Approaching vehicle lifted early",
        "Horn, bell and siren all protected",
        "Traffic trimmed, never silenced",
    ],
)

CAFE_CONVERSATION = DemoScene(
    key="cafe",
    title="Cafe Conversation",
    subtitle="One voice in a loud room",
    description=(
        "Music and competing conversation fill a cafe. With Person Lock on the "
        "person across the table, their voice is lifted above the room."
    ),
    mode=ListeningMode.SOCIAL,
    duration=12.0,
    layers=[
        SceneLayer("music", gain_db=-16.0),
        SceneLayer("crowd_speech", gain_db=-13.0),
        SceneLayer("human_speech", gain_db=-8.0, start=0.5, end=11.5),
    ],
    expect={
        "human_speech": "enhance",
        "crowd_speech": "suppress",
        "music": "reduce",
    },
    highlights=[
        "The person you chose comes forward",
        "Competing conversation pushed back",
        "Music reduced, not removed",
    ],
)


CLINIC_CONSULT = DemoScene(
    key="clinic",
    title="Clinic Consultation",
    subtitle="The words you cannot afford to mishear",
    description=(
        "A consultation room with the door open. A doctor explains a dose while "
        "a waiting room talks outside, an air-conditioner runs, and a television "
        "plays at the counter. A monitor alarms partway through. AURA should "
        "lift the doctor, push the room down, and never touch the alarm."
    ),
    mode=ListeningMode.CARE,
    duration=14.0,
    layers=[
        # The three sources that compete with the doctor, each a different kind
        # of mask: broadband (aircon), speech-shaped (waiting room), and
        # modulated (television).
        SceneLayer("mechanical_noise", gain_db=-9.0),
        SceneLayer("crowd_speech", gain_db=-11.0),
        SceneLayer("music", gain_db=-10.0, start=0.5),
        SceneLayer("human_speech", gain_db=-8.0, start=1.5, end=12.0),
        SceneLayer("alarm", gain_db=-4.0, start=8.5, end=10.5),
    ],
    expect={
        "human_speech": "enhance",
        "crowd_speech": "suppress",
        # Reduced, not suppressed: see the note in policy/modes.py CARE.
        "mechanical_noise": "reduce",
        "music": "reduce",
        "alarm": "preserve",
    },
    highlights=[
        "The doctor comes forward",
        "Waiting room, aircon and TV step back",
        "The monitor alarm is never suppressed",
    ],
)


SCENES: dict[str, DemoScene] = {
    s.key: s for s in (
        CLINIC_CONSULT, PSC_DEMO, JEEPNEY_COMMUTE, UNIVERSITY_CLASSROOM,
        URBAN_WALKING, CAFE_CONVERSATION,
    )
}


def get_scene(key: str) -> DemoScene:
    if key not in SCENES:
        raise KeyError(f"unknown demo scene {key!r}; have {sorted(SCENES)}")
    return SCENES[key]


def all_scenes() -> list[DemoScene]:
    return list(SCENES.values())


# --------------------------------------------------------------------------
# Comparison modes: Raw / ANC simulation / AURA
# --------------------------------------------------------------------------

def simulate_anc(signal: np.ndarray, sample_rate: int, *, reduction_db: float = -18.0) -> np.ndarray:
    """Simulate conventional active noise cancellation.

    Deliberately indiscriminate, because that is the point of the comparison:
    ANC attenuates by *frequency*, not by meaning. It takes out the low end
    hard and leaves speech partly intact -- and it removes the siren along with
    the engine, because it has no idea which is which.
    """
    from scipy.signal import butter, sosfilt

    x = np.asarray(signal, dtype=np.float64)
    nyq = sample_rate / 2.0
    # ANC is most effective at low frequencies and tapers off above ~1 kHz.
    sos_low = butter(2, min(900.0 / nyq, 0.99), btype="low", output="sos")
    low = sosfilt(sos_low, x)
    high = x - low
    gain = 10.0 ** (reduction_db / 20.0)
    # Broadband attenuation on top, so everything gets quieter overall.
    return ((low * gain + high * (gain ** 0.45)) * 0.9).astype(np.float32)


COMPARISON_MODES = {
    "raw": {
        "key": "raw",
        "label": "Raw",
        "description": "Everything audible, exactly as the microphone hears it.",
    },
    "anc": {
        "key": "anc",
        "label": "Noise Cancellation",
        "description": (
            "Conventional ANC. Attenuates by frequency with no idea what any "
            "sound means -- the siren goes quiet along with the engine."
        ),
    },
    "aura": {
        "key": "aura",
        "label": "AURA",
        "description": (
            "Selective Awareness. The speaker, the horn and the siren are kept; "
            "the engine and the crowd are suppressed."
        ),
    },
}

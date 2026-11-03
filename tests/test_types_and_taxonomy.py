"""Core contracts: the taxonomy registry and the value types."""

import numpy as np
import pytest

from aura.classification import taxonomy as tax
from aura.types import Action, LatencyBreakdown, Priority, SoundEvent, StreamKind


def test_required_mvp_categories_exist():
    """The brief names nine categories; all must be present."""
    required = {
        "human_speech", "crowd_speech", "vehicle_horn", "emergency_siren",
        "engine_noise", "general_traffic", "alarm", "music", "mechanical_noise",
    }
    assert required <= set(tax.category_keys())


def test_safety_categories_are_marked():
    for key in ("emergency_siren", "vehicle_horn", "alarm", "warning_signal"):
        assert tax.is_safety_critical(key), key
    assert not tax.is_safety_critical("music")


def test_registry_is_extensible():
    """Adding a category must not require touching any other module."""
    cat = tax.SoundCategory(
        key="dog_bark", label="Dog Bark", description="A dog barking.",
        stream=StreamKind.TRANSIENT, band_hz=(300.0, 5000.0), aliases=("Bark", "Dog"),
    )
    tax.register_category(cat, overwrite=True)
    assert tax.get_category("dog_bark").label == "Dog Bark"
    assert tax.resolve_alias("Bark") == "dog_bark"
    assert tax.stream_for("dog_bark") is StreamKind.TRANSIENT


def test_external_model_names_map_onto_the_taxonomy():
    """A neural backend must be pluggable without the policy engine changing."""
    assert tax.resolve_alias("Vehicle horn, car horn, honking") == "vehicle_horn"
    assert tax.resolve_alias("Ambulance (siren)") == "emergency_siren"
    assert tax.resolve_alias("Male speech, man speaking") == "human_speech"


def test_alias_matching_is_exact_not_substring():
    """Substring matching mapped ordinary objects onto safety-critical alerts.

    ``resolve_alias`` used to fall back to ``alias in name or name in alias``.
    Every pair below was a real mapping produced by that fallback against
    YAMNet's 521 AudioSet class names, three of them onto safety categories.
    A sound the taxonomy has no home for must resolve to None.
    """
    for audioset_name in (
        "Toothbrush",                           # "Toot"  -> vehicle_horn
        "Rain",                                 # in "train horn" -> warning_signal
        "Shuffling cards",                      # "Car"   -> approaching_vehicle
        "Belly laugh",                          # "Bell"  -> bicycle_bell
        "Bellow",
        "Crow",                                 # in "Crowd" -> crowd_speech
        "Wind instrument, woodwind instrument",  # "Wind" -> ambient
    ):
        assert tax.resolve_alias(audioset_name) != "vehicle_horn"
        assert tax.resolve_alias(audioset_name) not in tax.SAFETY_CRITICAL_KEYS

    # Rain is genuinely background, and is now mapped deliberately rather than
    # by accident; the rest have no home in this taxonomy.
    assert tax.resolve_alias("Rain") == "ambient"
    for unmapped in ("Toothbrush", "Shuffling cards", "Belly laugh", "Crow"):
        assert tax.resolve_alias(unmapped) is None


def test_comma_separated_audioset_names_resolve():
    """AudioSet names are synonym lists; any one synonym should be enough."""
    assert tax.resolve_alias("Vehicle horn, car horn, honking") == "vehicle_horn"
    assert tax.resolve_alias("Honk") == "vehicle_horn"
    assert tax.resolve_alias("  SIREN  ") == "emergency_siren"


def test_no_alias_is_claimed_by_two_categories():
    """A duplicate alias silently resolves to whichever category registered last."""
    assert not tax.ALIAS_COLLISIONS, (
        f"aliases claimed twice: {tax.ALIAS_COLLISIONS}"
    )


def test_alias_patterns_never_reach_a_safety_category():
    """Pattern matching is a convenience for broad families, not for alerts.

    ``alias_patterns`` exists because AudioSet spends ~150 classes on musical
    genres and instruments. A safety-critical category must stay on its
    explicit alias list, where every mapping is auditable: the cost of a wrong
    guess there is a false siren, not a mislabelled song.
    """
    for cat in tax.all_categories():
        if cat.safety_critical:
            assert not cat.alias_patterns, f"{cat.key} is safety-critical"
    for _pattern, key in tax._ALIAS_PATTERNS:
        assert not tax.is_safety_critical(key)


def test_generic_bell_is_not_an_alarm():
    """AudioSet emits bare "Bell" for church, cow and tubular bells.

    Mapping it to the safety-critical `alarm` fired on 38 of 40 church-bell
    recordings in ESC-50. The bells that really are alerts are listed by name.
    """
    assert tax.resolve_alias("Bell") is None
    assert tax.resolve_alias("Church bell") is None
    assert tax.resolve_alias("Cowbell") is None
    assert tax.resolve_alias("Alarm clock") == "alarm"
    assert tax.resolve_alias("Doorbell") == "alarm"


def test_approaching_vehicle_has_no_static_aliases():
    """Approach is a level trajectory, not a timbre.

    No AudioSet class separates a car coming towards you from one going away,
    so mapping "Car" here turned every parked taxi into a safety alert.
    """
    assert tax.get_category("approaching_vehicle").aliases == ()
    assert tax.resolve_alias("Car") != "approaching_vehicle"


def test_unknown_category_degrades_gracefully():
    cat = tax.get_category("something_unheard_of")
    assert cat.label == "Something Unheard Of"
    assert not cat.safety_critical


def test_action_and_priority_ordering():
    """max() over actions must pick the most audible -- safety relies on it."""
    assert max([Action.SUPPRESS, Action.PRESERVE, Action.REDUCE], key=lambda a: a.rank) is Action.PRESERVE
    assert Priority.CRITICAL.rank > Priority.HIGH.rank > Priority.NORMAL.rank > Priority.LOW.rank


def test_latency_breakdown_totals():
    b = LatencyBreakdown(capture_ms=10, separation_ms=1, classification_ms=2,
                         speaker_ms=0.5, policy_ms=0.1, mixing_ms=0.4, output_ms=8)
    assert b.inference_ms == pytest.approx(2.5)
    assert b.processing_ms == pytest.approx(4.0)
    assert b.total_ms == pytest.approx(22.0)


def test_sound_event_clamps_confidence():
    assert SoundEvent("music", 1.8).confidence == 1.0
    assert SoundEvent("music", -0.3).confidence == 0.0


def test_audio_frame_level():
    from aura.types import AudioFrame

    f = AudioFrame(np.full(160, 0.5, dtype=np.float32), 16000)
    assert f.dbfs == pytest.approx(-6.02, abs=0.1)
    assert f.duration_ms == pytest.approx(10.0)

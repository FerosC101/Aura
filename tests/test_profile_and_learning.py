"""Acoustic Profile, preference learning and context detection."""

import numpy as np
import pytest

from aura.context.engine import ContextEngine, MotionHint
from aura.profile.acoustic_profile import AcousticProfile, ProfileStore
from aura.profile.learning import MAX_BIAS_DB, PreferenceLearningEngine
from aura.types import ContextState, ListeningMode, SoundEvent

# -- learning ------------------------------------------------------------------


def test_repeated_adjustments_become_a_bias():
    """The worked example from the brief: announcements pushed up in Commute."""
    learner = PreferenceLearningEngine()
    for _ in range(6):
        learner.observe_adjustment(ListeningMode.COMMUTE, "announcement", +4.0)
    bias = learner.bias_for(ListeningMode.COMMUTE, "announcement")
    assert bias > 2.0
    # And it does not leak into other modes.
    assert learner.bias_for(ListeningMode.WALK, "announcement") == 0.0


def test_learning_is_bounded():
    learner = PreferenceLearningEngine()
    for _ in range(80):
        learner.observe_adjustment(ListeningMode.FOCUS, "music", +40.0)
    assert abs(learner.bias_for(ListeningMode.FOCUS, "music")) <= MAX_BIAS_DB


def test_never_learns_to_suppress_safety_sounds():
    learner = PreferenceLearningEngine()
    for _ in range(20):
        assert learner.observe_adjustment(ListeningMode.WALK, "emergency_siren", -8.0) is None
    assert learner.bias_for(ListeningMode.WALK, "emergency_siren") == 0.0


def test_safety_sounds_may_still_be_raised():
    learner = PreferenceLearningEngine()
    for _ in range(5):
        learner.observe_adjustment(ListeningMode.WALK, "emergency_siren", +3.0)
    assert learner.bias_for(ListeningMode.WALK, "emergency_siren") > 0


def test_tiny_adjustments_are_ignored_as_noise():
    learner = PreferenceLearningEngine()
    assert learner.observe_adjustment(ListeningMode.FOCUS, "music", 0.4) is None


def test_confidence_grows_with_evidence():
    learner = PreferenceLearningEngine()
    learner.observe_adjustment(ListeningMode.FOCUS, "music", +5.0)
    one = abs(learner.bias_for(ListeningMode.FOCUS, "music"))
    for _ in range(5):
        learner.observe_adjustment(ListeningMode.FOCUS, "music", +5.0)
    assert abs(learner.bias_for(ListeningMode.FOCUS, "music")) > one


def test_learning_can_be_disabled_and_forgotten():
    learner = PreferenceLearningEngine(enabled=False)
    assert learner.observe_adjustment(ListeningMode.FOCUS, "music", +5.0) is None
    learner.enabled = True
    for _ in range(4):
        learner.observe_adjustment(ListeningMode.FOCUS, "music", +5.0)
    assert learner.forget() == 1
    assert learner.bias_for(ListeningMode.FOCUS, "music") == 0.0


def test_learning_round_trips():
    a = PreferenceLearningEngine()
    for _ in range(4):
        a.observe_adjustment(ListeningMode.COMMUTE, "announcement", +4.0)
    b = PreferenceLearningEngine()
    b.load_dict(a.to_dict())
    assert b.bias_for(ListeningMode.COMMUTE, "announcement") == pytest.approx(
        a.bias_for(ListeningMode.COMMUTE, "announcement")
    )


# -- profile -------------------------------------------------------------------


def test_profile_persists_and_reloads(tmp_path):
    store = ProfileStore(tmp_path)
    profile, learner = AcousticProfile(user_name="Vince"), PreferenceLearningEngine()
    profile.set_override(ListeningMode.COMMUTE, "engine_noise", -20.0)
    profile.note_mode_use(ListeningMode.COMMUTE, 300.0)
    learner.observe_adjustment(ListeningMode.COMMUTE, "announcement", 4.0)
    store.save(profile, learner)

    reloaded, relearned = store.load()
    assert reloaded.user_name == "Vince"
    assert reloaded.overrides_for(ListeningMode.COMMUTE)["engine_noise"] == -20.0
    assert reloaded.most_used_mode is ListeningMode.COMMUTE
    assert relearned.records


def test_corrupt_profile_does_not_stop_startup(tmp_path):
    store = ProfileStore(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{ not json at all")
    profile, learner = store.load()
    assert isinstance(profile, AcousticProfile)
    assert not learner.records


def test_privacy_defaults_are_conservative():
    p = AcousticProfile()
    assert not p.privacy.store_speaker_embeddings
    assert not p.privacy.allow_cloud_inference
    assert not p.privacy.allow_name_awareness
    assert not p.privacy.allow_motion_context
    assert not p.privacy.auto_context_detection


def test_overrides_track_preferences():
    p = AcousticProfile()
    p.set_override(ListeningMode.COMMUTE, "announcement", 6.0)
    p.set_override(ListeningMode.COMMUTE, "engine_noise", -18.0)
    assert "announcement" in p.frequently_preserved()
    assert "engine_noise" in p.frequently_suppressed()
    p.set_override(ListeningMode.COMMUTE, "announcement", None)
    assert "announcement" not in p.overrides_for(ListeningMode.COMMUTE)


def test_public_view_hides_embeddings():
    p = AcousticProfile()
    p.speaker_profiles = [{"id": "x", "name": "V", "centroid": [0.1] * 60}]
    assert "centroid" not in str(p.public())


def test_delete_removes_the_file(tmp_path):
    store = ProfileStore(tmp_path)
    store.save(AcousticProfile(), PreferenceLearningEngine())
    assert store.exists()
    assert store.delete()
    assert not store.exists()


# -- context -------------------------------------------------------------------


def test_manual_context_overrides_detection():
    ctx = ContextEngine(auto_enabled=True)
    ctx.declare(ContextState.STUDYING)
    reading = ctx.read()
    assert reading.state is ContextState.STUDYING
    assert reading.source == "manual"
    assert reading.suggested_mode is ListeningMode.FOCUS


def test_auto_detection_is_off_by_default():
    ctx = ContextEngine()
    for _ in range(60):
        ctx.observe([SoundEvent("engine_noise", 0.9)], -18.0)
    assert ctx.read().state is ContextState.UNKNOWN


def test_detects_a_commute_acoustically():
    ctx = ContextEngine(auto_enabled=True)
    for _ in range(60):
        ctx.observe(
            [SoundEvent("engine_noise", 0.9), SoundEvent("announcement", 0.6),
             SoundEvent("crowd_speech", 0.4)], -18.0,
        )
    assert ctx.read().state is ContextState.COMMUTING


def test_detects_walking_from_street_sounds():
    ctx = ContextEngine(auto_enabled=True)
    rng = np.random.default_rng(0)
    for _ in range(60):
        ctx.observe(
            [SoundEvent("general_traffic", 0.8), SoundEvent("approaching_vehicle", 0.7),
             SoundEvent("vehicle_horn", 0.5)], -25.0 + rng.normal() * 6,
        )
    assert ctx.read().state is ContextState.WALKING


def test_motion_hints_require_opt_in():
    ctx = ContextEngine(auto_enabled=True)
    assert ctx.submit_motion(MotionHint(speed_mps=1.4)) is False
    ctx.allow_motion = True
    assert ctx.submit_motion(MotionHint(speed_mps=1.4)) is True

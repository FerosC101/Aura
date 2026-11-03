"""Policy, safety and the guarantees that must never be violated."""

import pytest

from aura.classification.taxonomy import SAFETY_CRITICAL_KEYS
from aura.policy.engine import AcousticPolicyEngine, PolicyContext
from aura.policy.modes import all_modes, get_mode
from aura.policy.safety import MIN_SAFETY_GAIN_DB, SafetyGuard
from aura.types import Action, GainDecision, ListeningMode, Priority, SoundEvent, StreamKind


@pytest.fixture
def policy():
    return AcousticPolicyEngine()


# -- the worked examples from the brief ---------------------------------------


def test_horn_in_walk_mode_is_critical(policy):
    r = policy.explain("vehicle_horn", PolicyContext(mode=ListeningMode.WALK))
    assert r["action"] in ("preserve", "enhance")
    assert r["priority"] == "critical"
    assert r["gain_db"] >= 0


def test_engine_in_commute_mode_is_suppressed(policy):
    r = policy.explain("engine_noise", PolicyContext(mode=ListeningMode.COMMUTE))
    assert r["action"] == "suppress"
    assert r["gain_db"] <= -14.0


def test_selected_speaker_in_social_mode_is_enhanced(policy):
    r = policy.explain("human_speech", PolicyContext(mode=ListeningMode.SOCIAL))
    assert r["action"] == "enhance"
    assert r["gain_db"] >= 4.0


# -- safety -------------------------------------------------------------------


@pytest.mark.parametrize("category", sorted(SAFETY_CRITICAL_KEYS))
@pytest.mark.parametrize("mode", list(ListeningMode))
def test_safety_sounds_are_never_silenced(policy, category, mode):
    """The central safety guarantee, across every category and every mode.

    Also applied with a hostile user override: someone dragging a siren to the
    bottom of the mixer must not be able to switch it off.
    """
    ctx = PolicyContext(mode=mode, user_overrides={category: -40.0}, suppression_strength=1.5)
    r = policy.explain(category, ctx)
    assert r["gain_db"] >= MIN_SAFETY_GAIN_DB, f"{category} in {mode.value} -> {r['gain_db']}"
    assert r["always_audible"]


def test_learned_preference_cannot_override_safety(policy):
    ctx = PolicyContext(
        mode=ListeningMode.FOCUS,
        learned_bias={"emergency_siren": -6.0},
        user_overrides={"emergency_siren": -30.0},
    )
    r = policy.explain("emergency_siren", ctx)
    assert r["gain_db"] >= MIN_SAFETY_GAIN_DB
    assert r["safety_override"]


def test_uncertain_detection_favours_audibility():
    """Doubt about a possible siren must resolve towards hearing it."""
    guard = SafetyGuard()
    decision = GainDecision("emergency_siren", Action.SUPPRESS, -18.0, Priority.LOW)
    event = SoundEvent("emergency_siren", 0.45, level_db=-30.0)
    verdict = guard.apply(decision, event, scene_level_db=-30.0)
    assert verdict.overridden
    assert verdict.decision.gain_db >= 0.0
    assert "uncertain" in verdict.decision.reason


def test_quiet_safety_sound_is_lifted_above_the_scene():
    guard = SafetyGuard()
    decision = GainDecision("emergency_siren", Action.PRESERVE, 0.0, Priority.NORMAL)
    quiet = SoundEvent("emergency_siren", 0.95, level_db=-48.0)
    verdict = guard.apply(decision, quiet, scene_level_db=-20.0)
    assert verdict.decision.gain_db > 0.0


def test_disabled_safety_category_is_not_protected():
    """Users may opt out in onboarding; the guard must honour that."""
    guard = SafetyGuard(enabled_categories={"emergency_siren"})
    assert guard.is_protected("emergency_siren")
    assert not guard.is_protected("vehicle_horn")


def test_non_safety_sounds_are_not_floored():
    guard = SafetyGuard()
    decision = GainDecision("engine_noise", Action.SUPPRESS, -18.0, Priority.LOW)
    verdict = guard.apply(decision, SoundEvent("engine_noise", 0.9), scene_level_db=-20.0)
    assert not verdict.overridden
    assert verdict.decision.gain_db == -18.0


# -- layering -----------------------------------------------------------------


def test_bypass_passes_everything_through(policy):
    events = [SoundEvent("engine_noise", 0.9), SoundEvent("human_speech", 0.9)]
    for d in policy.decide(events, PolicyContext(enabled=False)):
        assert d.gain_db == 0.0
        assert d.action is Action.PRESERVE


def test_user_override_beats_the_mode(policy):
    ctx = PolicyContext(mode=ListeningMode.COMMUTE, user_overrides={"engine_noise": 3.0})
    assert policy.explain("engine_noise", ctx)["gain_db"] == pytest.approx(3.0, abs=0.5)


def test_person_lock_promotes_and_ducks(policy):
    ctx = PolicyContext(mode=ListeningMode.SOCIAL, person_lock_active=True, person_lock_name="V")
    plain = policy.explain("human_speech", PolicyContext(mode=ListeningMode.SOCIAL))
    locked = policy.explain("human_speech", ctx)
    assert locked["gain_db"] > plain["gain_db"]
    ducked = policy.explain("crowd_speech", ctx)
    assert ducked["gain_db"] < policy.explain("crowd_speech", PolicyContext(mode=ListeningMode.SOCIAL))["gain_db"]


def test_low_confidence_produces_a_tentative_action(policy):
    """An unsure detection should trim a little, not gouge 16 dB."""
    strong = policy.decide([SoundEvent("engine_noise", 0.95)], PolicyContext(mode=ListeningMode.COMMUTE))[0]
    weak = policy.decide([SoundEvent("engine_noise", 0.40)], PolicyContext(mode=ListeningMode.COMMUTE))[0]
    assert abs(weak.gain_db) < abs(strong.gain_db)


def test_below_the_floor_nothing_happens(policy):
    d = policy.decide([SoundEvent("engine_noise", 0.2)], PolicyContext(mode=ListeningMode.COMMUTE))[0]
    assert d.gain_db == pytest.approx(0.0, abs=0.01)


def test_care_mode_lifts_the_talker_above_competing_speech(policy):
    """Care mode's whole reason to exist.

    In a clinic or a pharmacy the sound that makes the words hard to follow is
    usually *other speech* -- a waiting room, a television at the counter. A
    mode for those conversations has to beat Social on exactly that, or it is
    only Social with a different label.
    """
    ctx = PolicyContext(mode=ListeningMode.CARE)
    talker = policy.explain("human_speech", ctx)
    assert talker["action"] == "enhance"

    care = policy.explain("crowd_speech", ctx)["gain_db"]
    social = policy.explain("crowd_speech", PolicyContext(mode=ListeningMode.SOCIAL))["gain_db"]
    assert care < social, "Care must push competing speech below Social"


def test_care_mode_never_trades_safety_for_clarity(policy):
    """Quiet enough to hear a doctor is not quiet enough to miss a fire alarm."""
    ctx = PolicyContext(mode=ListeningMode.CARE)
    for category in SAFETY_CRITICAL_KEYS:
        r = policy.explain(category, ctx)
        assert r["gain_db"] >= MIN_SAFETY_GAIN_DB, category
        assert r["action"] in ("preserve", "enhance"), category


def test_walk_mode_suppresses_gently():
    """Over-suppression while walking is a safety issue, not a taste issue."""
    walk, focus = get_mode(ListeningMode.WALK), get_mode(ListeningMode.FOCUS)
    assert walk.suppression_scale < focus.suppression_scale


def test_every_mode_covers_every_category():
    from aura.classification.taxonomy import category_keys

    for mode in all_modes():
        for key in category_keys():
            assert isinstance(mode.intent_for(key), Action)


def test_name_awareness_lifts_speech(policy):
    base = policy.explain("human_speech", PolicyContext(mode=ListeningMode.FOCUS))
    lifted = policy.explain("human_speech", PolicyContext(mode=ListeningMode.FOCUS, name_detected=True))
    assert lifted["gain_db"] > base["gain_db"]


def test_decisions_carry_their_stream(policy):
    d = policy.decide([SoundEvent("engine_noise", 0.9, stream=StreamKind.ENVIRONMENT)],
                      PolicyContext(mode=ListeningMode.COMMUTE))[0]
    assert d.stream is StreamKind.ENVIRONMENT

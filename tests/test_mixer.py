"""The mixer: where decisions become audible change.

These are the tests that distinguish AURA from a dashboard. Each one asserts a
measured property of the output signal, not a label.
"""

import numpy as np
import pytest

from aura.audio.biquad import SHAPER_BANDS, MultibandShaper, SoftLimiter
from aura.audio.mixer import AudioMixer, _band_weights
from aura.types import Action, GainDecision, Priority, SeparatedAudio, StreamKind

SR, HOP = 16000, 320


def make_separated(**levels: float) -> SeparatedAudio:
    rng = np.random.default_rng(3)
    streams = {}
    for kind in StreamKind:
        amp = levels.get(kind.value, 0.1)
        streams[kind] = (rng.standard_normal(HOP) * amp).astype(np.float32)
    return SeparatedAudio(streams=streams, sample_rate=SR)


def settle(mixer, sep, decisions, n=60):
    out = None
    for _ in range(n):
        out, report = mixer.mix(sep, decisions)
    return out, report


def test_suppression_actually_reduces_the_signal():
    sep = make_separated()
    quiet = [GainDecision("crowd_speech", Action.SUPPRESS, -18.0, Priority.LOW,
                          stream=StreamKind.SECONDARY_SPEECH, confidence=0.9)]
    loud = [GainDecision("crowd_speech", Action.PRESERVE, 0.0, Priority.NORMAL,
                         stream=StreamKind.SECONDARY_SPEECH, confidence=0.9)]

    only_secondary = SeparatedAudio(
        streams={StreamKind.SECONDARY_SPEECH: sep.streams[StreamKind.SECONDARY_SPEECH]},
        sample_rate=SR,
    )
    a, _ = settle(AudioMixer(SR), only_secondary, quiet)
    b, _ = settle(AudioMixer(SR), only_secondary, loud)
    drop = 20 * np.log10(a.std() / max(b.std(), 1e-12))
    assert drop < -14.0, f"only {drop:.1f} dB of suppression"


def test_enhancement_raises_the_signal():
    sep = SeparatedAudio(
        streams={StreamKind.PRIMARY_SPEECH: (np.random.default_rng(1).standard_normal(HOP) * 0.05).astype(np.float32)},
        sample_rate=SR,
    )
    boosted, _ = settle(AudioMixer(SR), sep, [
        GainDecision("human_speech", Action.ENHANCE, 6.0, Priority.HIGH,
                     stream=StreamKind.PRIMARY_SPEECH, confidence=0.9)
    ])
    flat, _ = settle(AudioMixer(SR), sep, [
        GainDecision("human_speech", Action.PRESERVE, 0.0, Priority.NORMAL,
                     stream=StreamKind.PRIMARY_SPEECH, confidence=0.9)
    ])
    gain = 20 * np.log10(boosted.std() / max(flat.std(), 1e-12))
    assert gain == pytest.approx(6.0, abs=1.0)


def test_bypass_reproduces_the_input():
    mixer = AudioMixer(SR)
    sep = make_separated()
    dry = sum(sep.streams.values())
    out, _ = mixer.mix(sep, [], bypass=True)
    assert np.allclose(out, dry, atol=1e-5)


def test_environment_shaping_is_frequency_selective():
    """Engine suppression must not take the horn band with it.

    This is the property that makes "preserve the horn, suppress the engine"
    real rather than a display label: both live in the environment stream, so a
    single scalar gain could not express it.
    """
    mixer = AudioMixer(SR)
    sep = make_separated()
    for _ in range(80):
        mixer.mix(sep, [
            GainDecision("engine_noise", Action.SUPPRESS, -18.0, Priority.LOW,
                         stream=StreamKind.ENVIRONMENT, confidence=0.95)
        ])
    bands = np.asarray(mixer._band_db)
    low = bands[: 2]      # 60 Hz, 160 Hz -- engine territory
    high = bands[3:]      # 1 kHz and up -- horn / siren territory
    assert low.max() < -10.0, f"engine bands only {low}"
    assert abs(high).max() < 2.0, f"horn bands were disturbed: {high}"


def test_safety_decision_sets_a_floor_within_its_bands():
    mixer = AudioMixer(SR)
    sep = make_separated()
    decisions = [
        GainDecision("engine_noise", Action.SUPPRESS, -20.0, Priority.LOW,
                     stream=StreamKind.ENVIRONMENT, confidence=0.95),
        GainDecision("approaching_vehicle", Action.PRESERVE, 0.0, Priority.CRITICAL,
                     always_audible=True, stream=StreamKind.ENVIRONMENT, confidence=0.9),
    ]
    for _ in range(80):
        mixer.mix(sep, decisions)
    # The protected sound spans 50-4000 Hz, so those bands must not be gouged.
    assert np.asarray(mixer._band_db)[:4].min() >= -1.0


def test_critical_decision_wins_its_stream():
    """A siren sharing a stream with a suppression must not be averaged down."""
    gains = AudioMixer._group([
        GainDecision("crowd_speech", Action.SUPPRESS, -18.0, Priority.LOW,
                     stream=StreamKind.TRANSIENT, confidence=0.9),
        GainDecision("emergency_siren", Action.PRESERVE, 0.0, Priority.CRITICAL,
                     always_audible=True, stream=StreamKind.TRANSIENT, confidence=0.9),
    ])
    assert gains[StreamKind.TRANSIENT] == 0.0


def test_band_weights_follow_the_taxonomy():
    engine = _band_weights("engine_noise")
    horn = _band_weights("vehicle_horn")
    assert engine[0] > 0.9 and engine[-1] < 0.1   # engine is low
    assert horn[0] < 0.1 and horn[3] > 0.5        # horn is mid


def test_gain_changes_do_not_click():
    """A step change must be ramped, or the listener hears a click."""
    mixer = AudioMixer(SR)
    sep = SeparatedAudio(
        streams={StreamKind.PRIMARY_SPEECH: np.full(HOP, 0.3, dtype=np.float32)},
        sample_rate=SR,
    )
    loud = [GainDecision("human_speech", Action.ENHANCE, 6.0, Priority.HIGH,
                         stream=StreamKind.PRIMARY_SPEECH, confidence=0.9)]
    quiet = [GainDecision("human_speech", Action.SUPPRESS, -18.0, Priority.LOW,
                          stream=StreamKind.PRIMARY_SPEECH, confidence=0.9)]
    for _ in range(10):
        mixer.mix(sep, loud)
    out, _ = mixer.mix(sep, quiet)
    # No single-sample jump larger than a modest fraction of full scale.
    assert np.max(np.abs(np.diff(out))) < 0.05


def test_limiter_prevents_clipping():
    lim = SoftLimiter(SR, threshold_db=-1.0)
    out = lim.process(np.full(HOP, 4.0, dtype=np.float32))
    assert np.max(np.abs(out)) <= 10 ** (-1.0 / 20.0) + 1e-3
    assert lim.active


def test_shaper_hits_its_targets():
    """Band interaction is compensated, so a request of -18 dB measures -18 dB."""
    sh = MultibandShaper(SR)
    target = np.array([-18.0, -18.0, -6.0, 0.0, 0.0, 0.0])
    sh.set_gains(target)
    got = sh.response_db(np.asarray(SHAPER_BANDS))
    assert np.max(np.abs(got - target)) < 1.5, f"got {got}"


def test_flat_shaper_is_a_passthrough():
    sh = MultibandShaper(SR)
    x = np.random.default_rng(0).standard_normal(HOP)
    assert np.allclose(sh.process(x), x)

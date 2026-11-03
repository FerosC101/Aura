"""Sound-event detection.

MVP criterion 2: detect at least five meaningful sound classes.
"""

import numpy as np
import pytest

from aura.classification.dsp_classifier import DSPSoundClassifier
from aura.classification.features import FeatureExtractor
from aura.demo.synth import generate
from aura.types import AudioFrame

SR, HOP, WARMUP = 16000, 320, 25

# Sources the classifier is expected to find in isolation, with the fraction of
# frames it must appear in. These thresholds come from scripts/eval_classifier.py
# and exist to catch regressions, not to describe an aspiration.
EXPECTED = {
    "human_speech": 0.50,
    "crowd_speech": 0.70,
    "engine_noise": 0.80,
    "general_traffic": 0.70,
    "vehicle_horn": 0.70,
    "emergency_siren": 0.70,
    "alarm": 0.50,
    "music": 0.70,
    "mechanical_noise": 0.70,
    "announcement": 0.50,
    "approaching_vehicle": 0.70,
    "warning_signal": 0.50,
}


def run_classifier(category: str, seconds: float = 4.0):
    signal = generate(category, seconds, SR)
    clf = DSPSoundClassifier(SR)
    tallies: dict[str, int] = {}
    scored = 0
    for i in range(0, signal.size - HOP, HOP):
        events = clf.classify(AudioFrame(signal[i:i + HOP], SR, sequence=i // HOP))
        if i // HOP < WARMUP:
            continue
        scored += 1
        for e in events:
            tallies[e.category] = tallies.get(e.category, 0) + 1
    return tallies, max(scored, 1)


@pytest.mark.parametrize("category,min_recall", sorted(EXPECTED.items()))
def test_isolated_source_is_detected(category, min_recall):
    tallies, scored = run_classifier(category)
    recall = tallies.get(category, 0) / scored
    assert recall >= min_recall, (
        f"{category} detected in only {recall:.0%} of frames "
        f"(need {min_recall:.0%}); top: "
        f"{sorted(tallies.items(), key=lambda kv: -kv[1])[:3]}"
    )


def test_at_least_five_classes_detected_overall():
    """The headline MVP requirement, asserted directly."""
    found = {c for c in EXPECTED if run_classifier(c, 3.0)[0].get(c, 0) > 0}
    assert len(found) >= 5


def test_siren_and_horn_are_distinguished():
    """The hardest pair: both are loud tonal events, one sweeps its pitch."""
    for truth, rival in (("emergency_siren", "vehicle_horn"), ("vehicle_horn", "emergency_siren")):
        tallies, scored = run_classifier(truth)
        assert tallies.get(truth, 0) / scored > tallies.get(rival, 0) / scored, (
            f"{truth} was not preferred over {rival}"
        )


def test_silence_yields_no_foreground_detections():
    clf = DSPSoundClassifier(SR)
    silence = np.zeros(HOP, dtype=np.float32)
    events = []
    for i in range(60):
        events = clf.classify(AudioFrame(silence, SR, sequence=i))
    assert all(e.category == "ambient" for e in events), [e.category for e in events]


def test_classifier_is_multi_label():
    """Real scenes contain several sounds; one label per frame is not enough."""
    from aura.demo.synth import SceneLayer, render_scene

    mix = render_scene(
        [SceneLayer("engine_noise", -10.0), SceneLayer("human_speech", -8.0)], 4.0, SR
    )
    clf = DSPSoundClassifier(SR)
    best = 0
    for i in range(0, mix.size - HOP, HOP):
        best = max(best, len(clf.classify(AudioFrame(mix[i:i + HOP], SR))))
    assert best >= 2


def test_classifier_speed_fits_the_realtime_budget():
    import time

    signal = generate("general_traffic", 2.0, SR)
    clf = DSPSoundClassifier(SR)
    blocks = [signal[i:i + HOP] for i in range(0, signal.size - HOP, HOP)]
    for b in blocks[:20]:
        clf.classify(AudioFrame(b, SR))
    t0 = time.perf_counter()
    for b in blocks:
        clf.classify(AudioFrame(b, SR))
    per_frame_ms = (time.perf_counter() - t0) / len(blocks) * 1000.0
    budget_ms = 1000.0 * HOP / SR
    assert per_frame_ms < budget_ms * 0.5, f"{per_frame_ms:.2f} ms vs {budget_ms:.1f} ms block"


def test_features_are_finite_for_pathological_input():
    """Extreme input must not produce NaNs that would poison every later stage."""
    fx = FeatureExtractor(SR)
    for signal in (
        np.zeros(HOP, dtype=np.float32),
        np.ones(HOP, dtype=np.float32),
        np.full(HOP, -1.0, dtype=np.float32),
        (np.random.randn(HOP) * 1e-9).astype(np.float32),
    ):
        f = fx.extract(signal)
        for name in ("rms", "dbfs", "centroid_hz", "flatness", "entropy",
                     "tonality", "harmonicity", "modulation_depth", "stationarity"):
            assert np.isfinite(getattr(f, name)), f"{name} not finite"

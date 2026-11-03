"""Backend pluggability.

The brief requires that inference backends be replaceable. These tests assert
the interface contract rather than any particular model, and that an
unavailable backend degrades visibly instead of silently.
"""

import numpy as np
import pytest

from aura.classification.base import SoundClassifier
from aura.classification.dsp_classifier import DSPSoundClassifier
from aura.classification.registry import (
    build_classifier,
    build_separator,
    build_speaker_identifier,
    fallback_notes,
)
from aura.separation.base import SourceSeparator
from aura.speaker.base import SpeakerIdentifier
from aura.types import AudioFrame, SoundEvent, StreamKind

SR, HOP = 16000, 320


def test_unavailable_backend_falls_back_visibly():
    """A silent fallback would make it look like a neural model was running."""
    clf = build_classifier("onnx", SR, model_path=None)
    assert isinstance(clf, DSPSoundClassifier)
    note = fallback_notes().get("classifier", "")
    assert "onnx" in note and "DSP" in note


def test_unknown_speaker_backend_falls_back():
    emb = build_speaker_identifier("ecapa", SR)
    assert emb.is_available
    assert emb.dim > 0


def test_passthrough_separator_is_a_valid_backend():
    sep = build_separator("passthrough", SR, HOP)
    assert isinstance(sep, SourceSeparator)
    block = np.random.default_rng(0).standard_normal(HOP).astype(np.float32)
    out = sep.separate(block)
    assert np.allclose(out.streams[StreamKind.ENVIRONMENT], block)
    assert sep.algorithmic_delay_samples == 0


def test_a_custom_classifier_can_be_dropped_in():
    """The whole point of the abstraction: no other module needs to change."""

    class AlwaysHorn(SoundClassifier):
        name = "test-stub"

        def classify(self, frame, separated=None):
            return [SoundEvent("vehicle_horn", 0.99, StreamKind.TRANSIENT, frame.dbfs)]

    import tempfile

    from aura.config import EngineConfig
    from aura.pipeline.engine import AuraEngine

    engine = AuraEngine(EngineConfig(sample_rate=SR, block_size=HOP,
                                     profile_dir=tempfile.mkdtemp()))
    engine.classifier = AlwaysHorn()
    out = np.zeros(HOP, dtype=np.float32)
    engine._process(np.random.default_rng(1).standard_normal(HOP).astype(np.float32) * 0.1, out)
    engine.stop()

    categories = {c["category"] for c in engine.scene_dict()["sounds"]}
    assert categories == {"vehicle_horn"}
    assert np.isfinite(out).all()


def test_interfaces_expose_describe():
    for obj in (
        build_classifier("dsp", SR),
        build_separator("spectral", SR, HOP),
        build_speaker_identifier("spectral", SR),
    ):
        d = obj.describe()
        assert "name" in d and "available" in d


def test_classifier_never_raises_on_odd_input():
    """The contract says return an empty list, never throw."""
    clf = build_classifier("dsp", SR)
    for block in (
        np.zeros(HOP, dtype=np.float32),
        np.ones(HOP, dtype=np.float32),
        np.full(HOP, np.nan, dtype=np.float32),
    ):
        with np.errstate(all="ignore"):
            events = clf.classify(AudioFrame(block, SR))
        assert isinstance(events, list)


def test_base_similarity_handles_degenerate_vectors():
    assert SpeakerIdentifier.similarity(np.zeros(8), np.zeros(8)) == 0.0
    assert SpeakerIdentifier.similarity(np.ones(8), np.ones(4)) == 0.0
    v = np.array([1.0, 0.0, 0.0])
    assert SpeakerIdentifier.similarity(v, v) == pytest.approx(1.0)


def test_every_keyword_backend_can_forget():
    """The Privacy Center's delete-all must work whichever backend is active.

    ``forget`` originally existed only on the template spotter, so deleting all
    data would have raised with the ASR backend selected -- in the one code
    path where failing silently is least acceptable.
    """
    from aura.speech.name_awareness import KeywordSpotter, TemplateKeywordSpotter

    class Minimal(KeywordSpotter):
        name = "minimal"

        def detect(self, audio, sample_rate):
            return None

    assert Minimal().forget() == 0
    spotter = TemplateKeywordSpotter(SR)
    spotter.enrol("x", [np.random.default_rng(0).standard_normal(SR).astype(np.float32)], SR)
    assert spotter.forget() >= 0

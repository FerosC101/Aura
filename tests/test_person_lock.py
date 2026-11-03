"""Person Lock and Name Awareness."""

import numpy as np
import pytest

from aura.demo.synth import speech, spoken_word
from aura.speaker.person_lock import PersonLockManager, SpeakerProfile
from aura.speaker.spectral_embedder import SpectralSpeakerEmbedder
from aura.speech.name_awareness import NameAwareness, TemplateKeywordSpotter

SR, HOP = 16000, 320

VOICES = {
    "Vince": dict(f0=112.0, formants=(500.0, 1480.0, 2500.0, 3400.0)),
    "Maria": dict(f0=208.0, formants=(650.0, 1900.0, 2800.0, 3600.0)),
    "Ben": dict(f0=148.0, formants=(430.0, 1100.0, 2300.0, 3300.0)),
}


@pytest.fixture
def manager():
    mgr = PersonLockManager(SpectralSpeakerEmbedder(SR), SR)
    for name, cfg in VOICES.items():
        mgr.enrol(name, [speech(2.5, SR, seed=40 + 13 * i, syllable_rate=3.4 + 0.8 * i, **cfg)
                         for i in range(2)], SR)
    return mgr


def feed(mgr, signal):
    last = None
    for i in range(0, signal.size - HOP, HOP):
        last = mgr.update(signal[i:i + HOP], True)
    return last


def test_locks_onto_the_enrolled_speaker(manager):
    manager.select(manager.profiles[[p.id for p in manager.profiles.values()
                                     if p.name == "Vince"][0]].id)
    result = feed(manager, speech(5.0, SR, seed=999, syllable_rate=4.3, **VOICES["Vince"]))
    assert result.active
    assert result.name == "Vince"


@pytest.mark.parametrize("other", ["Maria", "Ben"])
def test_rejects_other_speakers(manager, other):
    vince = next(p for p in manager.profiles.values() if p.name == "Vince")
    manager.select(vince.id)
    manager.reset()
    result = feed(manager, speech(5.0, SR, seed=999, syllable_rate=4.3, **VOICES[other]))
    assert not result.active, f"{other} wrongly matched Vince at {result.similarity:.3f}"


def test_embeddings_separate_speakers():
    emb = SpectralSpeakerEmbedder(SR)
    vecs = {
        (n, c): emb.embed(speech(2.5, SR, seed=40 + c * 13, syllable_rate=3.4 + 0.8 * c, **cfg), SR)
        for n, cfg in VOICES.items() for c in range(3)
    }
    same = [emb.similarity(vecs[(n, i)], vecs[(n, j)])
            for n in VOICES for i in range(3) for j in range(i + 1, 3)]
    diff = [emb.similarity(vecs[(a, i)], vecs[(b, j)])
            for a in VOICES for b in VOICES if a < b for i in range(3) for j in range(3)]
    assert min(same) > max(diff), f"same {min(same):.3f} vs diff {max(diff):.3f}"


def test_enrolment_requires_consent(manager):
    with pytest.raises(PermissionError):
        manager.enrol("Nope", [speech(2.0, SR)], SR, consent=False)


def test_enrolment_rejects_unusable_audio(manager):
    with pytest.raises(ValueError):
        manager.enrol("Silence", [np.zeros(200, dtype=np.float32)], SR)


def test_profile_holds_no_audio(manager):
    """The privacy contract, asserted structurally."""
    profile = next(iter(manager.profiles.values()))
    serialized = profile.to_dict()
    assert set(serialized) == {
        "id", "name", "centroid", "n_samples", "within_spread",
        "backend", "created_at", "consented",
    }
    # The centroid is an embedding, orders of magnitude smaller than any audio.
    assert len(serialized["centroid"]) < 500
    assert "centroid" not in profile.public()


def test_profile_round_trips(manager):
    original = next(iter(manager.profiles.values()))
    restored = SpeakerProfile.from_dict(original.to_dict())
    assert restored.id == original.id
    assert np.allclose(restored.centroid, original.centroid)


def test_deleting_a_profile_clears_selection(manager):
    pid = manager.selected_id
    assert manager.delete(pid)
    assert manager.selected_id != pid


def test_clear_removes_everything(manager):
    assert manager.clear() == 3
    assert not manager.profiles
    assert manager.selected_id is None


def test_no_selection_means_no_lock():
    mgr = PersonLockManager(SpectralSpeakerEmbedder(SR), SR)
    assert not feed(mgr, speech(3.0, SR)).active


# -- Name Awareness ------------------------------------------------------------


@pytest.fixture
def spotter():
    s = TemplateKeywordSpotter(SR)
    s.enrol("Vince", [spoken_word("vince", 0.7, SR, f0=118.0, seed=x, rate_jitter=0.04)
                      for x in (5, 6)], SR)
    return s


@pytest.mark.parametrize("dur,f0,seed", [(0.70, 120.0, 9), (0.70, 200.0, 11), (0.88, 116.0, 13)])
def test_spots_the_enrolled_name(spotter, dur, f0, seed):
    hit = spotter.detect(spoken_word("vince", dur, SR, f0=f0, seed=seed, rate_jitter=0.06), SR)
    assert hit is not None and hit.word == "Vince"


@pytest.mark.parametrize("word", ["maria", "hello", "ben", "ana", "stop"])
def test_rejects_other_words(spotter, word):
    assert spotter.detect(spoken_word(word, 0.7, SR, f0=118.0, seed=7), SR) is None


def test_name_awareness_is_off_by_default(spotter):
    na = NameAwareness(spotter, SR)
    assert not na.enabled
    assert not na.update(spoken_word("vince", 0.7, SR), True)


def test_name_awareness_fires_and_holds(spotter):
    na = NameAwareness(spotter, SR, enabled=True)
    clip = spoken_word("vince", 0.7, SR, f0=122.0, seed=33, rate_jitter=0.05)
    stream = np.concatenate([np.zeros(int(0.2 * SR), np.float32), clip,
                             np.zeros(int(0.4 * SR), np.float32)])
    assert any(na.update(stream[i:i + HOP], True) for i in range(0, stream.size - HOP, HOP))
    assert na.last_hit is not None and na.active


def test_template_backend_does_not_transcribe(spotter):
    """Privacy: the default backend never turns speech into text."""
    assert spotter.transcribes is False

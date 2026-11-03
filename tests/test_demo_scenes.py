"""PSC Demo Mode and the Philippine presets.

Demo scenes are load-bearing: they are what gets shown when the live room is
unpredictable. If a scene stops behaving as advertised, a test should fail
rather than a judge noticing.
"""

import numpy as np
import pytest

from aura.config import EngineConfig
from aura.demo.scenes import COMPARISON_MODES, all_scenes, get_scene, simulate_anc
from aura.demo.synth import GENERATORS, SceneLayer, generate, render_scene
from aura.pipeline.engine import AuraEngine
from helpers import SR, band_energy_db, drive


def test_all_scenes_render():
    for scene in all_scenes():
        audio = scene.render(SR)
        assert audio.size == pytest.approx(int(scene.duration * SR), rel=0.01)
        assert np.isfinite(audio).all()
        assert 0.05 < np.max(np.abs(audio)) <= 1.0


def test_psc_scene_has_the_required_sources():
    """The brief names five: engine, crowd, a speaker, a horn and a siren."""
    categories = {layer.category for layer in get_scene("psc_demo").layers}
    assert {"engine_noise", "crowd_speech", "human_speech",
            "vehicle_horn", "emergency_siren"} <= categories


def test_philippine_presets_exist():
    keys = {s.key for s in all_scenes()}
    assert {"jeepney", "classroom", "urban_walk"} <= keys


def test_generators_cover_the_taxonomy():
    from aura.classification.taxonomy import category_keys

    assert set(GENERATORS) >= set(category_keys()) - {"dog_bark"}


def test_stems_sum_to_the_mix():
    mix, stems = render_scene(
        [SceneLayer("engine_noise", -10.0), SceneLayer("human_speech", -8.0)],
        3.0, SR, return_stems=True,
    )
    assert np.allclose(sum(stems.values()), mix, atol=1e-5)


def test_generation_is_deterministic():
    assert np.array_equal(generate("human_speech", 1.0, SR), generate("human_speech", 1.0, SR))


@pytest.mark.parametrize("key", [s.key for s in all_scenes()])
def test_scene_behaves_as_scripted(key, tmp_path):
    """Each scene's declared expectations must match what the engine does.

    Two sources are allowed to differ: acoustic masking is real, and a quiet
    bell under heavy traffic or road noise beneath a diesel engine genuinely
    may not be separable. The limit exists so a broad regression still fails.
    """
    scene = get_scene(key)
    engine = AuraEngine(EngineConfig(
        sample_rate=SR, block_size=320, mode=scene.mode,
        profile_dir=str(tmp_path / key), publish_rate_hz=1e6,
    ))

    seen: dict[str, dict[str, int]] = {}

    def collect(_s):
        for card in engine.scene_dict().get("sounds", []):
            actions = seen.setdefault(card["category"], {})
            actions[card["action"]] = actions.get(card["action"], 0) + 1

    engine.subscribe(collect)
    drive(engine, scene.render(SR))
    engine.stop()

    matched = 0
    misses = []
    for category, expected in scene.expect.items():
        actions = seen.get(category)
        if not actions:
            misses.append(f"{category}: not detected")
            continue
        modal = max(actions, key=lambda k: actions[k])
        if modal == expected:
            matched += 1
        else:
            misses.append(f"{category}: {modal} (wanted {expected})")

    assert matched >= len(scene.expect) - 2, (
        f"{key}: only {matched}/{len(scene.expect)} as scripted -- {misses}"
    )


def test_anc_simulation_is_indiscriminate():
    """The comparison is only fair if the ANC baseline behaves like real ANC.

    Real ANC attenuates by frequency with no idea what anything means, so it
    must take the siren down along with the engine.
    """
    scene = get_scene("psc_demo")
    mix = scene.render(SR)
    anc = simulate_anc(mix, SR)

    engine_change = band_energy_db(anc, SR, 20, 300) - band_energy_db(mix, SR, 20, 300)
    siren_change = band_energy_db(anc, SR, 500, 3000) - band_energy_db(mix, SR, 500, 3000)
    assert engine_change < -10.0
    assert siren_change < -3.0, "the ANC baseline was suspiciously kind to the siren"


def test_comparison_modes_are_declared():
    assert set(COMPARISON_MODES) == {"raw", "anc", "aura"}
    for meta in COMPARISON_MODES.values():
        assert meta["label"] and meta["description"]


def test_spoken_words_are_distinguishable():
    """Name Awareness testing depends on words differing phonetically."""
    from aura.demo.synth import spoken_word
    from aura.speaker.spectral_embedder import SpectralSpeakerEmbedder
    from aura.speech.name_awareness import _mfcc_sequence, dtw_distance

    emb = SpectralSpeakerEmbedder(SR)

    def seq(word, seed):
        clip = spoken_word(word, 0.7, SR, f0=118.0, seed=seed).astype(np.float64)
        return _mfcc_sequence(emb, clip)

    same = dtw_distance(seq("vince", 5), seq("vince", 9))
    other = dtw_distance(seq("vince", 5), seq("maria", 9))
    assert same < other

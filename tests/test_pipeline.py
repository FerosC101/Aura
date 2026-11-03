"""End-to-end pipeline behaviour.

These tests cover the MVP success criteria that only make sense for the whole
system: real transformation, latency accounting, mode switching, and the
guarantee that nothing persists raw audio.
"""

import json
import pathlib

import numpy as np
import pytest

from aura.demo.scenes import get_scene, simulate_anc
from aura.demo.synth import SceneLayer, render_scene
from aura.pipeline.engine import AuraEngine
from aura.types import ListeningMode
from helpers import HOP, SR, band_energy_db, drive


def test_engine_produces_audio(engine):
    signal = render_scene([SceneLayer("human_speech", -8.0)], 2.0, SR)
    out = drive(engine, signal)
    assert out.size > 0
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) > 1e-4


def test_output_never_clips(engine):
    loud = render_scene(
        [SceneLayer("human_speech", -1.0), SceneLayer("emergency_siren", -1.0)], 2.0, SR
    )
    out = drive(engine, loud)
    assert np.max(np.abs(out)) <= 1.0


def test_engine_suppressed_while_horn_band_survives(config):
    """MVP criteria 3, 4 and 5 together, measured on the signal.

    The claim is not that AURA makes things quieter -- ANC does that. It is
    that the engine band drops substantially *while* the horn/siren band is
    left alone. Both live in the same separated stream, so this only works if
    the spectral shaping is real.
    """
    scene = get_scene("psc_demo")
    config.mode = scene.mode
    engine = AuraEngine(config)
    mix = scene.render(SR)
    out = drive(engine, mix)
    engine.stop()

    n = min(out.size, mix.size)
    raw, processed = mix[:n], out[:n]

    engine_change = band_energy_db(processed, SR, 20, 300) - band_energy_db(raw, SR, 20, 300)
    horn_change = band_energy_db(processed, SR, 400, 3500) - band_energy_db(raw, SR, 400, 3500)

    assert engine_change < -6.0, f"engine band only moved {engine_change:.1f} dB"
    assert horn_change > -3.0, f"horn/siren band lost {horn_change:.1f} dB"
    assert engine_change < horn_change - 4.0, "suppression was not selective"


def test_aura_beats_anc_on_siren_audibility(config):
    """The headline comparison: ANC removes the siren, AURA keeps it."""
    scene = get_scene("psc_demo")
    config.mode = scene.mode
    engine = AuraEngine(config)
    mix = scene.render(SR)
    aura_out = drive(engine, mix)
    engine.stop()
    anc_out = simulate_anc(mix, SR)

    siren = next(lay for lay in scene.layers if lay.category == "emergency_siren")
    a, b = int((siren.start + 0.5) * SR), int((siren.end - 0.5) * SR)
    n = min(aura_out.size, mix.size, anc_out.size)
    a, b = min(a, n), min(b, n)

    base = band_energy_db(mix[a:b], SR, 500, 3000)
    aura_db = band_energy_db(aura_out[a:b], SR, 500, 3000) - base
    anc_db = band_energy_db(anc_out[a:b], SR, 500, 3000) - base
    assert aura_db > anc_db + 4.0, f"AURA {aura_db:.1f} dB vs ANC {anc_db:.1f} dB"
    assert aura_db > -3.0, "AURA attenuated the siren"


def test_bypass_is_transparent(engine):
    """"AURA off" must sound like the input, or every comparison is invalid."""
    engine.set_enabled(False)
    signal = render_scene(
        [SceneLayer("human_speech", -8.0), SceneLayer("engine_noise", -12.0)], 2.0, SR
    )
    out = drive(engine, signal)
    delay = engine.separator.algorithmic_delay_samples
    n = min(out.size - delay, signal.size - delay)
    error = out[delay:delay + n] - signal[:n]
    snr = 20 * np.log10(signal[:n].std() / max(error.std(), 1e-12))
    assert snr > 40, f"bypass is not transparent: {snr:.1f} dB"


def test_latency_is_accounted_end_to_end(engine):
    drive(engine, render_scene([SceneLayer("general_traffic", -12.0)], 2.0, SR))
    m = engine.metrics.snapshot()

    assert m.frames > 50
    assert m.total["mean_ms"] > 0
    # Every stage must be measured, not just the interesting ones.
    for stage in ("preprocess", "separation", "classification", "policy", "mixing"):
        assert stage in m.stages
    # Algorithmic delay must be included rather than quietly dropped.
    assert m.algorithmic_delay_ms == pytest.approx(1000.0 * HOP / SR, abs=0.1)
    assert m.total["mean_ms"] >= m.block_ms


def test_processing_fits_the_realtime_budget(engine):
    drive(engine, render_scene([SceneLayer("human_speech", -8.0),
                                SceneLayer("engine_noise", -12.0)], 3.0, SR))
    m = engine.metrics.snapshot()
    assert m.processing_load < 0.6, f"processing load {m.processing_load:.0%}"
    assert m.xruns == 0


@pytest.mark.parametrize("mode", list(ListeningMode))
def test_every_mode_runs(config, mode):
    config.mode = mode
    engine = AuraEngine(config)
    out = drive(engine, render_scene([SceneLayer("human_speech", -8.0),
                                      SceneLayer("engine_noise", -12.0)], 1.5, SR))
    engine.stop()
    assert np.isfinite(out).all()


def test_mode_switching_changes_the_mix(config):
    """Switching mode must change the audio, not just a label."""
    signal = render_scene([SceneLayer("crowd_speech", -8.0)], 2.5, SR)

    def run(mode):
        config.mode = mode
        eng = AuraEngine(config)
        out = drive(eng, signal)
        eng.stop()
        return out

    focus = run(ListeningMode.FOCUS)     # suppresses crowd
    walk = run(ListeningMode.WALK)       # only reduces it, gently
    n = min(focus.size, walk.size)
    assert focus[:n].std() < walk[:n].std() * 0.85


def test_scene_is_published(engine):
    drive(engine, render_scene([SceneLayer("engine_noise", -10.0)], 2.0, SR))
    scene = engine.scene_dict()
    assert scene["sounds"]
    card = scene["sounds"][0]
    assert {"category", "label", "confidence", "action", "gain_db", "reason"} <= set(card)


def test_subscribers_receive_scenes(engine):
    received = []
    engine.subscribe(received.append)
    drive(engine, render_scene([SceneLayer("human_speech", -8.0)], 1.5, SR))
    assert received


def test_broken_subscriber_cannot_break_audio(engine):
    def explode(_scene):
        raise RuntimeError("subscriber failure")

    engine.subscribe(explode)
    out = drive(engine, render_scene([SceneLayer("human_speech", -8.0)], 1.0, SR))
    assert np.isfinite(out).all() and out.size > 0


def test_status_is_complete(engine):
    s = engine.status()
    for key in ("running", "enabled", "mode", "config", "audio",
                "backends", "metrics", "context", "person_lock", "safety"):
        assert key in s


def test_no_raw_audio_is_persisted(engine, tmp_path):
    """MVP criterion 10, checked against what actually lands on disk."""
    drive(engine, render_scene([SceneLayer("human_speech", -8.0)], 2.0, SR))
    path = engine.store.save(engine.profile, engine.learner)
    text = pathlib.Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    assert "samples" not in text and "waveform" not in text
    # The only float arrays permitted are speaker embeddings.
    for speaker in data["profile"]["speaker_profiles"]:
        assert len(speaker["centroid"]) < 500
    # A profile holding audio would be orders of magnitude larger than this.
    assert len(text) < 60_000

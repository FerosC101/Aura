"""Source separation contracts."""

import numpy as np

from aura.demo.synth import SceneLayer, generate, render_scene
from aura.separation.spectral import SpectralSourceSeparator, WOLAProcessor
from aura.types import StreamKind

SR, HOP = 16000, 320


def test_streams_sum_back_to_the_input():
    """The mixer's whole model depends on this.

    If the streams do not sum to the input, unity gain on everything would not
    reproduce the original scene, and every gain the mixer applies would be
    measured against a distorted baseline.
    """
    sep = SpectralSourceSeparator(SR, HOP)
    signal = render_scene(
        [SceneLayer("human_speech", -8.0), SceneLayer("engine_noise", -12.0)], 3.0, SR
    )
    summed, original = [], []
    for i in range(0, signal.size - HOP, HOP):
        block = signal[i:i + HOP]
        out = sep.separate(block)
        summed.append(sum(out.streams.values()))
        original.append(block)

    y = np.concatenate(summed)
    x = np.concatenate(original)
    delay = sep.algorithmic_delay_samples
    error = y[delay:] - x[:-delay]
    snr = 20 * np.log10(x[:-delay].std() / max(error.std(), 1e-18))
    assert snr > 100, f"reconstruction SNR only {snr:.1f} dB"


def test_wola_is_cola_compliant():
    """Unity masks must reproduce the input exactly."""
    w = WOLAProcessor(HOP, n_streams=1)
    x = np.random.randn(HOP * 20).astype(np.float64)
    out = []
    for i in range(0, x.size - HOP, HOP):
        out.append(w.synthesise([w.analyse(x[i:i + HOP])])[0])
    y = np.concatenate(out)
    assert np.allclose(y[HOP:], x[: y.size - HOP], atol=1e-9)


def test_separator_reports_its_delay():
    sep = SpectralSourceSeparator(SR, HOP)
    assert sep.algorithmic_delay_samples == HOP


def test_emits_all_four_streams():
    sep = SpectralSourceSeparator(SR, HOP)
    out = sep.separate(np.random.randn(HOP).astype(np.float32) * 0.1)
    assert set(out.streams) == {
        StreamKind.PRIMARY_SPEECH, StreamKind.SECONDARY_SPEECH,
        StreamKind.TRANSIENT, StreamKind.ENVIRONMENT,
    }
    assert all(s.size == HOP for s in out.streams.values())


def _route(category: str, seconds: float = 3.0) -> dict[StreamKind, float]:
    sep = SpectralSourceSeparator(SR, HOP)
    signal = generate(category, seconds, SR)
    acc = {k: 0.0 for k in sep.streams}
    for i in range(0, signal.size - HOP, HOP):
        out = sep.separate(signal[i:i + HOP])
        for k, v in out.streams.items():
            acc[k] += float(np.sum(v.astype(np.float64) ** 2))
    total = sum(acc.values()) + 1e-18
    return {k: v / total for k, v in acc.items()}


def test_engine_noise_lands_in_the_environment_stream():
    """Suppressing an engine is only possible if it is isolated first."""
    assert _route("engine_noise")[StreamKind.ENVIRONMENT] > 0.9


def test_siren_lands_in_the_transient_stream():
    """A siren in the speech stream would be attenuated whenever speech is."""
    assert _route("emergency_siren")[StreamKind.TRANSIENT] > 0.6


def test_speech_reaches_the_speech_streams():
    r = _route("human_speech")
    assert r[StreamKind.PRIMARY_SPEECH] + r[StreamKind.SECONDARY_SPEECH] > 0.3
    # And is not mistakenly routed to the protected transient path.
    assert r[StreamKind.TRANSIENT] < 0.15


def test_separator_improves_speech_to_noise_ratio():
    """The headline separation claim, measured on a mixture."""
    layers = [SceneLayer("human_speech", -6.0), SceneLayer("engine_noise", -10.0)]
    mix, stems = render_scene(layers, 5.0, SR, return_stems=True)

    sep = SpectralSourceSeparator(SR, HOP)
    wolas = {k: WOLAProcessor(HOP, 1) for k in stems}
    out = {k: [] for k in stems}
    for i in range(0, mix.size - HOP, HOP):
        sep.separate(mix[i:i + HOP])
        mask = sep.last_masks[StreamKind.PRIMARY_SPEECH]
        for name in stems:
            spec = wolas[name].analyse(np.asarray(stems[name][i:i + HOP], dtype=np.float64))
            out[name].append(wolas[name].synthesise([spec * mask])[0])

    energy = {k: float(np.sum(np.concatenate(v) ** 2)) for k, v in out.items()}
    inp = {k: float(np.sum(v.astype(np.float64) ** 2)) for k, v in stems.items()}
    before = 10 * np.log10(inp["human_speech"] / inp["engine_noise"])
    after = 10 * np.log10(energy["human_speech"] / max(energy["engine_noise"], 1e-18))
    assert after - before > 8.0, f"SIR gain only {after - before:.1f} dB"


def test_reset_clears_state():
    sep = SpectralSourceSeparator(SR, HOP)
    loud = np.random.randn(HOP).astype(np.float32)
    for _ in range(10):
        sep.separate(loud)
    sep.reset()
    out = sep.separate(np.zeros(HOP, dtype=np.float32))
    assert all(np.allclose(v, 0.0) for v in out.streams.values())

"""End-to-end benchmark.

Runs a full demo scene through the complete engine and reports the three things
that decide whether AURA works:

1. **What it heard** -- detected categories and the action taken on each.
2. **What it did to the audio** -- measured dB change per source band, which is
   the claim that matters. A classifier that labels correctly but leaves the
   audio unchanged is a dashboard, not a product.
3. **How fast** -- per-stage latency and processing load.

It also compares AURA against a conventional ANC simulation on the metric that
distinguishes them: whether a siren survives.
"""
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from aura.config import EngineConfig
from aura.demo.scenes import get_scene, simulate_anc
from aura.pipeline.engine import AuraEngine

SR, HOP = 48000, 960
SCENE_KEY = sys.argv[1] if len(sys.argv) > 1 else "psc_demo"


def band_db(x: np.ndarray, sr: int, lo: float, hi: float,
            t0: float = 0.0, t1: float | None = None) -> float:
    """Energy in a frequency band over a time window, in dB."""
    a = int(t0 * sr)
    b = int(t1 * sr) if t1 is not None else x.size
    seg = x[a:b]
    if seg.size < 32:
        return -120.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return 10.0 * np.log10(float(spec[mask].sum()) + 1e-18)


scene = get_scene(SCENE_KEY)
print(f"=== {scene.title} -- {scene.subtitle} ===")
print(f"{scene.description}\n")

mix = scene.render(SR)
cfg = EngineConfig(sample_rate=SR, block_size=HOP, mode=scene.mode,
                   profile_dir=tempfile.mkdtemp(), publish_rate_hz=1000.0)
engine = AuraEngine(cfg)

captured: list[np.ndarray] = []
seen: dict[str, dict] = {}


def on_scene(_s):
    for card in engine.scene_dict().get("sounds", []):
        c = card["category"]
        rec = seen.setdefault(c, {"frames": 0, "conf": 0.0, "gain": 0.0,
                                  "actions": {}, "safety": False})
        rec["frames"] += 1
        rec["conf"] += card["confidence"]
        rec["gain"] += card["gain_db"]
        # The modal action, not the last one: a single boundary-straddling
        # frame should not decide how a source is reported.
        rec["actions"][card["action"]] = rec["actions"].get(card["action"], 0) + 1
        rec["safety"] |= card["safety_override"]


engine.subscribe(on_scene)
engine.start(signal=mix, realtime=False)
# OfflineAudioEngine with realtime=False still runs on its thread; drive it
# synchronously instead so the benchmark is deterministic.
engine.stop()

engine._audio = None
engine._running = True
engine._worker_stop.clear()
import threading

engine._worker = threading.Thread(target=engine._semantic_loop, daemon=True)
engine._worker.start()

t0 = time.perf_counter()
for i in range(0, mix.size - HOP, HOP):
    out = np.zeros(HOP, dtype=np.float32)
    engine._process(mix[i:i+HOP], out)
    captured.append(out.copy())
wall = time.perf_counter() - t0
engine.stop()

aura_out = np.concatenate(captured)
n = min(aura_out.size, mix.size)
raw, aura_out = mix[:n], aura_out[:n]
anc_out = simulate_anc(raw, SR)

print("WHAT AURA HEARD")
print(f"{'category':<22}{'frames':>8}{'conf':>8}{'action':>10}{'gain':>9}   expected")
print("-" * 78)
ok = 0
for cat, exp in scene.expect.items():
    r = seen.get(cat)
    if r is None:
        print(f"{cat:<22}{'--':>8}{'--':>8}{'not detected':>10}{'--':>9}   {exp}  MISS")
        continue
    conf = r["conf"] / r["frames"]
    gain = r["gain"] / r["frames"]
    action = max(r["actions"], key=lambda k: r["actions"][k])
    match = action == exp
    ok += match
    print(f"{cat:<22}{r['frames']:>8}{conf:>8.2f}{action:>10}{gain:>+9.1f}   "
          f"{exp:<9}{'OK' if match else 'differs'}")
total_frames = max((r["frames"] for r in seen.values()), default=1)
extra = [c for c, r in seen.items()
         if c not in scene.expect and r["frames"] > 0.2 * total_frames]
print("-" * 78)
print(f"{ok}/{len(scene.expect)} sources handled as scripted"
      + (f"   (also detected: {', '.join(extra)})" if extra else ""))

print("\nWHAT AURA DID TO THE AUDIO (dB change, input -> output)")
BANDS = {
    "engine / rumble  20-300 Hz": (20.0, 300.0),
    "speech body     300-1k Hz": (300.0, 1000.0),
    "speech clarity   1k-4k Hz": (1000.0, 4000.0),
    "horn / siren   400-3.5k Hz": (400.0, 3500.0),
}
print(f"{'band':<28}{'AURA':>10}{'ANC sim':>10}")
print("-" * 50)
for label, (lo, hi) in BANDS.items():
    a = band_db(aura_out, SR, lo, hi) - band_db(raw, SR, lo, hi)
    c = band_db(anc_out, SR, lo, hi) - band_db(raw, SR, lo, hi)
    print(f"{label:<28}{a:>+10.1f}{c:>+10.1f}")

# Safety comparison during the siren window, if the scene has one.
siren = next((lay for lay in scene.layers if lay.category == "emergency_siren"), None)
if siren is not None:
    t0s, t1s = siren.start + 0.5, (siren.end or scene.duration) - 0.5
    print(f"\nSIREN AUDIBILITY ({t0s:.1f}-{t1s:.1f}s, 500-3000 Hz)")
    base = band_db(raw, SR, 500, 3000, t0s, t1s)
    for name, sig in (("Raw", raw), ("ANC simulation", anc_out), ("AURA", aura_out)):
        print(f"  {name:<16}{band_db(sig, SR, 500, 3000, t0s, t1s) - base:>+7.1f} dB")

m = engine.metrics.snapshot()
print(f"\nLATENCY (block = {m.block_ms:.1f} ms)")
for stage in ("preprocess", "separation", "classification", "speaker", "policy", "mixing"):
    s = m.stages[stage]
    print(f"  {stage:<16}mean {s['mean_ms']:>6.3f} ms   p95 {s['p95_ms']:>6.3f} ms")
print(f"  {'-' * 46}")
compute_ms = sum(
    m.stages[s]["mean_ms"]
    for s in ("preprocess", "separation", "classification", "speaker", "policy", "mixing")
)
print(f"  processing        {compute_ms:.3f} ms/block")
print(f"  processing load   {m.processing_load*100:.1f}% of real time")
print(f"  algorithmic delay {m.algorithmic_delay_ms:.1f} ms")
print(f"  frames            {m.frames}   xruns {m.xruns}")
print(f"  wall clock        {wall:.2f}s for {n/SR:.1f}s audio "
      f"({n/SR/wall:.1f}x real time)")

"""Evaluate the classifier on ESC-50: real recordings, not the synthesiser.

``eval_classifier.py`` scores the classifier on audio from ``aura.demo.synth``
-- the same generators its thresholds were tuned against. That measures
self-consistency, not accuracy. This script scores it on ESC-50 (2000 real
field recordings, 50 classes, 5 s each, 44.1 kHz) so the numbers mean something
outside the fixture.

Three groups, because ESC-50 and the AURA taxonomy only partly overlap:

* **targets** -- ESC-50 classes that map onto a non-ambient AURA category.
  These give per-class recall: the honest accuracy number.
* **ambient-like** -- rain, wind, insects. Should come out ``ambient``; mostly
  a check that quiet backgrounds do not excite anything.
* **out-of-taxonomy** -- the rest. There is no right answer for a chainsaw in a
  taxonomy with no chainsaw, so these are scored only on what they *must not*
  do: fire a safety-critical category. A hearing device that cries "siren" at a
  crying baby is worse than one that says nothing.

Usage::

    python scripts/eval_esc50.py                     # all five folds
    python scripts/eval_esc50.py --limit 5           # 5 clips per class, quick
    python scripts/eval_esc50.py --folds 1 --json out.json

Get the dataset with ``--download`` (~600 MB) or point ``--dataset`` at an
existing ESC-50 checkout.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import urllib.request
import wave
import zipfile
from collections import Counter, defaultdict

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aura.classification.dsp_classifier import DSPSoundClassifier  # noqa: E402
from aura.classification.onnx_backend import WINDOW_SECONDS  # noqa: E402
from aura.classification.registry import build_classifier, fallback_notes  # noqa: E402
from aura.classification.taxonomy import SAFETY_CRITICAL_KEYS  # noqa: E402
from aura.types import AudioFrame  # noqa: E402

SR = 16000
HOP = 320
WARMUP = 25          # frames the classifier needs before its features settle
ACTIVE_FLOOR_DB = 30.0   # a frame counts if within this of the clip's loud frames
TARGET_RMS_DB = -20.0    # every clip is levelled here; see "Level" in the notes

DATA_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
DEFAULT_DIR = pathlib.Path.home() / ".cache" / "aura" / "esc50" / "ESC-50-master"
DEFAULT_MODEL = pathlib.Path.home() / ".cache" / "aura" / "models" / "yamnet.onnx"

# --------------------------------------------------------------------------
# ESC-50 -> AURA taxonomy
# --------------------------------------------------------------------------
# Only pairs that are defensible without squinting. Where ESC-50 is more
# specific than AURA (chainsaw, vacuum cleaner) the AURA category it rolls up
# into is used; where AURA has no home for a sound it stays out-of-taxonomy
# rather than being forced into the nearest bin, which would flatter the score.

TARGETS: dict[str, str] = {
    "siren": "emergency_siren",
    "car_horn": "vehicle_horn",
    "engine": "engine_noise",
    "clock_alarm": "alarm",
    "vacuum_cleaner": "mechanical_noise",
    "washing_machine": "mechanical_noise",
    "chainsaw": "mechanical_noise",
    "hand_saw": "mechanical_noise",
    "helicopter": "mechanical_noise",
}

AMBIENT_LIKE: frozenset[str] = frozenset({
    "rain", "wind", "sea_waves", "crickets", "insects", "chirping_birds",
})

# AURA categories ESC-50 cannot reach at all: it contains no speech and no
# music by design. Reported so the gap is visible rather than implied.
UNTESTABLE = (
    "human_speech", "crowd_speech", "announcement", "music",
    "general_traffic", "approaching_vehicle", "bicycle_bell", "warning_signal",
)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def fetch(dest: pathlib.Path) -> None:
    """Download and unpack ESC-50 next to ``dest`` if it is not there."""
    if (dest / "meta" / "esc50.csv").exists():
        return
    root = dest.parent
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "esc50.zip"
    if not zip_path.exists():
        print(f"downloading ESC-50 (~600 MB) -> {zip_path}")
        urllib.request.urlretrieve(DATA_URL, zip_path)  # noqa: S310  (fixed URL)
    print(f"unpacking -> {root}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)


def read_clip(path: pathlib.Path) -> np.ndarray:
    """Load a 16-bit WAV, downmix, and resample to the engine's rate."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != SR:
        g = np.gcd(rate, SR)
        x = resample_poly(x, SR // g, rate // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def level(x: np.ndarray) -> np.ndarray:
    """Normalise to a fixed RMS over the clip's non-silent part.

    ESC-50 levels vary by tens of dB between clips and the DSP classifier reads
    absolute level in places. On device an AGC does this job, so doing it here
    tests classification rather than gain staging. The alternative -- feeding
    raw levels -- mostly measures how loud the recordist stood.
    """
    rms = float(np.sqrt(np.mean(x[np.abs(x) > 1e-4] ** 2))) if np.any(np.abs(x) > 1e-4) else 0.0
    if rms <= 0.0:
        return x
    gain = (10.0 ** (TARGET_RMS_DB / 20.0)) / rms
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def frames_of(x: np.ndarray) -> list[np.ndarray]:
    return [x[i:i + HOP] for i in range(0, x.size - HOP + 1, HOP)]


def active_mask(blocks: list[np.ndarray]) -> list[bool]:
    """Which frames carry the event, not the silence ESC-50 pads clips with.

    A frame is active if its RMS is within ``ACTIVE_FLOOR_DB`` of the clip's
    90th-percentile frame. Without this, recall on a 5 s clip holding a 0.5 s
    car horn is dominated by 4.5 s of room tone and reads as failure.
    """
    rms = np.array([float(np.sqrt(np.mean(b ** 2)) + 1e-12) for b in blocks])
    ref = float(np.percentile(rms, 90))
    floor = ref * (10.0 ** (-ACTIVE_FLOOR_DB / 20.0))
    return [bool(r >= floor) for r in rms]


def make_classifier(backend: str, model_path: str | None):
    """Build a fresh classifier per clip, so no state leaks between recordings."""
    if backend == "dsp":
        return DSPSoundClassifier(SR), WARMUP
    clf = build_classifier(backend, SR, model_path=model_path)
    note = fallback_notes().get("classifier")
    if note:
        raise SystemExit(f"{backend} backend unavailable: {note}")
    # A neural backend reports nothing until its rolling window is full, so it
    # needs a longer warm-up than the DSP classifier's feature settling time.
    hops = int(WINDOW_SECONDS * SR / HOP) + 4
    return clf, max(WARMUP, hops)


def classify_clip(
    x: np.ndarray, backend: str = "dsp", model_path: str | None = None
) -> tuple[Counter, int, dict[str, float]]:
    """Run the classifier over one clip.

    Returns per-category frame counts, the number of frames scored, and the
    mean confidence per category over the frames where it fired.
    """
    clf, warmup = make_classifier(backend, model_path)
    blocks = frames_of(x)
    active = active_mask(blocks)
    counts: Counter = Counter()
    conf_sum: dict[str, float] = defaultdict(float)
    scored = 0
    for i, block in enumerate(blocks):
        events = clf.classify(AudioFrame(block, SR, sequence=i))
        if i < warmup or not active[i]:
            continue          # still fed to the classifier, just not scored
        scored += 1
        for e in events:
            counts[e.category] += 1
            conf_sum[e.category] += e.confidence
    conf = {k: conf_sum[k] / counts[k] for k in counts}
    return counts, max(scored, 1), conf


def control(backend: str = "dsp", model_path: str | None = None) -> int:
    """Score the synthesiser through this file's own pipeline.

    The point of a control: if resampling, levelling or the active-frame gate
    were quietly destroying the signal, ESC-50 would collapse for reasons that
    have nothing to do with the classifier. Feeding ``aura.demo.synth`` through
    the identical path should reproduce ``eval_classifier.py``. If this prints
    13/13 and ESC-50 does not, the gap is real.
    """
    from aura.demo.synth import GENERATORS, generate

    cats = [k for k in GENERATORS if k != "ambient"]
    print(f"{'SYNTHETIC SOURCE':<22}{'RAW':>7}{'LEVELLED':>10}")
    print("-" * 42)
    ok_raw = ok_lvl = 0
    for cat in cats:
        sig = generate(cat, 4.0, SR)
        raw, n, _ = classify_clip(sig, backend, model_path)
        lvl, m, _ = classify_clip(level(sig), backend, model_path)
        r, q = raw.get(cat, 0) / n, lvl.get(cat, 0) / m
        ok_raw += r >= 0.5
        ok_lvl += q >= 0.5
        print(f"{cat:<22}{r*100:>6.0f}%{q*100:>9.0f}%")
    print("-" * 42)
    print(f"recall raw {ok_raw}/{len(cats)}   levelled {ok_lvl}/{len(cats)}")
    if backend == "dsp":
        print("\nBoth should be 13/13: the DSP classifier was tuned on these very\n"
              "generators. Anything less means this harness, not the classifier,\n"
              "is losing the signal -- fix that before reading ESC-50.")
        return 0 if ok_lvl == len(cats) else 1
    print("\nA low score here is expected and is not a harness fault. A model\n"
          "trained on real recordings has never heard a synthesiser, so this\n"
          "measures how far aura/demo/synth.py sits from real audio -- the same\n"
          "gap that makes synthetic-only validation unsafe, seen from the other\n"
          "side. Judge a neural backend on ESC-50, not here.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DIR)
    ap.add_argument("--download", action="store_true", help="fetch ESC-50 if absent")
    ap.add_argument("--folds", default="1,2,3,4,5", help="comma-separated ESC-50 folds")
    ap.add_argument("--limit", type=int, default=0, help="clips per class (0 = all)")
    ap.add_argument("--hit", type=float, default=0.5,
                    help="fraction of active frames needed to count a clip as recalled")
    ap.add_argument("--json", type=pathlib.Path, help="write the full result here")
    ap.add_argument("--backend", default="dsp",
                    help="dsp (default) or onnx/yamnet")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="path to the ONNX model, for --backend onnx")
    ap.add_argument("--control", action="store_true",
                    help="score the synthesiser through this same path and exit")
    args = ap.parse_args()

    if args.control:
        return control(args.backend, args.model)
    if args.download:
        fetch(args.dataset)
    meta = args.dataset / "meta" / "esc50.csv"
    if not meta.exists():
        print(f"no ESC-50 at {args.dataset}\nrun with --download, or pass --dataset",
              file=sys.stderr)
        return 2

    folds = {int(f) for f in args.folds.split(",")}
    rows = [r for r in csv.DictReader(meta.open()) if int(r["fold"]) in folds]
    by_class: dict[str, list[str]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["filename"]):
        by_class[r["category"]].append(r["filename"])
    if args.limit:
        by_class = {k: v[:args.limit] for k, v in by_class.items()}

    audio = args.dataset / "audio"
    results: dict[str, dict] = {}
    print(f"ESC-50: {sum(len(v) for v in by_class.values())} clips, "
          f"folds {sorted(folds)}, {len(by_class)} classes, "
          f"backend {args.backend}\n")

    for esc_class in sorted(by_class):
        truth = TARGETS.get(esc_class)
        group = ("target" if truth else
                 "ambient-like" if esc_class in AMBIENT_LIKE else "out-of-taxonomy")
        clips = []
        for fname in by_class[esc_class]:
            counts, scored, conf = classify_clip(
                level(read_clip(audio / fname)), args.backend, args.model)
            clips.append({
                "file": fname,
                "frames": scored,
                "rates": {k: v / scored for k, v in counts.items()},
                "conf": conf,
            })
        results[esc_class] = {"group": group, "truth": truth, "clips": clips}
        print(f"  {esc_class:<18} {group:<16} {len(clips)} clips")

    report(results, args.hit)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.json}")
    return 0


def report(results: dict[str, dict], hit_frac: float) -> None:
    def rate(clip: dict, key: str) -> float:
        return clip["rates"].get(key, 0.0)

    print("\n" + "=" * 94)
    print("TARGETS -- recall against a real recording of the thing")
    print("=" * 94)
    print(f"{'ESC-50 CLASS':<18}{'AURA CATEGORY':<18}{'CLIP':>7}{'FRAME':>8}{'CONF':>7}"
          "   TOP CONFUSION")
    print("-" * 94)
    tgt_hits = tgt_n = 0
    for esc_class, r in sorted(results.items()):
        if r["group"] != "target":
            continue
        clips, truth = r["clips"], r["truth"]
        recalled = sum(rate(c, truth) >= hit_frac for c in clips)
        frame = float(np.mean([rate(c, truth) for c in clips]))
        confs = [c["conf"][truth] for c in clips if truth in c["conf"]]
        conf = float(np.mean(confs)) if confs else 0.0
        wrong: Counter = Counter()
        for c in clips:
            for k, v in c["rates"].items():
                if k not in (truth, "ambient") and v >= hit_frac:
                    wrong[k] += 1
        top = ", ".join(f"{k} {v}/{len(clips)}" for k, v in wrong.most_common(2)) or "--"
        tgt_hits += recalled
        tgt_n += len(clips)
        print(f"{esc_class:<18}{truth:<18}{recalled:>3}/{len(clips):<3}"
              f"{frame*100:>7.0f}%{conf*100:>6.0f}%   {top}")
    print("-" * 94)
    if tgt_n:
        print(f"clip recall on real audio: {tgt_hits}/{tgt_n} = {tgt_hits/tgt_n*100:.0f}%")

    # ---- safety-critical false alarms -------------------------------------
    print("\n" + "=" * 94)
    print("SAFETY-CRITICAL FALSE ALARMS -- on audio that is none of these")
    print("=" * 94)
    rows = []
    for esc_class, r in sorted(results.items()):
        if r["group"] == "target" and r["truth"] in SAFETY_CRITICAL_KEYS:
            continue   # a siren firing emergency_siren is the point, not a fault
        fired: Counter = Counter()
        for c in r["clips"]:
            for key in SAFETY_CRITICAL_KEYS:
                if rate(c, key) >= hit_frac:
                    fired[key] += 1
        if fired:
            rows.append((esc_class, r["group"], len(r["clips"]), fired))
    if rows:
        print(f"{'ESC-50 CLASS':<18}{'GROUP':<17}{'CLIPS':>6}   FIRED")
        print("-" * 94)
        for esc_class, group, n, fired in sorted(rows, key=lambda t: -sum(t[3].values())):
            txt = ", ".join(f"{k} {v}/{n}" for k, v in fired.most_common())
            print(f"{esc_class:<18}{group:<17}{n:>6}   {txt}")
    else:
        print("none -- no non-safety class triggered a safety category")
    total = sum(len(r["clips"]) for r in results.values()
                if not (r["group"] == "target" and r["truth"] in SAFETY_CRITICAL_KEYS))
    bad = sum(sum(f.values()) for _, _, _, f in rows)
    print("-" * 94)
    print(f"false-alarm clips: {bad}/{total} = {bad/max(total,1)*100:.1f}% of non-safety audio")

    # ---- coverage ---------------------------------------------------------
    print("\n" + "=" * 94)
    print("COVERAGE -- what ESC-50 cannot tell us")
    print("=" * 94)
    groups = Counter(r["group"] for r in results.values())
    for g, n in groups.most_common():
        print(f"  {g:<17} {n} ESC-50 classes")
    print(f"\n  AURA categories ESC-50 never exercises ({len(UNTESTABLE)}):")
    print(f"    {', '.join(UNTESTABLE)}")
    print("  ESC-50 contains no speech and no music by design, so the whole")
    print("  speech side of the taxonomy -- and Person Lock and Name Awareness")
    print("  with it -- needs a different corpus. UrbanSound8K reaches")
    print("  general_traffic and street music; speech needs LibriSpeech or AMI.")


if __name__ == "__main__":
    raise SystemExit(main())

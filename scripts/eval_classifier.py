"""Confusion matrix: run the classifier over each isolated synthetic source.

Reports recall (did we find the true source?) and the false positives it
triggered, so tuning changes can be judged on both at once.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from aura.classification.dsp_classifier import DSPSoundClassifier
from aura.demo.synth import GENERATORS, generate
from aura.types import AudioFrame

SR, HOP, WARMUP = 16000, 320, 25
targets = [k for k in GENERATORS if k != "ambient"]

def run(cat):
    sig = generate(cat, 4.0, SR)
    clf = DSPSoundClassifier(SR)
    tallies, scored = {}, 0
    for i in range(0, sig.size - HOP, HOP):
        ev = clf.classify(AudioFrame(sig[i:i+HOP], SR, sequence=i//HOP))
        if i // HOP < WARMUP:
            continue
        scored += 1
        for e in ev:
            tallies[e.category] = tallies.get(e.category, 0) + 1
    return tallies, max(scored, 1)

print(f"{'TRUE SOURCE':<22}{'RECALL':>8}   FALSE POSITIVES (>30% of frames)")
print("-" * 92)
rec_ok = fp_total = 0
for cat in targets:
    tal, n = run(cat)
    hit = tal.get(cat, 0) / n
    rec_ok += hit >= 0.5
    fps = sorted(((k, v / n) for k, v in tal.items() if k != cat and v / n > 0.30),
                 key=lambda kv: -kv[1])
    fp_total += len(fps)
    mark = "PASS" if hit >= 0.5 else "FAIL"
    txt = ", ".join(f"{k}:{v*100:.0f}%" for k, v in fps) or "-- none --"
    print(f"{cat:<22}{hit*100:>6.0f}% {mark}  {txt}")
print("-" * 92)
print(f"recall {rec_ok}/{len(targets)} sources   |   {fp_total} false-positive pairs above 30%")

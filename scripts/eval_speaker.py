"""Person Lock verification benchmark.

Measures the embedder's same/different-speaker separation and then checks that
the lock accepts the enrolled speaker and rejects everyone else.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from aura.demo.synth import speech
from aura.speaker.person_lock import PersonLockManager
from aura.speaker.spectral_embedder import SpectralSpeakerEmbedder

SR, HOP = 16000, 320
VOICES = {
    "Vince": dict(f0=112.0, formants=(500, 1480, 2500, 3400)),
    "Maria": dict(f0=208.0, formants=(650, 1900, 2800, 3600)),
    "Ben":   dict(f0=148.0, formants=(430, 1100, 2300, 3300)),
    "Ana":   dict(f0=178.0, formants=(600, 1700, 2650, 3500)),
}
emb = SpectralSpeakerEmbedder(SR)

vecs = {}
for name, cfg in VOICES.items():
    for c in range(3):
        vecs[(name, c)] = emb.embed(speech(2.5, SR, seed=40 + c * 13,
                                           syllable_rate=3.4 + 0.8 * c, **cfg), SR)
same = [emb.similarity(vecs[(n, i)], vecs[(n, j)])
        for n in VOICES for i in range(3) for j in range(i + 1, 3)]
diff = [emb.similarity(vecs[(a, i)], vecs[(b, j)])
        for a in VOICES for b in VOICES if a < b for i in range(3) for j in range(3)]
print(f"same-speaker  mean={np.mean(same):.4f}  min={np.min(same):.4f}")
print(f"diff-speaker  mean={np.mean(diff):.4f}  max={np.max(diff):.4f}  std={np.std(diff):.4f}")
print(f"-> suggested impostor_prior = ({np.mean(diff):.3f}, {np.std(diff):.3f})")
print(f"   equal-error separation: {'OK' if np.min(same) > np.max(diff) else 'OVERLAP'}\n")

mgr = PersonLockManager(emb, SR)
profs = {n: mgr.enrol(n, [speech(2.5, SR, seed=40 + c * 13, syllable_rate=3.4 + 0.8 * c, **cfg)
                          for c in range(2)], SR)
         for n, cfg in VOICES.items()}
mgr.select(profs["Vince"].id)

print(f"{'speaking':<10}{'locked':>8}{'similarity':>12}{'margin':>9}   expected")
print("-" * 55)
ok = 0
for name, cfg in VOICES.items():
    mgr.reset()
    sig = speech(5.0, SR, seed=999, syllable_rate=4.3, **cfg)
    last = None
    for i in range(0, sig.size - HOP, HOP):
        last = mgr.update(sig[i:i+HOP], True)
    expected = (name == "Vince")
    good = last.active == expected
    ok += good
    print(f"{name:<10}{str(last.active):>8}{last.similarity:>12.4f}{last.margin:>9.2f}"
          f"   {'lock' if expected else 'reject':<7} {'PASS' if good else 'FAIL'}")
print("-" * 55)
print(f"{ok}/{len(VOICES)} correct")

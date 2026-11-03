"""Name Awareness benchmark.

Measures the template spotter on isolated spoken words -- the regime it is
designed for: a name said as its own short utterance to get someone's
attention.

Words are synthesized as formant trajectories. The fixture has no consonants,
which carry much of what distinguishes real words, so these numbers if anything
understate what real recordings achieve.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from aura.demo.synth import spoken_word
from aura.speaker.spectral_embedder import SpectralSpeakerEmbedder
from aura.speech.name_awareness import (
    NameAwareness,
    TemplateKeywordSpotter,
    _mfcc_sequence,
    dtw_distance,
)

SR, HOP = 16000, 320
emb = SpectralSpeakerEmbedder(SR)
spotter = TemplateKeywordSpotter(SR)
spotter.enrol("Vince", [spoken_word("vince", 0.7, SR, f0=118, seed=s, rate_jitter=0.04)
                        for s in (5, 6)], SR)
print(f"enrolled: {spotter.words}   threshold={spotter.threshold}\n")


def distance(clip):
    seq = _mfcc_sequence(emb, np.asarray(clip, dtype=np.float64))
    return min(dtw_distance(t, seq) for t in spotter._templates["Vince"])


print(f"{'utterance':<28}{'spotted':>9}{'dist':>8}{'score':>8}   expected")
print("-" * 68)
cases = [
    ("vince, same voice",   spoken_word("vince", 0.70, SR, f0=120, seed=9,  rate_jitter=0.06), True),
    ("vince, higher voice", spoken_word("vince", 0.70, SR, f0=200, seed=11, rate_jitter=0.05), True),
    ("vince, slower",       spoken_word("vince", 0.88, SR, f0=116, seed=13, rate_jitter=0.05), True),
    ("vince, faster",       spoken_word("vince", 0.58, SR, f0=130, seed=21, rate_jitter=0.06), True),
    ("maria",               spoken_word("maria", 0.70, SR, f0=118, seed=7), False),
    ("hello",               spoken_word("hello", 0.70, SR, f0=118, seed=7), False),
    ("ben",                 spoken_word("ben",   0.62, SR, f0=118, seed=8), False),
    ("ana",                 spoken_word("ana",   0.66, SR, f0=118, seed=8), False),
    ("stop",                spoken_word("stop",  0.60, SR, f0=150, seed=8), False),
]
ok = 0
tgt, oth = [], []
for label, clip, expected in cases:
    d = distance(clip)
    (tgt if expected else oth).append(d)
    hit = spotter.detect(clip, SR)
    got = hit is not None
    ok += got == expected
    print(f"{label:<28}{str(got):>9}{d:>8.3f}{(f'{hit.score:.2f}' if hit else '-'):>8}   "
          f"{'spot' if expected else 'reject':<7} {'PASS' if got == expected else 'FAIL'}")
print("-" * 68)
print(f"{ok}/{len(cases)} correct")
print(f"target {min(tgt):.3f}-{max(tgt):.3f}   other {min(oth):.3f}-{max(oth):.3f}   "
      f"margin {min(oth) - max(tgt):+.3f}")

t0 = time.perf_counter()
for _ in range(20):
    spotter.detect(cases[0][1], SR)
print(f"detect(): {(time.perf_counter() - t0) / 20 * 1000:.1f} ms/call "
      f"(runs on the semantic worker thread, not the audio callback)")

na = NameAwareness(spotter, SR, enabled=True)
clip = spoken_word("vince", 0.7, SR, f0=122, seed=33, rate_jitter=0.05)
stream = np.concatenate([np.zeros(int(0.2 * SR), np.float32), clip,
                         np.zeros(int(0.4 * SR), np.float32)])
fired = any(na.update(stream[i:i+HOP], True) for i in range(0, stream.size - HOP, HOP))
print(f"streaming a spoken name -> fired: {fired}   "
      f"hit: {na.last_hit.as_dict() if na.last_hit else None}")

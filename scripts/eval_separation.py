"""Separation benchmark.

Two measurements, both on a realistic *mixture* rather than isolated sources:

1. Stream routing -- where each true source's energy ends up.
2. Interference rejection (SIR gain) -- how much better the speech-to-noise
   ratio is inside the primary-speech stream than in the raw mix.

The masks are estimated from the mixture and then applied to the known stems,
which is the standard way to attribute a mask-based separator's output.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from aura.demo.synth import SceneLayer, render_scene
from aura.separation.spectral import SpectralSourceSeparator, WOLAProcessor
from aura.types import StreamKind

SR, HOP, DUR = 16000, 320, 6.0
ORDER = [StreamKind.PRIMARY_SPEECH, StreamKind.SECONDARY_SPEECH,
         StreamKind.TRANSIENT, StreamKind.ENVIRONMENT]
LAYERS = [
    SceneLayer("human_speech", -6.0),
    SceneLayer("engine_noise", -10.0),
    SceneLayer("crowd_speech", -14.0),
    SceneLayer("vehicle_horn", -8.0, start=1.5, end=2.5),
    SceneLayer("emergency_siren", -10.0, start=3.5, end=5.5),
]

mix, stems = render_scene(LAYERS, DUR, SR, return_stems=True)
sep = SpectralSourceSeparator(SR, HOP)
names = list(stems)
# One WOLA per (stem, stream) so each stem can be resynthesised through the
# mixture's masks.
wolas = {n: WOLAProcessor(HOP, len(ORDER)) for n in names}
out = {n: {k: [] for k in ORDER} for n in names}

for i in range(0, mix.size - HOP, HOP):
    sep.separate(mix[i:i+HOP])
    masks = sep.last_masks
    for n in names:
        spec = wolas[n].analyse(np.asarray(stems[n][i:i+HOP], dtype=np.float64))
        ys = wolas[n].synthesise([spec * masks[k] for k in ORDER])
        for k, y in zip(ORDER, ys, strict=True):
            out[n][k].append(y)

energy = {n: {k: float(np.sum(np.concatenate(v)**2)) for k, v in d.items()} for n, d in out.items()}

print("STREAM ROUTING (% of each true source's energy)")
print(f"{'true source':<18}" + "".join(f"{k.value:>19}" for k in ORDER))
print("-" * 96)
for n in names:
    tot = sum(energy[n].values()) + 1e-18
    print(f"{n:<18}" + "".join(f"{100*energy[n][k]/tot:>18.1f}%" for k in ORDER))

def db(a, b): return 10*np.log10((a+1e-18)/(b+1e-18))

print("\nINTERFERENCE REJECTION")
noise_src = ["engine_noise", "crowd_speech"]
sp_in  = float(np.sum(stems["human_speech"].astype(np.float64)**2))
nz_in  = sum(float(np.sum(stems[n].astype(np.float64)**2)) for n in noise_src)
sp_out = energy["human_speech"][StreamKind.PRIMARY_SPEECH]
nz_out = sum(energy[n][StreamKind.PRIMARY_SPEECH] for n in noise_src)
print(f"  speech vs (engine+crowd)  in mix: {db(sp_in,nz_in):+6.1f} dB"
      f"   in primary_speech: {db(sp_out,nz_out):+6.1f} dB"
      f"   gain: {db(sp_out,nz_out)-db(sp_in,nz_in):+5.1f} dB")

# Transient events are windowed to when they are actually sounding -- the horn
# and the siren never overlap, so scoring either over the whole clip would
# count the other one as interference and understate both.
WINDOWS = {"vehicle_horn": (1.5, 2.5), "emergency_siren": (3.5, 5.5)}
for ev, (t0, t1) in WINDOWS.items():
    lo, hi = int(t0 * SR / HOP), int(t1 * SR / HOP)
    others = [n for n in names if n != ev]
    def win(n, k, lo=lo, hi=hi):
        return float(np.sum(np.concatenate(out[n][k][lo:hi]).astype(np.float64) ** 2))
    ev_in = float(np.sum(stems[ev][lo*HOP:hi*HOP].astype(np.float64) ** 2))
    ot_in = sum(float(np.sum(stems[n][lo*HOP:hi*HOP].astype(np.float64) ** 2)) for n in others)
    ev_out = win(ev, StreamKind.TRANSIENT)
    ot_out = sum(win(n, StreamKind.TRANSIENT) for n in others)
    print(f"  {ev:<16} vs rest  in mix: {db(ev_in,ot_in):+6.1f} dB"
          f"   in transient:      {db(ev_out,ot_out):+6.1f} dB"
          f"   gain: {db(ev_out,ot_out)-db(ev_in,ot_in):+5.1f} dB")

eng_in = float(np.sum(stems["engine_noise"].astype(np.float64)**2))
oth_in = sum(float(np.sum(stems[n].astype(np.float64)**2)) for n in names if n != "engine_noise")
eng_out = energy["engine_noise"][StreamKind.ENVIRONMENT]
oth_out = sum(energy[n][StreamKind.ENVIRONMENT] for n in names if n != "engine_noise")
print(f"  {'engine_noise':<16} vs rest  in mix: {db(eng_in,oth_in):+6.1f} dB"
      f"   in environment:    {db(eng_out,oth_out):+6.1f} dB"
      f"   gain: {db(eng_out,oth_out)-db(eng_in,oth_in):+5.1f} dB")

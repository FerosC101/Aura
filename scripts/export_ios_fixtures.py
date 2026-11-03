"""Export audio fixtures for the iOS engine.

The Swift port of the engine is validated against the same synthetic sources
the Python reference is benchmarked on, so the two can be compared number for
number. This writes:

* ``ios/AuraApp/Resources/Scenes/*.wav`` -- the demo scenes bundled in the app,
  plus a JSON manifest describing them.
* ``ios/AuraCore/Tests/AuraCoreTests/Fixtures/`` -- isolated sources, speaker
  clips and spoken words used by the Swift test suite, plus the Python
  engine's own results for the parity tests.

Everything is 16-bit mono WAV. Scenes and sources are rendered at 48 kHz, the
rate the app runs at on device.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aura.demo.scenes import all_scenes  # noqa: E402
from aura.demo.synth import GENERATORS, generate, speech, spoken_word  # noqa: E402

SR = 48000
SCENES = ROOT / "ios" / "AuraApp" / "Resources" / "Scenes"
FIX = ROOT / "ios" / "AuraCore" / "Tests" / "AuraCoreTests" / "Fixtures"


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main() -> None:
    manifest = []
    for scene in all_scenes():
        write_wav(SCENES / f"{scene.key}.wav", scene.render(SR), SR)
        manifest.append(scene.public())
    (SCENES / "scenes.json").write_text(json.dumps(manifest, indent=2))

    for key in GENERATORS:
        write_wav(FIX / "sources" / f"{key}.wav", generate(key, 3.0, SR), SR)

    voices = {
        "vince": dict(f0=112.0, formants=(500.0, 1480.0, 2500.0, 3400.0)),
        "maria": dict(f0=208.0, formants=(650.0, 1900.0, 2800.0, 3600.0)),
        "ben": dict(f0=148.0, formants=(430.0, 1100.0, 2300.0, 3300.0)),
        "ana": dict(f0=178.0, formants=(600.0, 1700.0, 2650.0, 3500.0)),
    }
    for name, cfg in voices.items():
        for c in range(2):
            clip = speech(2.5, SR, seed=40 + c * 13, syllable_rate=3.4 + 0.8 * c, **cfg)
            write_wav(FIX / "voices" / f"{name}_enrol{c}.wav", clip, SR)
        write_wav(FIX / "voices" / f"{name}_live.wav",
                  speech(5.0, SR, seed=999, syllable_rate=4.3, **cfg), SR)

    words = {
        "vince_enrol0": ("vince", 0.70, 118.0, 5, 0.04),
        "vince_enrol1": ("vince", 0.70, 118.0, 6, 0.04),
        "vince_same": ("vince", 0.70, 120.0, 9, 0.06),
        "vince_high": ("vince", 0.70, 200.0, 11, 0.05),
        "vince_slow": ("vince", 0.88, 116.0, 13, 0.05),
        "maria": ("maria", 0.70, 118.0, 7, 0.0),
        "hello": ("hello", 0.70, 118.0, 7, 0.0),
        "ben": ("ben", 0.62, 118.0, 8, 0.0),
        "ana": ("ana", 0.66, 118.0, 8, 0.0),
    }
    for key, (w, dur, f0, seed, jit) in words.items():
        write_wav(FIX / "words" / f"{key}.wav",
                  spoken_word(w, dur, SR, f0=f0, seed=seed, rate_jitter=jit), SR)

    total = sum(p.stat().st_size for p in (ROOT / "ios").rglob("*.wav"))
    print(f"wrote {len(manifest)} scenes, {len(GENERATORS)} sources, "
          f"{len(voices) * 3} voice clips, {len(words)} words  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

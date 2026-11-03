# Benchmarks

Every number in the README comes from a script in `scripts/`. All of them are
reproducible on a clean checkout with no model downloads.

Measured on an Apple M-series laptop, Python 3.12, 16 kHz / 20 ms blocks unless
stated otherwise.

```bash
python scripts/eval_classifier.py    # sound-event detection
python scripts/eval_separation.py    # stream routing + interference rejection
python scripts/eval_speaker.py       # Person Lock verification
python scripts/eval_name.py          # Name Awareness
python scripts/bench_end_to_end.py [scene]
```

## A note on the test signals

The sources are procedurally synthesized (`aura/demo/synth.py`) rather than
recorded. This is a real limitation and the numbers should be read with it in
mind — see [LIMITATIONS.md](LIMITATIONS.md). It buys three things that matter
for a reference implementation: ground truth is exact, the suite ships without
audio fixtures, and the demo is reproducible anywhere.

The synthesis is physically motivated rather than arbitrary. Speech is a
source-filter model with a jittered glottal pulse train through formant
resonators and a syllabic envelope. Sirens sweep their pitch; horns hold theirs.
Engines are a low harmonic stack with combustion noise. Words are trajectories
through vowel space, which is what makes keyword spotting testable at all.

---

## 1. Sound-event classification

`scripts/eval_classifier.py` runs each isolated source through the classifier
and reports the fraction of frames in which the true source was detected, plus
any other category firing in more than 30% of frames.

```
recall 13/13 sources   |   7 false-positive pairs above 30%
```

All 13 non-ambient categories are detected in at least 50% of frames; most
exceed 95%. Cost: **0.4 ms per frame**, about 2% of a 20 ms real-time budget.

Remaining confusions, and why they are tolerable:

| Confusion | Why |
|---|---|
| `warning_signal` / `bicycle_bell` → `ambient` | Both are duty-cycled; ambient is genuinely what is there during the gaps. |
| `warning_signal` / `bicycle_bell` → `vehicle_horn` | All are tonal events, and all three are safety-critical, so the policy outcome is identical. |
| `music` → `crowd_speech` / `engine_noise` | The synthetic chord organ is spectrally close to babble. A trained model is the fix. |

The detectors were tuned against measured feature distributions, not guessed.
Two findings drove the design:

- **Bin-wise spectral flatness is useless here** — it saturated near 0.00 for
  every source. Replaced with flatness over log-spaced band power *densities*,
  which restored a usable 0.00–0.43 spread.
- **Autocorrelation pitch tracking octave-errors on loud tones.** It reported
  *more* pitch variation for a steady horn (65 Hz) than for a sweeping siren
  (45 Hz) — the opposite of the truth. Replaced with dominant-spectral-peak
  tracking, which gives horn 50 Hz and siren 550 Hz. That single feature is
  what separates the two.

### 1a. The same classifier on real recordings

`scripts/eval_esc50.py` scores the classifier on [ESC-50][esc50]: 2000 field
recordings, 50 classes, five seconds each. Every number above is measured on
audio from `aura/demo/synth.py` — the generators the thresholds were tuned
against — so it reports self-consistency. This reports accuracy.

```bash
python scripts/eval_esc50.py --download      # ~600 MB, cached outside the repo
python scripts/eval_esc50.py --control       # methodology check, see below
```

**Clip recall: 70/360 = 19%**, against 13/13 on the synthesiser.

| ESC-50 class | AURA category | Recall | Top confusion |
|---|---|---:|---|
| `engine` | `engine_noise` | **32/40** | `general_traffic` 18/40 |
| `siren` | `emergency_siren` | **24/40** | `vehicle_horn` 13/40 |
| `hand_saw` | `mechanical_noise` | 9/40 | `general_traffic` 36/40 |
| `clock_alarm` | `alarm` | 2/40 | `emergency_siren` 22/40 |
| `chainsaw` | `mechanical_noise` | 2/40 | `music` 31/40 |
| `helicopter` | `mechanical_noise` | 1/40 | `engine_noise` 29/40 |
| `car_horn` | `vehicle_horn` | **0/40** | `engine_noise` 20/40 |
| `vacuum_cleaner` | `mechanical_noise` | 0/40 | `general_traffic` 29/40 |
| `washing_machine` | `mechanical_noise` | 0/40 | `music` 29/40 |

Only `engine_noise` and `emergency_siren` survive contact with real audio.
`car_horn` — safety-critical, and 100% recall on the synthesiser — is never
once detected in 40 real recordings.

**The false alarms are the more serious half.** On the 1880 clips that are not
a safety-critical sound, a safety category fired on **320 = 17%**:

| Real sound | Heard as | Rate |
|---|---|---:|
| `rooster` | `emergency_siren` | 38/40 |
| `crying_baby` | `emergency_siren` | 34/40 |
| `cat` | `emergency_siren` | 21/40 |
| `laughing` | `emergency_siren` | 17/40 |
| `door_wood_knock` | `approaching_vehicle` | 17/40 |
| `dog`, `hen` | `emergency_siren` | 15/40 each |

A device that misses the horn and then shouts *siren* at a crying baby is
failing in both directions at once. The pitch-peak feature that cleanly
separates horn from siren on synthetic tones (50 Hz vs 550 Hz, above) does not
survive real recordings, where animal and infant vocalisations sweep through
the same range a siren does.

**This is a classifier result, not a harness artefact.** `--control` pushes
`aura/demo/synth.py` through this script's own path — resample to 16 kHz, RMS
levelling, active-frame gating — and reproduces **13/13, levelling changing
nothing**. If that control ever drops below 13/13, fix the harness before
reading anything else on this page.

Methodology: clips are downmixed, resampled 44.1 → 16 kHz, and RMS-normalised
to −20 dBFS, because the DSP classifier reads absolute level in places and on
device an AGC does this job. Frames within 30 dB of the clip's 90th-percentile
frame are scored, so the silence ESC-50 pads clips with does not drown a
half-second horn. A clip counts as recalled when the true category fires in
≥50% of scored frames, matching `eval_classifier.py`.

**What ESC-50 cannot reach.** It contains no speech and no music by design, so
eight categories get no coverage here: `human_speech`, `crowd_speech`,
`announcement`, `music`, `general_traffic`, `approaching_vehicle`,
`bicycle_bell`, `warning_signal`. Person Lock and Name Awareness sit on the
speech path and are therefore still validated only against the synthesiser.
UrbanSound8K reaches `general_traffic` and street music; speech needs
LibriSpeech or AMI.

[esc50]: https://github.com/karolpiczak/ESC-50

### 1b. The ONNX backend on the same recordings

The DSP classifier's numbers above are what ships by default. The ONNX backend
hosts YAMNet behind the same interface:

```bash
mkdir -p ~/.cache/aura/models && cd ~/.cache/aura/models
curl -L -o yamnet.onnx https://huggingface.co/andrelgomes/yamnet-onnx/resolve/main/yamnet.onnx
curl -L -O https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv

python scripts/eval_esc50.py --backend onnx          # uses that path by default
aura live --classifier onnx --classifier-model ~/.cache/aura/models/yamnet.onnx
```

**ESC-50 fold 5, held out, identical clips through both backends:**

| | DSP (default) | YAMNet (ONNX) |
|---|---:|---:|
| Clip recall | 10/72 = **14%** | 36/72 = **50%** |
| Safety-critical false alarms | 74/376 = **19.7%** | 2/376 = **0.5%** |

Recall 3.6× better; false alerts 39× fewer. Per class, held out:

| ESC-50 class | AURA category | DSP | YAMNet |
|---|---|---:|---:|
| `clock_alarm` | `alarm` | 0/8 | **8/8** |
| `hand_saw` | `mechanical_noise` | 0/8 | **6/8** |
| `engine` | `engine_noise` | 6/8 | 5/8 |
| `siren` | `emergency_siren` | 3/8 | **5/8** |
| `vacuum_cleaner` | `mechanical_noise` | 0/8 | **5/8** |
| `chainsaw` | `mechanical_noise` | 1/8 | **4/8** |
| `car_horn` | `vehicle_horn` | 0/8 | **3/8** |
| `helicopter`, `washing_machine` | `mechanical_noise` | 0/8 | 0/8 |

**Development discipline.** Choosing which AudioSet names map onto the taxonomy
is tuning, and tuning against the set you then report is how the synthetic
numbers in §1 came to mean nothing. So the alias work was done on folds 1–4 and
fold 5 was not looked at until the mapping was frozen. Folds 1–4 score
145/288 = 50% recall at 0.5% false alarms; fold 5 scores 50% and 0.5%. The
agreement is the evidence that the mapping generalises rather than memorises.
Keep fold 5 held out for the next round too.

**What is still weak, honestly:**

- **`vehicle_horn` is the worst safety-critical class at 3/8**, usually lost to
  `engine_noise` — a horn in traffic is heard over an engine, and YAMNet
  reports the louder one. For a device whose job is to let a horn through,
  this is the number to improve next.
- `emergency_siren` also fires `alarm` on 4/8 siren clips. Both are
  safety-critical and both resolve to "keep audible", so the policy outcome is
  unchanged; it costs a correct label, not a correct action.
- `washing_machine` is 0/8 because **AudioSet has no washing-machine class** —
  no mapping can fix it, and YAMNet reasonably reports `engine_noise`.
  `helicopter` is 0/8 for the same reason in reverse: AudioSet has
  `Helicopter`, now mapped to `mechanical_noise`, but the engine-family labels
  score higher on these recordings. Both are taxonomy-boundary disputes rather
  than detection failures.

**Cost, measured in the engine at 48 kHz with 20 ms hops:**

| Backend | mean | p95 | max | over 20 ms budget |
|---|---:|---:|---:|---:|
| `dsp` | 1.39 ms | 1.69 ms | 3.80 ms | 0 / 299 |
| `onnx:yamnet` | **1.04 ms** | 3.12 ms | 3.46 ms | 0 / 299 |

The neural backend is *cheaper on average* than the heuristic one, because
inference runs on a rolling window every third hop while the DSP classifier
recomputes features on every hop. The p95 is the inference hop. Neither comes
close to the deadline on an M-series laptop — which is not the same claim as
on an iPhone, where none of this is yet ported (see
[LIMITATIONS.md](LIMITATIONS.md)).

**The synthesiser, from the other side.** `--backend onnx --control` scores
YAMNet on `aura/demo/synth.py` and gets **2/13**. A model trained on real
recordings cannot read the synthesiser, which is the §1a finding seen in a
mirror: the fixture and the real world are far apart, and a system tuned on
either alone will not survive the other.

## 2. Source separation

`scripts/eval_separation.py`. Masks are estimated from a five-source mixture and
then applied to the known stems, which is the standard way to attribute a
mask-based separator.

**Reconstruction.** With unity masks the four streams sum back to the input at
**153 dB SNR**. This is asserted by a test; without it, bypass would not be
transparent and every gain measurement would be against a distorted baseline.

**Routing** (percentage of each isolated source's energy per stream):

| Source | primary | secondary | transient | environment |
|---|---:|---:|---:|---:|
| `human_speech` | **86.4%** | 0.4% | 0.7% | 12.5% |
| `crowd_speech` | 49.0% | 5.5% | 2.3% | 43.2% |
| `engine_noise` | 0.2% | 1.1% | 0.0% | **98.7%** |
| `general_traffic` | 0.4% | 0.0% | 0.2% | **99.4%** |
| `emergency_siren` | 0.0% | 0.0% | **100.0%** | 0.0% |

**Interference rejection** on a mixture:

| | In the mix | In its stream | Gain |
|---|---:|---:|---:|
| Speech vs engine + crowd | −6.5 dB | +7.1 dB | **+13.6 dB** |
| Horn vs rest (while sounding) | +0.5 dB | +20.2 dB | **+19.7 dB** |
| Siren vs rest (while sounding) | +2.3 dB | +31.7 dB | **+29.4 dB** |
| Engine vs rest | −0.5 dB | +9.1 dB | **+9.6 dB** |

Transient events are scored only over the window in which they sound. Measuring
the horn across the whole clip counted the siren — which occurs at a different
time — as interference, and understated it by 20 dB.

**The voicing gate.** An SNR-only speech mask forced a bad trade: a fast noise
tracker captured only 44% of speech, and slowing it pulled 79% of road noise
*into* the speech stream. Gating the mask on voicing broke the trade — speech
rose to 86% while traffic leakage fell to 0.6%.

## 3. Person Lock

`scripts/eval_speaker.py`. Four synthetic speakers, two enrolment clips each.

```
same-speaker  mean=1.0000  min=1.0000
diff-speaker  mean=0.9529  max=0.9868

speaking    locked  similarity   margin   expected
Vince         True      0.9999     2.23   lock    PASS
Maria        False      0.9389    -0.60   reject  PASS
Ben          False      0.9647     0.40   reject  PASS
Ana          False      0.9752     0.38   reject  PASS
4/4 correct
```

The margin is the cohort-normalised score, not raw similarity. A fixed
threshold does not survive contact with real microphones: absolute scores drift
with room, distance and backend, and this embedder's impostor scores cluster at
0.95 while an ECAPA network's cluster near 0.0. Each backend therefore declares
its own measured `impostor_prior`, blended with the live cohort by shrinkage.

The threshold of 1.5 sits in the gap between the enrolled speaker at 2.2 and the
nearest impostor at 0.4.

## 4. Name Awareness

`scripts/eval_name.py`. One name enrolled from two examples.

```
target 0.186-0.212   other 0.365-0.484   margin +0.153
9/9 correct   detect(): 1.0 ms/call
```

Scoped honestly: this matches a name spoken as **its own short utterance** —
"Vince?", "Vince, over here" — which is how a name is normally used to get
someone's attention. Spotting a name buried mid-sentence in continuous speech
was measured and does **not** work reliably with template matching (the margin
collapsed to +0.001). That needs the `faster-whisper` backend.

## 5. End to end

`scripts/bench_end_to_end.py`. Full engine, five-source scene, Commute mode.

**What it heard** — 5/5 sources handled exactly as the scene scripted:

```
category                frames    conf    action     gain   expected
human_speech               186    0.65  preserve     +0.0   preserve OK
vehicle_horn                80    0.65  preserve     +0.0   preserve OK
emergency_siren            190    0.81  preserve     +0.0   preserve OK
engine_noise               550    0.91  suppress    -14.9   suppress OK
crowd_speech                48    0.61    reduce     -5.0   reduce   OK
```

**What it did to the audio:**

| Band | AURA | ANC simulation |
|---|---:|---:|
| Engine / rumble, 20–300 Hz | **−8.7 dB** | −18.4 dB |
| Speech body, 300 Hz–1 kHz | −1.3 dB | −10.4 dB |
| Speech clarity, 1–4 kHz | **+0.0 dB** | −7.8 dB |
| Horn / siren, 400–3500 Hz | **−0.6 dB** | −9.0 dB |

**Siren audibility** while it sounds (500–3000 Hz):

| | Change |
|---|---:|
| Raw | +0.0 dB |
| ANC simulation | **−8.0 dB** |
| AURA | **+0.6 dB** |

**Latency**, 20 ms blocks:

| Stage | Mean | p95 |
|---|---:|---:|
| Preprocess | 0.004 ms | 0.005 ms |
| Separation | 0.127 ms | 0.132 ms |
| Classification | 1.096 ms | 1.131 ms |
| Speaker hand-off | 0.005 ms | 0.005 ms |
| Policy | 0.016 ms | 0.019 ms |
| Mixing | 0.066 ms | 0.074 ms |
| **Total processing** | **1.31 ms** | — |
| Processing load | **6.6%** of real time | |
| Algorithmic delay (separator) | 20.0 ms | |
| Dropouts over 600 frames | **0** | |

Live, through CoreAudio, the measured end-to-end figure is **~45 ms**:
input buffer + 20 ms block + 1.3 ms processing + 20 ms overlap-add delay +
output buffer. Nothing is excluded to make the number look better.

## 6. Demo scenes

All five scenes run through the full engine, scored against their declared
expectations:

| Scene | Mode | As scripted |
|---|---|---:|
| PSC Demo | Commute | 5/5 |
| Jeepney / Bus | Commute | 5/6 |
| University Classroom | Focus | 4/4 |
| Urban Walking | Walk | 5/6 |
| Café Conversation | Social | 3/3 |

**22/24 overall.** The two misses are acoustic masking, documented in
[LIMITATIONS.md](LIMITATIONS.md): road noise beneath a louder diesel engine, and
a bicycle bell under heavy traffic.

## 7. Test suite

```
192 passed in 16.79s
```

Including an exhaustive safety matrix — every safety-critical category against
every listening mode, with a hostile user override — and tests that assert
measured properties of the output signal rather than labels.

# Limitations

What does not work yet, why, and what would fix it. Written plainly, because a
prototype that overstates itself is harder to build on than one that does not.

---

## 1. The classifier is heuristic, not learned

The default `SoundClassifier` combines DSP features through hand-tuned fuzzy
evidence rules. It reaches 13/13 recall on isolated synthetic sources at 0.4 ms
per frame, and it always runs — no download, no PyTorch.

It will not match a trained AudioSet model on real-world audio. Hand-tuned
thresholds are brittle in exactly the way learned features are not: unusual
rooms, unusual microphones and sources outside the taxonomy will degrade it.

**This is now measured, not predicted.** `scripts/eval_esc50.py` scores the
classifier on 2000 real ESC-50 recordings: **19% clip recall**, against 13/13
on the synthesiser. `car_horn` is detected in 0 of 40 real clips. Worse, a
safety-critical category fires on **17% of audio that contains no such sound** —
`rooster` reads as `emergency_siren` in 38 of 40 clips, `crying_baby` in 34.
See [BENCHMARKS.md §1a](BENCHMARKS.md) for the full table and the methodology
control. Treat the DSP classifier as a working reference implementation and a
demo engine, **not as something to put in front of a user who is crossing a
road.** The ONNX backend below is not an optimisation; on real audio it is the
difference between working and not.

**Fix, now wired up and measured.** `--classifier onnx --classifier-model
<yamnet.onnx>` runs YAMNet behind the same interface. On held-out ESC-50 it
takes recall from 14% to 50% and safety-critical false alarms from 19.7% to
0.5%, and it is *cheaper* per hop on average than the DSP classifier, because
inference runs every third hop on a rolling window. Setup and full numbers:
[BENCHMARKS.md §1b](BENCHMARKS.md).

Two caveats before calling this solved:

- **50% recall is better, not good.** `vehicle_horn` — the class most worth
  getting right — still only reaches 3/8, usually lost to `engine_noise`.
- **None of this is on the phone.** `AuraCore` has no ONNX or Core ML backend;
  the iOS app runs the DSP classifier, so the 14% figure is what ships on
  device today. Porting this is the single highest-value piece of work
  outstanding.

## 2. Validation uses synthetic audio

Every benchmark uses procedurally generated sources. That gives exact ground
truth and a suite that ships without fixtures, but it is not the same as
validation on real recordings, and the numbers should not be read as if it were.

Specific ways the fixture flatters the system:

- **No consonants.** The speech model is voiced-only, so fricatives, plosives
  and the coarticulation that makes real speech hard are all absent.
- **No reverberation, no room.** No early reflections, no late tail.
- **No microphone colouration**, no handling noise, no wind.
- **Clean statistics.** Real sources are messier than their models.

There is one place the fixture works *against* the system: with no consonants,
synthetic words are far less distinguishable than real ones, so the Name
Awareness numbers understate what a real recording would achieve.

**Partly addressed.** `scripts/eval_esc50.py` now evaluates the classifier on
ESC-50 and the result is in [BENCHMARKS.md §1a](BENCHMARKS.md): the synthetic
numbers did not survive, in the direction and roughly the magnitude this
section predicted. The separator, Person Lock, Name Awareness and the
end-to-end scenes are all still synthetic-only.

**Remaining fix:** UrbanSound8K for `general_traffic` and street music;
LibriSpeech or AMI for the speech path, which ESC-50 cannot reach because it
contains no speech; and real recordings from a jeepney, a classroom and a
street for the end-to-end scenes.

## 3. Separation prioritises latency over quality

The brief asked for exactly this trade. The spectral separator adds one hop
(20 ms) and 0.13 ms of compute, and achieves +13.6 dB speech SIR gain.

A neural separator (Conv-TasNet, SepFormer, Demucs) would do considerably
better on overlapping speech, at tens of milliseconds and orders of magnitude
more compute.

Known weaknesses of the current approach:

- **Babble splits across streams.** Crowd speech lands ~49% in the primary
  stream and ~43% in environment. The pitch-comb split cannot cleanly separate
  "one talker" from "many". The mixer compensates by blending speech decisions
  across the speech path, but this is a workaround for a separation limitation,
  not a solution to it.
- **Musical noise** at aggressive suppression. Temporal mask smoothing reduces
  it but does not eliminate it.
- **Only ~21% of a horn's energy** reaches the transient stream in a dense mix,
  though what does reach it is clean (+19.7 dB). The rest stays in environment
  and is attenuated with it.

## 4. Two demo scenes have masked sources

- **Jeepney: `general_traffic` not detected.** Road noise at −11 dB sits under a
  diesel engine at −10 dB with heavily overlapping spectra. The engine wins.
  Both are suppression targets in Commute mode, so the audible outcome is
  correct even though the label is not.
- **Urban Walk: `bicycle_bell` not detected.** A short, quiet, decaying
  high-frequency transient under continuous traffic. Genuinely masked.

Analysing transient detectors on the separator's isolated transient stream was
tried as a fix. It made things worse — residual speech harmonics and music
partials survive into that stream, and the siren detector fires on them, because
a sweeping partial looks like a sweeping siren once its context is gone. The
change was reverted and the reasoning recorded in the code.

## 5. Person Lock is weaker than a trained network

MFCC-statistics embeddings separate a handful of enrolled speakers (4/4 in the
benchmark) but will not match ECAPA-TDNN on:

- speakers with similar vocal tracts,
- a large enrolled set,
- channel mismatch between enrolment and use,
- noisy or reverberant conditions.

Cohort normalisation makes the decision robust to *score drift*, not to a
genuinely weak embedding.

**Fix:** `pip install -e ".[speaker]"` and `EngineConfig(speaker_backend="ecapa")`.

## 6. Name Awareness only spots isolated utterances

Template matching with DTW works well when the name is said on its own
(9/9, margin +0.153). Finding it mid-sentence was measured and does not work:
target and non-target distances overlapped with a +0.001 margin.

Several approaches were tried — subsequence DTW with free endpoints, span
constraints, sliding matched-length windows, local cepstral mean normalisation.
Free endpoints let the cheapest path skim a few similar frames; pinned endpoints
on arbitrary boundaries were too harsh. None produced a usable margin.

The threshold is therefore set for the isolated-utterance case, which is how a
name is normally used to get someone's attention.

**Fix:** `pip install -e ".[asr]"` and use `WhisperKeywordSpotter`. Note that
this transcribes speech locally, a meaningfully larger privacy step, which is
why it is a separate opt-in and not the default.

## 7. Automatic context detection is experimental

Off by default. The acoustic evidence rules are hand-weighted and were validated
only on synthetic scenes. Motion and place hints must be supplied by a client —
AURA never reaches for GPS itself — and require explicit opt-in.

Context never switches modes on its own; it only suggests.

**Fix:** a learned classifier over accumulated acoustic features, plus real
usage data.

## 8. Preference learning is a weighted average

Per-(mode, category) bias updated by EMA, bounded to ±6 dB, decaying with a
21-day half-life, and forbidden from ever learning to suppress a safety sound.

It cannot represent context beyond the mode, interactions between categories, or
time-of-day patterns. It is deliberately shaped as a bandit interface
(`observe_adjustment` is a reward keyed by context and arm; `bias_for` is the
current policy) so a contextual bandit can replace it without touching anything
else.

## 9. Mono only

The engine downmixes to mono. All spatial information is discarded, which means
no directional awareness — AURA can tell you a vehicle is approaching but not
from which side. For a walking-safety product that is a significant gap.

**Fix:** binaural processing with interaural time and level differences
preserved through the mixer. This is the single largest missing capability.

## 10. Biquad shaping is not linear phase

Zero-latency spectral shaping uses cascaded peaking filters, which introduce
phase distortion around the crossovers. Inaudible at moderate gains for a
monitoring path; at extreme settings it can colour transients.

A linear-phase FIR would avoid it at the cost of latency proportional to filter
length. For a live path the trade favours phase distortion, which is the choice
real audio products make.

## 11. Feedback with loudspeakers

`aura live` processing a microphone to speakers will howl. There is no echo
cancellation. `aura doctor` warns about this. Headphones are required.

**Fix:** an AEC stage before analysis.

## 12. Not validated on target hardware

Everything here was measured on a laptop. Latency, CPU load and thermal
behaviour on phones, earbuds or embedded DSPs are unknown. The Python engine is
a reference implementation, not a shipping runtime — production would need
JUCE/C++ with CoreAudio, AAudio/Oboe or a vendor DSP SDK.

---

## What is solid

For balance, the parts that are measured, tested and hold up:

- **The safety guarantee.** Exhaustively tested across every safety category ×
  every mode, including hostile overrides. A siren cannot be silenced.
- **Reconstruction.** Streams sum back to the input at 153 dB, so bypass is
  genuinely transparent and every gain is measured against a clean baseline.
- **Selective transformation.** −8.7 dB on the engine band while the horn/siren
  band moves −0.6 dB, measured on the output signal rather than asserted.
- **Real-time headroom.** 1.3 ms per 20 ms block, zero dropouts.
- **Privacy.** Enforced structurally — no field holds audio, and a test asserts
  what lands on disk.

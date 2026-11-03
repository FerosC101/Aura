# AURA — AI-Powered Selective Awareness

### Hear what matters. Silence what doesn't.

Noise cancellation removes sound by *frequency*. It cannot tell a siren from an
engine, so it takes both. AURA works out what each sound **is** and decides
whether you should hear it.

This repository contains:

- **The AURA app for iPhone** ([`ios/`](ios/README.md)): the engine running
  on-device in Swift, with the branded SwiftUI interface.
- **The reference engine** (`src/aura`): the Python implementation that the
  iPhone engine is ported from and checked against, plus the benchmark harness
  and a web control surface.

Both run on ordinary microphones and headphones, with no proprietary DSP or firmware.

```
  Raw                 ANC                  AURA
  ────────────        ────────────         ────────────
  engine   ▓▓▓▓▓▓     engine   ▓▓          engine   ▓
  crowd    ▓▓▓▓       crowd    ▓▓          crowd    ▓
  speaker  ▓▓▓▓▓      speaker  ▓▓          speaker  ▓▓▓▓▓
  horn     ▓▓▓▓▓▓     horn     ▓▓          horn     ▓▓▓▓▓▓
  siren    ▓▓▓▓▓▓     siren    ▓▓          siren    ▓▓▓▓▓▓
                      ↑ everything quieter  ↑ only what doesn't matter
```

---

## What it actually does

Measured end to end on the PSC demo scene (`scripts/bench_end_to_end.py`):

| Measurement | AURA | ANC simulation |
|---|---:|---:|
| Engine / rumble band (20–300 Hz) | **−8.7 dB** | −18.4 dB |
| Speech clarity band (1–4 kHz) | **+0.0 dB** | −7.8 dB |
| Horn / siren band (400–3500 Hz) | **−0.6 dB** | −9.0 dB |
| **Siren audibility while it sounds** | **+0.6 dB** | −8.0 dB |

That last row is the product. ANC makes the siren 8 dB harder to hear. AURA
leaves it exactly where it was, while removing the engine underneath it.

| Performance | |
|---|---:|
| Processing per 20 ms block | 1.3 ms (6.6% of real time) |
| Measured end-to-end latency, live | ~45 ms |
| Dropouts over 600 frames | 0 |

Latency is reported honestly: it includes the CoreAudio input and output
buffers and the separator's algorithmic delay, not just the compute time.

---

## Quick start

Requires Python 3.10–3.13 and PortAudio (`brew install portaudio` on macOS).

```bash
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && pip install -e ".[dev]"

aura doctor                    # check audio devices and backends
aura demo psc_demo             # play a scripted scene through the full engine
aura live --mode commute       # process your microphone  ← use headphones
```

**Use headphones for `aura live`.** Processing a microphone to loudspeakers
feeds back.

### The full interface

```bash
cd ui && npm install && npm run build && cd ..
aura serve                     # http://127.0.0.1:8000
```

The browser is a control surface only — audio never passes through it. Routing
live audio through WebAudio would add tens of milliseconds to a pipeline whose
entire processing budget is about one millisecond, so the Python engine owns
the device directly and the UI receives events over a WebSocket.

---

## The pipeline

```
microphone → buffering → preprocessing → separation → classification
          → speaker identification → relevance scoring → gain assignment
          → reconstruction → headphones
```

Two threads, split by deadline:

- **The audio callback** runs separation, classification, policy and mixing —
  1.3 ms against a 20 ms budget.
- **A semantic worker** runs Person Lock and Name Awareness, which need a
  1–1.5 s window and would blow the audio deadline. It publishes conclusions
  the audio thread reads; a stale-by-one-window answer is harmless, an audio
  dropout is not.

### Why separation matters

To raise a voice while lowering the bus it is sitting in, the two must first
become independent signals. The separator splits the input into four streams
that sum back to it (verified to 153 dB):

| Stream | What lands there | Measured |
|---|---|---:|
| `primary_speech` | The dominant voice | 86% of speech |
| `secondary_speech` | Competing talkers, babble | — |
| `transient` | Horns, sirens, alarms, bells | 100% of a siren |
| `environment` | Engine, road, room tone | 99% of engine noise |

Separation quality, measured as interference rejection on a mixture:

| | Gain |
|---|---:|
| Speech vs engine + crowd | **+13.6 dB** |
| Horn in the transient stream | **+19.7 dB** |
| Siren in the transient stream | **+29.4 dB** |
| Engine in the environment stream | **+9.6 dB** |

### Why the mixer is not an equaliser

Engine rumble and a car horn share the environment stream, so one scalar gain
cannot express "drop the rumble, keep the horn". Each category's decision is
projected onto six shaping bands in proportion to its spectral footprint, and
applied with cascaded biquads — **zero added latency**, unlike a second STFT
stage. Band interaction is compensated, so a request for −18 dB measures −18 dB.

---

## Listening modes

| Mode | Keeps | Removes |
|---|---|---|
| **Focus** | Your speaker, your name, alarms, emergencies | Surrounding conversation, traffic, café noise |
| **Commute** | Conductor, announcements, horns, sirens | Engine rumble, road noise, crowd |
| **Walk** | Approaching vehicles, horns, sirens, bells | Repetitive city noise — gently, never removed |
| **Social** | The person in front of you | Competing conversation, background |
| **Custom** | Whatever you set | Whatever you set |

Modes are declarative tables, not code. A new mode is a dictionary.

Walk mode suppresses at 0.7× the normal depth. Over-suppression while walking
is a safety problem, not a matter of taste.

---

## The safety layer

`SafetyGuard` is the last component to touch any decision and the only one that
cannot be overridden — not by a mode, not by the mixer, not by learned
preferences. Three rules:

1. **Floor.** A safety-critical sound is never attenuated below −3 dB.
2. **Lift.** A quiet siren in a loud scene is boosted to clear the mix.
3. **Doubt favours audibility.** When confidence is uncertain (0.30–0.72), a
   *possibly* safety-critical sound is preserved rather than suppressed. A
   false alarm costs a moment of unwanted noise; a missed siren costs more.

This is enforced by a test matrix over every safety category × every mode,
including a hostile user override dragging a siren to −40 dB.

The preference learner is separately forbidden from ever learning to suppress a
safety sound, however often it is asked.

---

## Person Lock

Record a few seconds of a trusted voice; AURA generates an embedding, **discards
the audio**, and afterwards lifts that person above everyone else.

Matching uses cohort normalisation rather than a fixed threshold: absolute
similarity scores drift with room, distance and backend, so the live score is
compared against a running estimate of what *non-matching* speech scores in the
same conditions. Benchmarked at 4/4 correct with the enrolled speaker at margin
2.2 and the nearest impostor at 0.4.

---

## Privacy

Local-first, and enforced structurally rather than by policy:

- **Never stored:** raw or encoded audio, transcripts, logs of what was heard.
- **Stored:** preferences, mixer positions, learned adjustments, speaker
  embeddings (only with explicit consent), trigger words.
- The default Name Awareness backend matches sound patterns and **never
  transcribes**. The ASR backend is a separate, explicit opt-in.
- Cloud inference is not implemented. There is no code path that uploads audio.

`SpeakerProfile` has no field that can hold a waveform, and a test asserts that
what lands on disk contains no audio arrays.

The Privacy Center in the UI shows what is processed, what is stored, which
profiles exist, and offers one-click deletion of everything.

---

## Demo Mode

A live demo cannot rely on the room containing an ambulance.

```bash
aura demo list
aura demo psc_demo       # engine, crowd, a speaker, a horn, a siren
aura demo jeepney        # EDSA rush hour: conductor + announcements + horns
aura demo classroom      # lecturer over fans and hallway chatter
aura demo urban_walk     # crossing a busy street
aura demo cafe           # one voice in a loud room
```

Each scene declares what AURA *should* do with every source, and the test suite
asserts it — so a scene that stops behaving as advertised fails a test rather
than surprising a judge. In the UI, each scene can be toggled between **Raw**,
**Noise Cancellation** and **AURA** with the same source material.

---

## Backends

Every AI component sits behind an interface. The defaults are dependency-free,
so the engine runs on a clean machine with no model download:

| Component | Default | Optional |
|---|---|---|
| `SoundClassifier` | DSP evidence, 14 categories, 0.4 ms | YAMNet / PANNs / AST via ONNX (`[onnx]`) — **3.6× the recall on real recordings**, see [BENCHMARKS §1b](docs/BENCHMARKS.md) |
| `SourceSeparator` | Spectral WOLA, 4 streams | Conv-TasNet / Demucs (`[torch]`) |
| `SpeakerIdentifier` | MFCC statistics | ECAPA-TDNN (`[speaker]`) |
| Keyword spotting | DTW templates, no transcription | faster-whisper (`[asr]`) |

An unavailable backend falls back to the default **and says so** in the UI
status panel. A silent fallback would make it look like a neural model was
running when it was not.

Adding a sound category means registering one `SoundCategory`; the policy
engine, mixer, safety guard and UI all pick it up automatically.

---

## Development

```bash
pytest                              # 192 tests
python scripts/eval_classifier.py   # per-source recall and false positives
python scripts/eval_separation.py   # stream routing and interference rejection
python scripts/eval_speaker.py      # Person Lock verification
python scripts/eval_name.py         # Name Awareness
python scripts/bench_end_to_end.py  # the full pipeline, measured
ruff check src tests && mypy src
```

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components, threading, data flow
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — how every number here was measured
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — what does not work yet, and why

---

## Where this goes

The engine is deliberately hardware-independent. The long-term target is a
Semantic Hearing SDK running inside earbuds, headphones, smart glasses, helmets,
workplace headsets and accessibility devices — with AURA as the intelligence
layer between people and the acoustic world.

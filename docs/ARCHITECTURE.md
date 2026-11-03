# Architecture

## Why a native engine with a web UI

The brief asked for the lowest practical audio latency and explicitly warned
against forcing the application into a browser. So audio never enters one.

The Python engine opens a single full-duplex PortAudio stream and does all its
work inside the driver callback. The browser is a control surface that receives
acoustic-scene events over a WebSocket and sends commands back over REST. The
two are fully decoupled: closing the UI does not disturb the audio path, and a
stalled WebSocket cannot reach back and stall the callback.

The alternative — WebAudio capture, a round trip to the server, WebAudio
playback — would add tens of milliseconds of buffering and resampling to a
pipeline whose entire processing cost is 1.3 ms. That is the wrong trade by an
order of magnitude.

## Data flow

```
 ┌──────────────────────── audio callback thread ────────────────────────┐
 │                                                                       │
 │  AudioCaptureEngine ──► AudioPreprocessor ──► SourceSeparator         │
 │   (PortAudio duplex)     (DC, downmix)        (WOLA, 4 streams)       │
 │                                                     │                 │
 │                                                     ▼                 │
 │                                              SoundClassifier          │
 │                                              (multi-label events)     │
 │                                                     │                 │
 │                                                     ▼                 │
 │                                          AcousticPolicyEngine         │
 │                                          (mode, prefs, lock, ctx)     │
 │                                                     │                 │
 │                                                     ▼                 │
 │                                              SafetyGuard              │
 │                                          (floor, lift, doubt)         │
 │                                                     │                 │
 │                                                     ▼                 │
 │                    AudioMixer ──► AudioOutputEngine ──► headphones    │
 │           (stream gains + band shaping + limiter)                     │
 │                                                                       │
 │  MetricsMonitor times every stage. AudioBufferManager keeps rings.    │
 └───────────────────────────────────────────────────────────────────────┘
            │  speech stream handed off                ▲ results read
            ▼                                          │
 ┌─────────────────── semantic worker thread ──────────────────────┐
 │  SpeakerIdentifier ──► PersonLockManager   NameAwareness        │
 │  (1.5 s windows)       (cohort-normalised) (1.0 s windows)      │
 └─────────────────────────────────────────────────────────────────┘
            │
            ▼  scene published at a fixed rate
 ┌─────────────────── asyncio / FastAPI ───────────────────────────┐
 │  ConnectionManager ──► WebSocket clients                        │
 │  REST: modes, mixer, profile, speakers, privacy, demo           │
 └─────────────────────────────────────────────────────────────────┘
```

## The threading split

The division is by **deadline**, not by module.

Anything with a hard 20 ms deadline and bounded cost runs on the audio thread:
separation (0.13 ms), classification (1.1 ms), policy (0.02 ms), mixing
(0.07 ms). Handing these to a worker would add at least one more block of
buffering — a 50% latency increase — for no benefit.

Anything that needs a long window runs on the semantic worker. A speaker
embedding needs 1.5 s of speech and costs several milliseconds; a keyword match
needs 1 s and costs ~1 ms but spikes. Either would cause an audible dropout on
the audio thread. These are also inherently slow-rate decisions — who is
speaking does not change every 20 ms — so a stale-by-one-window answer is
harmless.

The worker publishes into plain attributes (`_lock_result`, `_name_active`).
Assignment of an immutable object is atomic under the GIL, so the audio thread
reads them without a lock. Audio blocks are handed to the worker through a
bounded list; if the worker falls behind, the *oldest* audio is dropped, because
a backlog is latency rather than safety.

## Components

| Component | File | Responsibility |
|---|---|---|
| `AudioCaptureEngine` / `AudioOutputEngine` | `audio/capture.py` | Duplex, split and offline transports |
| `AudioBufferManager` | `audio/buffer.py` | Ring buffers, peak-decimated waveform for the UI |
| `AudioPreprocessor` | `audio/preprocess.py` | Downmix, DC removal; AGC only on the analysis copy |
| `SourceSeparator` | `separation/` | Splits into 4 streams that sum back to the input |
| `SoundClassifier` | `classification/` | Multi-label, stream-tagged sound events |
| `SpeakerIdentifier` | `speaker/` | Voice embeddings for Person Lock |
| `ContextEngine` | `context/engine.py` | Situation detection; manual always wins |
| `SafetyGuard` | `policy/safety.py` | Minimum audibility — the only non-overridable layer |
| `AcousticPolicyEngine` | `policy/engine.py` | Detections + context → gain decisions |
| `AudioMixer` | `audio/mixer.py` | Applies gains; stream scalars + band shaping + limiter |
| `AcousticProfile` | `profile/acoustic_profile.py` | The only thing written to disk |
| `PreferenceLearningEngine` | `profile/learning.py` | Learns from corrections, bounded and decaying |
| `MetricsMonitor` | `metrics/monitor.py` | Per-stage latency, load, dropouts |

Components communicate only through the value types in `types.py`
(`AudioFrame`, `SeparatedAudio`, `SoundEvent`, `GainDecision`, `AcousticScene`),
which is what makes any of them replaceable.

## The policy stack

Each layer can only be overridden by the one beneath it:

1. **Mode intent** — the declarative table for the active mode.
2. **User override** — an explicit mixer position beats the mode.
3. **Learned preference** — a bounded nudge from repeated corrections.
4. **Person Lock** — promote the locked voice, duck what competes with it.
5. **Name Awareness** — a temporary lift when the user's name is heard.
6. **Confidence scaling** — act tentatively on tentative detections.
7. **SafetyGuard** — absolute, and last.

The ordering is the design. Learned behaviour can shape the mix; it can never
talk the system out of a safety floor.

**Confidence scaling** deserves a note: a detection at 40% confidence does not
get 40% of a label, it gets a gain scaled towards 0 dB. An unsure "that might be
an engine" trims a few dB rather than gouging 16 dB out of what may be a voice.

## Separation, in detail

Weighted overlap-add with square-root Hann windows at 50% overlap, which
satisfies the COLA condition — with unity masks the output is a delayed copy of
the input to 153 dB. That matters because "AURA off" and "AURA on, everything
preserved" must sound identical, or every gain is measured against a distorted
baseline.

There is no lookahead: the analysis window only reaches backwards, so the added
algorithmic delay is exactly one hop (20 ms), and it is reported in the latency
figure rather than hidden.

Order of operations, which was arrived at empirically:

1. **Transients first**, from the full spectrum. Running the speech mask first
   routed sirens into the speech stream — a siren is a loud tone well above the
   noise floor, exactly what a Wiener mask is built to keep — and a siren in
   the speech stream is attenuated the moment the user reduces speech.
2. **Voiced harmonics are shielded** from transient extraction, so speech
   partials are not stripped into the protected stream.
3. **The speech mask** is a decision-directed Wiener gain over a tracked noise
   floor, **gated by voicing**. The gate is essential: an SNR mask cannot tell a
   voice from a lorry, so relaxing the noise tracker enough to keep speech also
   pulls road noise into the speech stream. With the gate, speech capture went
   from 44% to 86% while traffic leakage fell to 0.6%.
4. **The remaining speech splits by pitch comb** into primary and secondary.

## A note on labels versus streams

The classifier's labels and the separator's streams do not correspond exactly,
and the mixer is written to know this. The pitch-comb split puts roughly half of
a crowd into the *primary* stream, so applying only primary-labelled decisions
there would leave most of a room's conversation untouched while the UI claimed
it was suppressed.

The mixer therefore blends all speech decisions by confidence across the speech
path — unless one has been explicitly promoted. Person Lock raises its speaker
to HIGH priority, and that decision takes the primary stream outright, which is
what keeps "lift this voice, duck the rest" working.

## Extending the taxonomy

```python
from aura.classification.taxonomy import SoundCategory, register_category
from aura.types import StreamKind

register_category(SoundCategory(
    key="dog_bark", label="Dog Bark", description="A dog barking nearby.",
    stream=StreamKind.TRANSIENT, band_hz=(300.0, 5000.0),
    aliases=("Bark", "Dog", "Animal"),
))
```

Nothing else changes. The policy engine falls back to each mode's default
intent, the mixer derives band weights from `band_hz`, the safety guard reads
`safety_critical`, and the UI renders it. `aliases` let a neural backend's class
names map on without the policy engine learning about AudioSet.

## Swapping a backend

```python
EngineConfig(classifier_backend="onnx", classifier_model_path="yamnet.onnx")
EngineConfig(speaker_backend="ecapa")
EngineConfig(separator_backend="passthrough")   # latency baseline
```

`classification/registry.py` builds each component and records a note when a
requested backend is unavailable. The note is surfaced in the UI, because a
silent fallback would misrepresent what is running.

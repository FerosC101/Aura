# Presentation mockups

Device-sized frames of the AURA app for slides and press material.
Regenerate with `ios/tools/mockups.sh` — never edit these by hand, or the deck
and the app drift apart.

- **Device:** iPhone 17 Pro simulator, 1206 × 2622 px (3× native)
- **Status bar:** pinned to 9:41, full signal, charged, so every frame matches
- **Engine:** real. Each frame is the actual engine running a scripted scene,
  not a design comp — the decisions shown are the ones AURA made.

| File | Screen | Notes |
|---|---|---|
| `01-onboarding-0…5.png` | Onboarding | Six steps |
| `02-home-paused.png` | Home, idle | Care mode selected, engine off |
| `03-home-listening.png` | Home, live | Clinic scene running |
| `04-voice-focus.png` | Voice Focus | Three enrolled voices, one held |
| `05-modes.png` | Modes | Care, Focus, Commute, Walk, Social, Custom |
| `06-insights.png` | Insights | Live scene analysis |
| `07-demo-clinic.png` | PSC Demo — clinic | **The healthcare story** |
| `08-demo-commute.png` | PSC Demo — commute | The original street scene |
| `09-you.png` | You | Profile, voices, privacy |

## The clinic scene

`07-demo-clinic.png` is the frame the pitch rests on. A consultation room:
doctor, waiting room, air-conditioning, television, monitor alarm. The engine
scores **5/5 as scripted**, in both the Python reference and the Swift port:

| Source | Action | Gain |
|---|---|---:|
| The person speaking | enhance | **+3.5 dB** |
| Waiting room | suppress | −14.3 dB |
| Television | reduce | −6.9 dB |
| Air-conditioning | reduce | −5.9 dB |
| Monitor alarm | preserve | **+0.0 dB** |

## Before using these to make a claim

The scenes are synthetic, and the on-device classifier is measured at **14%
recall on real recordings** with `vehicle_horn` at 0/40 — see
[`docs/BENCHMARKS.md` §1a](../../../docs/BENCHMARKS.md). These frames honestly
show what AURA does on scripted audio. They are not evidence of accuracy in a
real clinic, and the speech path has no real-audio validation at all yet.

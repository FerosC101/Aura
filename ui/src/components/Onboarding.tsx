import { useState } from "react";
import { api } from "../lib/api";
import type { Category } from "../lib/types";
import { LogoMark, ShieldIcon } from "./Icons";

const USES = [
  { key: "studying", label: "Studying", mode: "focus" },
  { key: "commuting", label: "Commuting", mode: "commute" },
  { key: "walking", label: "Walking", mode: "walk" },
  { key: "conversations", label: "Conversations", mode: "social" },
  { key: "work", label: "Work", mode: "focus" },
];

export function Onboarding({
  categories,
  onDone,
}: {
  categories: Category[];
  onDone: () => void;
}) {
  const safety = categories.filter((c) => c.safety_critical);
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [uses, setUses] = useState<string[]>([]);
  // Every safety sound starts protected. Opting out is a deliberate act.
  const [safeOn, setSafeOn] = useState<string[]>(safety.map((c) => c.key));
  const [lockVoice, setLockVoice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const toggle = (arr: string[], v: string) =>
    arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v];

  async function finish() {
    setBusy(true);
    try {
      const preferred = USES.find((u) => u.key === uses[0])?.mode ?? "commute";
      await api.onboarding({
        user_name: name,
        primary_uses: uses,
        safety_categories: safeOn,
        preferred_mode: preferred,
        completed: true,
      });
      if (lockVoice) {
        await api.enrolSpeaker({ name: lockVoice, consent: true, demo_voice: lockVoice.toLowerCase() });
      }
      onDone();
    } finally {
      setBusy(false);
    }
  }

  const screens = [
    {
      title: "Welcome to AURA",
      body: (
        <>
          <p style={{ fontSize: 17, lineHeight: 1.6, color: "var(--text)" }}>
            Noise cancellation blocks the world.
            <br />
            <strong>AURA learns what parts of it matter.</strong>
          </p>
          <p className="muted">
            Instead of removing sound by frequency, AURA works out what each sound{" "}
            <em>is</em> — and decides whether you should hear it.
          </p>
        </>
      ),
    },
    {
      title: "Selective Awareness",
      body: (
        <>
          <p className="muted">
            AURA listens to everything happening around you at once, identifies each source, and
            builds a different mix for you in real time.
          </p>
          <div className="stack" style={{ gap: 8, marginTop: 14 }}>
            {[
              ["Preserved", "The voice in front of you, announcements, your name", "var(--preserve)"],
              ["Enhanced", "Anything that warns you — horns, sirens, bells", "var(--enhance)"],
              ["Reduced", "Traffic, crowds, repetitive background", "var(--reduce)"],
              ["Suppressed", "Engine rumble, machinery, fans", "var(--suppress)"],
            ].map(([k, v, c]) => (
              <div key={k} className="row" style={{ gap: 11 }}>
                <span style={{ width: 3, height: 30, borderRadius: 2, background: c, flex: "none" }} />
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: c as string }}>{k}</div>
                  <div className="faint" style={{ fontSize: 12 }}>{v}</div>
                </div>
              </div>
            ))}
          </div>
        </>
      ),
    },
    {
      title: "What will you use AURA for?",
      body: (
        <>
          <p className="muted">Pick any that apply. This sets your starting mode.</p>
          <div className="stack" style={{ gap: 7, marginTop: 12 }}>
            {USES.map((u) => (
              <label key={u.key} className="check">
                <input type="checkbox" checked={uses.includes(u.key)} onChange={() => setUses(toggle(uses, u.key))} />
                <span style={{ fontSize: 14 }}>{u.label}</span>
              </label>
            ))}
          </div>
        </>
      ),
    },
    {
      title: "Sounds that always get through",
      body: (
        <>
          <p className="muted">
            These override every other setting. AURA will never silence them, even if a mode or a
            slider says otherwise.
          </p>
          <div className="stack" style={{ gap: 7, marginTop: 12 }}>
            {safety.map((c) => (
              <label key={c.key} className="check">
                <input type="checkbox" checked={safeOn.includes(c.key)} onChange={() => setSafeOn(toggle(safeOn, c.key))} />
                <span>
                  <span className="row" style={{ gap: 7, fontSize: 14 }}>
                    <span style={{ color: "var(--safety)", display: "flex" }}><ShieldIcon size={13} /></span>
                    {c.label}
                  </span>
                  <div className="faint" style={{ fontSize: 11.5 }}>{c.description}</div>
                </span>
              </label>
            ))}
          </div>
          {safeOn.length < safety.length && (
            <div className="banner warn" style={{ marginTop: 12, fontSize: 12 }}>
              Turning a safety sound off means AURA may quieten it. We recommend leaving them all on.
            </div>
          )}
        </>
      ),
    },
    {
      title: "Person Lock (optional)",
      body: (
        <>
          <p className="muted">
            Lock onto one person's voice and AURA will lift them above everyone else. A short sample
            is turned into a mathematical fingerprint; the recording itself is discarded immediately.
          </p>
          <div className="row" style={{ flexWrap: "wrap", gap: 7, marginTop: 12 }}>
            {["Vince", "Maria", "Ben", "Ana"].map((v) => (
              <button
                key={v}
                className={`btn sm ${lockVoice === v ? "primary" : ""}`}
                onClick={() => setLockVoice(lockVoice === v ? null : v)}
              >
                {v}
              </button>
            ))}
          </div>
          <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
            These are synthesized demo voices. You can skip this and set it up later.
          </p>
        </>
      ),
    },
    {
      title: "Microphone access",
      body: (
        <>
          <p className="muted">AURA needs your microphone to hear the room.</p>
          <div className="banner info" style={{ marginTop: 12, display: "block" }}>
            <strong>AURA processes environmental audio in real time. Raw audio is not stored by default.</strong>
            <ul className="list" style={{ marginTop: 10 }}>
              <li>Everything runs on this machine. Nothing is uploaded.</li>
              <li>Audio is analysed frame by frame and discarded immediately.</li>
              <li>Only your preferences and settings are saved to disk.</li>
            </ul>
          </div>
          <div style={{ marginTop: 14 }}>
            <label className="metric-label">Your name (optional, for Name Awareness)</label>
            <input
              type="text"
              value={name}
              placeholder="Vince"
              onChange={(e) => setName(e.target.value)}
              style={{ marginTop: 6 }}
            />
          </div>
        </>
      ),
    },
  ];

  const current = screens[step];
  const last = step === screens.length - 1;

  return (
    <div className="overlay">
      <div className="modal">
        <div className="row" style={{ gap: 12, marginBottom: 20 }}>
          <div className="logo-mark"><LogoMark /></div>
          <div>
            <div className="logo-name" style={{ fontSize: 17 }}>AURA</div>
            <div className="logo-sub">Hear what matters. Silence what doesn't.</div>
          </div>
        </div>

        <h1 style={{ fontSize: 23, margin: "0 0 12px", fontWeight: 650 }}>{current.title}</h1>
        <div style={{ minHeight: 230 }}>{current.body}</div>

        <div className="spread" style={{ marginTop: 22 }}>
          <div className="row" style={{ gap: 5 }}>
            {screens.map((_, i) => (
              <span
                key={i}
                style={{
                  width: i === step ? 18 : 6, height: 6, borderRadius: 3,
                  background: i === step ? "var(--accent)" : "var(--line-strong)",
                  transition: "all 0.22s",
                }}
              />
            ))}
          </div>
          <div className="row" style={{ gap: 8 }}>
            {step > 0 && <button className="btn" onClick={() => setStep(step - 1)}>Back</button>}
            <button
              className="btn primary"
              disabled={busy}
              onClick={() => (last ? finish() : setStep(step + 1))}
            >
              {last ? (busy ? "Setting up…" : "Enable microphone & start") : "Continue"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

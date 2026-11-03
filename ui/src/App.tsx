import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./lib/api";
import type { Category, MixerRow, ModeProfile, Profile, Scene, Status } from "./lib/types";
import { AcousticSceneView } from "./components/AcousticScene";
import { DemoMode } from "./components/DemoMode";
import { LogoMark, ShieldIcon } from "./components/Icons";
import { Mixer } from "./components/Mixer";
import { Onboarding } from "./components/Onboarding";
import { PersonLockPanel } from "./components/PersonLock";
import { PrivacyCenter } from "./components/PrivacyCenter";
import { StatusPanel } from "./components/StatusPanel";
import { Waveform } from "./components/Waveform";

type Tab = "mixer" | "person" | "demo";

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [modes, setModes] = useState<ModeProfile[]>([]);
  const [mixer, setMixer] = useState<MixerRow[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [tab, setTab] = useState<Tab>("mixer");
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api.status());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Initial load.
  useEffect(() => {
    (async () => {
      try {
        const [s, m, c, p, mx] = await Promise.all([
          api.status(), api.modes(), api.categories(), api.profile(), api.mixer(),
        ]);
        setStatus(s);
        setModes(m.modes);
        setCategories(c.categories);
        setProfile(p);
        setMixer(mx.categories);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  // Live scene stream. Classifications arrive over the WebSocket; the heavier
  // status payload is polled, so a slow status call can never delay a frame.
  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.event === "scene") setScene(msg.data as Scene);
      else if (msg.event === "status") setStatus(msg.data as Status);
    };
    ws.onerror = () => setError("Lost the connection to the AURA engine.");
    return () => ws.close();
  }, []);

  useEffect(() => {
    const t = setInterval(refreshStatus, 1000);
    return () => clearInterval(t);
  }, [refreshStatus]);

  async function guard(fn: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      await refreshStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!status) {
    return (
      <div className="app">
        <div className="empty" style={{ minHeight: "60vh" }}>
          {error ?? "Connecting to the AURA engine…"}
        </div>
      </div>
    );
  }

  const running = status.running;
  const enabled = status.enabled;
  const sounds = scene?.sounds ?? [];
  const safetyActive = scene?.safety_override_active ?? false;

  return (
    <div className="app">
      {profile && !profile.onboarded && (
        <Onboarding
          categories={categories}
          onDone={async () => {
            setProfile(await api.profile());
            await guard(() => api.start({}));
          }}
        />
      )}
      {showPrivacy && <PrivacyCenter onClose={() => setShowPrivacy(false)} />}

      <header className="header">
        <div className="logo">
          <div className="logo-mark"><LogoMark /></div>
          <div>
            <div className="logo-name">AURA</div>
            <div className="logo-sub row" style={{ gap: 7 }}>
              <span className={`status-dot ${running && enabled ? "on" : ""}`} />
              {running
                ? enabled ? "Selective Awareness Active" : "Bypassed — hearing everything"
                : "Engine stopped"}
            </div>
          </div>
        </div>

        <div className="row" style={{ gap: 9 }}>
          <button className="btn sm ghost" onClick={() => setShowPrivacy(true)}>
            <span className="row" style={{ gap: 6 }}><ShieldIcon size={13} /> Privacy</span>
          </button>
          <button
            className={`btn sm ${running ? "" : "primary"}`}
            disabled={busy}
            onClick={() => guard(() => (running ? api.stop() : api.start({})))}
          >
            {running ? "Stop" : "Start engine"}
          </button>
          <div className="toggle">
            <span className="toggle-label" style={{ color: enabled ? "var(--accent)" : "var(--text-faint)" }}>
              AURA
            </span>
            <button
              className={`switch ${enabled ? "on" : ""}`}
              aria-label="Master AURA toggle"
              disabled={busy}
              onClick={() => guard(() => api.setEnabled(!enabled))}
            />
          </div>
        </div>
      </header>

      {error && (
        <div className="banner safety" style={{ marginBottom: 14 }}>
          {error}
          <button className="btn sm ghost" style={{ marginLeft: "auto" }} onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}
      {!running && (
        <div className="banner info" style={{ marginBottom: 14 }}>
          The engine is stopped. Start it to process your microphone, or open Demo Mode to play a
          scripted scene.
        </div>
      )}
      {safetyActive && (
        <div className="banner safety" style={{ marginBottom: 14 }}>
          <ShieldIcon size={16} />
          <strong>Safety Override</strong>
          <span className="faint">A safety-critical sound is being held audible.</span>
        </div>
      )}

      <section style={{ marginBottom: 16 }}>
        <div className="modes">
          {modes.map((m) => (
            <button
              key={m.mode}
              className={`mode ${status.mode === m.mode ? "active" : ""}`}
              disabled={busy}
              onClick={() =>
                guard(async () => {
                  const r = await api.setMode(m.mode);
                  setMixer(r.mixer.categories);
                })
              }
            >
              <div className="mode-name">{m.label}</div>
              <div className="mode-tag">{m.tagline}</div>
            </button>
          ))}
        </div>
      </section>

      <div className="grid grid-main">
        <div className="stack">
          <section className="card">
            <div className="card-title">
              <h2>Live Acoustic Scene</h2>
              <div className="row" style={{ gap: 7 }}>
                {scene?.person_lock.active && (
                  <span className="pill live">Person Lock · {scene.person_lock.name}</span>
                )}
                {scene?.name_awareness.active && <span className="pill live">Name heard</span>}
                <span className={`pill ${running ? "live" : ""}`}>
                  {running ? `${sounds.length} detected` : "idle"}
                </span>
              </div>
            </div>
            <AcousticSceneView sounds={sounds} running={running} />
          </section>

          <section className="card">
            <div className="card-title">
              <h2>Input vs Output</h2>
              <span className="faint mono" style={{ fontSize: 11 }}>
                in {scene?.levels.input_db.toFixed(0) ?? "–"} dB · out{" "}
                {scene?.levels.output_db.toFixed(0) ?? "–"} dB
              </span>
            </div>
            <Waveform input={status.waveform.input} output={status.waveform.output} />
          </section>
        </div>

        <div className="stack">
          <section className="card">
            <div className="tabs" style={{ marginBottom: 15 }}>
              {(["mixer", "person", "demo"] as Tab[]).map((t) => (
                <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
                  {t === "mixer" ? "Mixer" : t === "person" ? "Person Lock" : "Demo"}
                </button>
              ))}
            </div>

            {tab === "mixer" && (
              <Mixer rows={mixer} onChange={setMixer} safetyFloorDb={status.safety.min_gain_db} />
            )}
            {tab === "person" && (
              <PersonLockPanel
                state={status.person_lock}
                storeConsent={Boolean(profile?.privacy.store_speaker_embeddings)}
                onChange={() => refreshStatus()}
              />
            )}
            {tab === "demo" && <DemoMode onStarted={refreshStatus} />}
          </section>

          <section className="card">
            <div className="card-title">
              <h2>Status</h2>
              <span className="pill">{status.mode_profile.label}</span>
            </div>
            <StatusPanel status={status} />
          </section>
        </div>
      </div>

      <footer style={{ marginTop: 26, textAlign: "center" }}>
        <div className="faint" style={{ fontSize: 11.5 }}>
          AURA processes environmental audio locally in real time. Raw audio is not stored.
        </div>
      </footer>
    </div>
  );
}

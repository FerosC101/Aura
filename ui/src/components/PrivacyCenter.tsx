import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { PrivacyInfo } from "../lib/types";
import { ShieldIcon } from "./Icons";

const LABELS: Record<string, { title: string; note: string }> = {
  store_speaker_embeddings: {
    title: "Store speaker embeddings",
    note: "Keep Person Lock profiles between sessions. Vectors only — never audio.",
  },
  allow_name_awareness: {
    title: "Name Awareness",
    note: "Watch for your name in speech. Matching is done on sound patterns, not transcription.",
  },
  auto_context_detection: {
    title: "Automatic context detection",
    note: "Let AURA suggest a mode from what it hears. Experimental.",
  },
  allow_motion_context: {
    title: "Motion and place hints",
    note: "Use speed and place category to improve context detection, if a client supplies them.",
  },
  allow_cloud_inference: {
    title: "Cloud inference",
    note: "Not implemented. AURA is local-first; this would require a separate, explicit opt-in.",
  },
  retain_scene_history: {
    title: "Keep a scene history",
    note: "Store a log of detected sound categories over time for later review.",
  },
};

export function PrivacyCenter({ onClose }: { onClose: () => void }) {
  const [info, setInfo] = useState<PrivacyInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  useEffect(() => {
    api.privacy().then(setInfo);
  }, []);

  async function toggle(key: string, value: boolean) {
    setBusy(true);
    try {
      await api.setPrivacy({ [key]: value });
      setInfo(await api.privacy());
    } finally {
      setBusy(false);
    }
  }

  async function wipe() {
    setBusy(true);
    try {
      const r = await api.deleteData();
      setDone(
        `Deleted ${r.speaker_profiles_deleted} speaker profile(s) and ` +
          `${r.learned_preferences_deleted} learned preference(s).`,
      );
      setInfo(await api.privacy());
      setConfirming(false);
    } finally {
      setBusy(false);
    }
  }

  if (!info) return null;

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="spread" style={{ marginBottom: 6 }}>
          <h1 className="row" style={{ gap: 10, fontSize: 21, margin: 0, fontWeight: 650 }}>
            <span style={{ color: "var(--accent)", display: "flex" }}><ShieldIcon size={19} /></span>
            Privacy Center
          </h1>
          <button className="btn sm ghost" onClick={onClose}>Close</button>
        </div>
        <p className="muted" style={{ marginTop: 0 }}>
          AURA is local-first. Audio is processed on this machine and discarded.
        </p>

        <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 14, margin: "18px 0" }}>
          <Section title="What AURA processes" items={info.processes} tone="var(--accent)" />
          <Section title="What AURA stores" items={info.stores} tone="var(--reduce)" />
        </div>
        <Section title="What AURA never stores" items={info.never_stores} tone="var(--enhance)" />

        <h2 className="metric-label" style={{ margin: "22px 0 10px" }}>Permissions</h2>
        <div className="stack" style={{ gap: 7 }}>
          {Object.entries(info.settings).map(([key, value]) => {
            const meta = LABELS[key] ?? { title: key, note: "" };
            const disabled = busy || key === "allow_cloud_inference";
            return (
              <label key={key} className="check" style={{ opacity: disabled && key === "allow_cloud_inference" ? 0.55 : 1 }}>
                <input type="checkbox" checked={value} disabled={disabled} onChange={(e) => toggle(key, e.target.checked)} />
                <span>
                  <div style={{ fontSize: 13.5, fontWeight: 550 }}>{meta.title}</div>
                  <div className="faint" style={{ fontSize: 11.5 }}>{meta.note}</div>
                </span>
              </label>
            );
          })}
        </div>

        <h2 className="metric-label" style={{ margin: "22px 0 10px" }}>Saved speaker profiles</h2>
        {info.speaker_profiles.length ? (
          <div className="stack" style={{ gap: 6 }}>
            {info.speaker_profiles.map((p) => (
              <div key={p.id} className="spread" style={{ fontSize: 12.5, padding: "8px 11px", background: "var(--bg-input)", borderRadius: 8, border: "1px solid var(--line)" }}>
                <span>{p.name}</span>
                <span className="faint mono">{p.embedding_dim}-d · {p.backend}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="faint" style={{ fontSize: 12.5, margin: 0 }}>None saved.</p>
        )}

        <h2 className="metric-label" style={{ margin: "22px 0 10px" }}>Your data</h2>
        <p className="faint" style={{ fontSize: 12, marginTop: 0 }}>
          Stored at <span className="mono">{info.profile_path}</span>
        </p>
        {done && <div className="banner info" style={{ marginBottom: 10 }}>{done}</div>}
        {confirming ? (
          <div className="banner safety" style={{ display: "block" }}>
            <strong>Delete everything AURA has saved?</strong>
            <div style={{ fontSize: 12, margin: "6px 0 11px" }}>
              Speaker embeddings, learned preferences, trigger words and mixer positions. This cannot be undone.
            </div>
            <div className="row" style={{ gap: 8 }}>
              <button className="btn sm danger" disabled={busy} onClick={wipe}>Yes, delete it all</button>
              <button className="btn sm" onClick={() => setConfirming(false)}>Cancel</button>
            </div>
          </div>
        ) : (
          <button className="btn danger" disabled={busy} onClick={() => setConfirming(true)}>
            Delete all stored data
          </button>
        )}
      </div>
    </div>
  );
}

function Section({ title, items, tone }: { title: string; items: string[]; tone: string }) {
  return (
    <div>
      <h2 className="metric-label" style={{ color: tone, marginBottom: 8 }}>{title}</h2>
      <ul className="list">
        {items.map((s) => <li key={s}>{s}</li>)}
      </ul>
    </div>
  );
}

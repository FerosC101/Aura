import type { Action, SoundCard } from "../lib/types";
import { SoundIcon } from "./Icons";

const ACTION_COLOR: Record<Action, string> = {
  enhance: "var(--enhance)",
  preserve: "var(--preserve)",
  reduce: "var(--reduce)",
  suppress: "var(--suppress)",
};

const ACTION_VERB: Record<Action, string> = {
  enhance: "Enhanced",
  preserve: "Preserved",
  reduce: "Reduced",
  suppress: "Suppressed",
};

export function AcousticSceneView({ sounds, running }: { sounds: SoundCard[]; running: boolean }) {
  if (!sounds.length) {
    return (
      <div className="empty">
        {running ? "Listening for sounds…" : "Start the engine to hear the scene"}
      </div>
    );
  }
  return (
    <div className="sounds">
      {sounds.map((s) => (
        <SoundRow key={s.category} sound={s} />
      ))}
    </div>
  );
}

function SoundRow({ sound }: { sound: SoundCard }) {
  const color = ACTION_COLOR[sound.action];
  const gain = sound.gain_db;
  return (
    <div className={`sound ${sound.action}${sound.safety_override ? " safety" : ""}`}>
      <div className="sound-icon" style={{ color }}>
        <SoundIcon name={sound.icon} />
      </div>

      <div style={{ minWidth: 0 }}>
        <div className="sound-name">{sound.label}</div>
        <div className="sound-meta">
          {Math.round(sound.confidence * 100)}% confidence
          <span className="faint"> · {sound.reason}</span>
        </div>
        <div className="conf-bar">
          <div
            className="conf-fill"
            style={{ width: `${Math.round(sound.confidence * 100)}%`, background: color }}
          />
        </div>
      </div>

      <div className="verdict">
        <div className="verdict-action" style={{ color }}>
          {ACTION_VERB[sound.action]}
        </div>
        <div className="verdict-gain mono" style={{ color: gain === 0 ? "var(--text-dim)" : color }}>
          {gain > 0 ? "+" : ""}
          {gain.toFixed(1)} dB
        </div>
        {sound.safety_override && <div className="badge">Safety override</div>}
        {!sound.safety_override && sound.safety_critical && (
          <div className="badge" style={{ background: "rgba(94,234,212,0.13)", color: "var(--preserve)" }}>
            Safety priority
          </div>
        )}
      </div>
    </div>
  );
}

import { useState } from "react";
import { api } from "../lib/api";
import type { PersonLockState } from "../lib/types";
import { LockIcon } from "./Icons";

const DEMO_VOICES = ["Vince", "Maria", "Ben", "Ana"];

/**
 * Person Lock panel.
 *
 * Enrolment needs a voice sample. In a browser we cannot reach the engine's
 * microphone (the engine owns it natively, which is the whole latency
 * argument), so this offers synthetic demo voices. A real deployment enrols
 * from the engine's own capture; the API accepts raw samples for that.
 */
export function PersonLockPanel({
  state,
  onChange,
  storeConsent,
}: {
  state: PersonLockState;
  onChange: (s: PersonLockState) => void;
  storeConsent: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(fn: () => Promise<PersonLockState>) {
    setBusy(true);
    setError(null);
    try {
      onChange(await fn());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const selected = state.profiles.find((p) => p.id === state.selected_id);

  return (
    <div className="stack" style={{ gap: 13 }}>
      {state.active && selected ? (
        <div className="banner info">
          <span style={{ color: "var(--enhance)", display: "flex" }}>
            <LockIcon size={16} />
          </span>
          <div>
            <strong>Person Lock Active</strong>
            <div className="faint" style={{ fontSize: 12 }}>
              {selected.name} · similarity {state.similarity.toFixed(2)}
            </div>
          </div>
        </div>
      ) : (
        <div className="banner warn" style={{ background: "transparent", borderColor: "var(--line)", color: "var(--text-dim)" }}>
          <LockIcon size={16} />
          <div>
            {selected
              ? `Watching for ${selected.name} — not speaking right now`
              : "No speaker selected"}
          </div>
        </div>
      )}

      {state.profiles.length > 0 && (
        <div className="stack" style={{ gap: 7 }}>
          {state.profiles.map((p) => (
            <div
              key={p.id}
              className="spread"
              style={{
                padding: "9px 12px",
                background: p.id === state.selected_id ? "var(--bg-hover)" : "var(--bg-input)",
                border: `1px solid ${p.id === state.selected_id ? "rgba(34,211,238,0.35)" : "var(--line)"}`,
                borderRadius: "var(--radius-sm)",
              }}
            >
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 13.5, fontWeight: 550 }}>{p.name}</div>
                <div className="faint" style={{ fontSize: 11 }}>
                  {p.n_samples} sample{p.n_samples === 1 ? "" : "s"} · {p.embedding_dim}-d embedding
                </div>
              </div>
              <div className="row" style={{ gap: 6 }}>
                {p.id !== state.selected_id && (
                  <button className="btn sm" disabled={busy} onClick={() => run(() => api.selectSpeaker(p.id))}>
                    Select
                  </button>
                )}
                <button className="btn sm danger" disabled={busy} onClick={() => run(() => api.deleteSpeaker(p.id))}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div>
        <div className="faint" style={{ fontSize: 11.5, marginBottom: 7 }}>
          Enrol a demo voice (no recording leaves your machine; only the embedding is kept)
        </div>
        <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
          {DEMO_VOICES.filter((v) => !state.profiles.some((p) => p.name === v)).map((v) => (
            <button
              key={v}
              className="btn sm"
              disabled={busy}
              onClick={() =>
                run(async () =>
                  (await api.enrolSpeaker({ name: v, consent: true, demo_voice: v.toLowerCase() })).person_lock,
                )
              }
            >
              + {v}
            </button>
          ))}
        </div>
      </div>

      {!storeConsent && state.profiles.length > 0 && (
        <div className="faint" style={{ fontSize: 11.5 }}>
          Embeddings are held in memory only. Turn on “Store speaker embeddings” in the Privacy
          Center to keep them between sessions.
        </div>
      )}
      {error && <div className="banner safety">{error}</div>}
    </div>
  );
}

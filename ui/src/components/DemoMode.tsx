import { useEffect, useState } from "react";
import { api, type Comparison } from "../lib/api";
import type { DemoScene } from "../lib/types";

const ACTION_COLOR: Record<string, string> = {
  enhance: "var(--enhance)",
  preserve: "var(--preserve)",
  reduce: "var(--reduce)",
  suppress: "var(--suppress)",
};

/**
 * PSC Demo Mode.
 *
 * Plays a scripted multi-source scene so the concept can be shown reliably
 * regardless of what the room happens to contain, and lets a judge A/B it
 * against raw audio and a conventional ANC simulation. The same source
 * material goes through all three chains, so the comparison is honest.
 */
export function DemoMode({ onStarted }: { onStarted: () => void }) {
  const [scenes, setScenes] = useState<DemoScene[]>([]);
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [active, setActive] = useState<string>("psc_demo");
  const [comparison, setComparison] = useState<string>("aura");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.demoScenes().then((r) => {
      setScenes(r.scenes);
      setComparisons(r.comparisons);
    });
  }, []);

  async function play(sceneKey: string, cmp: string) {
    setBusy(true);
    setError(null);
    try {
      await api.startDemo(sceneKey, cmp);
      setActive(sceneKey);
      setComparison(cmp);
      onStarted();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const scene = scenes.find((s) => s.key === active);

  return (
    <div className="stack" style={{ gap: 14 }}>
      <div className="row" style={{ flexWrap: "wrap", gap: 7 }}>
        {scenes.map((s) => (
          <button
            key={s.key}
            className={`btn sm ${s.key === active ? "primary" : ""}`}
            disabled={busy}
            onClick={() => play(s.key, comparison)}
          >
            {s.title}
          </button>
        ))}
      </div>

      {scene && (
        <>
          <div>
            <div style={{ fontSize: 14.5, fontWeight: 600 }}>{scene.subtitle}</div>
            <p className="muted" style={{ fontSize: 12.5, margin: "5px 0 0" }}>{scene.description}</p>
          </div>

          <div className="tabs">
            {comparisons.map((c) => (
              <button
                key={c.key}
                className={`tab ${c.key === comparison ? "active" : ""}`}
                disabled={busy}
                onClick={() => play(active, c.key)}
              >
                {c.label}
              </button>
            ))}
          </div>
          <p className="faint" style={{ fontSize: 11.5, margin: 0 }}>
            {comparisons.find((c) => c.key === comparison)?.description}
          </p>

          <div>
            <div className="metric-label" style={{ marginBottom: 8 }}>Sources in this scene</div>
            <div className="stack" style={{ gap: 5 }}>
              {scene.layers.map((l, i) => {
                const expected = scene.expect[l.category];
                return (
                  <div key={`${l.category}-${i}`} className="spread" style={{ fontSize: 12 }}>
                    <span className="muted">{l.category.replace(/_/g, " ")}</span>
                    <span className="row" style={{ gap: 9 }}>
                      <span className="faint mono">
                        {l.start.toFixed(0)}–{l.end.toFixed(0)}s
                      </span>
                      {expected && (
                        <span style={{ color: ACTION_COLOR[expected], fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                          {expected}
                        </span>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <ul className="list">
            {scene.highlights.map((h) => <li key={h}>{h}</li>)}
          </ul>
        </>
      )}

      {error && <div className="banner safety">{error}</div>}
    </div>
  );
}

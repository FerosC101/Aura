import { useMemo, useState } from "react";
import type { MixerRow } from "../lib/types";
import { api } from "../lib/api";
import { ShieldIcon, SoundIcon } from "./Icons";

const ACTION_COLOR: Record<string, string> = {
  enhance: "var(--enhance)",
  preserve: "var(--preserve)",
  reduce: "var(--reduce)",
  suppress: "var(--suppress)",
};

/**
 * Per-category gain control.
 *
 * Moving a slider does two things: it sets an explicit override for the current
 * mode, and it teaches the preference learner (the engine records the
 * *difference* from what it would have chosen). Safety-critical categories can
 * be raised but not pushed below the guard's floor, and the slider reflects
 * that rather than letting the user drag into a value that will be ignored.
 */
export function Mixer({
  rows,
  onChange,
  safetyFloorDb,
}: {
  rows: MixerRow[];
  onChange: (rows: MixerRow[]) => void;
  safetyFloorDb: number;
}) {
  const [pending, setPending] = useState<Record<string, number>>({});
  const [showAll, setShowAll] = useState(false);

  const visible = useMemo(() => {
    const primary = [
      "human_speech", "crowd_speech", "announcement", "vehicle_horn",
      "emergency_siren", "engine_noise", "general_traffic", "alarm",
    ];
    return showAll ? rows : rows.filter((r) => primary.includes(r.category));
  }, [rows, showAll]);

  async function commit(category: string, value: number | null) {
    const res = await api.setGain(category, value);
    onChange(res.categories);
    setPending((p) => {
      const next = { ...p };
      delete next[category];
      return next;
    });
  }

  return (
    <div>
      {visible.map((row) => {
        const min = row.safety_critical ? safetyFloorDb : -24;
        const live = pending[row.category] ?? row.override_db ?? row.gain_db;
        const color = ACTION_COLOR[row.action] ?? "var(--text)";
        return (
          <div className="mixer-row" key={row.category}>
            <div className="mixer-head">
              <span className="row" style={{ gap: 8 }}>
                <span style={{ color, display: "flex" }}>
                  <SoundIcon name={row.icon} size={14} />
                </span>
                <span className="mixer-name">{row.label}</span>
                {row.safety_critical && (
                  <span style={{ color: "var(--safety)", display: "flex" }} title="Protected by the Safety Guard">
                    <ShieldIcon size={12} />
                  </span>
                )}
                {row.override_db !== null && <span className="pill">manual</span>}
                {Math.abs(row.learned_db) >= 0.5 && (
                  <span className="pill" title="Learned from your adjustments">
                    learned {row.learned_db > 0 ? "+" : ""}
                    {row.learned_db.toFixed(1)}
                  </span>
                )}
              </span>
              <span className="mixer-value mono" style={{ color }}>
                {live > 0 ? "+" : ""}
                {live.toFixed(1)} dB
              </span>
            </div>

            <input
              type="range"
              min={min}
              max={12}
              step={1}
              value={live}
              onChange={(e) =>
                setPending((p) => ({ ...p, [row.category]: Number(e.target.value) }))
              }
              onMouseUp={(e) => commit(row.category, Number((e.target as HTMLInputElement).value))}
              onTouchEnd={(e) => commit(row.category, Number((e.target as HTMLInputElement).value))}
              onKeyUp={(e) => commit(row.category, Number((e.target as HTMLInputElement).value))}
            />

            {row.override_db !== null && (
              <button className="btn ghost sm" style={{ marginTop: 7 }} onClick={() => commit(row.category, null)}>
                Reset to {row.action}
              </button>
            )}
          </div>
        );
      })}

      <button className="btn ghost sm" style={{ marginTop: 12 }} onClick={() => setShowAll((v) => !v)}>
        {showAll ? "Show fewer" : `Show all ${rows.length} categories`}
      </button>
    </div>
  );
}

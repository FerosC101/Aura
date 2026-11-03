import type { Status } from "../lib/types";

function Metric({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: string }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={{ color: tone }}>
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
    </div>
  );
}

/**
 * Latency and load.
 *
 * The headline figure is end-to-end and includes the driver buffers and the
 * separator's algorithmic delay, not just the time spent computing — reporting
 * only the compute time would understate what the listener actually
 * experiences by an order of magnitude.
 */
export function StatusPanel({ status }: { status: Status }) {
  const m = status.metrics;
  const load = m.processing_load;
  const loadTone = load > 0.85 ? "var(--suppress)" : load > 0.6 ? "var(--reduce)" : "var(--enhance)";
  const stages: [string, string][] = [
    ["Capture", "capture"],
    ["Separation", "separation"],
    ["Inference", "classification"],
    ["Policy", "policy"],
    ["Mixing", "mixing"],
    ["Output", "output"],
  ];

  return (
    <div className="stack" style={{ gap: 14 }}>
      <div className="metric-grid">
        <Metric label="Total latency" value={m.total.mean_ms.toFixed(1)} unit="ms" tone="var(--accent)" />
        <Metric label="p95" value={m.total.p95_ms.toFixed(1)} unit="ms" />
        <Metric label="AI load" value={(load * 100).toFixed(0)} unit="%" tone={loadTone} />
        <Metric label="Dropouts" value={String(m.xruns)} tone={m.xruns ? "var(--reduce)" : undefined} />
      </div>

      <div>
        <div className="metric-label" style={{ marginBottom: 8 }}>Per-stage</div>
        <div className="stack" style={{ gap: 5 }}>
          {stages.map(([label, key]) => {
            const ms = m.latency[`${key}_ms`] ?? m.stages[key]?.mean_ms ?? 0;
            const pct = Math.min(100, (ms / Math.max(m.total.mean_ms, 1)) * 100);
            return (
              <div key={key}>
                <div className="spread" style={{ fontSize: 11.5 }}>
                  <span className="muted">{label}</span>
                  <span className="mono faint">{ms.toFixed(2)} ms</span>
                </div>
                <div className="conf-bar" style={{ marginTop: 3 }}>
                  <div className="conf-fill" style={{ width: `${pct}%`, background: "var(--accent-deep)" }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="stack" style={{ gap: 5, fontSize: 12 }}>
        <div className="spread">
          <span className="faint">Input</span>
          <span className="muted" style={{ textAlign: "right" }}>{status.audio.input_device ?? "—"}</span>
        </div>
        <div className="spread">
          <span className="faint">Output</span>
          <span className="muted" style={{ textAlign: "right" }}>{status.audio.output_device ?? "—"}</span>
        </div>
        <div className="spread">
          <span className="faint">Block</span>
          <span className="mono muted">
            {m.block_ms.toFixed(1)} ms @ {(status.config.sample_rate / 1000).toFixed(1)} kHz
          </span>
        </div>
        <div className="spread">
          <span className="faint">Classifier</span>
          <span className="mono muted">{status.backends.classifier.name}</span>
        </div>
        <div className="spread">
          <span className="faint">Separator</span>
          <span className="mono muted">{status.backends.separator.name}</span>
        </div>
      </div>

      {Object.entries(status.backends.fallbacks).map(([k, note]) => (
        <div className="banner warn" key={k} style={{ fontSize: 11.5 }}>
          {note}
        </div>
      ))}
    </div>
  );
}

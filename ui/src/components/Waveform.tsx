import { useEffect, useRef } from "react";

interface Props {
  input: number[];
  output: number[];
  height?: number;
}

/**
 * Input against output, drawn as mirrored envelopes.
 *
 * The point of the view is the *difference* between the two: where AURA has
 * removed energy the cyan output sits visibly inside the grey input, which is
 * the clearest single picture of the product working. The data is already
 * peak-decimated by the engine, so this only has to draw it.
 */
export function Waveform({ input, output, height = 88 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const mid = h / 2;
    const draw = (data: number[], color: string, fill: string) => {
      if (!data.length) return;
      const step = w / data.length;
      // Mirrored envelope: up from the centre, then back along the bottom.
      ctx.beginPath();
      data.forEach((v, i) => {
        const y = mid - Math.min(v, 1) * (mid - 3);
        i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * step, y);
      });
      for (let i = data.length - 1; i >= 0; i--) {
        ctx.lineTo(i * step, mid + Math.min(data[i], 1) * (mid - 3));
      }
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    draw(input, "rgba(153,161,179,0.5)", "rgba(153,161,179,0.13)");
    draw(output, "rgba(34,211,238,0.95)", "rgba(34,211,238,0.2)");

    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();
  }, [input, output, height]);

  return (
    <div>
      <canvas ref={ref} style={{ width: "100%", height, display: "block" }} />
      <div className="row" style={{ gap: 16, marginTop: 8, fontSize: 11 }}>
        <span className="row" style={{ gap: 6 }}>
          <i style={{ width: 9, height: 3, borderRadius: 2, background: "rgba(153,161,179,0.6)" }} />
          <span className="faint">Microphone</span>
        </span>
        <span className="row" style={{ gap: 6 }}>
          <i style={{ width: 9, height: 3, borderRadius: 2, background: "var(--accent)" }} />
          <span className="faint">AURA output</span>
        </span>
      </div>
    </div>
  );
}

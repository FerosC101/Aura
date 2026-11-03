// Inline SVG glyphs, one per taxonomy icon key. Inline so there is no icon
// font to load and so each one inherits currentColor.

const S = { fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round", strokeLinejoin: "round" } as const;

export function SoundIcon({ name, size = 17 }: { name: string; size?: number }) {
  const p = { width: size, height: size, viewBox: "0 0 24 24", ...S };
  switch (name) {
    case "speech":
      return <svg {...p}><path d="M12 4v10M8 7v4M16 6v6M4 9v2M20 9v2" /></svg>;
    case "crowd":
      return <svg {...p}><circle cx="8" cy="9" r="2.4" /><circle cx="16" cy="9" r="2.4" /><path d="M3.5 18a4.5 4.5 0 0 1 9 0M11.5 18a4.5 4.5 0 0 1 9 0" /></svg>;
    case "horn":
      return <svg {...p}><path d="M4 10v4h3l5 3.5v-11L7 10H4Z" /><path d="M16 9.5a3.5 3.5 0 0 1 0 5M18.6 7a7 7 0 0 1 0 10" /></svg>;
    case "siren":
      return <svg {...p}><path d="M7 16a5 5 0 0 1 10 0Z" /><path d="M4 19h16M12 5V3M6.4 7.4 5 6M17.6 7.4 19 6" /></svg>;
    case "engine":
      return <svg {...p}><rect x="3" y="9" width="13" height="8" rx="1.6" /><path d="M16 12h2.5l2.5 3v2h-5M6 9V6.5h5V9M6 20v-3M13 20v-3" /></svg>;
    case "traffic":
      return <svg {...p}><path d="M4 20 8 4M20 20 16 4M12 5v2M12 11v2M12 17v2" /></svg>;
    case "alarm":
      return <svg {...p}><path d="M18 15V10a6 6 0 0 0-12 0v5l-1.6 2.5h15.2Z" /><path d="M10 20.5a2.2 2.2 0 0 0 4 0" /></svg>;
    case "music":
      return <svg {...p}><path d="M9 18V6l11-2v12" /><circle cx="6.5" cy="18" r="2.5" /><circle cx="17.5" cy="16" r="2.5" /></svg>;
    case "construction":
      return <svg {...p}><path d="m13.5 4 6.5 6.5-2.5 2.5L11 6.5 13.5 4Z" /><path d="M10 7.5 3.5 14v6.5H10L16.5 14" /></svg>;
    case "announcement":
      return <svg {...p}><path d="M4 9v6h3.5L15 19V5L7.5 9H4Z" /><path d="M18.5 9.5a4 4 0 0 1 0 5" /></svg>;
    case "bell":
      return <svg {...p}><circle cx="12" cy="12" r="7.5" /><circle cx="12" cy="12" r="2" /><path d="M12 4.5v3M12 16.5v3M4.5 12h3M16.5 12h3" /></svg>;
    case "approach":
      return <svg {...p}><path d="M3 12h11M10 8l4 4-4 4" /><path d="M17 6v12M20.5 8.5v7" /></svg>;
    case "warning":
      return <svg {...p}><path d="M12 4 2.8 19.5h18.4L12 4Z" /><path d="M12 10v4M12 16.8v.2" /></svg>;
    case "ambient":
      return <svg {...p}><path d="M3 12c2-4 4 4 6 0s4 4 6 0 4 4 6 0" /></svg>;
    default:
      return <svg {...p}><path d="M3 12h3l3-6 3 12 3-9 3 3h3" /></svg>;
  }
}

export function LogoMark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#04161b" strokeWidth={2} strokeLinecap="round">
      <path d="M4 13h2.5l2-6 3 12 3-9 2 3H20" />
    </svg>
  );
}

export function ShieldIcon({ size = 14 }: { size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" {...S}><path d="M12 3 5 6v6c0 4.2 2.9 7.9 7 9 4.1-1.1 7-4.8 7-9V6l-7-3Z" /></svg>;
}

export function LockIcon({ size = 14 }: { size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" {...S}><rect x="5" y="11" width="14" height="9" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></svg>;
}

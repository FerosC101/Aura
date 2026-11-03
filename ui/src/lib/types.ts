// Mirrors the payloads produced by aura.pipeline.engine and aura.server.app.

export type Action = "enhance" | "preserve" | "reduce" | "suppress";
export type ModeKey = "focus" | "commute" | "walk" | "social" | "custom";

export interface SoundCard {
  category: string;
  label: string;
  icon: string;
  confidence: number;
  level_db: number;
  stream: string;
  safety_critical: boolean;
  action: Action;
  gain_db: number;
  priority: string;
  safety_override: boolean;
  reason: string;
}

export interface Scene {
  sequence: number;
  timestamp: number;
  mode: ModeKey;
  context: string;
  enabled: boolean;
  sounds: SoundCard[];
  person_lock: { active: boolean; name: string | null; similarity: number };
  name_awareness: { enabled: boolean; active: boolean; words: string[] };
  safety_override_active: boolean;
  levels: { input_db: number; output_db: number };
  mix: {
    stream_gain_db: Record<string, number>;
    environment_band_db: number[];
    environment_bands_hz: number[];
    limiter_active: boolean;
  };
}

export interface StageStat { mean_ms: number; p50_ms: number; p95_ms: number; max_ms: number }

export interface Status {
  running: boolean;
  enabled: boolean;
  mode: ModeKey;
  mode_profile: ModeProfile;
  suppression_strength: number;
  config: Record<string, unknown> & { sample_rate: number; block_ms: number };
  audio: {
    transport: string | null;
    running: boolean;
    input_device?: string | null;
    output_device?: string | null;
    xruns?: number;
    input_latency_ms?: number;
    output_latency_ms?: number;
  };
  backends: {
    classifier: { name: string; categories: string[] };
    separator: { name: string; streams: string[] };
    speaker: { name: string; dim: number };
    name_awareness: { name: string; transcribes: boolean; words: string[] };
    fallbacks: Record<string, string>;
  };
  metrics: {
    latency: Record<string, number>;
    stages: Record<string, StageStat>;
    total: StageStat;
    processing_load: number;
    frames: number;
    xruns: number;
    block_ms: number;
    device_latency_ms: { input: number; output: number };
    algorithmic_delay_ms: number;
  };
  context: {
    state: string; confidence: number; suggested_mode: ModeKey;
    source: string; auto_enabled: boolean; allow_motion: boolean; manual: string | null;
  };
  person_lock: PersonLockState;
  safety: { min_gain_db: number; protected: string[]; labels: Record<string, string> };
  waveform: { input: number[]; output: number[] };
}

export interface ModeProfile {
  mode: ModeKey;
  label: string;
  tagline: string;
  description: string;
  intents: Record<string, Action>;
  default: Action;
  suppression_scale: number;
}

export interface SpeakerProfile {
  id: string; name: string; n_samples: number; backend: string;
  created_at: number; consented: boolean; embedding_dim: number;
}

export interface PersonLockState {
  selected_id: string | null;
  profiles: SpeakerProfile[];
  active: boolean;
  name: string | null;
  similarity: number;
  margin?: number;
}

export interface MixerRow {
  category: string; label: string; action: Action; gain_db: number;
  priority: string; safety_override: boolean; always_audible: boolean;
  reason: string; override_db: number | null; safety_critical: boolean;
  stream: string; icon: string; description: string; learned_db: number;
}

export interface DemoScene {
  key: string; title: string; subtitle: string; description: string;
  duration: number; mode: ModeKey; expect: Record<string, string>;
  highlights: string[];
  layers: { category: string; gain_db: number; start: number; end: number }[];
}

export interface PrivacyInfo {
  settings: Record<string, boolean>;
  processes: string[];
  stores: string[];
  never_stores: string[];
  profile_path: string;
  speaker_profiles: SpeakerProfile[];
}

export interface Profile {
  user_name: string; onboarded: boolean; preferred_mode: ModeKey;
  primary_uses: string[]; safety_categories: string[]; trigger_words: string[];
  suppression_strength: number; privacy: Record<string, boolean>;
  most_used_mode: ModeKey; frequently_preserved: string[]; frequently_suppressed: string[];
}

export interface Category {
  key: string; label: string; description: string;
  safety_critical: boolean; stream: string; band_hz: number[]; icon: string;
}

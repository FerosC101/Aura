import type {
  Category, DemoScene, MixerRow, ModeProfile, PersonLockState,
  PrivacyInfo, Profile, Scene, Status,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* the body was not JSON; the status text will do */
    }
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  status: () => request<Status>("/api/status"),
  scene: () => request<Scene>("/api/scene"),
  modes: () => request<{ modes: ModeProfile[]; current: string }>("/api/modes"),
  categories: () => request<{ categories: Category[] }>("/api/categories"),
  devices: () => request<{ available: boolean; devices: DeviceInfo[] }>("/api/devices"),

  start: (body: { input_device?: string | number | null; demo_scene?: string | null } = {}) =>
    post<Status>("/api/engine/start", body),
  stop: () => post<Status>("/api/engine/stop"),
  setEnabled: (enabled: boolean) => post<{ enabled: boolean }>("/api/engine/enabled", { enabled }),
  setMode: (mode: string) => post<{ mode: string; mixer: { categories: MixerRow[] } }>("/api/mode", { mode }),
  setSuppression: (strength: number) => post<{ strength: number }>("/api/suppression", { strength }),

  mixer: () => request<{ mode: string; categories: MixerRow[] }>("/api/mixer"),
  setGain: (category: string, gain_db: number | null) =>
    post<{ mode: string; categories: MixerRow[] }>("/api/mixer", { category, gain_db }),
  learning: () => request<{ enabled: boolean; preferences: LearnedPref[] }>("/api/learning"),
  forgetLearning: () => request<{ forgotten: number }>("/api/learning", { method: "DELETE" }),

  speakers: () => request<PersonLockState>("/api/speakers"),
  enrolSpeaker: (body: { name: string; consent: boolean; demo_voice?: string; clips?: Clip[] }) =>
    post<{ person_lock: PersonLockState }>("/api/speakers", body),
  selectSpeaker: (profile_id: string | null) =>
    post<PersonLockState>("/api/speakers/select", { profile_id }),
  deleteSpeaker: (id: string) => request<PersonLockState>(`/api/speakers/${id}`, { method: "DELETE" }),

  enrolName: (body: { word: string; demo_word?: string; clips?: Clip[] }) =>
    post<unknown>("/api/name/enrol", body),
  setNameAwareness: (enabled: boolean) => post<{ enabled: boolean }>("/api/name/enabled", { enabled }),

  setContext: (state: string | null) => post<unknown>("/api/context", { state }),

  profile: () => request<Profile>("/api/profile"),
  onboarding: (body: unknown) => post<Profile>("/api/onboarding", body),

  privacy: () => request<PrivacyInfo>("/api/privacy"),
  setPrivacy: (body: Record<string, boolean>) =>
    request<Record<string, boolean>>("/api/privacy", { method: "PATCH", body: JSON.stringify(body) }),
  deleteData: () => request<Record<string, number | boolean>>("/api/privacy/data", { method: "DELETE" }),

  demoScenes: () => request<{ scenes: DemoScene[]; comparisons: Comparison[] }>("/api/demo/scenes"),
  startDemo: (scene: string, comparison: string) =>
    post<{ status: Status }>("/api/demo/start", { scene, comparison }),
};

export interface DeviceInfo {
  index: number; name: string; max_input_channels: number;
  max_output_channels: number; default_samplerate: number;
  hostapi: string; is_default_input: boolean; is_default_output: boolean;
}
export interface Clip { samples: number[]; sample_rate: number }
export interface Comparison { key: string; label: string; description: string }
export interface LearnedPref {
  mode: string; category: string; effective_db: number;
  direction: string; observations: number;
}

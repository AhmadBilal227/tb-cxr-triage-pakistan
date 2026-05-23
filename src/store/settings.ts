import { useSyncExternalStore } from 'react';
import type { Settings } from '@/lib/types';
import { DEFAULT_SETTINGS, SETTINGS_STORAGE_KEY } from '@/lib/defaults';

/**
 * Dependency-free reactive settings store backed by localStorage.
 *
 * BYOK keys live in localStorage by design (frontend-only app). The Settings
 * drawer carries the matching "any JS on this page can read these" warning.
 */

type Listener = () => void;

/** Local-dev BYOK seeding from .env.local. Empty strings in production builds. */
const ENV_KEYS = {
  openaiKey: import.meta.env.VITE_OPENAI_KEY ?? '',
  hfToken: import.meta.env.VITE_HF_TOKEN ?? '',
  replicateToken: import.meta.env.VITE_REPLICATE_TOKEN ?? '',
};

function withEnvFallback(s: Settings): Settings {
  // Use an env key only when the stored/default value for that field is empty.
  return {
    ...s,
    openaiKey: s.openaiKey || ENV_KEYS.openaiKey,
    hfToken: s.hfToken || ENV_KEYS.hfToken,
    replicateToken: s.replicateToken || ENV_KEYS.replicateToken,
  };
}

function loadInitial(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return withEnvFallback(DEFAULT_SETTINGS);
    const parsed = JSON.parse(raw) as Partial<Settings>;
    // Deep-merge so new fields added in later versions inherit defaults.
    return withEnvFallback({
      ...DEFAULT_SETTINGS,
      ...parsed,
      overrides: { ...DEFAULT_SETTINGS.overrides, ...(parsed.overrides ?? {}) },
      models: { ...DEFAULT_SETTINGS.models, ...(parsed.models ?? {}) },
    });
  } catch {
    return withEnvFallback(DEFAULT_SETTINGS);
  }
}

let state: Settings = loadInitial();
const listeners = new Set<Listener>();

function emit(): void {
  for (const l of listeners) l();
}

function persist(): void {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Storage full / disabled — settings simply won't persist this session.
  }
}

export const settingsStore = {
  get(): Settings {
    return state;
  },
  subscribe(listener: Listener): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  /** Shallow-patch top-level fields. */
  set(patch: Partial<Settings>): void {
    state = { ...state, ...patch };
    persist();
    emit();
  },
  /** Patch the nested model-override block. */
  setOverride(patch: Partial<Settings['overrides']>): void {
    state = { ...state, overrides: { ...state.overrides, ...patch } };
    persist();
    emit();
  },
  setModels(patch: Partial<Settings['models']>): void {
    state = { ...state, models: { ...state.models, ...patch } };
    persist();
    emit();
  },
  reset(): void {
    state = DEFAULT_SETTINGS;
    persist();
    emit();
  },
};

/** React hook: re-renders on any settings change. */
export function useSettings(): Settings {
  return useSyncExternalStore(settingsStore.subscribe, settingsStore.get, settingsStore.get);
}

// ---------------------------------------------------------------------------
// Derived capability checks (used to gate stages + show warnings)
// ---------------------------------------------------------------------------

export interface Capabilities {
  hasOpenAI: boolean;
  hasHF: boolean;
  hasReplicate: boolean;
  /** Stage 3 can run only if an embedding provider is configured. */
  hasEmbedding: boolean;
  embeddingVia: 'hf-endpoint' | 'replicate' | null;
}

export function deriveCapabilities(s: Settings): Capabilities {
  const hasReplicate = s.replicateToken.trim().length > 0;
  const hasHfEndpoint = s.overrides.embeddingEndpointUrl.trim().length > 0;
  const hasReplicateEmbed =
    hasReplicate && s.overrides.embeddingReplicate.trim().length > 0;
  return {
    hasOpenAI: s.openaiKey.trim().length > 0,
    hasHF: s.hfToken.trim().length > 0,
    hasReplicate,
    hasEmbedding: hasHfEndpoint || hasReplicateEmbed,
    embeddingVia: hasHfEndpoint ? 'hf-endpoint' : hasReplicateEmbed ? 'replicate' : null,
  };
}

export function useCapabilities(): Capabilities {
  return deriveCapabilities(useSettings());
}

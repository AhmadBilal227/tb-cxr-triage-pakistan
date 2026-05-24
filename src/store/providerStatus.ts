import { useSyncExternalStore } from 'react';
import type { Provider } from '@/lib/types';

/**
 * In-memory, ephemeral per-provider status (NOT persisted to localStorage).
 *
 * Records the result of the LAST call to each provider so the Settings drawer
 * can show a human-readable status line (e.g. "HF Inference: last call 401
 * unauthorized — token missing or invalid"). Without this the user has to open
 * DevTools to see why perception failed.
 *
 * Recorded by:
 *   - the HF / Replicate / OpenAI provider clients on success/failure
 *   - the orchestrator on a fully-unconfigured run (no-token short-circuit)
 *
 * Consumed by:
 *   - SettingsDrawer (per-key status lines)
 *
 * Failure-mode tags are kept stable so tests + UI can match on them. The
 * `note` carries optional model-id / endpoint / status-code context for the
 * user to act on without DevTools.
 */

export type ProviderStatusState =
  | 'unknown' // never called
  | 'not-configured' // no token / key for this provider
  | 'ok' // last call succeeded
  | 'unauthorized' // 401 / 403 — bad or missing token
  | 'model-unsupported' // 400/410 on hf-inference: model retired from router
  | 'rate-limited' // 429
  | 'network' // CORS / DNS / TLS / abort
  | 'other-error'; // anything else (5xx, 4xx not above, parse error)

export interface ProviderStatus {
  state: ProviderStatusState;
  /** Optional short note carrying model id, endpoint URL, or HTTP code. */
  note?: string;
  /** epoch ms of the last update; UI may surface "X seconds ago" later. */
  at: number | null;
}

const INITIAL_STATUS: ProviderStatus = { state: 'unknown', at: null };

type Listener = () => void;

let state: Record<Provider, ProviderStatus> = {
  hf: { ...INITIAL_STATUS },
  replicate: { ...INITIAL_STATUS },
  openai: { ...INITIAL_STATUS },
};

const listeners = new Set<Listener>();

function emit(): void {
  for (const l of listeners) l();
}

export const providerStatusStore = {
  get(): Record<Provider, ProviderStatus> {
    return state;
  },
  subscribe(listener: Listener): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  /** Replace the status for one provider. UI re-renders. */
  set(provider: Provider, status: Omit<ProviderStatus, 'at'>): void {
    state = { ...state, [provider]: { ...status, at: Date.now() } };
    emit();
  },
  /** Test seam: reset all to 'unknown'. */
  reset(): void {
    state = {
      hf: { ...INITIAL_STATUS },
      replicate: { ...INITIAL_STATUS },
      openai: { ...INITIAL_STATUS },
    };
    emit();
  },
};

export function useProviderStatus(): Record<Provider, ProviderStatus> {
  return useSyncExternalStore(
    providerStatusStore.subscribe,
    providerStatusStore.get,
    providerStatusStore.get,
  );
}

/**
 * Translate a HTTP status + response text into a ProviderStatusState tag.
 *
 * The hf-inference router signals retired models with HTTP 400 + a body whose
 * `.error` is "Model not supported by provider hf-inference" — this is a
 * CONFIG problem, not an auth problem, so it gets its own tag rather than
 * being collapsed into "unauthorized" or "other-error".
 */
export function classifyHttpFailure(
  status: number,
  bodyText: string,
): { state: ProviderStatusState; humanReason: string } {
  const lower = bodyText.toLowerCase();
  if (status === 401 || status === 403) {
    return { state: 'unauthorized', humanReason: `${status} unauthorized — token missing or invalid` };
  }
  if (status === 429) {
    return { state: 'rate-limited', humanReason: `${status} rate-limited` };
  }
  // hf-inference retirement signal — model browsable on Hub but not deployed.
  if (
    (status === 400 || status === 404 || status === 410) &&
    (lower.includes('model not supported by provider') ||
      lower.includes('no longer supported') ||
      lower.includes('model not found') ||
      lower.includes('deprecated'))
  ) {
    return {
      state: 'model-unsupported',
      humanReason: `${status} model not available on this provider — update the model id in Settings`,
    };
  }
  return { state: 'other-error', humanReason: `${status} ${bodyText.slice(0, 120)}` };
}

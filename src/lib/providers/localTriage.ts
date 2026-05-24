/**
 * LOCAL-MODE TRIAGE PROVIDER (Milestone 22).
 *
 * Talks to the FastAPI server in training/server.py. On the local-mode path the
 * server returns a CALIBRATED `tb_prob` from the deployed TBHeadT2 (temperature-
 * scaled under T from data/tb_threshold_t2.json) plus an `s_inactive` from the
 * InactiveSequelaeHead (temperature-scaled under T_sequelae from data/tb_inactive_meta.json).
 * Those numbers are the ones the LODO experiments validated — the deployed
 * threshold (thr@95sens = 0.6105) is OURS, fit on a labeled validation split of
 * the training data, so it is the legitimate operating point to compare `tb_prob`
 * against. This is the single place in the project where the validated threshold
 * is allowed to do real work in the browser.
 *
 * ERROR PATHS (surfaced to providerStatusStore so the SettingsDrawer banner can
 * tell the user exactly what to do, never DevTools):
 *   - 'connection-refused' — fetch threw / server not running. Banner says
 *      `Run \`uvicorn training.server:app --port 8000\` in the repo root.`
 *   - 'server-error'       — 5xx from the server. We surface the JSON body so
 *      the user sees the actual engine_run_failed reason.
 *   - 'schema-error'       — the server returned 200 but the body doesn't match
 *      TriageResult. Indicates a contract drift — we bail rather than guess.
 *   - 'ok'                 — clean run.
 *
 * The provider intentionally does NOT collapse into `ClassifierResult` (the
 * normalized HF/Replicate shape). The local pathway carries calibration metadata,
 * s_inactive, image-quality warnings, and audit pins that the M21 orchestrator
 * combine doesn't model. Keep this its own type and convert at the seam.
 */
import type { Provider } from '@/lib/types';
import { providerStatusStore } from '@/store/providerStatus';
import { ProviderError } from './errors';

// ---------------------------------------------------------------------------
// Wire shape — MUST match training/triage_core.TriageResult.to_dict()
// (Python -> JSON). If you change one, change the other.
// ---------------------------------------------------------------------------
export interface LocalTriageAudit {
  model_id: string;
  model_sha: string;
  calibration: { T: number; thr_at_95sens: number; T_sequelae: number };
  git_sha: string;
  version: number;
  timestamp: string;
}

export interface LocalTriageResult {
  /** Calibrated TB probability under T (the validated head's temperature). */
  tb_prob: number;
  /** Raw model logit BEFORE Platt/temperature calibration. Pinned for the audit trail. */
  tb_logit: number;
  /** Calibrated s_inactive under T_sequelae. */
  s_inactive: number;
  /** Server-side deterministic verdict from sequelaeEscalation rule. */
  verdict: 'tb' | 'no_tb' | 'abstain';
  /** The threshold the server compared `tb_prob` to (typically thr@95sens=0.6105). */
  decided_at_threshold: number;
  /** Non-null when the safety net fired (e.g. scar-shape escalation). */
  safety_net_applied: string | null;
  /** Per-stage warnings the engine emitted while preprocessing (non-fatal). */
  image_quality: { warnings: string[] };
  /** ms per stage + total — recorded for the UI's trace panel. */
  latency_ms: Record<string, number>;
  audit: LocalTriageAudit;
}

export interface LocalHealthOk {
  ok: true;
  engine: 'ready';
  model_sha: string;
  git_sha: string;
  calibration: { T: number; thr_at_95sens: number; T_sequelae: number; version: number };
}

export interface LocalHealthDown {
  ok: false;
  engine: 'unreachable';
  reason: string;
  hint: string;
}

export type LocalHealth = LocalHealthOk | LocalHealthDown;

/**
 * Provider id used in `providerStatusStore` + the verdict trace. Keep in sync
 * with the Provider union in types.ts.
 */
export const LOCAL_TRIAGE_PROVIDER: Provider = 'local-triage';

// Soft fetch timeout — the engine targets sub-second on warm M4, so a 30s
// budget covers a cold first-load (Rad-DINO + TXRV load on first /triage) AND
// the small fan-in from the browser. Cold first request can hit 5-6 s.
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Subset of the TriageResult schema validator. We do NOT use a heavy schema
 * library here — strict TS gives us shape checking once the result is typed, and
 * keeping this hand-rolled means a single dep-free file the test suite can mock.
 */
function isLocalTriageResult(o: unknown): o is LocalTriageResult {
  if (o === null || typeof o !== 'object') return false;
  const r = o as Record<string, unknown>;
  if (typeof r.tb_prob !== 'number' || !Number.isFinite(r.tb_prob)) return false;
  if (typeof r.tb_logit !== 'number' || !Number.isFinite(r.tb_logit)) return false;
  if (typeof r.s_inactive !== 'number' || !Number.isFinite(r.s_inactive)) return false;
  if (r.verdict !== 'tb' && r.verdict !== 'no_tb' && r.verdict !== 'abstain') return false;
  if (typeof r.decided_at_threshold !== 'number') return false;
  if (r.safety_net_applied !== null && typeof r.safety_net_applied !== 'string') return false;
  const iq = r.image_quality as Record<string, unknown> | null;
  if (iq === null || typeof iq !== 'object' || !Array.isArray(iq.warnings)) return false;
  if (typeof r.latency_ms !== 'object' || r.latency_ms === null) return false;
  const audit = r.audit as Record<string, unknown> | null;
  if (audit === null || typeof audit !== 'object') return false;
  if (typeof audit.model_id !== 'string') return false;
  if (typeof audit.model_sha !== 'string') return false;
  if (typeof audit.git_sha !== 'string') return false;
  return true;
}

export class LocalTriageError extends ProviderError {
  constructor(message: string, status?: number, raw?: unknown) {
    super(message, LOCAL_TRIAGE_PROVIDER, status, raw);
    this.name = 'LocalTriageError';
  }
}

/**
 * Quick health probe. Used by the SettingsDrawer when the user toggles local
 * mode ON, so we can show "engine: ready · model_sha: ..." vs "engine: unreachable —
 * run uvicorn ...". Does NOT throw on a downed server — returns the typed
 * LocalHealthDown so the caller can render an actionable line.
 */
export async function localHealth(baseUrl: string = 'http://localhost:8000'): Promise<LocalHealth> {
  try {
    const resp = await fetch(`${baseUrl.replace(/\/+$/, '')}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!resp.ok) {
      // server up but health check failed (likely engine_unavailable 503)
      const body = await resp.text();
      providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
        state: 'server-error',
        note: `${resp.status} ${body.slice(0, 120)}`,
      });
      return {
        ok: false,
        engine: 'unreachable',
        reason: `HTTP ${resp.status}: ${body.slice(0, 200)}`,
        hint: 'Check the uvicorn shell for the engine load error.',
      };
    }
    const data = (await resp.json()) as LocalHealth;
    if (data.ok && data.engine === 'ready') {
      providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
        state: 'ok',
        note: `model_sha: ${data.model_sha.slice(0, 14)}… · git: ${data.git_sha.slice(0, 7)}`,
      });
      return data;
    }
    providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
      state: 'server-error',
      note: 'engine not ready',
    });
    return data;
  } catch (err) {
    const reason = (err as Error).message;
    providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
      state: 'connection-refused',
      note: reason,
    });
    return {
      ok: false,
      engine: 'unreachable',
      reason,
      hint: 'Run `PYTORCH_ENABLE_MPS_FALLBACK=1 training/.venv/bin/python -m uvicorn training.server:app --port 8000` in the repo root.',
    };
  }
}

export interface LocalTriageOptions {
  /** Hard deadline (ms). Default 30s. Cold-load Rad-DINO on first /triage can take ~5s. */
  timeoutMs?: number;
  /** Abort signal — propagates the orchestrator's cancellation. */
  signal?: AbortSignal;
}

/**
 * POST a CXR blob to the local FastAPI server's /triage endpoint and return a
 * typed LocalTriageResult. Updates the providerStatusStore on every call so the
 * SettingsDrawer banner is always honest about the last result.
 */
export async function localTriage(
  blob: Blob,
  baseUrl: string = 'http://localhost:8000',
  opts: LocalTriageOptions = {},
): Promise<LocalTriageResult> {
  const url = `${baseUrl.replace(/\/+$/, '')}/triage`;
  const form = new FormData();
  // Filename matters only because FastAPI's UploadFile requires `.filename` to be
  // non-empty before it accepts the field as a file.
  form.append('file', blob, 'cxr.bin');

  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const ac = new AbortController();
  // Compose AbortControllers: if the caller supplied one, abort ours when theirs aborts.
  if (opts.signal) {
    if (opts.signal.aborted) ac.abort();
    else opts.signal.addEventListener('abort', () => ac.abort(), { once: true });
  }
  const timeoutId = setTimeout(() => ac.abort(), timeoutMs);

  let resp: Response;
  try {
    resp = await fetch(url, { method: 'POST', body: form, signal: ac.signal });
  } catch (err) {
    clearTimeout(timeoutId);
    // fetch threw — connection refused (server down), DNS, TLS, CORS, or abort.
    // We tag all of those as 'connection-refused' because the actionable hint is
    // identical: start the server / verify the URL.
    const message = (err as Error).message || 'fetch failed';
    providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
      state: 'connection-refused',
      note: message,
    });
    throw new LocalTriageError(
      `connection refused: ${message} — is the local server running on ${baseUrl}?`,
    );
  }
  clearTimeout(timeoutId);

  if (!resp.ok) {
    // Try to surface the JSON body; fall back to text. 5xx is a true server error,
    // 4xx (bad image, bad base64) is also surfaced under server-error since from
    // the BROWSER side they're indistinguishable in remedy ("look at the shell").
    let bodyText = '';
    try {
      bodyText = await resp.text();
    } catch {
      // ignore — body unreadable
    }
    providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
      state: 'server-error',
      note: `${resp.status} ${bodyText.slice(0, 140)}`,
    });
    throw new LocalTriageError(
      `server error ${resp.status}: ${bodyText.slice(0, 200)}`,
      resp.status,
      bodyText,
    );
  }

  let data: unknown;
  try {
    data = await resp.json();
  } catch (err) {
    providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
      state: 'schema-error',
      note: 'response was not valid JSON',
    });
    throw new LocalTriageError(`schema error: response was not valid JSON: ${(err as Error).message}`);
  }

  if (!isLocalTriageResult(data)) {
    providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
      state: 'schema-error',
      note: 'response did not match TriageResult shape',
    });
    throw new LocalTriageError(
      'schema error: server response did not match TriageResult shape',
      200,
      data,
    );
  }

  providerStatusStore.set(LOCAL_TRIAGE_PROVIDER, {
    state: 'ok',
    note: `verdict: ${data.verdict} · ${data.latency_ms.total ?? 0}ms`,
  });
  return data;
}

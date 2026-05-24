/**
 * localTriage provider tests (Milestone 22).
 *
 * The four required behaviors:
 *   1. ok            — server returns a valid TriageResult; we parse it and tag
 *                       providerStatusStore['local-triage'] as 'ok'.
 *   2. schema-error  — server returns 200 but the body is missing/malformed; the
 *                       provider throws LocalTriageError and tags 'schema-error'.
 *   3. server-error  — server returns 5xx; the provider throws and tags
 *                       'server-error' with the body slice in `note`.
 *   4. connection-refused — fetch throws (server not running); the provider
 *                       throws and tags 'connection-refused' with the actionable
 *                       hint.
 *
 * Uses vi.fn() to mock fetch — no network, no Python server needed.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  LOCAL_TRIAGE_PROVIDER,
  LocalTriageError,
  localTriage,
  type LocalTriageResult,
} from './localTriage';
import { providerStatusStore } from '@/store/providerStatus';

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: init?.status ?? 200,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });
}

function validResultPayload(over: Partial<LocalTriageResult> = {}): LocalTriageResult {
  return {
    tb_prob: 0.92,
    tb_logit: 4.2,
    s_inactive: 0.1,
    verdict: 'tb',
    decided_at_threshold: 0.6105,
    safety_net_applied: null,
    image_quality: { warnings: [] },
    latency_ms: { harmonize: 8, seg: 120, rad_dino: 110, txrv: 60, heads: 5, total: 303 },
    audit: {
      model_id: 'tb_head_t2',
      model_sha: 'sha256:abc',
      calibration: { T: 1.5915, thr_at_95sens: 0.6105, T_sequelae: 1.1313 },
      git_sha: 'deadbee',
      version: 1,
      timestamp: '2026-05-24T00:00:00.000Z',
    },
    ...over,
  };
}

const fakeBlob = new Blob([new Uint8Array([1, 2, 3])], { type: 'image/jpeg' });

beforeEach(() => {
  providerStatusStore.reset();
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('localTriage — four error paths', () => {
  it('(1) ok: parses a well-formed TriageResult and tags providerStatusStore as ok', async () => {
    const payload = validResultPayload();
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(payload)));

    const r = await localTriage(fakeBlob, 'http://localhost:8000');
    expect(r.tb_prob).toBeCloseTo(0.92, 6);
    expect(r.verdict).toBe('tb');
    expect(r.audit.model_sha).toBe('sha256:abc');
    const status = providerStatusStore.get()[LOCAL_TRIAGE_PROVIDER];
    expect(status.state).toBe('ok');
    expect(status.note).toContain('verdict: tb');
  });

  it('(2) schema-error: 200 with missing required field throws and tags schema-error', async () => {
    // Strip the `verdict` field — the isLocalTriageResult guard fails on this.
    const bad = { ...validResultPayload(), verdict: undefined } as unknown;
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(bad)));

    await expect(localTriage(fakeBlob, 'http://localhost:8000')).rejects.toBeInstanceOf(
      LocalTriageError,
    );
    expect(providerStatusStore.get()[LOCAL_TRIAGE_PROVIDER].state).toBe('schema-error');
  });

  it('(2b) schema-error: 200 with non-JSON body throws and tags schema-error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('not json at all', { status: 200 })),
    );

    await expect(localTriage(fakeBlob, 'http://localhost:8000')).rejects.toBeInstanceOf(
      LocalTriageError,
    );
    expect(providerStatusStore.get()[LOCAL_TRIAGE_PROVIDER].state).toBe('schema-error');
  });

  it('(3) server-error: 503 from /triage throws and tags server-error with body slice', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: { error: 'engine_unavailable', reason: 'no weights' } }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    let err: unknown;
    try {
      await localTriage(fakeBlob, 'http://localhost:8000');
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(LocalTriageError);
    expect((err as LocalTriageError).status).toBe(503);
    const status = providerStatusStore.get()[LOCAL_TRIAGE_PROVIDER];
    expect(status.state).toBe('server-error');
    expect(status.note ?? '').toContain('engine_unavailable');
  });

  it('(4) connection-refused: fetch throws and we tag connection-refused with the hint message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      }),
    );

    let err: unknown;
    try {
      await localTriage(fakeBlob, 'http://localhost:8000');
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(LocalTriageError);
    expect((err as Error).message).toMatch(/connection refused/i);
    const status = providerStatusStore.get()[LOCAL_TRIAGE_PROVIDER];
    expect(status.state).toBe('connection-refused');
    expect(status.note).toContain('Failed to fetch');
  });
});

describe('localTriage — base URL normalization', () => {
  it('strips trailing slashes from baseUrl', async () => {
    let capturedUrl: string | null = null;
    const fetchMock = vi.fn(async (input: unknown) => {
      capturedUrl = typeof input === 'string' ? input : String(input);
      return jsonResponse(validResultPayload());
    });
    vi.stubGlobal('fetch', fetchMock);
    await localTriage(fakeBlob, 'http://localhost:8000///');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(capturedUrl).toBe('http://localhost:8000/triage');
  });
});

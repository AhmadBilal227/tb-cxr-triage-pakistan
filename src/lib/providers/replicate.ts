import { blobToDataURL, sleep } from '@/lib/utils';
import { ReplicateError } from './errors';

/**
 * Replicate API — per-stage FALLBACK.
 *
 * Predictions are asynchronous: create a prediction, then poll its status URL
 * until succeeded/failed. We cap the whole poll at 60s with exponential backoff.
 *
 * Auth: modern Replicate uses `Authorization: Bearer <token>`.
 *
 * CORS note: Replicate's REST API does not always send permissive CORS headers.
 * Direct browser calls may be blocked depending on account/edge config; this is
 * documented in the README "Known limits" section. We attempt the call directly
 * per the frontend-only constraint and surface any CORS failure in the trace.
 */

// Always go through a same-origin proxy: Replicate's REST API does not send permissive
// CORS headers, so direct browser calls are blocked. Dev = Vite proxy (vite.config.ts).
// Prod = Netlify rewrite (netlify.toml: /replicate/* -> api.replicate.com/:splat 200).
const REPLICATE_BASE = '/replicate/v1';
const MAX_POLL_MS = 60_000;
const POLL_START_MS = 700;
const POLL_MAX_INTERVAL_MS = 5_000;

type PredictionStatus = 'starting' | 'processing' | 'succeeded' | 'failed' | 'canceled';

interface Prediction {
  id: string;
  status: PredictionStatus;
  output?: unknown;
  error?: string | null;
  urls?: { get?: string };
}

export interface ReplicateResult {
  raw: unknown; // prediction.output
  latencyMs: number;
}

/**
 * Run a Replicate model (by version hash) against an image.
 * @param input extra input fields merged into the request (e.g. a different image key).
 */
export async function replicatePredict(
  token: string,
  version: string,
  image: Blob,
  input: Record<string, unknown> = {},
): Promise<ReplicateResult> {
  if (!token) throw new ReplicateError('Replicate token missing');
  if (!version) throw new ReplicateError('Replicate model version missing');

  const start = performance.now();
  const dataUrl = await blobToDataURL(image);

  // 1. Create the prediction.
  let created: Prediction;
  try {
    const res = await fetch(`${REPLICATE_BASE}/predictions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ version, input: { image: dataUrl, ...input } }),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new ReplicateError(
        `Replicate create ${res.status}: ${text.slice(0, 240)}`,
        res.status,
        text,
      );
    }
    created = (await res.json()) as Prediction;
  } catch (err) {
    if (err instanceof ReplicateError) throw err;
    throw new ReplicateError(
      `network error creating Replicate prediction (${(err as Error).message}). May be CORS — see README.`,
    );
  }

  // 2. Poll until terminal or 60s budget exhausted.
  // Derive the poll URL from our base (not prediction.urls.get, which is an absolute
  // api.replicate.com URL that would bypass the dev proxy and hit CORS).
  const getUrl = `${REPLICATE_BASE}/predictions/${created.id}`;
  let current = created;
  let interval = POLL_START_MS;

  while (current.status === 'starting' || current.status === 'processing') {
    if (performance.now() - start > MAX_POLL_MS) {
      throw new ReplicateError(`Replicate prediction exceeded 60s budget (id ${current.id})`);
    }
    await sleep(interval);
    interval = Math.min(interval * 1.6, POLL_MAX_INTERVAL_MS);

    const res = await fetch(getUrl, {
      headers: { Authorization: `Bearer ${token}` },
    }).catch((err: Error) => {
      throw new ReplicateError(`network error polling Replicate (${err.message})`);
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new ReplicateError(`Replicate poll ${res.status}: ${text.slice(0, 240)}`, res.status);
    }
    current = (await res.json()) as Prediction;
  }

  if (current.status !== 'succeeded') {
    throw new ReplicateError(
      `Replicate prediction ${current.status}: ${current.error ?? 'no detail'}`,
      undefined,
      current,
    );
  }

  return { raw: current.output, latencyMs: performance.now() - start };
}

import { sleep } from '@/lib/utils';
import { HfError } from './errors';
import { classifyHttpFailure, providerStatusStore } from '@/store/providerStatus';

/**
 * Hugging Face Inference API — PRIMARY perception layer.
 *
 * Cold-start protocol (per spec):
 *   - 503 response carries JSON `{ error, estimated_time }`.
 *   - Wait estimated_time + buffer, then retry. Max 3 retries.
 *   - Any non-2xx after retries OR a projected total wait > 20s => failure
 *     (caller falls back to Replicate).
 *
 * The budget check happens BEFORE sleeping so a model reporting a long
 * estimated_time fails fast instead of blocking the pipeline.
 */

// HuggingFace retired api-inference.huggingface.co in favor of the router.
// Serverless inference now lives under the hf-inference provider on the router.
const HF_BASE = 'https://router.huggingface.co/hf-inference/models';
const MAX_RETRIES = 3;
const MAX_TOTAL_WAIT_MS = 20_000;
const BUFFER_MS = 1500;

export interface HfResult {
  raw: unknown;
  latencyMs: number;
  /** total time we spent waiting on cold-start, for the trace note */
  coldStartWaitMs: number;
}

/**
 * Run image classification against a free serverless HF model.
 * Sends the raw image bytes as the request body (HF image pipelines accept binary).
 */
export async function hfImageInference(
  token: string,
  model: string,
  image: Blob,
): Promise<HfResult> {
  if (!token) {
    providerStatusStore.set('hf', { state: 'not-configured', note: 'no token' });
    throw new HfError('Hugging Face token missing');
  }
  if (!model) {
    providerStatusStore.set('hf', { state: 'not-configured', note: 'no model id' });
    throw new HfError('Hugging Face model id missing');
  }

  const url = `${HF_BASE}/${model}`;
  const start = performance.now();
  let totalWaitMs = 0;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    let res: Response;
    try {
      res = await fetch(url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: 'application/json',
          'Content-Type': image.type || 'application/octet-stream',
        },
        body: image,
      });
    } catch (err) {
      // Network / CORS failure — not retryable in a useful way.
      providerStatusStore.set('hf', {
        state: 'network',
        note: `network: ${(err as Error).message.slice(0, 80)}`,
      });
      throw new HfError(
        `network error calling HF (${(err as Error).message}). If this is CORS, the model may not be reachable from the browser.`,
      );
    }

    if (res.ok) {
      const raw: unknown = await res.json();
      providerStatusStore.set('hf', { state: 'ok', note: `model ${model}` });
      return {
        raw,
        latencyMs: performance.now() - start,
        coldStartWaitMs: totalWaitMs,
      };
    }

    if (res.status === 503) {
      // Model loading. Parse estimated_time and decide whether the wait fits the budget.
      const body = (await res.json().catch(() => ({}))) as {
        estimated_time?: number;
        error?: string;
      };
      const estSec = typeof body.estimated_time === 'number' ? body.estimated_time : 5;
      const waitMs = estSec * 1000 + BUFFER_MS;

      // Budget check BEFORE sleeping (spec: >20s total wait => fail).
      if (attempt === MAX_RETRIES || totalWaitMs + waitMs > MAX_TOTAL_WAIT_MS) {
        throw new HfError(
          `model loading (est ${estSec}s); projected wait ${(
            (totalWaitMs + waitMs) /
            1000
          ).toFixed(0)}s exceeds 20s budget — falling back`,
          503,
          body,
        );
      }

      totalWaitMs += waitMs;
      await sleep(waitMs);
      continue;
    }

    // Any other non-2xx is a hard failure for this provider.
    const text = await res.text().catch(() => '');
    const classified = classifyHttpFailure(res.status, text);
    providerStatusStore.set('hf', {
      state: classified.state,
      note: `${classified.humanReason} (model ${model})`,
    });
    throw new HfError(`HF ${res.status}: ${text.slice(0, 240)}`, res.status, text);
  }

  throw new HfError('HF retries exhausted');
}

/**
 * Call a user-supplied HF Inference Endpoint (dedicated, paid) for embeddings.
 * Used for google/cxr-foundation, which is NOT available on free serverless.
 * Endpoint URL and response shape are user-configured; we return the raw payload
 * and let the caller's parser extract the vector.
 */
export async function hfEndpointEmbedding(
  token: string,
  endpointUrl: string,
  image: Blob,
): Promise<HfResult> {
  if (!token) throw new HfError('Hugging Face token missing');
  if (!endpointUrl) throw new HfError('HF Inference Endpoint URL not configured');

  const start = performance.now();
  let res: Response;
  try {
    res = await fetch(endpointUrl, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/json',
        'Content-Type': image.type || 'application/octet-stream',
      },
      body: image,
    });
  } catch (err) {
    throw new HfError(`network error calling HF endpoint (${(err as Error).message})`);
  }

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    const classified = classifyHttpFailure(res.status, text);
    providerStatusStore.set('hf', {
      state: classified.state,
      note: `endpoint: ${classified.humanReason}`,
    });
    throw new HfError(`HF endpoint ${res.status}: ${text.slice(0, 240)}`, res.status, text);
  }
  const raw: unknown = await res.json();
  providerStatusStore.set('hf', { state: 'ok', note: 'endpoint embedding' });
  return { raw, latencyMs: performance.now() - start, coldStartWaitMs: 0 };
}

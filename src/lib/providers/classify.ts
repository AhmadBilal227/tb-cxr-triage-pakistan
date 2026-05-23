import type { ClassifierResult, ClassifierProvider, Provider, StageConfig } from '@/lib/types';
import { clamp } from '@/lib/utils';
import { hfImageInference, hfEndpointEmbedding } from './hf';
import { replicatePredict } from './replicate';
import { parseEmbedding } from './parsers';
import { NotConfiguredError, ProviderError } from './errors';

export interface ClassifyDeps {
  hfToken: string;
  replicateToken: string;
  /** Called when the primary (HF) fails and we attempt the Replicate fallback. */
  onFallback?: (from: Provider, to: Provider, reason: string) => void;
}

/**
 * The core provider abstraction the spec requires.
 *
 * Tries the HF primary; on any failure, falls back to the configured Replicate
 * model (if a token + slug exist). Normalizes both to ClassifierResult. Fallback
 * is announced via deps.onFallback so the trace UI can animate it.
 */
export async function classifyWithFallback(
  image: Blob,
  stage: StageConfig,
  deps: ClassifyDeps,
): Promise<ClassifierResult> {
  // --- Primary: Hugging Face ---
  try {
    const hf = await hfImageInference(deps.hfToken, stage.primary.model, image);
    return {
      tb_prob: clamp(stage.primary.parseOutput(hf.raw), 0, 1),
      raw: hf.raw,
      provider_used: 'hf',
      latency_ms: hf.latencyMs,
    };
  } catch (primaryErr) {
    const reason = (primaryErr as Error).message;

    // --- Fallback: Replicate ---
    if (!stage.fallback) {
      throw new ProviderError(
        `HF failed (${reason}) and no Replicate fallback is configured for this stage.`,
        'hf',
      );
    }
    if (!deps.replicateToken) {
      throw new NotConfiguredError(
        `HF failed (${reason}) and Replicate token is missing — fallback disabled.`,
      );
    }
    deps.onFallback?.('hf', 'replicate', reason);

    const rep = await replicatePredict(deps.replicateToken, stage.fallback.version, image);
    return {
      tb_prob: clamp(stage.fallback.parseOutput(rep.raw), 0, 1),
      raw: rep.raw,
      provider_used: 'replicate',
      latency_ms: rep.latencyMs,
    };
  }
}

export interface EmbedConfig {
  hfToken: string;
  replicateToken: string;
  endpointUrl: string; // HF Inference Endpoint for cxr-foundation
  replicateModel: string; // CLIP slug (informational)
  replicateVersion: string;
  onFallback?: (from: Provider, to: Provider, reason: string) => void;
}

export interface EmbedResult {
  embedding: number[];
  provider_used: ClassifierProvider;
  latency_ms: number;
}

/**
 * Embedding with HF Inference Endpoint primary -> Replicate CLIP fallback.
 * Throws NotConfiguredError if neither is configured (Stage 3 then skips with a banner).
 */
export async function embedWithFallback(image: Blob, cfg: EmbedConfig): Promise<EmbedResult> {
  const hasEndpoint = cfg.endpointUrl.trim().length > 0;
  const hasReplicate = cfg.replicateToken.trim().length > 0 && cfg.replicateVersion.trim().length > 0;

  if (!hasEndpoint && !hasReplicate) {
    throw new NotConfiguredError(
      'No embedding provider configured (set the CXR-Foundation HF Inference Endpoint URL or a Replicate CLIP fallback).',
    );
  }

  if (hasEndpoint) {
    try {
      const hf = await hfEndpointEmbedding(cfg.hfToken, cfg.endpointUrl, image);
      const embedding = parseEmbedding(hf.raw);
      if (embedding.length === 0) throw new Error('endpoint returned no parseable embedding vector');
      return { embedding, provider_used: 'hf', latency_ms: hf.latencyMs };
    } catch (err) {
      if (!hasReplicate) throw err;
      cfg.onFallback?.('hf', 'replicate', (err as Error).message);
    }
  }

  // Replicate CLIP fallback (or sole provider).
  const rep = await replicatePredict(cfg.replicateToken, cfg.replicateVersion, image);
  const embedding = parseEmbedding(rep.raw);
  if (embedding.length === 0) throw new ProviderError('Replicate returned no parseable embedding', 'replicate');
  return { embedding, provider_used: 'replicate', latency_ms: rep.latencyMs };
}

// ---------------------------------------------------------------------------
// Vector math for kNN retrieval (Stage 3)
// ---------------------------------------------------------------------------

export function cosineSimilarity(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

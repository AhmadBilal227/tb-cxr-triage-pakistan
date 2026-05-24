import type { Provider } from '@/lib/types';
import { replicatePredict } from './replicate';
import { parseEmbedding } from './parsers';
import { NotConfiguredError, ProviderError } from './errors';

/**
 * RAG embedding + vector math.
 *
 * Milestone 23 removed Hugging Face from the runtime path. The classifier
 * fallback (`classifyWithFallback`) was deleted alongside it — the perception
 * pipeline is now single-path (local-triage primary, gpt-5.5 vision primary on
 * deploy, Replicate as the only configurable BYO classifier slot wired directly
 * from the orchestrator). What remains here is the retrieval embedding path —
 * the kNN corpus uses a Replicate CLIP model when one is configured.
 *
 * `embedWithFallback` keeps its name for API compatibility but is now a single-
 * provider path (Replicate). The "fallback" parameter list is preserved so the
 * orchestrator's onFallback callback signature stays stable — fallbacks no
 * longer fire, but a future BYO embedding provider can slot in here.
 */

export interface EmbedConfig {
  replicateToken: string;
  replicateModel: string; // CLIP slug (informational)
  replicateVersion: string;
  onFallback?: (from: Provider, to: Provider, reason: string) => void;
}

export interface EmbedResult {
  embedding: number[];
  provider_used: 'replicate';
  latency_ms: number;
}

/**
 * Embedding via the Replicate CLIP model. Throws NotConfiguredError if the
 * Replicate token or version is missing (Stage 3 then skips with a banner).
 */
export async function embedWithFallback(image: Blob, cfg: EmbedConfig): Promise<EmbedResult> {
  const hasReplicate =
    cfg.replicateToken.trim().length > 0 && cfg.replicateVersion.trim().length > 0;

  if (!hasReplicate) {
    throw new NotConfiguredError(
      'No embedding provider configured (set a Replicate token + CLIP model in Settings).',
    );
  }

  const rep = await replicatePredict(cfg.replicateToken, cfg.replicateVersion, image);
  const embedding = parseEmbedding(rep.raw);
  if (embedding.length === 0) {
    throw new ProviderError('Replicate returned no parseable embedding', 'replicate');
  }
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

import type { PipelineRun } from './types';

export const DISCLAIMER =
  'Research preview. Not a medical device. Not for diagnostic use. ' +
  'Outputs are produced by third-party AI models called directly from the browser and must not ' +
  'inform any clinical decision.';

export interface SessionExport {
  disclaimer: string;
  generatedAt: string;
  app: string;
  runs: ExportedRun[];
  session_fallback_rate: number;
}

interface ExportedRun {
  id: string;
  imageName: string;
  timestamp: string;
  verdict: string | null;
  confidence: number | null;
  auto_abstained: boolean | null;
  fallback_rate: number;
  model_versions: Record<string, string>;
  provider_log: PipelineRun['providerLog'];
  ensemble: {
    weighted_score: number;
    std: number;
    disagreement: number;
    members: { id: string; tb_prob: number | null; provider_used: string | null }[];
  } | null;
  retrieval: { embedding_provider: string | null; skipped: boolean; neighbors: { label: number; similarity: number }[] } | null;
  halted: { reason: string; stage: string } | null;
}

export function buildSessionExport(runs: PipelineRun[]): SessionExport {
  const exported: ExportedRun[] = runs.map((r) => ({
    id: r.id,
    imageName: r.imageName,
    timestamp: new Date(r.createdAt).toISOString(),
    verdict: r.adjudication?.verdict ?? null,
    confidence: r.adjudication?.confidence ?? null,
    auto_abstained: r.adjudication?.auto_abstained ?? null,
    fallback_rate: r.fallbackRate,
    model_versions: r.modelVersions,
    provider_log: r.providerLog,
    ensemble: r.ensemble
      ? {
          weighted_score: r.ensemble.weightedScore,
          std: r.ensemble.std,
          disagreement: r.ensemble.disagreement,
          members: r.ensemble.members.map((m) => ({
            id: m.id,
            tb_prob: m.tb_prob,
            provider_used: m.provider_used,
          })),
        }
      : null,
    retrieval: r.rag
      ? {
          embedding_provider: r.rag.embedding_provider,
          skipped: r.rag.skipped,
          neighbors: r.rag.neighbors.map((n) => ({ label: n.label, similarity: n.similarity })),
        }
      : null,
    halted: r.halted,
  }));

  const sessionFallback =
    runs.length > 0 ? runs.reduce((a, r) => a + r.fallbackRate, 0) / runs.length : 0;

  return {
    disclaimer: DISCLAIMER,
    generatedAt: new Date().toISOString(),
    app: 'tb-triage research preview',
    runs: exported,
    session_fallback_rate: sessionFallback,
  };
}

export function downloadJSON(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  triggerDownload(blob, filename);
}

export function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

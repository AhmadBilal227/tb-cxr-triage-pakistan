import type { Settings } from './types';
import { embedWithFallback } from './providers/classify';
import { addLabeledCase } from './db';

/**
 * Labeled set format (per README):
 *   - a CSV with header `filename,label` where label ∈ {0,1}
 *   - the referenced image files
 * Imported together via a multi-file / folder picker; we match CSV rows to files
 * by basename, then (if an embedding provider is configured) compute + store each
 * embedding so the case becomes a kNN candidate.
 */

export function parseLabelCsv(text: string): Map<string, 0 | 1> {
  const map = new Map<string, 0 | 1>();
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (const line of lines) {
    const [rawName, rawLabel] = line.split(',').map((c) => c.trim());
    if (!rawName || rawLabel === undefined) continue;
    if (/^filename$/i.test(rawName)) continue; // header
    const base = basename(rawName);
    const label = rawLabel === '1' ? 1 : rawLabel === '0' ? 0 : null;
    if (label === null) continue;
    map.set(base, label);
  }
  return map;
}

function basename(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] ?? path;
}

export interface ImportProgress {
  done: number;
  total: number;
  current: string;
  embedded: boolean;
}

export interface ImportSummary {
  imported: number;
  embedded: number;
  failed: number;
  skippedNoMatch: number;
  embeddingConfigured: boolean;
}

/**
 * Import a labeled set into the RAG corpus.
 * @param source 'import' for labeled sets; the verdict-card feedback uses addLabeledCase directly.
 */
export async function importLabeledSet(
  files: File[],
  settings: Settings,
  onProgress?: (p: ImportProgress) => void,
): Promise<ImportSummary> {
  const csvFile = files.find((f) => f.name.toLowerCase().endsWith('.csv'));
  if (!csvFile) throw new Error('No .csv found in selection. Include a CSV with `filename,label` rows.');

  const labels = parseLabelCsv(await csvFile.text());
  const imageFiles = files.filter((f) => /\.(png|jpe?g|webp|bmp|gif)$/i.test(f.name));
  const byName = new Map(imageFiles.map((f) => [basename(f.name), f]));

  const embeddingConfigured =
    settings.overrides.embeddingEndpointUrl.trim().length > 0 ||
    (settings.replicateToken.trim().length > 0 &&
      settings.overrides.embeddingReplicate.trim().length > 0);

  const summary: ImportSummary = {
    imported: 0,
    embedded: 0,
    failed: 0,
    skippedNoMatch: 0,
    embeddingConfigured,
  };

  const entries = [...labels.entries()];
  let done = 0;
  for (const [filename, label] of entries) {
    const file = byName.get(filename);
    done++;
    if (!file) {
      summary.skippedNoMatch++;
      onProgress?.({ done, total: entries.length, current: filename, embedded: false });
      continue;
    }

    let embedding: number[] | null = null;
    let provider: 'hf' | 'replicate' | null = null;
    if (embeddingConfigured) {
      try {
        const e = await embedWithFallback(file, {
          hfToken: settings.hfToken,
          replicateToken: settings.replicateToken,
          endpointUrl: settings.overrides.embeddingEndpointUrl,
          replicateModel: settings.overrides.embeddingReplicate,
          replicateVersion: settings.overrides.embeddingReplicateVersion,
        });
        embedding = e.embedding;
        provider = e.provider_used;
        summary.embedded++;
      } catch {
        summary.failed++;
      }
    }

    await addLabeledCase({
      filename,
      blob: file,
      embedding,
      embedding_provider: provider,
      label,
      source: 'import',
    });
    summary.imported++;
    onProgress?.({ done, total: entries.length, current: filename, embedded: embedding !== null });
  }

  return summary;
}

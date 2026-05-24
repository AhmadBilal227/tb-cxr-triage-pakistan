/**
 * Tolerant output parsers for the embedding + (optional) Replicate classifier slot.
 *
 * Milestone 23 removed Hugging Face. The two parsers that were specific to the
 * HF heads (`parseTbProb` over a label/score array, `parseGeneralCxrTbProb`)
 * were deleted along with the HF provider — the orchestrator no longer routes
 * through them. What remains is the embedding shape extractor (used by the
 * Replicate CLIP path for kNN retrieval) and the detection-box extractor (kept
 * so a future BYO Replicate detection slot can surface bounding boxes through
 * the existing UI seam).
 */

/** Extract a YOLOv8 bounding box if a detection model returned detections. */
export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
  label?: string;
  score?: number;
}

export function parseBoxes(raw: unknown): BBox[] {
  const out: BBox[] = [];
  const pushFrom = (arr: unknown[]): void => {
    for (const b of arr) {
      if (!b || typeof b !== 'object') continue;
      const o = b as Record<string, unknown>;
      // [x1,y1,x2,y2] under various keys
      if (
        typeof o.xmin === 'number' &&
        typeof o.ymin === 'number' &&
        typeof o.xmax === 'number' &&
        typeof o.ymax === 'number'
      ) {
        out.push({
          x: o.xmin,
          y: o.ymin,
          w: o.xmax - o.xmin,
          h: o.ymax - o.ymin,
          label: typeof o.label === 'string' ? o.label : undefined,
          score: typeof o.score === 'number' ? o.score : undefined,
        });
      }
    }
  };
  if (raw && typeof raw === 'object') {
    const o = raw as Record<string, unknown>;
    if (Array.isArray(o.boxes)) pushFrom(o.boxes);
    if (Array.isArray(o.predictions)) pushFrom(o.predictions);
  }
  if (Array.isArray(raw)) pushFrom(raw);
  return out;
}

/** Extract an embedding vector from various provider shapes. */
export function parseEmbedding(raw: unknown): number[] {
  // Flat numeric array
  if (Array.isArray(raw) && raw.every((x) => typeof x === 'number')) {
    return raw as number[];
  }
  // Nested [[...]] (feature-extraction APIs often return [seq][dim] or [1][dim])
  if (Array.isArray(raw) && Array.isArray(raw[0])) {
    const first = raw[0] as unknown[];
    if (first.every((x) => typeof x === 'number')) return first as number[];
    // [tokens][dim] -> mean-pool
    if (Array.isArray(first[0])) {
      const rows = raw as number[][];
      const dim = (rows[0] ?? []).length;
      const mean = new Array<number>(dim).fill(0);
      for (const row of rows) {
        for (let i = 0; i < dim; i++) {
          mean[i] = (mean[i] ?? 0) + (row[i] ?? 0) / rows.length;
        }
      }
      return mean;
    }
  }
  if (raw && typeof raw === 'object') {
    const o = raw as Record<string, unknown>;
    if (Array.isArray(o.embedding) && o.embedding.every((x) => typeof x === 'number')) {
      return o.embedding as number[];
    }
    // OpenAI-style { data: [{ embedding: [...] }] }
    if (Array.isArray(o.data) && o.data[0] && typeof o.data[0] === 'object') {
      const e = (o.data[0] as { embedding?: unknown }).embedding;
      if (Array.isArray(e) && e.every((x) => typeof x === 'number')) return e as number[];
    }
  }
  return [];
}

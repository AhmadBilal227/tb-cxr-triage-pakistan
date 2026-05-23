import { clamp } from '@/lib/utils';

/**
 * Tolerant output parsers. BYOK means we cannot assume a fixed label vocabulary,
 * so each parser inspects several plausible shapes and degrades gracefully.
 */

interface LabelScore {
  label: string;
  score: number;
}

function asLabelScores(raw: unknown): LabelScore[] {
  if (Array.isArray(raw)) {
    const out: LabelScore[] = [];
    for (const item of raw) {
      if (
        item &&
        typeof item === 'object' &&
        'label' in item &&
        'score' in item &&
        typeof (item as LabelScore).score === 'number'
      ) {
        out.push({ label: String((item as LabelScore).label), score: (item as LabelScore).score });
      }
    }
    return out;
  }
  return [];
}

const TB_LABEL = /tub|(^|[^a-z])tb([^a-z]|$)|positive|abnormal|^1$/i;
const NEG_LABEL = /normal|negative|healthy|^0$/i;

/** TB-specific classifier (Stage 2A). Returns probability of TB in [0,1]. */
export function parseTbProb(raw: unknown): number {
  const scores = asLabelScores(raw);
  if (scores.length > 0) {
    const tb = scores.find((s) => TB_LABEL.test(s.label));
    if (tb) return clamp(tb.score, 0, 1);
    const neg = scores.find((s) => NEG_LABEL.test(s.label));
    if (neg) return clamp(1 - neg.score, 0, 1);
    // Unknown labels: assume highest-scoring class is the "positive" one only if 2-class.
    if (scores.length === 2) {
      const sorted = [...scores].sort((a, b) => b.score - a.score);
      return clamp(sorted[0]?.score ?? 0, 0, 1);
    }
  }
  // Direct numeric probability.
  if (typeof raw === 'number') return clamp(raw, 0, 1);
  if (raw && typeof raw === 'object') {
    const o = raw as Record<string, unknown>;
    for (const k of ['tb_prob', 'probability', 'score', 'confidence']) {
      if (typeof o[k] === 'number') return clamp(o[k] as number, 0, 1);
    }
  }
  return 0;
}

/**
 * General CXR classifier (Stage 2B). Extracts a TB-related signal from a broader
 * pathology distribution. If no TB-related label exists, returns 0 (no TB signal
 * from this head) rather than fabricating one.
 */
export function parseGeneralCxrTbProb(raw: unknown): number {
  const scores = asLabelScores(raw);
  if (scores.length > 0) {
    const tb = scores.find((s) => /tubercul/i.test(s.label));
    if (tb) return clamp(tb.score, 0, 1);
    // Some CXR heads emit a generic "abnormal"/"finding" class — use as a weak proxy.
    const abn = scores.find((s) => /abnormal|finding|consolidation|opacity|nodule/i.test(s.label));
    if (abn) return clamp(abn.score * 0.6, 0, 1); // discounted: not TB-specific
    return 0;
  }
  // YOLOv8 detection style: { boxes: [...] } — presence of relevant detections.
  if (raw && typeof raw === 'object' && 'boxes' in (raw as object)) {
    const boxes = (raw as { boxes?: unknown[] }).boxes;
    if (Array.isArray(boxes) && boxes.length > 0) return 0.4;
    return 0;
  }
  return 0;
}

/** Extract a YOLOv8 bounding box if the general classifier returned detections. */
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
  // Nested [[...]] (HF feature-extraction often returns [seq][dim] or [1][dim])
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

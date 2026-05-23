import type { JsonSchemaFormat } from '@/lib/providers/openai';
import type { EnsembleResult, RagResult, Verdict } from '@/lib/types';

// --- Stage 1: Quality gate ---
export const QUALITY_PROMPT =
  'Is this a frontal chest X-ray suitable for TB screening? ' +
  'Reply JSON only: {is_cxr: bool, quality: "good"|"poor"|"unreadable", reason: string}';

export const QUALITY_SCHEMA: JsonSchemaFormat = {
  name: 'quality_gate',
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      is_cxr: { type: 'boolean' },
      quality: { type: 'string', enum: ['good', 'poor', 'unreadable'] },
      reason: { type: 'string' },
    },
    required: ['is_cxr', 'quality', 'reason'],
  },
};

// --- Stage 2C: VLM ensemble member ---
// Structured chain-of-thought, zone-by-zone read with a screening prior, returning a
// directly verbalized calibrated probability (arXiv:2508.12455, 2312.04344). The VLM
// is called K times (self-consistency) and the mean is used; the spread is uncertainty.
export interface VlmResult {
  reasoning: string;
  findings: string[];
  tb_probability: number; // 0..1
  image_quality?: 'good' | 'limited' | 'unreadable';
}

export const VLM_PROMPT = [
  'You are a sensitive safety-net reader for tuberculosis screening in a high-prevalence setting.',
  'Systematically inspect each lung zone (right and left upper, mid, lower) plus the hila, mediastinum and pleura.',
  'TB-associated signs: upper-zone consolidation, cavitation, miliary or nodular pattern, hilar or',
  'mediastinal lymphadenopathy, pleural effusion, fibrosis or volume loss, tree-in-bud.',
  'Reason step by step about each zone, then give a calibrated probability that this radiograph shows tuberculosis.',
  'This is a SCREEN: subtle or equivocal findings matter — do NOT dismiss them, and when genuinely',
  'uncertain lean toward a higher probability rather than clearing the patient.',
  'Respond JSON only: {reasoning: string, findings: string[], tb_probability: number between 0 and 1, image_quality: "good"|"limited"|"unreadable"}',
].join(' ');

export const VLM_SCHEMA: JsonSchemaFormat = {
  name: 'vlm_read',
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      reasoning: { type: 'string' },
      findings: { type: 'array', items: { type: 'string' } },
      tb_probability: { type: 'number' },
      image_quality: { type: 'string', enum: ['good', 'limited', 'unreadable'] },
    },
    required: ['reasoning', 'findings', 'tb_probability', 'image_quality'],
  },
};

/** Map a VLM read into a tb_prob in [0,1] (now a directly verbalized probability). */
export function vlmToProb(v: VlmResult): number {
  return Math.max(0, Math.min(1, Number(v.tb_probability)));
}

// --- Stage 4: Adjudicator ---
export interface AdjudicationRaw {
  verdict: Verdict;
  confidence: number; // 0..100
  rationale: string;
  abstain_reason?: string;
}

export interface AdjudicatorContext {
  ensemble: EnsembleResult;
  rag: RagResult;
  providersUsed: string[];
  replicateFallbackCount: number;
}

export function buildAdjudicatorPrompt(ctx: AdjudicatorContext): string {
  const members = ctx.ensemble.members.map((m) => ({
    model: m.label,
    tb_prob: m.tb_prob,
    provider_used: m.provider_used,
    status: m.status,
    findings: m.findings,
  }));
  const neighbors = ctx.rag.neighbors.map((n) => ({
    filename: n.filename,
    label: n.label === 1 ? 'TB' : 'NOT_TB',
    similarity: Number(n.similarity.toFixed(3)),
  }));

  return [
    'You are a cautious TB chest X-ray triage adjudicator acting as a safety net. You are',
    'shown the query chest X-ray image plus the outputs of independent perception models, a',
    'weighted ensemble score, and the most similar labeled cases retrieved by embedding',
    'similarity. This is a SCREEN: missing TB is far worse than a false alarm. When signals',
    'conflict or evidence is weak, prefer "abstain" (refer to a human) over "no_tb" — never',
    'clear a patient on weak evidence. Only return "no_tb" when the image is clearly normal',
    'and the models agree.',
    '',
    'ENSEMBLE MEMBERS:',
    JSON.stringify(members, null, 2),
    '',
    `WEIGHTED ENSEMBLE SCORE (0..1): ${ctx.ensemble.weightedScore.toFixed(3)}`,
    `ENSEMBLE STD: ${ctx.ensemble.std.toFixed(3)}`,
    `ENSEMBLE DISAGREEMENT (max-min): ${ctx.ensemble.disagreement.toFixed(3)}`,
    '',
    'NEAREST LABELED CASES (cosine similarity):',
    neighbors.length ? JSON.stringify(neighbors, null, 2) : '(none — retrieval skipped or empty corpus)',
    '',
    `PROVIDERS USED PER STAGE: ${ctx.providersUsed.join(', ') || 'n/a'}`,
    `STAGES THAT FELL BACK TO REPLICATE: ${ctx.replicateFallbackCount}`,
    '',
    'Respond JSON only:',
    '{verdict: "tb"|"no_tb"|"abstain", confidence: 0-100, rationale: string (<=3 sentences), abstain_reason?: string}',
  ].join('\n');
}

export const ADJUDICATION_SCHEMA: JsonSchemaFormat = {
  name: 'adjudication',
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      verdict: { type: 'string', enum: ['tb', 'no_tb', 'abstain'] },
      confidence: { type: 'number' },
      rationale: { type: 'string' },
      abstain_reason: { type: 'string' },
    },
    required: ['verdict', 'confidence', 'rationale'],
  },
};

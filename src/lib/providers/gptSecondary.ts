/**
 * GPT SECONDARY OBSERVATIONS (Phase B, post-M24).
 *
 * The third (and most cautious) GPT-after-our-model stage. Sibling to
 * M22 (`vlmTriage` borderline verifier) and M24 (`gptInterpreter`
 * radiology-report narrator). This stage runs gpt-5.5 vision ONCE on the
 * raw X-ray and asks for structured observations the TB-specific trained
 * head ignores BY DESIGN:
 *
 *   1. image_quality        — projection, rotation, exposure, motion
 *   2. support_devices      — ET tubes, central lines, pacemakers, leads
 *   3. cardiomediastinal    — silhouette / mediastinal / hilar gestalt
 *   4. other_incidentals    — non-TB findings (fractures, masses, etc.)
 *
 * Hard contract: this stage NEVER influences the verdict. The pipeline
 * already produced it; the safety net (M19 sequelae + M21 VLM verifier +
 * M26 asymmetric-evidence) already protected it. This stage's only job is
 * to surface side information that would otherwise be discarded.
 *
 * Honesty pin: every rendered observation carries an explicit "VLM
 * observation — not validated, advisory only" disclosure. Per the
 * literature (Jiang et al. 2024: GPT-4V cannot reliably generate de novo
 * radiology reports), this stage WILL miss findings and WILL occasionally
 * invent them. Calling that out in the UI is non-negotiable.
 *
 * Cost / latency: ~$0.01-0.04 per call, ~5-15s. On-demand only.
 *
 * Audit pins: prompt_hash + schema_version + model_id_from_response in
 * the return envelope so portfolio export can join a secondary-obs result
 * to the exact prompt+schema that produced it.
 */
import type { JsonSchemaFormat } from '@/lib/providers/openai';
import { openaiJSON } from '@/lib/providers/openai';
import { OpenAIError } from '@/lib/providers/errors';
import type { LocalTriageResult } from './localTriage';

export const GPT_SECONDARY_SCHEMA_VERSION = 'gpt-secondary-v1' as const;

export interface ImageQualityObservation {
  /** True iff the image is adequate for primary interpretation without caveats. */
  adequate: boolean;
  /** Specific concerns the radiologist should know about. Empty when adequate. */
  concerns: string[];
}

export interface SecondaryObservations {
  image_quality: ImageQualityObservation;
  /** Lines, tubes, pacemakers, ECG leads, surgical clips. Empty if none seen. */
  support_devices: string[];
  /** Cardiomediastinal observations beyond TB scope (silhouette size, mediastinal width, hilar gestalt). */
  cardiomediastinal_notes: string[];
  /** Non-TB incidental findings (fractures, masses, prior surgery, etc.). Empty if none. */
  other_incidentals: string[];
  /** What the VLM explicitly could not assess; surfaced for the radiologist. */
  limitations: string[];
}

export const GPT_SECONDARY_SCHEMA: JsonSchemaFormat = {
  name: 'secondary_observations',
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      image_quality: {
        type: 'object',
        additionalProperties: false,
        properties: {
          adequate: { type: 'boolean' },
          concerns: { type: 'array', items: { type: 'string' } },
        },
        required: ['adequate', 'concerns'],
      },
      support_devices: { type: 'array', items: { type: 'string' } },
      cardiomediastinal_notes: { type: 'array', items: { type: 'string' } },
      other_incidentals: { type: 'array', items: { type: 'string' } },
      limitations: { type: 'array', items: { type: 'string' } },
    },
    required: [
      'image_quality',
      'support_devices',
      'cardiomediastinal_notes',
      'other_incidentals',
      'limitations',
    ],
  },
};

/**
 * System prompt — extremely conservative tone. The model is being asked to
 * inspect the image for things OUTSIDE the TB-classifier's scope; this is
 * the highest-hallucination-risk stage of the pipeline, so the prompt
 * trades recall for precision (be wrong-by-omission, never invent).
 *
 * Grounding source: Fleischner Society terminology + MAIRA-2-style
 * conservative-listing posture + Jiang et al. 2024 (GPT-4V false-positive
 * findings are the dominant failure mode).
 */
export const GPT_SECONDARY_PROMPT = [
  'You are a board-certified radiologist running a SECONDARY observation pass on a single frontal chest radiograph. A validated AI screening pipeline has already produced the TB verdict. You DO NOT diagnose TB; that has been decided. Your job is to surface NON-TB information the TB-specific head ignores by design, so a clinician reviewing the case has the full picture.',
  '',
  'TASK',
  'List observations in four categories:',
  '- image_quality: projection (AP / PA / lordotic), rotation, exposure, motion blur, lung-field cutoff, scapular overlay. Set adequate=true when the image is adequate for primary interpretation without caveats.',
  '- support_devices: tubes, lines, leads, clips, pacemakers, prosthetic hardware. Specify position when visible (e.g. "ET tube tip approximately at the carina").',
  '- cardiomediastinal_notes: cardiac silhouette size, mediastinal width, hilar configuration, aortic contour — observations beyond the TB-specific head\'s scope. Skip if all unremarkable.',
  '- other_incidentals: NON-TB findings — fractures, mass lesions, post-surgical changes, calcifications, gas patterns, foreign bodies. Skip TB-suggestive findings (those belong in the main report).',
  '',
  'CONSERVATIVE POSTURE (HIGHEST PRIORITY)',
  '- This is the highest-hallucination-risk stage of the pipeline (Jiang et al. 2024: GPT-4V false-positive findings are the dominant failure mode). PREFER WRONG-BY-OMISSION OVER INVENTION.',
  '- Only list observations you can clearly identify. If unsure, omit and add a line to "limitations" instead ("cardiac silhouette not clearly assessable due to AP projection magnification").',
  '- Each item is one short sentence. Use Fleischner Society terminology.',
  '- An empty array in any category is fine and expected. Empty support_devices when none seen. Empty other_incidentals when none seen.',
  '',
  'NEGATIVE CONSTRAINTS',
  '- Do NOT comment on TB likelihood. Do not say "consistent with TB", "suspicious for TB", or similar. The TB verdict is already produced.',
  '- Do NOT invent measurements (no cardiothoracic ratios, no lesion sizes in cm — VLM cannot reliably measure).',
  '- Do NOT speculate on patient history, symptoms, or demographics.',
  '- Do NOT use first-person voice.',
  '',
  'Output ONLY the JSON schema fields. No chain-of-thought. No commentary.',
].join('\n');

function fnv1aHex(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export const GPT_SECONDARY_PROMPT_HASH = fnv1aHex(GPT_SECONDARY_PROMPT);
export const GPT_SECONDARY_SCHEMA_HASH = fnv1aHex(JSON.stringify(GPT_SECONDARY_SCHEMA.schema));

// ---------------------------------------------------------------------------
// Validator — clamps malformed output to a safe shape.
// ---------------------------------------------------------------------------
function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === 'string');
}

function clampStringArray(v: unknown): string[] {
  return isStringArray(v) ? v.filter((s) => s.length > 0).slice(0, 20) : [];
}

function normalizeImageQuality(raw: unknown): ImageQualityObservation {
  const r = (raw ?? {}) as Record<string, unknown>;
  const adequate = typeof r.adequate === 'boolean' ? r.adequate : true;
  const concerns = clampStringArray(r.concerns);
  return { adequate, concerns };
}

export function validateAndNormalize(raw: unknown): SecondaryObservations {
  const r = (raw ?? {}) as Record<string, unknown>;
  return {
    image_quality: normalizeImageQuality(r.image_quality),
    support_devices: clampStringArray(r.support_devices),
    cardiomediastinal_notes: clampStringArray(r.cardiomediastinal_notes),
    other_incidentals: clampStringArray(r.other_incidentals),
    limitations: clampStringArray(r.limitations),
  };
}

// ---------------------------------------------------------------------------
// Audit + call shape
// ---------------------------------------------------------------------------
export interface GptSecondaryAuditPins {
  prompt_hash: string;
  schema_version: string;
  schema_hash: string;
  model_id_from_response: string;
}

export interface GptSecondaryCall {
  observations: SecondaryObservations;
  audit: GptSecondaryAuditPins;
  latencyMs: number;
  fellBack: boolean;
  modelUsed: string;
}

export interface GptSecondaryOpts {
  apiKey: string;
  primaryModel: string;
  fallbackModel: string;
  imageDataUrl: string;
  /**
   * The trained-pipeline result; included so the prompt can affirm the TB
   * verdict was already produced (defensive: keeps the model from drifting
   * into TB-diagnosis territory).
   */
  localResult: LocalTriageResult;
}

/**
 * Build the user-message context. Deliberately minimal: the VLM should NOT
 * be anchored on the pipeline's specific numbers (that risks the model
 * regurgitating them as findings). Just confirm the verdict has been made
 * and the head is the authority on TB; the VLM is here only for NON-TB
 * side information.
 */
function buildContextMessage(local: LocalTriageResult): string {
  const lines: string[] = [];
  lines.push('CONTEXT: a validated TB triage pipeline has already produced the verdict for this image.');
  lines.push(`verdict_already_decided_as: ${local.verdict}`);
  lines.push('YOU are NOT re-deciding TB. List ONLY non-TB observations per the categories above.');
  return lines.join('\n');
}

/**
 * Single deterministic call to gpt-5.5 vision. The model receives the
 * image + a short context message; it returns the structured observations.
 */
export async function gptSecondary(opts: GptSecondaryOpts): Promise<GptSecondaryCall> {
  if (!opts.apiKey) {
    throw new OpenAIError('OpenAI API key missing — cannot run secondary observations');
  }
  const fullPrompt = `${GPT_SECONDARY_PROMPT}\n\n${buildContextMessage(opts.localResult)}`;

  const res = await openaiJSON<unknown>({
    apiKey: opts.apiKey,
    model: opts.primaryModel,
    fallbackModel: opts.fallbackModel,
    prompt: fullPrompt,
    imageDataUrl: opts.imageDataUrl,
    schema: GPT_SECONDARY_SCHEMA,
  });

  const envelope = res.raw as { model?: unknown } | null;
  const modelIdFromResponse =
    envelope && typeof envelope.model === 'string' && envelope.model.length > 0
      ? envelope.model
      : res.modelUsed;

  if (res.data === null || typeof res.data !== 'object') {
    throw new OpenAIError(
      `gpt-secondary returned non-object data (model=${modelIdFromResponse})`,
    );
  }

  const observations = validateAndNormalize(res.data);

  return {
    observations,
    audit: {
      prompt_hash: GPT_SECONDARY_PROMPT_HASH,
      schema_version: GPT_SECONDARY_SCHEMA_VERSION,
      schema_hash: GPT_SECONDARY_SCHEMA_HASH,
      model_id_from_response: modelIdFromResponse,
    },
    latencyMs: res.latencyMs,
    fellBack: res.fellBack,
    modelUsed: res.modelUsed,
  };
}

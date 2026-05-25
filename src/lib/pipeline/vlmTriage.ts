/**
 * VLM PRIMARY TRIAGE (Milestone 21).
 *
 * After M20 made it explicit that `microsoft/rad-dino` and `torchxrayvision` /
 * `Owos/tb-classifier` are no longer hosted on the hf-inference free serverless
 * router, the deployed browser app no longer has a working primary perception
 * head. The local ONNX heads we exported in M19 (`public/models/tb_head_t2.onnx`
 * + `sequelae_head.onnx`) stay on disk as honest "ready when Phase B feasible"
 * evidence but cannot execute in the browser today — they consume Rad-DINO patch
 * tokens + TXRV logits that no in-browser pathway produces.
 *
 * This module is the bridge. The OpenAI Responses API's `gpt-5.5` vision model
 * is promoted from "ensemble member" to PRIMARY perception, called with a
 * structured-output JSON schema so the orchestrator never parses prose.
 *
 * It is also, deliberately, a STEP DOWN in measured accuracy from our M15 head
 * (LODO AUROC 0.925). The VLM is uncalibrated, has not seen the project's mimic
 * stress test, and may overreact to scar/fibrosis/pleural-thickening (which is
 * exactly the failure mode the sequelae head was meant to fix). Every claim the
 * UI makes about the VLM verdict must reflect that.
 *
 * SCHEMA DESIGN NOTES (per the M21 hostile-critic brief):
 *   - `tb_score_uncalibrated` (NOT `tb_prob`). The name is the safety contract:
 *     this number has not been mapped onto a labeled dataset's operating point.
 *   - `scar_shape_score_uncalibrated` (NOT `s_inactive`). Different model,
 *     different distribution; the threshold from the sequelae ONNX head's own
 *     scar probe (0.7126) is NOT valid here.
 *   - No `chain_of_thought` field. Forcing a structured schema with a SHORT
 *     `short_rationale` discourages explicit-CoT prompting (CoT in radiology
 *     VLMs has been shown to invent findings).
 *   - `safety_flags` and `refusal_or_limitation` are first-class so the model
 *     can abstain on portable AP, pediatric film, or visible artifact without
 *     having to encode that into `tb_screen_result`.
 *
 * AUDIT TRAIL: every successful submission stores `{prompt_hash, schema_version,
 * model_id_from_response, image_preprocessing_version}`. The orchestrator pins
 * this into `PipelineRun.modelVersions` so an exported run records exactly which
 * prompt+schema+model produced the verdict.
 */
import type { JsonSchemaFormat } from '@/lib/providers/openai';
import { openaiJSON } from '@/lib/providers/openai';
import { OpenAIError } from '@/lib/providers/errors';

// ---------------------------------------------------------------------------
// Schema versioning — bump when the prompt or schema changes meaningfully so the
// audit trail can join a verdict to the exact triage contract that produced it.
// ---------------------------------------------------------------------------
export const VLM_TRIAGE_SCHEMA_VERSION = 'vlm-triage-v1' as const;
export const VLM_IMAGE_PREPROCESSING_VERSION = 'browser-passthrough-v1' as const;

// ---------------------------------------------------------------------------
// The single typed shape every downstream consumer sees.
// ---------------------------------------------------------------------------
export type TbScreenResult = 'screen_positive' | 'screen_negative' | 'abstain';
export type ImageQuality = 'diagnostic' | 'limited' | 'nondiagnostic';
export type Projection = 'pa_ap' | 'lateral' | 'unknown';
export type ConfidenceBand = 'low' | 'medium' | 'high';

export interface TriageSubmission {
  image_quality: ImageQuality;
  projection: Projection;
  tb_screen_result: TbScreenResult;
  /** 0..1 — name MUST be `_uncalibrated`, never `tb_prob`. */
  tb_score_uncalibrated: number;
  confidence_band: ConfidenceBand;
  /** 0..1 — heuristic; NOT the sequelae head's `s_inactive`. */
  scar_shape_score_uncalibrated: number;
  /** Free-text feature tags, e.g. ['apical_fibrosis','pleural_thickening']. */
  mimic_features_present: string[];
  /** Short free-text findings list (location-tagged when possible). */
  abnormality_localization: string[];
  /** e.g. ['pediatric','portable_ap','artifact']. */
  safety_flags: string[];
  /** 1-2 sentences MAX, evidence-grounded; NOT chain-of-thought. */
  short_rationale: string;
  /** null when the model proceeded; non-null when it refused/limited. */
  refusal_or_limitation: string | null;
  /** Echoed from the request so the wire doesn't drift from the audit trail. */
  model_version_seen_by_client: string;
}

/**
 * The JSON schema we pass to the Responses API's structured-output mode.
 *
 * NOTE: enum constraints get the model 90% of the way to "doesn't free-text the
 * screen_result"; the post-parse validator (`validateAndNormalizeSubmission`)
 * catches the remaining slop without crashing the whole call.
 */
export const VLM_TRIAGE_SCHEMA: JsonSchemaFormat = {
  name: 'submit_triage',
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      image_quality: { type: 'string', enum: ['diagnostic', 'limited', 'nondiagnostic'] },
      projection: { type: 'string', enum: ['pa_ap', 'lateral', 'unknown'] },
      tb_screen_result: {
        type: 'string',
        enum: ['screen_positive', 'screen_negative', 'abstain'],
      },
      tb_score_uncalibrated: { type: 'number' },
      confidence_band: { type: 'string', enum: ['low', 'medium', 'high'] },
      scar_shape_score_uncalibrated: { type: 'number' },
      mimic_features_present: { type: 'array', items: { type: 'string' } },
      abnormality_localization: { type: 'array', items: { type: 'string' } },
      safety_flags: { type: 'array', items: { type: 'string' } },
      short_rationale: { type: 'string' },
      refusal_or_limitation: { type: ['string', 'null'] },
      model_version_seen_by_client: { type: 'string' },
    },
    required: [
      'image_quality',
      'projection',
      'tb_screen_result',
      'tb_score_uncalibrated',
      'confidence_band',
      'scar_shape_score_uncalibrated',
      'mimic_features_present',
      'abnormality_localization',
      'safety_flags',
      'short_rationale',
      'refusal_or_limitation',
      'model_version_seen_by_client',
    ],
  },
};

/**
 * Boring, constrained policy prompt. NO "expert radiologist" persona. NO chain-
 * of-thought scaffolding. NO injected priors from our LODO numbers (those belong
 * in the disclosure, not in the inference prompt — passing them anchors the
 * model on expected priors rather than reading the image).
 */
export const VLM_TRIAGE_PROMPT_PRIMARY = [
  'You are a structured-output triage helper for a research preview tuberculosis screen.',
  'This is NOT diagnosis. Output ONLY the schema fields; do not narrate.',
  'When uncertain or when image quality is limited or nondiagnostic, choose ABSTAIN rather than screen_negative.',
  'Do not assume demographic priors. Do not infer clinical history from the image.',
  'Do not claim your scores are calibrated.',
  'tb_score_uncalibrated is a soft 0..1 indicator only; it has not been validated against labels.',
  'scar_shape_score_uncalibrated is a separate heuristic for healed fibrosis / pleural thickening / volume loss.',
  'mimic_features_present should list specific visible patterns (e.g. apical_fibrosis, pleural_thickening, healed_scar, calcified_granuloma) when seen.',
  'safety_flags should include pediatric, portable_ap, lateral_only, artifact, rotation, or other quality concerns.',
  'Keep short_rationale to one or two sentences, citing visible evidence; do not include step-by-step reasoning.',
].join(' ');

/**
 * Verifier prompt: same schema, intentionally different phrasing. Used only when
 * the primary result is borderline (see TASK B in the M21 brief). Single
 * independent call, not three majority-vote samples (correlated self-sampling
 * looks like ensembling but isn't).
 */
export const VLM_TRIAGE_PROMPT_VERIFIER = [
  'Second-opinion structured triage check for a research preview tuberculosis screen.',
  'Inspect the radiograph independently. This is NOT diagnosis. Output ONLY the schema fields.',
  'Be willing to disagree with prior readers. If image quality is limited or you are unsure, return ABSTAIN.',
  'Prefer ABSTAIN over screen_negative when in doubt.',
  'Do not invent findings; if no concerning pattern is visible, say so plainly in short_rationale.',
  'No demographic priors. No inferred clinical history. No calibration claim.',
].join(' ');

// ---------------------------------------------------------------------------
// Prompt fingerprinting for the audit trail.
// ---------------------------------------------------------------------------
/**
 * Cheap deterministic FNV-1a 32-bit hash, hex-encoded. Good enough to spot when
 * a prompt or schema string changes between deploys; not a security primitive.
 */
function fnv1aHex(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export const VLM_PROMPT_HASH_PRIMARY = fnv1aHex(VLM_TRIAGE_PROMPT_PRIMARY);
export const VLM_PROMPT_HASH_VERIFIER = fnv1aHex(VLM_TRIAGE_PROMPT_VERIFIER);
export const VLM_SCHEMA_HASH = fnv1aHex(JSON.stringify(VLM_TRIAGE_SCHEMA.schema));

// ---------------------------------------------------------------------------
// Tolerant validator. The Responses API structured-output mode is meant to
// guarantee shape, but: (a) older API revisions may have shifted syntax;
// (b) a fallback model may not honor strict mode the same way. Validate every
// field rather than trusting the schema unconditionally.
// ---------------------------------------------------------------------------
function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === 'string');
}

function clamp01(n: unknown): number {
  if (typeof n !== 'number' || !Number.isFinite(n)) return 0;
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

const QUALITY_VALUES: readonly ImageQuality[] = ['diagnostic', 'limited', 'nondiagnostic'];
const PROJECTION_VALUES: readonly Projection[] = ['pa_ap', 'lateral', 'unknown'];
const SCREEN_VALUES: readonly TbScreenResult[] = [
  'screen_positive',
  'screen_negative',
  'abstain',
];
const BAND_VALUES: readonly ConfidenceBand[] = ['low', 'medium', 'high'];

function pickEnum<T extends string>(v: unknown, allowed: readonly T[], fallback: T): T {
  if (typeof v === 'string' && (allowed as readonly string[]).includes(v)) return v as T;
  return fallback;
}

/**
 * Normalize a raw structured-output payload into the strict `TriageSubmission`
 * shape, applying the deterministic "force ABSTAIN" guards from the M21 brief.
 *
 * Forced ABSTAIN when:
 *   - `image_quality !== 'diagnostic'`
 *   - `projection === 'unknown'` AND the model said `screen_positive`
 *     (radiograph orientation unknown but suspecting TB anyway → not safe)
 *   - `confidence_band === 'low'`
 *
 * These are POST-MODEL deterministic rails. The model gets to advise; the rails
 * decide. (Same ethos as the existing `screeningPolicy` in the orchestrator.)
 */
export function validateAndNormalizeSubmission(
  raw: unknown,
  modelIdFromResponse: string,
): TriageSubmission {
  const r = (raw ?? {}) as Record<string, unknown>;

  const image_quality = pickEnum<ImageQuality>(r.image_quality, QUALITY_VALUES, 'limited');
  const projection = pickEnum<Projection>(r.projection, PROJECTION_VALUES, 'unknown');
  let tb_screen_result = pickEnum<TbScreenResult>(
    r.tb_screen_result,
    SCREEN_VALUES,
    'abstain',
  );
  const confidence_band = pickEnum<ConfidenceBand>(r.confidence_band, BAND_VALUES, 'low');

  const tb_score_uncalibrated = clamp01(r.tb_score_uncalibrated);
  const scar_shape_score_uncalibrated = clamp01(r.scar_shape_score_uncalibrated);

  const mimic_features_present = isStringArray(r.mimic_features_present)
    ? r.mimic_features_present
    : [];
  const abnormality_localization = isStringArray(r.abnormality_localization)
    ? r.abnormality_localization
    : [];
  const safety_flags = isStringArray(r.safety_flags) ? r.safety_flags : [];

  const short_rationale =
    typeof r.short_rationale === 'string' && r.short_rationale.length > 0
      ? r.short_rationale
      : '(no rationale returned)';

  const refusal_or_limitation =
    typeof r.refusal_or_limitation === 'string' && r.refusal_or_limitation.length > 0
      ? r.refusal_or_limitation
      : null;

  // Deterministic forced-abstain rails. Apply AFTER pulling raw enum values so
  // the original model intent is preserved in the audit trail (model says
  // screen_positive but we forced ABSTAIN → recorded in short_rationale below).
  const forcedAbstainReasons: string[] = [];
  if (image_quality !== 'diagnostic') {
    forcedAbstainReasons.push(`image_quality=${image_quality}`);
  }
  if (projection === 'unknown' && tb_screen_result === 'screen_positive') {
    forcedAbstainReasons.push('projection=unknown with positive screen');
  }
  if (confidence_band === 'low') {
    forcedAbstainReasons.push('confidence_band=low');
  }

  let mergedRationale = short_rationale;
  if (forcedAbstainReasons.length > 0 && tb_screen_result !== 'abstain') {
    tb_screen_result = 'abstain';
    mergedRationale = `[forced abstain: ${forcedAbstainReasons.join(', ')}] ${short_rationale}`;
  }

  return {
    image_quality,
    projection,
    tb_screen_result,
    tb_score_uncalibrated,
    confidence_band,
    scar_shape_score_uncalibrated,
    mimic_features_present,
    abnormality_localization,
    safety_flags,
    short_rationale: mergedRationale,
    refusal_or_limitation,
    model_version_seen_by_client:
      typeof r.model_version_seen_by_client === 'string' && r.model_version_seen_by_client.length
        ? r.model_version_seen_by_client
        : modelIdFromResponse,
  };
}

// ---------------------------------------------------------------------------
// Audit-trail surface (stored in PipelineRun.modelVersions and event stream).
// ---------------------------------------------------------------------------
export interface VlmAuditPins {
  prompt_hash: string;
  schema_version: string;
  schema_hash: string;
  model_id_from_response: string;
  image_preprocessing_version: string;
}

export interface VlmTriageCall {
  submission: TriageSubmission;
  audit: VlmAuditPins;
  latencyMs: number;
  fellBack: boolean;
  /** The model slot we ACTUALLY called (post-fallback), as returned by openaiJSON. */
  modelUsed: string;
}

export interface VlmTriageOpts {
  apiKey: string;
  primaryModel: string;
  fallbackModel: string;
  imageDataUrl: string;
  /** 'primary' | 'verifier'. Verifier uses the different-phrasing prompt. */
  role: 'primary' | 'verifier';
}

/**
 * Single deterministic call to gpt-5.5 vision via the Responses API structured-
 * output mode. NO self-consistency K-sampling — that pattern is correlated self-
 * sampling and was the M4 hack used when this model was only the ensemble's
 * THIRD member. Now that it is the PRIMARY perception, we lean on the schema
 * instead, and add the borderline-band consistency check in the orchestrator.
 */
export async function vlmTriage(opts: VlmTriageOpts): Promise<VlmTriageCall> {
  const prompt =
    opts.role === 'primary' ? VLM_TRIAGE_PROMPT_PRIMARY : VLM_TRIAGE_PROMPT_VERIFIER;
  const promptHash =
    opts.role === 'primary' ? VLM_PROMPT_HASH_PRIMARY : VLM_PROMPT_HASH_VERIFIER;

  const res = await openaiJSON<unknown>({
    apiKey: opts.apiKey,
    model: opts.primaryModel,
    fallbackModel: opts.fallbackModel,
    prompt,
    imageDataUrl: opts.imageDataUrl,
    schema: VLM_TRIAGE_SCHEMA,
  });

  // Try to recover the model id the SERVER ACTUALLY USED — the Responses API
  // echoes a `model` field in the response envelope. Falls back to the slot we
  // requested if the envelope shape shifts.
  const envelope = res.raw as { model?: unknown } | null;
  const modelIdFromResponse =
    envelope && typeof envelope.model === 'string' && envelope.model.length > 0
      ? envelope.model
      : res.modelUsed;

  if (res.data === null || typeof res.data !== 'object') {
    throw new OpenAIError(
      `VLM triage returned non-object data (model=${modelIdFromResponse})`,
    );
  }

  const submission = validateAndNormalizeSubmission(res.data, modelIdFromResponse);

  const audit: VlmAuditPins = {
    prompt_hash: promptHash,
    schema_version: VLM_TRIAGE_SCHEMA_VERSION,
    schema_hash: VLM_SCHEMA_HASH,
    model_id_from_response: modelIdFromResponse,
    image_preprocessing_version: VLM_IMAGE_PREPROCESSING_VERSION,
  };

  return {
    submission,
    audit,
    latencyMs: res.latencyMs,
    fellBack: res.fellBack,
    modelUsed: res.modelUsed,
  };
}

// ---------------------------------------------------------------------------
// Borderline / consistency-check predicates (pure; orchestrator drives the call).
// ---------------------------------------------------------------------------
/**
 * Lower edge of the "borderline" band where the verifier call fires.
 *
 * M26 — widened from 0.35 to 0.20. The M24 diagnostic surfaced TB cases
 * where the model returned mid-range scores (tb_prob 0.48, 0.60 on
 * tb_blind_04 / tb_blind_01) and was wrong; widening the lower edge of
 * the band catches MORE uncertain cases at the cost of a small bump in
 * gpt-5.5 vision verifier calls (the verifier only fires when borderline,
 * so the cost increase is proportional to the slice of cases sitting in
 * [0.20, 0.35] — empirically ~7% of LODO calls; on a 13k cache that's
 * ~900 extra calls but on a single browser session it's ~0). Trade a
 * few extra GPT calls for the chance of catching mid-zone TB the
 * validated head reads as low-probability.
 *
 * Aligned with `asymmetricEvidence.TB_PROB_LOW_THRESHOLD` (also 0.20)
 * so the local-path and vlm-path borderlines share a single number.
 */
export const VLM_BORDERLINE_LOW = 0.20 as const;
/** Upper edge — above this the primary's "positive" stands without a second opinion. */
export const VLM_BORDERLINE_HIGH = 0.65 as const;

/**
 * Feature tags that should trigger a verifier call regardless of score, because
 * the VLM is known to over-react to scar-shaped findings (M18 mimic context).
 * Substring match, case-insensitive: lets the model use `apical_fibrosis`,
 * `Fibrosis` or `apical fibrosis` interchangeably.
 */
export const SCAR_TRIGGER_TAGS = [
  'fibrosis',
  'pleural_thickening',
  'pleural thickening',
  'healed_scar',
  'healed scar',
  'scar',
] as const;

export function hasScarMimicFlag(features: string[]): boolean {
  const lc = features.map((f) => f.toLowerCase());
  return lc.some((f) => SCAR_TRIGGER_TAGS.some((t) => f.includes(t)));
}

export function isBorderlineForConsistencyCheck(s: TriageSubmission): boolean {
  if (s.confidence_band === 'low') return true;
  if (hasScarMimicFlag(s.mimic_features_present)) return true;
  const score = s.tb_score_uncalibrated;
  return score >= VLM_BORDERLINE_LOW && score <= VLM_BORDERLINE_HIGH;
}

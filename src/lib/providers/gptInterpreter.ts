/**
 * GPT-AS-INTERPRETER (Milestone 24).
 *
 * The opposite of M21's `vlmTriage`. M21 asks gpt-5.5 vision to GIVE A VERDICT on
 * the image (because no in-browser perception backbone exists). M24's interpreter
 * asks gpt-5.5 vision to TRANSLATE THE VALIDATED MODEL'S STRUCTURED OUTPUT INTO
 * A CLINICIAN-STYLE PARAGRAPH. The verdict already exists — it came from the
 * Rad-DINO + TXRV + TBHeadT2 + InactiveSequelaeHead pipeline running locally
 * under its calibrated temperatures. The model is forbidden from disagreeing
 * with it; it may only describe the evidence the local model flagged.
 *
 * This is the keep-AUROC-0.922-load-bearing design. The validated model decides;
 * GPT narrates. Anything stronger than that crosses the line from honest
 * enrichment into laundering a VLM's opinion as a calibrated verdict.
 *
 * COST/LATENCY: ~$0.01-0.04 per call, ~5-15s. On-demand only (the UI fires this
 * on a "Generate clinician report" click; result is cached in component state so
 * re-opening Details doesn't re-bill).
 *
 * AUDIT PINS: prompt_hash + schema_version + model_id_from_response are returned
 * in the call envelope so any future portfolio export can join a narrative to
 * the exact prompt+schema that produced it.
 */
import type { JsonSchemaFormat } from '@/lib/providers/openai';
import { openaiJSON } from '@/lib/providers/openai';
import { OpenAIError } from '@/lib/providers/errors';
import type { LocalTriageResult } from './localTriage';

// ---------------------------------------------------------------------------
// Schema versioning — bump when prompt or schema changes meaningfully.
// ---------------------------------------------------------------------------
export const GPT_INTERPRETER_SCHEMA_VERSION = 'gpt-interpreter-v1' as const;

export type ConfidenceQualifier = 'low' | 'moderate' | 'high';
export type DifferentialLikelihood = 'consider' | 'less_likely';

export interface DifferentialAlternative {
  label: string;
  likelihood: DifferentialLikelihood;
  /** 1 sentence from the local-model evidence. */
  rationale: string;
}

export interface ClinicianReport {
  /** 1-2 sentences, evidence-grounded, evidence-only. NOT a diagnosis claim. */
  finding: string;
  /** Zones the LOCAL MODEL flagged (e.g. ['upper_r','hilar']). Subset of the zones we passed in. */
  key_regions: string[];
  confidence_qualifier: ConfidenceQualifier;
  top_differential_alternatives: DifferentialAlternative[];
  /** Non-null when the model refused or noted a limitation it could not bridge. */
  refusal_or_limitation: string | null;
}

// ---------------------------------------------------------------------------
// The JSON schema we pass to the Responses API's structured-output mode.
// ---------------------------------------------------------------------------
export const GPT_INTERPRETER_SCHEMA: JsonSchemaFormat = {
  name: 'clinician_report',
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      finding: { type: 'string' },
      key_regions: { type: 'array', items: { type: 'string' } },
      confidence_qualifier: { type: 'string', enum: ['low', 'moderate', 'high'] },
      top_differential_alternatives: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            label: { type: 'string' },
            likelihood: { type: 'string', enum: ['consider', 'less_likely'] },
            rationale: { type: 'string' },
          },
          required: ['label', 'likelihood', 'rationale'],
        },
      },
      refusal_or_limitation: { type: ['string', 'null'] },
    },
    required: [
      'finding',
      'key_regions',
      'confidence_qualifier',
      'top_differential_alternatives',
      'refusal_or_limitation',
    ],
  },
};

/**
 * The system prompt is the keep-the-discipline contract. It DOES NOT let the
 * model freelance on the image; every sentence the model writes must ground
 * itself in the local pipeline's structured output. The verdict is FIXED.
 *
 * Notes on what is INTENTIONALLY ABSENT:
 *   - No "expert radiologist" persona — anchors the model to a register it
 *     does not have the training data to occupy.
 *   - No chain-of-thought scaffolding. Structured output discourages CoT.
 *   - No LODO numbers in the prompt — those belong in the disclosure under
 *     the rendered narrative, not in the inference context (anchoring).
 */
export const GPT_INTERPRETER_PROMPT = [
  'You are translating the structured output of a validated TB-triage pipeline (Rad-DINO + TorchXRayVision + TBHeadT2 + InactiveSequelaeHead, LODO AUROC 0.922) into a short clinician-style narrative.',
  'You are NOT diagnosing the image. The pipeline already produced the verdict, the calibrated tb_prob, and the per-zone probabilities. Your job is to put the pipeline\'s OWN evidence into a 1-2 sentence paragraph and a small differential.',
  'STRICT RULES:',
  '- Use ONLY zones the pipeline flagged in the per-zone probabilities (the highest probabilities are the regions to mention by name).',
  '- Use ONLY pathology features the pipeline\'s TXRV head ranked in the top scores (do NOT introduce findings the pipeline did not score).',
  '- Do NOT contradict the pipeline verdict. If the verdict is "tb", describe the evidence consistent with that verdict; if "no_tb", describe the absence of evidence; if "abstain", describe the borderline / ambiguity.',
  '- top_differential_alternatives ranks alternatives by FIT TO THE PIPELINE EVIDENCE, not fit to the image alone.',
  '- key_regions MUST be a subset of the zone keys the user provides (e.g. ["upper_r","hilar"]).',
  '- confidence_qualifier reflects the calibrated tb_prob (high near 0/1, moderate in mid-band, low in the borderline band 0.35-0.65).',
  '- refusal_or_limitation is non-null ONLY when the image data the pipeline preprocessed is degraded or you cannot honestly map its scores into a narrative.',
  'No demographic priors. No inferred clinical history. No claim that your narrative is calibrated. No chain-of-thought — output ONLY the schema fields.',
].join(' ');

// FNV-1a 32-bit hash for audit pinning (mirrors vlmTriage's fnv1aHex).
function fnv1aHex(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export const GPT_INTERPRETER_PROMPT_HASH = fnv1aHex(GPT_INTERPRETER_PROMPT);
export const GPT_INTERPRETER_SCHEMA_HASH = fnv1aHex(JSON.stringify(GPT_INTERPRETER_SCHEMA.schema));

// ---------------------------------------------------------------------------
// Validator — defense against the model returning shape-valid-but-semantically-
// drifted output. (Strict mode usually guarantees shape; semantic checks are
// our own contract enforcement.)
// ---------------------------------------------------------------------------
function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === 'string');
}

const QUALIFIER_VALUES: readonly ConfidenceQualifier[] = ['low', 'moderate', 'high'];
const LIKELIHOOD_VALUES: readonly DifferentialLikelihood[] = ['consider', 'less_likely'];

function pickEnum<T extends string>(v: unknown, allowed: readonly T[], fallback: T): T {
  if (typeof v === 'string' && (allowed as readonly string[]).includes(v)) return v as T;
  return fallback;
}

function clipString(v: unknown, fallback: string): string {
  return typeof v === 'string' && v.length > 0 ? v : fallback;
}

function validateAndNormalizeReport(raw: unknown, allowedZones: readonly string[]): ClinicianReport {
  const r = (raw ?? {}) as Record<string, unknown>;
  const finding = clipString(r.finding, '(no finding returned)');
  // Subset filter: zones the model named MUST be a subset of the zones we passed in
  // (so the model cannot invent zone keys we never told it about).
  const allowed = new Set(allowedZones);
  const key_regions = isStringArray(r.key_regions)
    ? r.key_regions.filter((z) => allowed.has(z))
    : [];
  const confidence_qualifier = pickEnum<ConfidenceQualifier>(
    r.confidence_qualifier,
    QUALIFIER_VALUES,
    'low',
  );
  const rawDiffs = Array.isArray(r.top_differential_alternatives)
    ? r.top_differential_alternatives
    : [];
  const top_differential_alternatives: DifferentialAlternative[] = rawDiffs
    .map((d): DifferentialAlternative | null => {
      if (d === null || typeof d !== 'object') return null;
      const o = d as Record<string, unknown>;
      const label = typeof o.label === 'string' ? o.label : null;
      if (label === null) return null;
      return {
        label,
        likelihood: pickEnum<DifferentialLikelihood>(o.likelihood, LIKELIHOOD_VALUES, 'less_likely'),
        rationale: clipString(o.rationale, ''),
      };
    })
    .filter((d): d is DifferentialAlternative => d !== null)
    .slice(0, 3);
  const refusal_or_limitation =
    typeof r.refusal_or_limitation === 'string' && r.refusal_or_limitation.length > 0
      ? r.refusal_or_limitation
      : null;
  return {
    finding,
    key_regions,
    confidence_qualifier,
    top_differential_alternatives,
    refusal_or_limitation,
  };
}

// ---------------------------------------------------------------------------
// Audit + call shape
// ---------------------------------------------------------------------------
export interface GptInterpreterAuditPins {
  prompt_hash: string;
  schema_version: string;
  schema_hash: string;
  model_id_from_response: string;
}

export interface GptInterpreterCall {
  report: ClinicianReport;
  audit: GptInterpreterAuditPins;
  latencyMs: number;
  fellBack: boolean;
  modelUsed: string;
}

export interface GptInterpreterOpts {
  apiKey: string;
  primaryModel: string;
  fallbackModel: string;
  imageDataUrl: string;
  /** The local pipeline's structured output — the only ground truth the model is allowed to translate. */
  localResult: LocalTriageResult;
}

/**
 * Build the user-message text the model sees. This is where we PIN the
 * evidence the model is allowed to talk about: the calibrated verdict, the
 * threshold, the top-K zones, the top-K TXRV findings. We do NOT pass the
 * raw 8x8 grid (the model would invent grid-cell labels we cannot back).
 */
export function buildEvidenceMessage(local: LocalTriageResult): string {
  const lines: string[] = [];
  lines.push('VALIDATED PIPELINE OUTPUT (translate this, do not re-diagnose):');
  lines.push(`verdict: ${local.verdict}`);
  lines.push(`tb_prob (calibrated under T=${local.audit.calibration.T.toFixed(3)}): ${local.tb_prob.toFixed(4)}`);
  lines.push(`decided_at_threshold: ${local.decided_at_threshold.toFixed(4)}`);
  lines.push(`s_inactive (scar/sequelae score, under T_seq=${local.audit.calibration.T_sequelae.toFixed(3)}): ${local.s_inactive.toFixed(4)}`);
  if (local.safety_net_applied) {
    lines.push(`safety_net_applied: ${local.safety_net_applied}`);
  }

  // Top-5 zones by probability so the model has both the headline (top zone) and the
  // 'less probable but mentionable' tier. Below the top 5, the values are usually <0.05
  // and would clutter the differential.
  if (local.zonal_scores) {
    const sortedZones = Object.entries(local.zonal_scores)
      .filter(([, v]) => typeof v === 'number')
      .sort((a, b) => (b[1] as number) - (a[1] as number));
    lines.push('per-zone calibrated TB probabilities (key: probability):');
    for (const [k, v] of sortedZones) {
      lines.push(`  ${k}: ${(v as number).toFixed(4)}`);
    }
  }

  // Top-5 TXRV findings (the model has 18; the top-5 are what the head's fusion lever
  // weighted most — and what the narrative should mention).
  if (local.txrv_pathologies) {
    const top = Object.entries(local.txrv_pathologies)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    lines.push('top-5 TorchXRayVision findings (label: probability — input features, NOT independent diagnoses):');
    for (const [k, v] of top) {
      lines.push(`  ${k}: ${v.toFixed(4)}`);
    }
  }

  if (local.image_quality.warnings.length > 0) {
    lines.push('preprocessing warnings:');
    for (const w of local.image_quality.warnings) lines.push(`  - ${w}`);
  }

  lines.push('');
  lines.push('Now produce the clinician_report JSON. Mention zones by name; do not invent zone keys not listed above. Differential alternatives must be grounded in the TXRV findings or the scar/sequelae score, not in pixels.');
  return lines.join('\n');
}

/**
 * Single deterministic call to gpt-5.5 vision. The model receives BOTH the
 * image AND the evidence text — image so it can ground its language in what
 * it actually sees, evidence so its narrative is anchored to the validated
 * pipeline's numbers (not its own image impression). Temperature defaults
 * (the Responses API's default is fine; we don't override it here because the
 * project's existing `openaiJSON` helper doesn't expose a temperature knob).
 */
export async function gptInterpreter(opts: GptInterpreterOpts): Promise<GptInterpreterCall> {
  if (!opts.apiKey) {
    throw new OpenAIError('OpenAI API key missing — cannot generate clinician report');
  }
  const evidenceMessage = buildEvidenceMessage(opts.localResult);
  const fullPrompt = `${GPT_INTERPRETER_PROMPT}\n\n${evidenceMessage}`;

  const res = await openaiJSON<unknown>({
    apiKey: opts.apiKey,
    model: opts.primaryModel,
    fallbackModel: opts.fallbackModel,
    prompt: fullPrompt,
    imageDataUrl: opts.imageDataUrl,
    schema: GPT_INTERPRETER_SCHEMA,
  });

  // Echo the model id from the response envelope when available; fall back to
  // the slot we asked for (the openaiJSON helper already does fallback for us).
  const envelope = res.raw as { model?: unknown } | null;
  const modelIdFromResponse =
    envelope && typeof envelope.model === 'string' && envelope.model.length > 0
      ? envelope.model
      : res.modelUsed;

  if (res.data === null || typeof res.data !== 'object') {
    throw new OpenAIError(
      `gpt-interpreter returned non-object data (model=${modelIdFromResponse})`,
    );
  }

  const allowedZones = Object.keys(opts.localResult.zonal_scores ?? {});
  const report = validateAndNormalizeReport(res.data, allowedZones);

  return {
    report,
    audit: {
      prompt_hash: GPT_INTERPRETER_PROMPT_HASH,
      schema_version: GPT_INTERPRETER_SCHEMA_VERSION,
      schema_hash: GPT_INTERPRETER_SCHEMA_HASH,
      model_id_from_response: modelIdFromResponse,
    },
    latencyMs: res.latencyMs,
    fellBack: res.fellBack,
    modelUsed: res.modelUsed,
  };
}

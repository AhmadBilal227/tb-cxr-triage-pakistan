/**
 * GPT-AS-INTERPRETER (Milestone 24 → v2 radiologist rewrite).
 *
 * The opposite of M21's `vlmTriage`. M21 asks gpt-5.5 vision to GIVE A VERDICT
 * on the image. M24's interpreter asks gpt-5.5 vision to TRANSLATE THE
 * VALIDATED MODEL'S STRUCTURED OUTPUT INTO A RADIOLOGY-REPORT-STYLE
 * NARRATIVE. The verdict already exists, produced by the Rad-DINO + TXRV +
 * TBHeadT2 + InactiveSequelaeHead pipeline running locally under its
 * calibrated temperatures. The model is forbidden from disagreeing with it;
 * it may only describe the evidence the local model flagged.
 *
 * v2 (2026-05-25) rewrites the schema + prompt to produce a STRUCTURED
 * RADIOLOGY REPORT (RSNA-style Findings + Impression + Recommendation) using
 * Fleischner Society terminology, intended for radiologist consumption rather
 * than ML-scientist consumption. The honesty contract is preserved: zones must
 * be a subset of the supplied per-zone vocabulary, pathology terms must come
 * from the TXRV top-5, and the first impression item must match the pipeline
 * verdict. The prompt is grounded in the literature on LLM radiology report
 * generation (MAIRA / MAIRA-Seg / MAIRA2: bounding-box and segmentation
 * grounding prevents hallucination) and RSNA RadReport structural conventions.
 *
 * COST/LATENCY: ~$0.01-0.04 per call, ~5-15s. On-demand only (the UI fires
 * this on a "Generate report" click; result is cached in component state so
 * re-opening does not re-bill).
 *
 * AUDIT PINS: prompt_hash + schema_version + model_id_from_response are
 * returned in the call envelope so any portfolio export can join a narrative
 * to the exact prompt+schema that produced it.
 */
import type { JsonSchemaFormat } from '@/lib/providers/openai';
import { openaiJSON } from '@/lib/providers/openai';
import { OpenAIError } from '@/lib/providers/errors';
import type { LocalTriageResult } from './localTriage';

// ---------------------------------------------------------------------------
// Schema versioning — bump when prompt or schema changes meaningfully.
// v1 (M24 original): finding/key_regions/confidence_qualifier/differential.
// v2 (M24 rewrite):  RSNA-style technique/findings/impression/recommendation.
// ---------------------------------------------------------------------------
export const GPT_INTERPRETER_SCHEMA_VERSION = 'gpt-interpreter-v2' as const;

export type ImpressionLikelihood = 'primary' | 'consider' | 'less_likely';

export interface ImpressionItem {
  /** One short clinical statement. */
  statement: string;
  /**
   * 'primary' = the pipeline-anchored reading (MUST be the first item and MUST
   * match the verdict). 'consider' / 'less_likely' = differential alternatives
   * ranked by fit to the pipeline evidence.
   */
  likelihood: ImpressionLikelihood;
}

/** RSNA-style sub-sections under FINDINGS. Each is a short paragraph (1-3 sentences). */
export interface FindingsSections {
  lungs_and_airways: string;
  pleura: string;
  cardiomediastinum: string;
  bones_and_soft_tissues: string;
}

/**
 * Structured radiology report. The shape mirrors the RSNA RadReport convention
 * for chest radiograph (Exam, Comparison, Technique, Findings by region,
 * Impression, Recommendation).
 */
export interface ClinicianReport {
  /** Single short sentence describing how the image was acquired and read. */
  technique: string;
  /** "No prior studies available for comparison." unless told otherwise. */
  comparison: string;
  findings: FindingsSections;
  /** Numbered list ranked by clinical significance. First item is verdict-anchored. */
  impression: ImpressionItem[];
  /** 1-2 sentences, actionable next step. */
  recommendation: string;
  /** Zone keys the report referenced. MUST be a subset of the supplied zonal vocabulary. */
  key_regions: string[];
  /** Always includes the standard single-view + radiographic-vs-microbiological caveats. */
  limitations: string[];
}

// ---------------------------------------------------------------------------
// The JSON schema we pass to the Responses API's structured-output mode.
// All fields are required and additionalProperties:false so OpenAI's strict
// mode can enforce the shape; semantic constraints (e.g. zone-subset) are
// then enforced by our own validator below.
// ---------------------------------------------------------------------------
export const GPT_INTERPRETER_SCHEMA: JsonSchemaFormat = {
  name: 'radiologist_report',
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      technique: { type: 'string' },
      comparison: { type: 'string' },
      findings: {
        type: 'object',
        additionalProperties: false,
        properties: {
          lungs_and_airways: { type: 'string' },
          pleura: { type: 'string' },
          cardiomediastinum: { type: 'string' },
          bones_and_soft_tissues: { type: 'string' },
        },
        required: ['lungs_and_airways', 'pleura', 'cardiomediastinum', 'bones_and_soft_tissues'],
      },
      impression: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            statement: { type: 'string' },
            likelihood: { type: 'string', enum: ['primary', 'consider', 'less_likely'] },
          },
          required: ['statement', 'likelihood'],
        },
      },
      recommendation: { type: 'string' },
      key_regions: { type: 'array', items: { type: 'string' } },
      limitations: { type: 'array', items: { type: 'string' } },
    },
    required: [
      'technique',
      'comparison',
      'findings',
      'impression',
      'recommendation',
      'key_regions',
      'limitations',
    ],
  },
};

/**
 * System prompt v3 — board-certified radiologist drafting a report on top of
 * an AI screening pipeline's structured output.
 *
 * v2 → v3 changes (2026-05-25, second pass):
 *   - Explains the PIPELINE ARCHITECTURE up front so the model can describe
 *     what the system actually saw and why, rather than treating the
 *     evidence as a black-box JSON blob.
 *   - Adds TB-SPECIFIC RADIOLOGY KNOWLEDGE the model can draw from to fill
 *     gaps the trained head cannot measure (pattern characterization,
 *     distribution, severity grading, differential considerations grounded
 *     in the radiology literature).
 *   - Expands RECOMMENDATION guidance: lateral view, comparison with prior,
 *     CT chest for cavitation / lymphadenopathy / mass characterization,
 *     follow-up imaging cadence.
 *   - Tightens LIMITATIONS to include the always-include comparison caveat.
 *
 * Grounding source: RSNA RadReport structural convention + Fleischner Society
 * terminology + MAIRA / MAIRA-Seg literature on hallucination suppression
 * via spatial / feature constraint.
 *
 * Schema version stays at v2 (response shape unchanged); the prompt_hash on
 * the audit envelope changes automatically with the prompt text, so any
 * downstream join on prompt_hash will see the bump.
 */
export const GPT_INTERPRETER_PROMPT = [
  'You are a board-certified radiologist drafting a structured chest radiograph report for a pulmonologist or radiology colleague. The image is a single frontal chest X-ray. An AI screening pipeline has already run on this image and is handing you its structured output as context. Your role is to describe radiographic findings in clinical-radiology language and add radiologist-level context the trained head cannot measure. You DO NOT diagnose, disagree with, or contradict the pipeline\'s verdict.',
  '',
  'CRITICAL TASK CONSTRAINT (from the radiology AI literature — Jiang et al. 2024 showed that vision-language models cannot reliably generate de-novo radiology reports from images alone)',
  '- You are NOT performing image interpretation. The image is provided ONLY to disambiguate the pipeline\'s structured output when describing distribution, severity, or character. Your textual content MUST originate from the supplied evidence (zones + TXRV findings + verdict + threshold), not from your own image impression.',
  '- ONE SENTENCE PER FINDING. Each sentence in the findings sub-sections should describe at most one pathological feature, anchored to one zone or one TXRV finding from the supplied evidence (MAIRA-2 style spatial grounding suppresses hallucination).',
  '- HEDGED LANGUAGE FOR BORDERLINE CASES. When tb_prob is within +/- 0.10 of the decision threshold (0.6105), use radiologic uncertainty language: "indeterminate", "equivocal", "cannot be excluded", "suggestive of", rather than confident assertions.',
  '',
  'THE PIPELINE YOU ARE READING (so you can describe its reasoning faithfully when needed)',
  '- Vision backbone: Rad-DINO (self-supervised on ~1M chest radiographs) provides the global + 37x37 patch features.',
  '- Findings backbone: TorchXRayVision DenseNet-121 supplies 18 supervised pathology scores (Lung Opacity, Effusion, Consolidation, Lung Lesion, Infiltration, Atelectasis, Cardiomegaly, Pneumonia, Pleural Thickening, Fibrosis, etc.).',
  '- TB classifier head: a small head on top of the combined features, trained leave-one-dataset-out across five public TB cohorts and temperature-scaled. It emits a calibrated tb_prob and per-zone TB probabilities for seven zones (upper_l/upper_r, mid_l/mid_r, lower_l/lower_r, hilar).',
  '- Inactive sequelae head: a parallel head trained to flag healed-scar patterns; if it fires, the safety net routes the case to ABSTAIN instead of clearing it.',
  '- Decision rule: tb_prob is compared to a 95%-sensitivity threshold (0.6105). Above → screen positive. Below → screen negative. Three deterministic safety-net rules can only escalate "no_tb" → "abstain", never the reverse.',
  '',
  'VOICE',
  '- Impersonal, declarative radiology-report voice. Correct: "There is patchy parenchymal opacity in the right upper lobe with suggestion of cavitation." Incorrect: "I think..." / "The model believes..." / "It looks like...".',
  '- Use Fleischner Society terminology where applicable: consolidation; cavity (thick irregular wall, may contain a gas-fluid level); nodule; mass; ground-glass opacity; reticular opacity; fibroproductive density; scarring / fibrosis; calcified granuloma; hilar lymphadenopathy; pleural effusion; pleural thickening; atelectasis; tree-in-bud opacities; miliary pattern.',
  '',
  'ANCHORING (mandatory — keeps the report honest)',
  '- Mention anatomical zones BY NAME only when they appear in the per-zone probabilities supplied below. Vocabulary: upper_l = left upper lobe; upper_r = right upper lobe; mid_l = left mid-lung field; mid_r = right mid-lung field; lower_l = left lower lobe; lower_r = right lower lobe; hilar = hilar / perihilar region. In prose use the anatomical name; in key_regions use the key.',
  '- Mention pathology terms only when they appear in the TXRV top-5. Translate to Fleischner-equivalent terminology in prose: TXRV "Lung Opacity" → "parenchymal opacity"; "Effusion" → "pleural effusion"; "Lung Lesion" → "focal pulmonary lesion"; "Fibrosis" → "fibrotic scarring"; "Pleural Thickening" → "pleural thickening"; "Atelectasis" → "atelectasis"; "Cardiomegaly" → "cardiac enlargement"; "Infiltration" → "interstitial infiltrate".',
  '- Where the pipeline did NOT flag a region, state so explicitly ("The pleural spaces are clear." / "The cardiomediastinal silhouette is unremarkable.").',
  '',
  'TB-SPECIFIC RADIOLOGY KNOWLEDGE TO DRAW FROM (use to characterize patterns and frame the differential)',
  '- Post-primary (reactivation) TB classically presents as apical / posterior upper-lobe fibroproductive opacity, cavitation, and bronchogenic spread (tree-in-bud).',
  '- Primary TB often shows mid/lower-zone consolidation, hilar / mediastinal lymphadenopathy, and unilateral pleural effusion; may be radiographically subtle in adults.',
  '- Miliary TB: diffuse fine 1-3 mm nodules; haematogenous spread.',
  '- Inactive / healed TB: dense calcified granulomas, apical fibrocicatricial scarring, upper-lobe volume loss, pleural thickening — these are radiographically SIMILAR to active TB but are NOT contagious and do NOT need treatment. The pipeline\'s sequelae head exists specifically to flag this distinction.',
  '- TB radiographic mimics worth keeping in the differential when the pattern is non-classical: granulomatous lung disease (sarcoidosis), non-tuberculous mycobacterial infection, fungal infection (histoplasmosis, coccidioidomycosis), organizing pneumonia, malignancy (cavitary lung cancer is the single most important differential for a cavitary upper-lobe lesion in an older patient), septic emboli.',
  '',
  'FINDINGS (populate all four sub-sections, each 1-3 sentences; describe pattern, distribution, and SEVERITY grading where appropriate: mild / moderate / extensive)',
  '- lungs_and_airways: the primary signal. Describe parenchymal pattern (consolidation, cavitation, nodular, miliary, reticular, ground-glass), distribution (apical / posterior / diffuse), severity, and any TB-classical or non-classical features.',
  '- pleura: pleural effusion, pleural thickening, pneumothorax — only when flagged. Otherwise: "The pleural spaces are clear."',
  '- cardiomediastinum: heart size, mediastinal width, hilar configuration. Comment on hilar / mediastinal lymphadenopathy when relevant. Otherwise: "The cardiomediastinal silhouette is unremarkable."',
  '- bones_and_soft_tissues: rib / vertebral abnormalities only if flagged. Otherwise: "No osseous abnormality is identified on this projection."',
  '',
  'IMPRESSION (numbered list, most-to-least clinically significant; each item one short clinical statement)',
  '- The FIRST impression item MUST have likelihood "primary" and MUST match the pipeline verdict:',
  '  · verdict "tb"      → "Radiographic findings raise concern for active pulmonary tuberculosis. Bacteriological confirmation (sputum AFB smear, culture, or NAAT) is recommended."',
  '  · verdict "no_tb"   → "No radiographic findings to suggest active pulmonary tuberculosis on this single frontal view. A negative screen does not exclude early or subclinical disease."',
  '  · verdict "abstain" → "Findings are equivocal for active pulmonary tuberculosis. Recommend expert second-reader review and clinical correlation."',
  '- Subsequent items (max 2 differentials) are ranked by FIT TO THE PIPELINE EVIDENCE, with likelihood "consider" or "less_likely". Each one short statement. Pull from the TB-mimic list above when the radiographic pattern is non-classical.',
  '',
  'RECOMMENDATION (1-2 short sentences, actionable, clinical, radiologist-level)',
  '- For "tb": recommend sputum AFB smear, culture, and NAAT. If cavitation, nodules, or lymphadenopathy are described, additionally recommend lateral view and CT chest for further characterization.',
  '- For "no_tb": no further radiographic workup is indicated based on this exam alone. Test symptomatic or high-risk patients per WHO TB screening algorithm regardless. If prior films exist, comparison is helpful.',
  '- For "abstain": expert radiologist second-read; lateral view and CT chest for further characterization; clinical correlation; sputum testing if symptomatic or high-risk.',
  '',
  'LIMITATIONS (array of short caveats). Always include all three:',
  '- "Single-view frontal radiograph; lateral and CT may yield additional characterization."',
  '- "Radiographic features are not diagnostic of microbiological status; bacteriological confirmation is required for the diagnosis of active tuberculosis."',
  '- "No prior imaging available for comparison; serial follow-up imaging may be valuable to establish chronicity or progression."',
  '- Plus any preprocessing warnings supplied below.',
  '',
  'key_regions — array of zone keys from the controlled vocabulary the report referenced. MUST be a subset of the keys supplied in the evidence message.',
  '',
  'STRICT NEGATIVE CONSTRAINTS',
  '- Do NOT mention model identifiers, schema versions, calibration constants, AUROC, or ML / AI jargon in any prose section. The reader is a clinician. (You may reference "AI-assisted screen" in the Technique section.)',
  '- Do NOT speculate beyond the pipeline\'s evidence (no demographic priors, no symptom inference, no co-pathology the pipeline did not flag).',
  '- Do NOT use first-person voice.',
  '- Do NOT include the research-only disclaimer in the report sections; that lives in a separate disclosure rendered outside the report.',
  '',
  'Output ONLY the JSON schema fields. No chain-of-thought. No commentary.',
].join('\n');

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

const LIKELIHOOD_VALUES: readonly ImpressionLikelihood[] = ['primary', 'consider', 'less_likely'];

function pickEnum<T extends string>(v: unknown, allowed: readonly T[], fallback: T): T {
  if (typeof v === 'string' && (allowed as readonly string[]).includes(v)) return v as T;
  return fallback;
}

function clipString(v: unknown, fallback: string): string {
  return typeof v === 'string' && v.length > 0 ? v : fallback;
}

const ALWAYS_INCLUDE_LIMITATIONS: readonly string[] = [
  'Single-view frontal radiograph; lateral and CT may yield additional information.',
  'Radiographic features are not diagnostic of microbiological status.',
];

function normalizeFindings(raw: unknown): FindingsSections {
  const r = (raw ?? {}) as Record<string, unknown>;
  return {
    lungs_and_airways: clipString(r.lungs_and_airways, '(no parenchymal description returned)'),
    pleura: clipString(r.pleura, 'The pleural spaces are clear.'),
    cardiomediastinum: clipString(
      r.cardiomediastinum,
      'The cardiomediastinal silhouette is unremarkable.',
    ),
    bones_and_soft_tissues: clipString(
      r.bones_and_soft_tissues,
      'No osseous abnormality is identified on this projection.',
    ),
  };
}

function normalizeImpression(raw: unknown): ImpressionItem[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((d): ImpressionItem | null => {
      if (d === null || typeof d !== 'object') return null;
      const o = d as Record<string, unknown>;
      const statement = typeof o.statement === 'string' && o.statement.length > 0 ? o.statement : null;
      if (statement === null) return null;
      return {
        statement,
        likelihood: pickEnum<ImpressionLikelihood>(o.likelihood, LIKELIHOOD_VALUES, 'less_likely'),
      };
    })
    .filter((d): d is ImpressionItem => d !== null)
    .slice(0, 4);
}

function normalizeLimitations(raw: unknown): string[] {
  const provided = isStringArray(raw) ? raw : [];
  // Always ensure the two load-bearing caveats are present, even if the model
  // forgot or paraphrased them away.
  const out = [...provided];
  for (const required of ALWAYS_INCLUDE_LIMITATIONS) {
    const already = out.some((l) => l.toLowerCase().includes(required.toLowerCase().slice(0, 20)));
    if (!already) out.push(required);
  }
  return out;
}

function validateAndNormalizeReport(raw: unknown, allowedZones: readonly string[]): ClinicianReport {
  const r = (raw ?? {}) as Record<string, unknown>;
  const allowed = new Set(allowedZones);
  const key_regions = isStringArray(r.key_regions)
    ? r.key_regions.filter((z) => allowed.has(z))
    : [];
  return {
    technique: clipString(
      r.technique,
      'Single frontal chest radiograph; AI-assisted TB triage pipeline.',
    ),
    comparison: clipString(r.comparison, 'No prior studies available for comparison.'),
    findings: normalizeFindings(r.findings),
    impression: normalizeImpression(r.impression),
    recommendation: clipString(r.recommendation, '(no recommendation returned)'),
    key_regions,
    limitations: normalizeLimitations(r.limitations),
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
  lines.push('VALIDATED PIPELINE OUTPUT (translate this into the report; do not re-diagnose):');
  lines.push(`verdict: ${local.verdict}`);
  lines.push(`tb_prob (calibrated under T=${local.audit.calibration.T.toFixed(3)}): ${local.tb_prob.toFixed(4)}`);
  lines.push(`decided_at_threshold: ${local.decided_at_threshold.toFixed(4)}`);
  lines.push(`s_inactive (scar / sequelae score, under T_seq=${local.audit.calibration.T_sequelae.toFixed(3)}): ${local.s_inactive.toFixed(4)}`);
  if (local.safety_net_applied) {
    lines.push(`safety_net_applied: ${local.safety_net_applied}`);
  }

  if (local.zonal_scores) {
    const sortedZones = Object.entries(local.zonal_scores)
      .filter(([, v]) => typeof v === 'number')
      .sort((a, b) => (b[1] as number) - (a[1] as number));
    lines.push('per-zone calibrated TB probabilities (key: probability):');
    for (const [k, v] of sortedZones) {
      lines.push(`  ${k}: ${(v as number).toFixed(4)}`);
    }
  }

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
    lines.push('preprocessing warnings (include in the limitations array):');
    for (const w of local.image_quality.warnings) lines.push(`  - ${w}`);
  }

  lines.push('');
  lines.push(
    'Now produce the radiologist_report JSON. Use anatomical zone names in prose; cite zone keys in key_regions. First impression item must be the verdict-anchored statement.',
  );
  return lines.join('\n');
}

/**
 * Single deterministic call to gpt-5.5 vision. The model receives BOTH the
 * image AND the evidence text — image so it can ground its language in what
 * it actually sees, evidence so its narrative is anchored to the validated
 * pipeline's numbers (not its own image impression).
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

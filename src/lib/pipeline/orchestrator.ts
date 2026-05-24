import type {
  Adjudication,
  EnsembleMember,
  EnsembleResult,
  PipelineEvent,
  PipelineRun,
  ProviderLogEntry,
  QualityResult,
  RagNeighbor,
  RagResult,
  Settings,
  Verdict,
} from '@/lib/types';
import { ENSEMBLE_WEIGHTS, KNN_K } from '@/lib/defaults';
import { deriveCapabilities } from '@/store/settings';
import { blobToDataURL, clamp, shortId } from '@/lib/utils';
import { classifyWithFallback, cosineSimilarity, embedWithFallback } from '@/lib/providers/classify';
import { openaiJSON } from '@/lib/providers/openai';
import { getEmbeddedCases } from '@/lib/db';
import { buildGeneralStageConfig, buildTbStageConfig } from './stageConfigs';
import { QUALITY_PROMPT, QUALITY_SCHEMA } from './prompts';
import {
  isBorderlineForConsistencyCheck,
  vlmTriage,
  type TbScreenResult,
  type TriageSubmission,
  type VlmAuditPins,
  type VlmTriageCall,
} from './vlmTriage';
import { applyVlmEscalation } from './vlmEscalation';

type Emit = (e: PipelineEvent) => void;

/**
 * ORCHESTRATOR — Milestone 21 PIVOT (VLM-PRIMARY, see CASE_STUDY M21 + brief).
 *
 * Before M21: a three-member perception ensemble (HF TB head + HF general CXR
 * head + gpt-5.5 vision self-consistency K=3) feeding a streamed gpt-5.5
 * adjudicator. M20 confirmed both HF heads are off the free hf-inference router
 * (HTTP 400 "Model not supported by provider hf-inference"). The deployed app
 * had no working primary perception.
 *
 * After M21: gpt-5.5 vision via the Responses API's structured-output schema
 * IS the primary perception. The submission carries enough structured signal
 * (screen_result, uncalibrated scores, mimic features, safety flags) that the
 * streamed adjudicator stage is no longer the load-bearing decision step — the
 * VLM submission + a path-specific safety net IS the decision. We keep the
 * quality gate, the optional HF perception members (when configured), and the
 * RAG retrieval for the audit trail, but the verdict is built directly from
 * the submission and the M21 escalation rule.
 *
 * Why no self-consistency K-sample here: that pattern was a calibration trick
 * when the VLM was the THIRD ensemble member. Now that it is the PRIMARY,
 * correlated self-sampling looks like ensembling but isn't. Instead we fire
 * ONE deterministic primary call, and for borderline / scar-mimic results,
 * ONE independent verifier call with different prompt phrasing. If primary
 * and verifier disagree on screen_result → abstain. See vlmTriage.ts.
 */

const SEVERITY: Record<Verdict, number> = { no_tb: 0, abstain: 1, tb: 2 };
const FROM_SEVERITY: Verdict[] = ['no_tb', 'abstain', 'tb'];
/** Take the more cautious (higher-severity) verdict: tb > abstain > no_tb. */
function mostCautious(...verdicts: Verdict[]): Verdict {
  return FROM_SEVERITY[Math.max(...verdicts.map((v) => SEVERITY[v] ?? 1))] ?? 'abstain';
}

/** Map the VLM submission's `tb_screen_result` enum into the project Verdict union. */
function screenResultToVerdict(r: TbScreenResult): Verdict {
  if (r === 'screen_positive') return 'tb';
  if (r === 'screen_negative') return 'no_tb';
  return 'abstain';
}

/**
 * Confidence rendered to the verdict card. Maps the VLM's `confidence_band` +
 * `tb_score_uncalibrated` onto a 0..100 value. Deliberately conservative so the
 * dial never reads "97% confident" on an uncalibrated VLM output.
 */
function confidenceFromBand(s: TriageSubmission): number {
  const base = s.confidence_band === 'high' ? 75 : s.confidence_band === 'medium' ? 55 : 35;
  // Nudge slightly by how extreme the score is — a 0.95 score in "high" band
  // earns a few extra points without ever exceeding 90 (we are NOT calibrated).
  const score = s.tb_score_uncalibrated;
  const extreme = Math.abs(score - 0.5) * 2; // 0..1
  return clamp(Math.round(base + extreme * 15), 0, 90);
}

interface VlmCombineResult {
  adjudication: Adjudication;
  /** Synthetic ensemble result so the existing UI Details view still has numbers to show. */
  ensemble: EnsembleResult;
}

/**
 * Build the Adjudication + a synthetic EnsembleResult from a VLM submission and
 * (optional) verifier submission. Applies:
 *   1. consistency-check abstain when primary + verifier disagree on screen_result,
 *   2. the M21 VLM-path escalate-not-clear rule (vlmEscalation.ts),
 *   3. mostCautious combine across the model's own verdict + the rules above.
 */
function combineVlmIntoAdjudication(args: {
  primary: VlmTriageCall;
  verifier: VlmTriageCall | null;
  hfMembers: EnsembleMember[];
}): VlmCombineResult {
  const { primary, verifier, hfMembers } = args;
  const subm = primary.submission;

  const modelVerdict = screenResultToVerdict(subm.tb_screen_result);

  const consistencyRan = verifier !== null;
  const consistencyDisagreed =
    verifier !== null &&
    verifier.submission.tb_screen_result !== subm.tb_screen_result;

  const reasons: string[] = [];
  let preEscalateVerdict = modelVerdict;
  if (consistencyDisagreed && verifier) {
    preEscalateVerdict = 'abstain';
    reasons.push(
      `consistency check disagreed (primary=${subm.tb_screen_result}, verifier=${verifier.submission.tb_screen_result})`,
    );
  }

  if (subm.refusal_or_limitation) {
    reasons.push(`VLM limitation: ${subm.refusal_or_limitation}`);
    preEscalateVerdict = mostCautious(preEscalateVerdict, 'abstain');
  }

  const vlmEsc = applyVlmEscalation({
    verdict: preEscalateVerdict,
    tbScoreUncalibrated: subm.tb_score_uncalibrated,
    scarShapeScoreUncalibrated: subm.scar_shape_score_uncalibrated,
    mimicFeatures: subm.mimic_features_present,
  });
  if (vlmEsc.escalated && vlmEsc.reason) reasons.push(vlmEsc.reason);

  const finalVerdict = vlmEsc.verdict;
  const escalated = finalVerdict !== modelVerdict;

  const audit: VlmAuditPins & {
    consistency_check_ran: boolean;
    consistency_check_disagreed: boolean;
  } = {
    ...primary.audit,
    consistency_check_ran: consistencyRan,
    consistency_check_disagreed: consistencyDisagreed,
  };

  const adjudication: Adjudication = {
    verdict: finalVerdict,
    confidence: confidenceFromBand(subm),
    rationale: escalated
      ? `[Safety net: ${modelVerdict} → ${finalVerdict}] ${subm.short_rationale}`
      : subm.short_rationale,
    abstain_reason:
      finalVerdict === 'abstain'
        ? `Refer: ${reasons.join('; ') || 'weak or uncertain evidence'}.`
        : undefined,
    auto_abstained: finalVerdict === 'abstain' && modelVerdict !== 'abstain',
    auto_abstain_reasons: reasons,
    perception_unavailable: false,
    perception_path: 'vlm-primary',
    vlm_audit: audit,
    screening: {
      policyVerdict: finalVerdict,
      modelVerdict,
      fusedProb: subm.tb_score_uncalibrated,
      vlmProb: subm.tb_score_uncalibrated,
      vlmUncertainty: subm.confidence_band === 'low' ? 0.3 : subm.confidence_band === 'medium' ? 0.15 : 0.05,
    },
  };

  // Synthetic ensemble: the VLM is one "member" with weight 1 in the UI; the
  // optional HF members are passed through so the trace still shows whatever
  // signal we got. Keeps VerdictCard's "Details & stats" useful without
  // pretending the HF heads contributed to the decision.
  const vlmMember: EnsembleMember = {
    id: 'vlm',
    label: 'GPT-5.5 Vision (primary)',
    weight: 1,
    status: 'done',
    tb_prob: subm.tb_score_uncalibrated,
    provider_used: 'openai',
    latency_ms: primary.latencyMs,
    raw: subm,
    findings: subm.abnormality_localization,
    uncertainty: subm.confidence_band === 'low' ? 0.3 : subm.confidence_band === 'medium' ? 0.15 : 0.05,
    samples: consistencyRan ? 2 : 1,
  };

  const members: EnsembleMember[] = [vlmMember, ...hfMembers];
  const returning = members.filter((m) => m.tb_prob !== null);
  const probs = returning.map((m) => m.tb_prob as number);
  const disagreement = probs.length > 1 ? Math.max(...probs) - Math.min(...probs) : 0;
  const mean = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0;
  const variance = probs.length
    ? probs.reduce((a, b) => a + (b - mean) ** 2, 0) / probs.length
    : 0;
  const replicateFallbackCount = members.filter((m) => m.provider_used === 'replicate').length;
  const ensemble: EnsembleResult = {
    members,
    weightedScore: subm.tb_score_uncalibrated,
    std: Math.sqrt(variance),
    disagreement,
    replicateFallbackCount,
  };

  return { adjudication, ensemble };
}

/**
 * Run the full pipeline. Emits PipelineEvents live; resolves with the
 * aggregate PipelineRun for persistence + export.
 *
 * STAGE FLOW (M21):
 *   1. quality       — gpt-5.5 vision quality gate (unchanged from M1).
 *   2. ensemble.vlm  — gpt-5.5 vision primary triage (NEW; structured-output schema).
 *      + ensemble.vlm — verifier call when borderline (NEW; same stage id, different prompt).
 *      ensemble.tb / ensemble.general run best-effort when configured; reported in the trace.
 *   3. rag           — retrieval (unchanged).
 *   4. adjudicate    — REPLACED. On the VLM path the submission IS the adjudication;
 *      the stage emits done immediately with the synthesized verdict.
 *   5. verdict       — finalize.
 */
export async function runPipeline(
  image: Blob,
  imageName: string,
  settings: Settings,
  emit: Emit,
): Promise<PipelineRun> {
  const caps = deriveCapabilities(settings);
  const providerLog: ProviderLogEntry[] = [];
  const run: PipelineRun = {
    id: shortId(),
    createdAt: Date.now(),
    imageName,
    quality: null,
    ensemble: null,
    rag: null,
    adjudication: null,
    halted: null,
    providerLog,
    fallbackRate: 0,
    modelVersions: {
      perception_primary: 'gpt-5.5-vision (Responses API, structured-output)',
      tb_classifier_hf: settings.overrides.tbClassifierHf || '(unset — VLM-primary path)',
      general_cxr_hf: settings.overrides.generalCxrHf || '(unset)',
      adjudicator: settings.models.adjudicator,
      adjudicator_fallback: settings.models.adjudicatorFallback,
      embedding_endpoint: settings.overrides.embeddingEndpointUrl || '(none)',
    },
  };

  if (!caps.hasOpenAI) {
    run.halted = {
      reason: 'OpenAI API key not set — required for the gpt-5.5 vision primary perception.',
      stage: 'quality',
    };
    emit({ type: 'halted', reason: run.halted.reason, stage: 'quality' });
    return run;
  }

  const dataUrl = await blobToDataURL(image);

  // ----------------------------------------------------------------------
  // Stage 1 — Quality gate (gpt-5.5 vision, no fallback)
  // ----------------------------------------------------------------------
  emit({ type: 'stage_status', stage: 'quality', status: 'running' });
  try {
    const q = await openaiJSON<QualityResult>({
      apiKey: settings.openaiKey,
      model: settings.models.adjudicator,
      fallbackModel: settings.models.adjudicatorFallback,
      prompt: QUALITY_PROMPT,
      imageDataUrl: dataUrl,
      schema: QUALITY_SCHEMA,
    });
    run.quality = q.data;
    providerLog.push({ stage: 'quality', provider_used: 'openai', latency_ms: q.latencyMs, fell_back: q.fellBack });
    emit({ type: 'quality_done', result: q.data });
    emit({ type: 'stage_status', stage: 'quality', status: 'done' });

    if (!q.data.is_cxr || q.data.quality === 'unreadable') {
      run.halted = { reason: q.data.reason || 'Image rejected by quality gate.', stage: 'quality' };
      emit({ type: 'halted', reason: run.halted.reason, stage: 'quality' });
      return run;
    }
  } catch (err) {
    const message = (err as Error).message;
    run.halted = { reason: `Quality gate failed: ${message}`, stage: 'quality' };
    emit({ type: 'error', stage: 'quality', message });
    emit({ type: 'halted', reason: run.halted.reason, stage: 'quality' });
    return run;
  }

  // ----------------------------------------------------------------------
  // Stage 2 — Perception
  //   2A. (primary)   VLM triage via Responses API structured-output.
  //   2B. (verifier)  ONE extra independent call ONLY if borderline.
  //   2C. (auxiliary) HF TB head + general CXR head — best-effort. NOT decision-driving.
  // ----------------------------------------------------------------------
  const classifyDeps = {
    hfToken: settings.hfToken,
    replicateToken: settings.replicateToken,
  };

  // 2C kicks off in parallel with the VLM call — purely advisory in the trace.
  const tbMemberP: Promise<EnsembleMember | null> = (async () => {
    if (!caps.hasHF || !settings.overrides.tbClassifierHf.trim()) return null;
    emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'running' });
    try {
      const r = await classifyWithFallback(image, buildTbStageConfig(settings), {
        ...classifyDeps,
        onFallback: (from, to, reason) => {
          emit({ type: 'fallback_fired', stage: 'ensemble.tb', from, to });
          emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'fallback', note: reason });
        },
      });
      const m: EnsembleMember = {
        id: 'tb',
        label: 'TB Classifier (auxiliary)',
        weight: ENSEMBLE_WEIGHTS.tb,
        status: 'done',
        tb_prob: r.tb_prob,
        provider_used: r.provider_used,
        latency_ms: r.latency_ms,
        raw: r.raw,
      };
      providerLog.push({
        stage: 'ensemble.tb',
        provider_used: r.provider_used,
        latency_ms: r.latency_ms,
        fell_back: r.provider_used === 'replicate',
      });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'done' });
      return m;
    } catch (err) {
      const message = (err as Error).message;
      const m: EnsembleMember = {
        id: 'tb',
        label: 'TB Classifier (auxiliary)',
        weight: ENSEMBLE_WEIGHTS.tb,
        status: 'error',
        tb_prob: null,
        provider_used: null,
        latency_ms: null,
        raw: null,
        error: message,
      };
      providerLog.push({ stage: 'ensemble.tb', provider_used: null, latency_ms: null, fell_back: false });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'error', stage: 'ensemble.tb', message });
      emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'error' });
      return m;
    }
  })();

  const generalMemberP: Promise<EnsembleMember | null> = (async () => {
    if (!caps.hasHF || !settings.overrides.generalCxrHf.trim()) return null;
    emit({ type: 'stage_status', stage: 'ensemble.general', status: 'running' });
    try {
      const r = await classifyWithFallback(image, buildGeneralStageConfig(settings), {
        ...classifyDeps,
        onFallback: (from, to, reason) => {
          emit({ type: 'fallback_fired', stage: 'ensemble.general', from, to });
          emit({ type: 'stage_status', stage: 'ensemble.general', status: 'fallback', note: reason });
        },
      });
      const m: EnsembleMember = {
        id: 'general',
        label: 'General CXR (auxiliary)',
        weight: ENSEMBLE_WEIGHTS.general,
        status: 'done',
        tb_prob: r.tb_prob,
        provider_used: r.provider_used,
        latency_ms: r.latency_ms,
        raw: r.raw,
      };
      providerLog.push({
        stage: 'ensemble.general',
        provider_used: r.provider_used,
        latency_ms: r.latency_ms,
        fell_back: r.provider_used === 'replicate',
      });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'stage_status', stage: 'ensemble.general', status: 'done' });
      return m;
    } catch (err) {
      const message = (err as Error).message;
      const m: EnsembleMember = {
        id: 'general',
        label: 'General CXR (auxiliary)',
        weight: ENSEMBLE_WEIGHTS.general,
        status: 'error',
        tb_prob: null,
        provider_used: null,
        latency_ms: null,
        raw: null,
        error: message,
      };
      providerLog.push({ stage: 'ensemble.general', provider_used: null, latency_ms: null, fell_back: false });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'error', stage: 'ensemble.general', message });
      emit({ type: 'stage_status', stage: 'ensemble.general', status: 'error' });
      return m;
    }
  })();

  // 2A. Primary VLM triage. THIS IS THE DECISION-DRIVING CALL.
  emit({ type: 'stage_status', stage: 'ensemble.vlm', status: 'running' });
  let primary: VlmTriageCall;
  try {
    primary = await vlmTriage({
      apiKey: settings.openaiKey,
      primaryModel: settings.models.adjudicator,
      fallbackModel: settings.models.adjudicatorFallback,
      imageDataUrl: dataUrl,
      role: 'primary',
    });
    providerLog.push({
      stage: 'ensemble.vlm',
      provider_used: 'openai',
      latency_ms: primary.latencyMs,
      fell_back: primary.fellBack,
    });
    run.modelVersions['vlm_model_used'] = primary.audit.model_id_from_response;
    run.modelVersions['vlm_prompt_hash'] = primary.audit.prompt_hash;
    run.modelVersions['vlm_schema_version'] = primary.audit.schema_version;
    run.modelVersions['vlm_schema_hash'] = primary.audit.schema_hash;
    run.modelVersions['vlm_image_preprocessing'] = primary.audit.image_preprocessing_version;
  } catch (err) {
    const message = (err as Error).message;
    emit({ type: 'error', stage: 'ensemble.vlm', message });
    emit({ type: 'stage_status', stage: 'ensemble.vlm', status: 'error' });
    const adjudication: Adjudication = {
      verdict: 'abstain',
      confidence: 0,
      rationale:
        'gpt-5.5 vision primary perception failed; defaulting to abstain. The deployed app needs an OpenAI key with vision access.',
      abstain_reason: message,
      auto_abstained: true,
      auto_abstain_reasons: [`vlm primary error: ${message}`],
      perception_unavailable: true,
      perception_path: 'perception-unavailable',
    };
    run.adjudication = adjudication;
    emit({ type: 'adjudicate_done', result: adjudication });
    emit({ type: 'stage_status', stage: 'adjudicate', status: 'error' });
    return run;
  }

  // 2B. Verifier call (consistency check) ONLY when borderline / scar-mimic.
  let verifier: VlmTriageCall | null = null;
  if (isBorderlineForConsistencyCheck(primary.submission)) {
    emit({
      type: 'stage_status',
      stage: 'ensemble.vlm',
      status: 'running',
      note: 'borderline result — running consistency check',
    });
    try {
      verifier = await vlmTriage({
        apiKey: settings.openaiKey,
        primaryModel: settings.models.adjudicator,
        fallbackModel: settings.models.adjudicatorFallback,
        imageDataUrl: dataUrl,
        role: 'verifier',
      });
      providerLog.push({
        stage: 'ensemble.vlm',
        provider_used: 'openai',
        latency_ms: verifier.latencyMs,
        fell_back: verifier.fellBack,
      });
    } catch (err) {
      // A failed verifier doesn't poison the verdict — it just means the
      // consistency check didn't run, which is recorded in the audit pins.
      emit({ type: 'error', stage: 'ensemble.vlm', message: `verifier failed: ${(err as Error).message}` });
    }
  }
  emit({ type: 'stage_status', stage: 'ensemble.vlm', status: 'done' });

  // Wait for the optional HF auxiliary calls to settle so the trace shows them.
  const [tbAux, generalAux] = await Promise.all([tbMemberP, generalMemberP]);
  const hfMembers: EnsembleMember[] = [tbAux, generalAux].filter(
    (m): m is EnsembleMember => m !== null,
  );

  // Combine: VLM submission + verifier + escalation -> Adjudication.
  const combined = combineVlmIntoAdjudication({ primary, verifier, hfMembers });
  run.ensemble = combined.ensemble;
  emit({ type: 'ensemble_member', member: combined.ensemble.members[0] as EnsembleMember });
  emit({ type: 'ensemble_done', result: combined.ensemble });

  // ----------------------------------------------------------------------
  // Stage 3 — RAG retrieval (kNN over labeled corpus). Display-only on the VLM path.
  // ----------------------------------------------------------------------
  let rag: RagResult = { neighbors: [], embedding_provider: null, skipped: false };
  if (!caps.hasEmbedding) {
    rag = {
      neighbors: [],
      embedding_provider: null,
      skipped: true,
      skipReason:
        'No embedding provider configured. Set the CXR-Foundation HF Inference Endpoint URL or a Replicate CLIP fallback in Settings to enable retrieval.',
    };
    emit({ type: 'stage_status', stage: 'rag', status: 'skipped', note: rag.skipReason });
    emit({ type: 'rag_done', result: rag });
  } else {
    emit({ type: 'stage_status', stage: 'rag', status: 'running' });
    try {
      const embed = await embedWithFallback(image, {
        hfToken: settings.hfToken,
        replicateToken: settings.replicateToken,
        endpointUrl: settings.overrides.embeddingEndpointUrl,
        replicateModel: settings.overrides.embeddingReplicate,
        replicateVersion: settings.overrides.embeddingReplicateVersion,
        onFallback: (from, to, reason) => {
          emit({ type: 'fallback_fired', stage: 'rag', from, to });
          emit({ type: 'stage_status', stage: 'rag', status: 'fallback', note: reason });
        },
      });
      providerLog.push({
        stage: 'rag',
        provider_used: embed.provider_used,
        latency_ms: embed.latency_ms,
        fell_back: embed.provider_used === 'replicate',
      });

      const corpus = await getEmbeddedCases();
      const scored = corpus
        .map((c) => ({ c, sim: cosineSimilarity(embed.embedding, c.embedding ?? []) }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, KNN_K);
      const neighbors: RagNeighbor[] = scored.map(({ c, sim }) => ({
        filename: c.filename,
        similarity: clamp(sim, 0, 1),
        label: c.label,
        thumbUrl: URL.createObjectURL(c.blob),
      }));
      rag = { neighbors, embedding_provider: embed.provider_used, skipped: false };
      emit({ type: 'rag_done', result: rag });
      emit({ type: 'stage_status', stage: 'rag', status: 'done' });
    } catch (err) {
      const message = (err as Error).message;
      rag = { neighbors: [], embedding_provider: null, skipped: true, skipReason: `Retrieval failed: ${message}` };
      emit({ type: 'error', stage: 'rag', message });
      emit({ type: 'stage_status', stage: 'rag', status: 'error' });
      emit({ type: 'rag_done', result: rag });
    }
  }
  run.rag = rag;

  // ----------------------------------------------------------------------
  // Stage 4 — Adjudication (synthetic on the VLM path; the submission IS the adjudication).
  // ----------------------------------------------------------------------
  emit({ type: 'stage_status', stage: 'adjudicate', status: 'running' });
  run.adjudication = combined.adjudication;
  emit({ type: 'adjudicate_done', result: combined.adjudication });
  emit({ type: 'stage_status', stage: 'adjudicate', status: 'done' });

  // ----------------------------------------------------------------------
  // Stage 5 — finalize: fallback rate + verdict
  // ----------------------------------------------------------------------
  const classifierStages = providerLog.filter((p) =>
    ['ensemble.tb', 'ensemble.general', 'rag'].includes(p.stage),
  );
  const fellBack = classifierStages.filter((p) => p.fell_back).length;
  run.fallbackRate = classifierStages.length ? fellBack / classifierStages.length : 0;
  emit({ type: 'stage_status', stage: 'verdict', status: 'done' });

  return run;
}

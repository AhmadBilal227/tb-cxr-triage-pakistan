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
import { KNN_K } from '@/lib/defaults';
import { deriveCapabilities } from '@/store/settings';
import { blobToDataURL, clamp, shortId } from '@/lib/utils';
import { cosineSimilarity, embedWithFallback } from '@/lib/providers/classify';
import { openaiJSON } from '@/lib/providers/openai';
import { getEmbeddedCases } from '@/lib/db';
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
import {
  applySequelaeEscalation,
  DEFAULT_BORDERLINE_HIGH,
} from './sequelaeEscalation';
import { localTriage, type LocalTriageResult } from '@/lib/providers/localTriage';

type Emit = (e: PipelineEvent) => void;

/**
 * ORCHESTRATOR — Milestone 23 (HF removed).
 *
 * Pipeline shape after M23:
 *   1. quality       — gpt-5.5 vision quality gate (OpenAI key required).
 *   2. perception    — single-path, in priority order:
 *        a) LOCAL-MODE PRIMARY (when settings.localMode === true AND the FastAPI
 *           server is reachable). Validated Rad-DINO + TXRV + TBHeadT2 +
 *           InactiveSequelaeHead under the calibrated T/T_sequelae. Optional
 *           gpt-5.5 verifier fires only on borderline tb_prob or scar-mimic
 *           s_inactive; disagreement forces ABSTAIN. Emits under stage id
 *           `ensemble.tb` (the existing UI trace card carries it).
 *        b) VLM-PRIMARY (gpt-5.5 vision via Responses API structured-output).
 *           Used when local mode is off OR the local server is unreachable.
 *           ONE deterministic primary call + ONE verifier call when borderline.
 *           Emits under stage id `ensemble.vlm`.
 *   3. rag           — kNN retrieval (Replicate-only embeddings). Display-only.
 *   4. adjudicate    — synthetic; the perception submission IS the adjudication.
 *   5. verdict       — finalize.
 *
 * Hugging Face was removed in M23 — the free hf-inference router dropped every
 * default classifier we relied on through 2024-2025, and we already have a
 * strictly better local path (the M22 server runs the validated 0.922-AUROC
 * trained model). There is no "ensemble" anymore: the perception is single-
 * path. The synthetic EnsembleResult emitted to the UI is preserved as a
 * single-member shape so existing trace cards keep rendering — but the
 * pretense of a multi-member vote is gone, and the orchestrator no longer fires
 * any HF call, holds any HF token, or wires any HF model id.
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

// ---------------------------------------------------------------------------
// Local-mode helpers (Milestone 22, unchanged in M23).
//
// localBorderline: when the calibrated `tb_prob` from the local server sits
// inside [0.35, 0.65] OR `s_inactive` exceeds the scar-recall threshold (0.7126),
// we ALSO fire the gpt-5.5 vision call — it doesn't decide, it consistency-checks.
// ---------------------------------------------------------------------------
const LOCAL_BORDERLINE_LOW = 0.35;
const LOCAL_BORDERLINE_HIGH_FOR_GPT_CHECK = 0.65;
const LOCAL_S_INACTIVE_TRIGGER_FOR_GPT_CHECK = 0.7126;

export function isLocalBorderlineForGptCheck(local: LocalTriageResult): boolean {
  if (
    local.tb_prob >= LOCAL_BORDERLINE_LOW &&
    local.tb_prob <= LOCAL_BORDERLINE_HIGH_FOR_GPT_CHECK
  ) {
    return true;
  }
  if (local.s_inactive >= LOCAL_S_INACTIVE_TRIGGER_FOR_GPT_CHECK) {
    return true;
  }
  return false;
}

/** Map the local pipeline's deterministic verdict back to TbScreenResult so the
 *  GPT verifier consistency check uses the SAME alphabet (screen_positive |
 *  screen_negative | abstain). */
function localVerdictToScreen(v: Verdict): TbScreenResult {
  if (v === 'tb') return 'screen_positive';
  if (v === 'no_tb') return 'screen_negative';
  return 'abstain';
}

interface LocalCombineResult {
  adjudication: Adjudication;
  ensemble: EnsembleResult;
}

/**
 * Build the Adjudication + synthetic EnsembleResult from a local-server
 * TriageResult and (optional) gpt-5.5 verifier submission.
 *
 * Decision flow:
 *   1. Base verdict = local.verdict (already calibrated under T and run through
 *      the deterministic sequelaeEscalation rule on the SERVER side).
 *   2. If a verifier ran and disagreed on screen_result → ABSTAIN.
 *   3. Re-apply the sequelae escalation client-side as a belt-and-braces step:
 *      idempotent on the server side; only differs if the server-side rule
 *      ever drifts.
 *
 * Confidence is rendered from `tb_prob`: above the validated threshold (0.6105)
 * scales linearly to ~95; below the threshold scales linearly to ~5. This
 * is the ONLY place in the project that allows a "high-confidence" number,
 * because it IS a calibrated probability from a validated head.
 */
export function combineLocalIntoAdjudication(args: {
  local: LocalTriageResult;
  gptVerifier: VlmTriageCall | null;
}): LocalCombineResult {
  const { local, gptVerifier } = args;

  const baseVerdict: Verdict = local.verdict;
  const reasons: string[] = [];
  if (local.safety_net_applied) {
    reasons.push(`server safety net: ${local.safety_net_applied}`);
  }

  // Verifier consistency check — disagreement forces ABSTAIN. We map the local
  // verdict back to the screen_result enum so the comparison is apples-to-apples.
  let finalVerdict: Verdict = baseVerdict;
  let verifierDisagreed = false;
  if (gptVerifier) {
    const localScreen = localVerdictToScreen(baseVerdict);
    const verifierScreen = gptVerifier.submission.tb_screen_result;
    if (verifierScreen !== localScreen) {
      verifierDisagreed = true;
      finalVerdict = 'abstain';
      reasons.push(
        `gpt verifier disagreed (local=${localScreen}, verifier=${verifierScreen})`,
      );
    }
  }

  // Belt-and-braces: re-apply the sequelae escalation client-side. Server
  // already ran it — this is here so a future server-rule drift surfaces
  // immediately rather than silently.
  const clientEsc = applySequelaeEscalation({
    verdict: finalVerdict,
    tbProb: local.tb_prob,
    sInactive: local.s_inactive,
    borderlineHigh: DEFAULT_BORDERLINE_HIGH,
  });
  if (clientEsc.escalated) {
    finalVerdict = clientEsc.verdict;
    if (clientEsc.reason) reasons.push(`client safety net escalated: ${clientEsc.reason}`);
  }

  // Confidence: linear ramp on tb_prob, anchored at the validated threshold.
  // Capped at 95 — even the validated head has measured ~10% mimic FPR on NIH
  // stress, so we never claim 100.
  const distFromThr = local.tb_prob - local.decided_at_threshold;
  const span = local.tb_prob >= local.decided_at_threshold
    ? Math.max(0.001, 1 - local.decided_at_threshold)
    : Math.max(0.001, local.decided_at_threshold);
  const confidencePct = Math.round(clamp(50 + (distFromThr / span) * 45, 5, 95));

  const safetyNetApplied = local.safety_net_applied !== null || clientEsc.escalated || verifierDisagreed;
  const rationale = verifierDisagreed
    ? `[Verifier disagreement: ${baseVerdict} → ${finalVerdict}] tb_prob ${local.tb_prob.toFixed(3)} ` +
      `vs threshold ${local.decided_at_threshold.toFixed(3)}; s_inactive ${local.s_inactive.toFixed(3)}.`
    : `Local validated pipeline: tb_prob ${local.tb_prob.toFixed(3)} (calibrated under T=` +
      `${local.audit.calibration.T.toFixed(3)}) vs validated threshold ` +
      `${local.decided_at_threshold.toFixed(3)}; s_inactive ${local.s_inactive.toFixed(3)} ` +
      `(under T_seq=${local.audit.calibration.T_sequelae.toFixed(3)}).`;

  const adjudication: Adjudication = {
    verdict: finalVerdict,
    confidence: confidencePct,
    rationale,
    abstain_reason:
      finalVerdict === 'abstain'
        ? `Refer: ${reasons.join('; ') || 'safety net or verifier flagged for review'}.`
        : undefined,
    auto_abstained: finalVerdict === 'abstain' && baseVerdict !== 'abstain',
    auto_abstain_reasons: reasons,
    perception_unavailable: false,
    perception_path: 'local-onnx-via-server',
    vlm_audit: gptVerifier
      ? {
          ...gptVerifier.audit,
          consistency_check_ran: true,
          consistency_check_disagreed: verifierDisagreed,
        }
      : undefined,
    screening: {
      policyVerdict: finalVerdict,
      modelVerdict: baseVerdict,
      fusedProb: local.tb_prob,
      vlmProb: gptVerifier?.submission.tb_score_uncalibrated ?? null,
      vlmUncertainty: gptVerifier
        ? gptVerifier.submission.confidence_band === 'low'
          ? 0.3
          : gptVerifier.submission.confidence_band === 'medium'
            ? 0.15
            : 0.05
        : 0,
    },
    // M24: surface every validated-model intermediate the server emits. The UI
    // detail components (BoxEvidenceHeatmap, ZonalBars, PathologyList) render
    // only the sub-fields that exist, so dropping any one is graceful.
    local_enrichment: {
      box_evidence_grid: local.box_evidence_grid,
      zonal_scores: local.zonal_scores,
      txrv_pathologies: local.txrv_pathologies,
      crop_box: local.crop_box,
      inversion_detected: local.inversion_detected,
    },
  };

  // Synthetic ensemble: ONE member representing the local head (with its
  // calibrated tb_prob), plus the GPT verifier as a SECOND member when it ran.
  // Provider tag uses 'local-triage' for the local member (M23 — no more 'hf'
  // pseudo-tag).
  const localMember: EnsembleMember = {
    id: 'tb',
    label: 'Local validated head (Rad-DINO + TXRV + TBHeadT2)',
    weight: 1,
    status: 'done',
    tb_prob: local.tb_prob,
    provider_used: 'local-triage',
    latency_ms: local.latency_ms.total ?? null,
    raw: local,
    findings: local.image_quality.warnings,
  };
  const members: EnsembleMember[] = [localMember];
  if (gptVerifier) {
    members.push({
      id: 'vlm',
      label: 'GPT-5.5 Vision (verifier)',
      weight: 0,
      status: 'done',
      tb_prob: gptVerifier.submission.tb_score_uncalibrated,
      provider_used: 'openai',
      latency_ms: gptVerifier.latencyMs,
      raw: gptVerifier.submission,
      uncertainty:
        gptVerifier.submission.confidence_band === 'low'
          ? 0.3
          : gptVerifier.submission.confidence_band === 'medium'
            ? 0.15
            : 0.05,
      samples: 1,
    });
  }
  const probs = members.filter((m) => m.tb_prob !== null).map((m) => m.tb_prob as number);
  const meanProb = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : local.tb_prob;
  const variance = probs.length
    ? probs.reduce((a, b) => a + (b - meanProb) ** 2, 0) / probs.length
    : 0;
  const ensemble: EnsembleResult = {
    members,
    weightedScore: local.tb_prob,
    std: Math.sqrt(variance),
    disagreement: probs.length > 1 ? Math.max(...probs) - Math.min(...probs) : 0,
    replicateFallbackCount: 0,
  };

  // Suppress unused warning when safetyNetApplied isn't user-visible elsewhere;
  // the value flows through reasons[] and the audit pins.
  void safetyNetApplied;

  return { adjudication, ensemble };
}

/**
 * Build the Adjudication + a synthetic EnsembleResult from a VLM submission and
 * (optional) verifier submission. Applies:
 *   1. consistency-check abstain when primary + verifier disagree on screen_result,
 *   2. the M21 VLM-path escalate-not-clear rule (vlmEscalation.ts),
 *   3. mostCautious combine across the model's own verdict + the rules above.
 *
 * M23: dropped the `hfMembers` pass-through arg — HF auxiliary stages no longer
 * exist, so the synthetic ensemble is a single-member shape (the VLM).
 */
function combineVlmIntoAdjudication(args: {
  primary: VlmTriageCall;
  verifier: VlmTriageCall | null;
}): VlmCombineResult {
  const { primary, verifier } = args;
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

  // Synthetic ensemble: ONE member (the VLM). M23 — no auxiliary members.
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

  const ensemble: EnsembleResult = {
    members: [vlmMember],
    weightedScore: subm.tb_score_uncalibrated,
    std: 0,
    disagreement: 0,
    replicateFallbackCount: 0,
  };

  return { adjudication, ensemble };
}

/**
 * Run the full pipeline. Emits PipelineEvents live; resolves with the
 * aggregate PipelineRun for persistence + export.
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
      adjudicator: settings.models.adjudicator,
      adjudicator_fallback: settings.models.adjudicatorFallback,
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
  // Stage 2a — LOCAL-MODE PRIMARY (when localMode on AND server reachable).
  // ----------------------------------------------------------------------
  if (settings.localMode) {
    emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'running', note: 'local mode' });
    let localResult: LocalTriageResult | null = null;
    try {
      localResult = await localTriage(image, settings.localServerUrl);
      providerLog.push({
        stage: 'ensemble.tb',
        provider_used: 'local-triage',
        latency_ms: localResult.latency_ms.total ?? null,
        fell_back: false,
      });
      run.modelVersions['perception_primary'] =
        `local-onnx-via-server (model_sha=${localResult.audit.model_sha.slice(0, 14)}…, ` +
        `git=${localResult.audit.git_sha.slice(0, 7)})`;
      run.modelVersions['local_calibration_T'] = localResult.audit.calibration.T.toFixed(6);
      run.modelVersions['local_calibration_thr_at_95sens'] =
        localResult.audit.calibration.thr_at_95sens.toFixed(6);
      run.modelVersions['local_calibration_T_sequelae'] =
        localResult.audit.calibration.T_sequelae.toFixed(6);
      emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'done' });
    } catch (err) {
      const message = (err as Error).message;
      emit({ type: 'error', stage: 'ensemble.tb', message });
      emit({
        type: 'stage_status',
        stage: 'ensemble.tb',
        status: 'fallback',
        note: 'local server unreachable; falling back to gpt-5.5 vision primary',
      });
      localResult = null;
    }

    if (localResult !== null) {
      // Optional GPT verifier — only on borderline or scar-shape.
      let gptVerifier: VlmTriageCall | null = null;
      if (isLocalBorderlineForGptCheck(localResult)) {
        emit({
          type: 'stage_status',
          stage: 'ensemble.vlm',
          status: 'running',
          note: 'local result borderline — running gpt verifier as consistency check',
        });
        try {
          gptVerifier = await vlmTriage({
            apiKey: settings.openaiKey,
            primaryModel: settings.models.adjudicator,
            fallbackModel: settings.models.adjudicatorFallback,
            imageDataUrl: dataUrl,
            role: 'verifier',
          });
          providerLog.push({
            stage: 'ensemble.vlm',
            provider_used: 'openai',
            latency_ms: gptVerifier.latencyMs,
            fell_back: gptVerifier.fellBack,
          });
          emit({ type: 'stage_status', stage: 'ensemble.vlm', status: 'done' });
        } catch (e) {
          emit({
            type: 'error',
            stage: 'ensemble.vlm',
            message: `gpt verifier failed: ${(e as Error).message}`,
          });
          // verifier failure is not fatal — the local verdict still stands
        }
      }

      const combined = combineLocalIntoAdjudication({ local: localResult, gptVerifier });
      run.ensemble = combined.ensemble;
      emit({ type: 'ensemble_member', member: combined.ensemble.members[0] as EnsembleMember });
      if (combined.ensemble.members[1]) {
        emit({ type: 'ensemble_member', member: combined.ensemble.members[1] });
      }
      emit({ type: 'ensemble_done', result: combined.ensemble });

      // RAG runs display-only when local mode is the source-of-truth path.
      const rag: RagResult = {
        neighbors: [],
        embedding_provider: null,
        skipped: true,
        skipReason: 'Local mode: retrieval display-only and currently skipped (M22).',
      };
      emit({ type: 'stage_status', stage: 'rag', status: 'skipped', note: rag.skipReason });
      emit({ type: 'rag_done', result: rag });
      run.rag = rag;

      emit({ type: 'stage_status', stage: 'adjudicate', status: 'running' });
      run.adjudication = combined.adjudication;
      emit({ type: 'adjudicate_done', result: combined.adjudication });
      emit({ type: 'stage_status', stage: 'adjudicate', status: 'done' });
      run.fallbackRate = 0;
      emit({ type: 'stage_status', stage: 'verdict', status: 'done' });
      return run;
    }
    // else: local call failed -> fall through to the VLM-primary block
  }

  // ----------------------------------------------------------------------
  // Stage 2b — VLM-PRIMARY (used when local mode is off OR local mode is on
  // but the server was unreachable).
  //   2A. (primary)   VLM triage via Responses API structured-output.
  //   2B. (verifier)  ONE extra independent call ONLY if borderline.
  // ----------------------------------------------------------------------
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

  // Combine: VLM submission + verifier + escalation -> Adjudication.
  const combined = combineVlmIntoAdjudication({ primary, verifier });
  run.ensemble = combined.ensemble;
  emit({ type: 'ensemble_member', member: combined.ensemble.members[0] as EnsembleMember });
  emit({ type: 'ensemble_done', result: combined.ensemble });

  // ----------------------------------------------------------------------
  // Stage 3 — RAG retrieval (kNN over labeled corpus). Display-only on the VLM path.
  // Replicate-only embeddings (M23 — HF endpoint removed).
  // ----------------------------------------------------------------------
  let rag: RagResult = { neighbors: [], embedding_provider: null, skipped: false };
  if (!caps.hasEmbedding) {
    rag = {
      neighbors: [],
      embedding_provider: null,
      skipped: true,
      skipReason:
        'No embedding provider configured. Set a Replicate token + CLIP model in Settings to enable retrieval.',
    };
    emit({ type: 'stage_status', stage: 'rag', status: 'skipped', note: rag.skipReason });
    emit({ type: 'rag_done', result: rag });
  } else {
    emit({ type: 'stage_status', stage: 'rag', status: 'running' });
    try {
      const embed = await embedWithFallback(image, {
        replicateToken: settings.replicateToken,
        replicateModel: settings.overrides.embeddingReplicate,
        replicateVersion: settings.overrides.embeddingReplicateVersion,
      });
      providerLog.push({
        stage: 'rag',
        provider_used: embed.provider_used,
        latency_ms: embed.latency_ms,
        fell_back: false,
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
  const classifierStages = providerLog.filter((p) => p.stage === 'rag');
  const fellBack = classifierStages.filter((p) => p.fell_back).length;
  run.fallbackRate = classifierStages.length ? fellBack / classifierStages.length : 0;
  emit({ type: 'stage_status', stage: 'verdict', status: 'done' });

  return run;
}

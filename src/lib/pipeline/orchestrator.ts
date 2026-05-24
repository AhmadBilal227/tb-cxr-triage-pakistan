import type {
  Adjudication,
  CalibrationParams,
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
import { applyCalibration, effectiveWeights, fuseLogOdds } from '@/lib/calibration';
import {
  ABSTAIN_RULES,
  ENSEMBLE_WEIGHTS,
  KNN_K,
  SCREENING_POLICY,
  SELF_CONSISTENCY_K,
} from '@/lib/defaults';
import { deriveCapabilities } from '@/store/settings';
import { blobToDataURL, clamp, shortId } from '@/lib/utils';
import { classifyWithFallback, cosineSimilarity, embedWithFallback } from '@/lib/providers/classify';
import { openaiJSON, openaiStream } from '@/lib/providers/openai';
import { getEmbeddedCases } from '@/lib/db';
import { buildGeneralStageConfig, buildTbStageConfig } from './stageConfigs';
import {
  AdjudicationRaw,
  buildAdjudicatorPrompt,
  QUALITY_PROMPT,
  QUALITY_SCHEMA,
  VLM_PROMPT,
  VLM_SCHEMA,
  VlmResult,
  vlmToProb,
} from './prompts';
import { applySequelaeEscalation } from './sequelaeEscalation';

type Emit = (e: PipelineEvent) => void;

function popStats(xs: number[]): { mean: number; std: number } {
  if (xs.length === 0) return { mean: 0, std: 0 };
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const variance = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
  return { mean, std: Math.sqrt(variance) };
}

const SEVERITY: Record<Verdict, number> = { no_tb: 0, abstain: 1, tb: 2 };
const FROM_SEVERITY: Verdict[] = ['no_tb', 'abstain', 'tb'];

/**
 * Screening-biased decision policy. Asymmetric on purpose: a low bar to flag, a high
 * bar to clear. fusedProb includes the CNN once it is live; the vlmSafetyThreshold lets
 * the VLM escalate on its own (catching CNN false negatives) without being able to veto.
 * When fitted calibration params are present, uses their conformal thresholds; otherwise
 * falls back to the hard-coded SCREENING_POLICY constants.
 */
function screeningPolicy(
  fusedProb: number,
  vlmProb: number | null,
  vlmUncertainty: number,
  cal: CalibrationParams | null,
): { verdict: Verdict; reason: string } {
  const tauLow = cal?.conformal.tauLow ?? SCREENING_POLICY.negClear;
  const tauHigh = cal?.conformal.tauHigh ?? SCREENING_POLICY.tbFlag;
  const vlmSafe = cal?.vlmSafetyThreshold ?? SCREENING_POLICY.vlmSafetyThreshold;
  const flag = fusedProb >= tauHigh || (vlmProb !== null && vlmProb >= vlmSafe);
  if (flag) {
    return { verdict: 'tb', reason: `flag (fused ${fusedProb.toFixed(2)} ≥ τ_high ${tauHigh.toFixed(2)} or VLM ≥ ${vlmSafe.toFixed(2)})` };
  }
  if (fusedProb < tauLow && vlmUncertainty <= SCREENING_POLICY.maxClearUncertainty) {
    return { verdict: 'no_tb', reason: '' };
  }
  return { verdict: 'abstain', reason: `prob ${fusedProb.toFixed(2)} in [τ_low ${tauLow.toFixed(2)}, τ_high ${tauHigh.toFixed(2)}) band` };
}

/** Take the more cautious (higher-severity) verdict: tb > abstain > no_tb. */
function mostCautious(...verdicts: Verdict[]): Verdict {
  return FROM_SEVERITY[Math.max(...verdicts.map((v) => SEVERITY[v] ?? 1))] ?? 'abstain';
}

/** Deterministic guardrails. Returns the reasons an abstain is forced (empty => allowed to stand). */
function evaluateAbstainRules(
  modelConfidence: number,
  ensemble: EnsembleResult,
  rag: RagResult,
): string[] {
  const reasons: string[] = [];
  if (modelConfidence < ABSTAIN_RULES.minConfidence) {
    reasons.push(`model confidence ${modelConfidence} < ${ABSTAIN_RULES.minConfidence}`);
  }
  if (ensemble.std > ABSTAIN_RULES.maxEnsembleStd) {
    reasons.push(`ensemble std ${ensemble.std.toFixed(2)} > ${ABSTAIN_RULES.maxEnsembleStd}`);
  }
  const top1 = rag.neighbors[0]?.similarity ?? 1;
  if (
    top1 < ABSTAIN_RULES.minTop1Similarity &&
    ensemble.disagreement > ABSTAIN_RULES.maxDisagreementForLowSim
  ) {
    reasons.push(
      `top-1 retrieval similarity ${top1.toFixed(2)} < ${ABSTAIN_RULES.minTop1Similarity} and ensemble disagreement ${ensemble.disagreement.toFixed(2)} > ${ABSTAIN_RULES.maxDisagreementForLowSim}`,
    );
  }
  if (ensemble.replicateFallbackCount >= ABSTAIN_RULES.maxReplicateFallbacks) {
    reasons.push(
      `${ensemble.replicateFallbackCount} stages fell back to Replicate (degraded inference quality)`,
    );
  }
  return reasons;
}

/**
 * Run the full 5-stage pipeline. Emits PipelineEvents live; resolves with the
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
      tb_classifier_hf: settings.overrides.tbClassifierHf,
      general_cxr_hf: settings.overrides.generalCxrHf,
      adjudicator: settings.models.adjudicator,
      adjudicator_fallback: settings.models.adjudicatorFallback,
      embedding_endpoint: settings.overrides.embeddingEndpointUrl || '(none)',
    },
  };

  if (!caps.hasOpenAI) {
    run.halted = { reason: 'OpenAI API key not set — required for quality gate, VLM, and adjudication.', stage: 'quality' };
    emit({ type: 'halted', reason: run.halted.reason, stage: 'quality' });
    return run;
  }
  if (!caps.hasHF) {
    run.halted = { reason: 'Hugging Face token not set — required for the perception ensemble.', stage: 'ensemble.tb' };
    emit({ type: 'halted', reason: run.halted.reason, stage: 'ensemble.tb' });
    return run;
  }

  const dataUrl = await blobToDataURL(image);

  // ----------------------------------------------------------------------
  // Stage 1 — Quality gate (GPT-5.5 vision, no fallback)
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
  // Stage 2 — Perception ensemble (parallel, each HF -> Replicate fallback)
  // ----------------------------------------------------------------------
  const classifyDeps = {
    hfToken: settings.hfToken,
    replicateToken: settings.replicateToken,
  };

  emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'running' });
  emit({ type: 'stage_status', stage: 'ensemble.general', status: 'running' });
  emit({ type: 'stage_status', stage: 'ensemble.vlm', status: 'running' });

  const tbMember = (async (): Promise<EnsembleMember> => {
    try {
      const r = await classifyWithFallback(image, buildTbStageConfig(settings), {
        ...classifyDeps,
        onFallback: (from, to, reason) => {
          emit({ type: 'fallback_fired', stage: 'ensemble.tb', from, to });
          emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'fallback', note: reason });
        },
      });
      const m: EnsembleMember = {
        id: 'tb', label: 'TB Classifier', weight: ENSEMBLE_WEIGHTS.tb, status: 'done',
        tb_prob: r.tb_prob, provider_used: r.provider_used, latency_ms: r.latency_ms, raw: r.raw,
      };
      providerLog.push({ stage: 'ensemble.tb', provider_used: r.provider_used, latency_ms: r.latency_ms, fell_back: r.provider_used === 'replicate' });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'done' });
      return m;
    } catch (err) {
      const message = (err as Error).message;
      const m: EnsembleMember = {
        id: 'tb', label: 'TB Classifier', weight: ENSEMBLE_WEIGHTS.tb, status: 'error',
        tb_prob: null, provider_used: null, latency_ms: null, raw: null, error: message,
      };
      providerLog.push({ stage: 'ensemble.tb', provider_used: null, latency_ms: null, fell_back: false });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'error', stage: 'ensemble.tb', message });
      emit({ type: 'stage_status', stage: 'ensemble.tb', status: 'error' });
      return m;
    }
  })();

  const generalMember = (async (): Promise<EnsembleMember> => {
    try {
      const r = await classifyWithFallback(image, buildGeneralStageConfig(settings), {
        ...classifyDeps,
        onFallback: (from, to, reason) => {
          emit({ type: 'fallback_fired', stage: 'ensemble.general', from, to });
          emit({ type: 'stage_status', stage: 'ensemble.general', status: 'fallback', note: reason });
        },
      });
      const m: EnsembleMember = {
        id: 'general', label: 'General CXR', weight: ENSEMBLE_WEIGHTS.general, status: 'done',
        tb_prob: r.tb_prob, provider_used: r.provider_used, latency_ms: r.latency_ms, raw: r.raw,
      };
      providerLog.push({ stage: 'ensemble.general', provider_used: r.provider_used, latency_ms: r.latency_ms, fell_back: r.provider_used === 'replicate' });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'stage_status', stage: 'ensemble.general', status: 'done' });
      return m;
    } catch (err) {
      const message = (err as Error).message;
      const m: EnsembleMember = {
        id: 'general', label: 'General CXR', weight: ENSEMBLE_WEIGHTS.general, status: 'error',
        tb_prob: null, provider_used: null, latency_ms: null, raw: null, error: message,
      };
      providerLog.push({ stage: 'ensemble.general', provider_used: null, latency_ms: null, fell_back: false });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'error', stage: 'ensemble.general', message });
      emit({ type: 'stage_status', stage: 'ensemble.general', status: 'error' });
      return m;
    }
  })();

  const vlmMember = (async (): Promise<EnsembleMember> => {
    try {
      // Self-consistency: K independent reads. Mean = calibrated probability, spread = uncertainty.
      const samples = await Promise.allSettled(
        Array.from({ length: SELF_CONSISTENCY_K }, () =>
          openaiJSON<VlmResult>({
            apiKey: settings.openaiKey,
            model: settings.models.adjudicator,
            fallbackModel: settings.models.adjudicatorFallback,
            prompt: VLM_PROMPT,
            imageDataUrl: dataUrl,
            schema: VLM_SCHEMA,
          }),
        ),
      );
      const ok = samples.flatMap((s) => (s.status === 'fulfilled' ? [s.value] : []));
      if (ok.length === 0) {
        const rej = samples.find((s) => s.status === 'rejected');
        throw rej && rej.status === 'rejected' ? (rej.reason as Error) : new Error('all VLM samples failed');
      }
      const probs = ok.map((r) => vlmToProb(r.data));
      const { mean, std } = popStats(probs);
      const latency = Math.max(...ok.map((r) => r.latencyMs));
      const m: EnsembleMember = {
        id: 'vlm', label: 'GPT-5.5 Vision', weight: ENSEMBLE_WEIGHTS.vlm, status: 'done',
        tb_prob: mean, provider_used: 'openai', latency_ms: latency,
        raw: ok[0]?.data, findings: ok[0]?.data.findings, uncertainty: std, samples: ok.length,
      };
      providerLog.push({ stage: 'ensemble.vlm', provider_used: 'openai', latency_ms: latency, fell_back: ok.some((r) => r.fellBack) });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'stage_status', stage: 'ensemble.vlm', status: 'done' });
      return m;
    } catch (err) {
      const message = (err as Error).message;
      const m: EnsembleMember = {
        id: 'vlm', label: 'GPT-5.5 Vision', weight: ENSEMBLE_WEIGHTS.vlm, status: 'error',
        tb_prob: null, provider_used: null, latency_ms: null, raw: null, error: message,
      };
      providerLog.push({ stage: 'ensemble.vlm', provider_used: null, latency_ms: null, fell_back: false });
      emit({ type: 'ensemble_member', member: m });
      emit({ type: 'error', stage: 'ensemble.vlm', message });
      emit({ type: 'stage_status', stage: 'ensemble.vlm', status: 'error' });
      return m;
    }
  })();

  const members = await Promise.all([tbMember, generalMember, vlmMember]);
  const returning = members.filter((m) => m.tb_prob !== null);
  const noPerception = returning.length === 0;
  const probs = returning.map((m) => m.tb_prob as number);
  // Use the fitted calibration only when it is complete; an incomplete fit (too few
  // per-class samples) has meaningless thresholds and must not override the validated
  // default policy. When null, the legacy arithmetic-mean + SCREENING_POLICY path runs.
  const cal = settings.calibration && !settings.calibration.conformal.incomplete
    ? settings.calibration
    : null;
  const calProbOf = (m: EnsembleMember): number => {
    const raw = m.tb_prob as number;
    const c = cal?.perModel[m.id];
    return c ? applyCalibration(raw, c) : raw;
  };
  let weightedScore: number;
  if (cal) {
    const present = returning.map((m) => m.id);
    const w = effectiveWeights(present, cal.fusion.weights, cal.fusion.mode);
    weightedScore = fuseLogOdds(
      returning.map((m) => ({ id: m.id, prob: calProbOf(m) })),
      w,
      cal.fusion.mode === 'fitted' ? cal.fusion.bias : 0,
    );
  } else {
    const totalWeight = returning.reduce((a, m) => a + m.weight, 0) || 1;
    weightedScore = returning.reduce((a, m) => a + m.weight * (m.tb_prob as number), 0) / totalWeight;
  }
  const { std } = popStats(probs);
  const disagreement = probs.length ? Math.max(...probs) - Math.min(...probs) : 0;
  const replicateFallbackCount = members.filter((m) => m.provider_used === 'replicate').length;

  const ensemble: EnsembleResult = {
    members, weightedScore, std, disagreement, replicateFallbackCount,
  };
  run.ensemble = ensemble;
  emit({ type: 'ensemble_done', result: ensemble });

  // ----------------------------------------------------------------------
  // Stage 3 — RAG retrieval (kNN over labeled corpus)
  // ----------------------------------------------------------------------
  let rag: RagResult = { neighbors: [], embedding_provider: null, skipped: false };
  if (!caps.hasEmbedding) {
    rag = {
      neighbors: [], embedding_provider: null, skipped: true,
      skipReason: 'No embedding provider configured. Set the CXR-Foundation HF Inference Endpoint URL or a Replicate CLIP fallback in Settings to enable retrieval.',
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
      providerLog.push({ stage: 'rag', provider_used: embed.provider_used, latency_ms: embed.latency_ms, fell_back: embed.provider_used === 'replicate' });

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
  // Stage 4 — GPT-5.5 adjudicator (single streamed call) + deterministic guardrails
  // ----------------------------------------------------------------------
  emit({ type: 'stage_status', stage: 'adjudicate', status: 'running' });
  const providersUsed = providerLog.map((p) => `${p.stage}:${p.provider_used ?? 'none'}`);
  let adjudication: Adjudication;
  try {
    const stream = await openaiStream({
      apiKey: settings.openaiKey,
      model: settings.models.adjudicator,
      fallbackModel: settings.models.adjudicatorFallback,
      prompt: buildAdjudicatorPrompt({ ensemble, rag, providersUsed, replicateFallbackCount }),
      imageDataUrl: dataUrl,
      onToken: (t) => emit({ type: 'adjudicate_token', token: t }),
    });

    const parsed = JSON.parse(
      stream.text.slice(stream.text.indexOf('{'), stream.text.lastIndexOf('}') + 1),
    ) as AdjudicationRaw;

    const VALID_VERDICTS: Verdict[] = ['tb', 'no_tb', 'abstain'];
    const modelVerdict: Verdict = VALID_VERDICTS.includes(parsed.verdict) ? parsed.verdict : 'abstain';
    const modelConfidence = Number.isFinite(parsed.confidence) ? clamp(parsed.confidence, 0, 100) : 0;

    // Safety-net combine: screening policy (calibrated, sensitivity-first) + deterministic
    // guardrails + the model's own verdict — take the MOST CAUTIOUS. The model and policy
    // can escalate; neither can clear a flagged case. (tb > abstain > no_tb)
    const vlmM = members.find((m) => m.id === 'vlm');
    const vlmRaw = vlmM?.tb_prob ?? null;
    const vlmUncertainty = vlmM?.uncertainty ?? 0;
    // vlmSafetyThreshold is fit on the CALIBRATED VLM prob, so the policy must compare
    // against the calibrated value. Keep fusedProb and vlmProb consistently both-calibrated
    // (when cal active) or both-raw (legacy path).
    const vlmProbForPolicy =
      cal && vlmRaw !== null && cal.perModel.vlm
        ? applyCalibration(vlmRaw, cal.perModel.vlm)
        : vlmRaw;
    const policy = screeningPolicy(ensemble.weightedScore, vlmProbForPolicy, vlmUncertainty, cal);
    const guardrailReasons = evaluateAbstainRules(modelConfidence, ensemble, rag);

    if (noPerception) {
      guardrailReasons.push('no perception model returned a result');
    }
    const baseVerdict = mostCautious(
      policy.verdict,
      modelVerdict,
      guardrailReasons.length > 0 ? 'abstain' : 'no_tb',
      noPerception ? 'abstain' : 'no_tb',
    );
    // Sequelae escalate-not-clear (Milestone 19). Today `s_inactive` is always null —
    // the browser-side feature pathway for Rad-DINO + TXRV is not built (Phase B gap,
    // documented in CASE_STUDY M19). The interface is locked down so wiring is a
    // one-line change once those features land.
    const sInactive: number | null = null;
    const seqEsc = applySequelaeEscalation({
      verdict: baseVerdict,
      tbProb: ensemble.weightedScore,
      sInactive,
      borderlineHigh: cal?.conformal.tauHigh,
    });
    const finalVerdict = seqEsc.verdict;
    const reasons = [
      ...(policy.verdict !== 'no_tb' && policy.reason ? [policy.reason] : []),
      ...guardrailReasons,
      ...(seqEsc.escalated && seqEsc.reason ? [seqEsc.reason] : []),
    ];
    const escalated = finalVerdict !== modelVerdict;
    adjudication = {
      verdict: finalVerdict,
      confidence: modelConfidence,
      rationale: escalated
        ? `[Safety net: ${modelVerdict} → ${finalVerdict}] ${parsed.rationale}`
        : parsed.rationale,
      abstain_reason:
        finalVerdict === 'abstain'
          ? `Refer: ${reasons.join('; ') || 'weak or uncertain evidence'}.`
          : parsed.abstain_reason,
      auto_abstained: finalVerdict === 'abstain' && modelVerdict !== 'abstain',
      auto_abstain_reasons: reasons,
      // Honesty contract: when every perception member errored, the verdict is forced
      // to abstain by the safety net above. Flag the case so VerdictCard renders the
      // dedicated "configure an API key" state instead of a misleading uncertain card.
      perception_unavailable: noPerception,
      screening: {
        policyVerdict: policy.verdict,
        modelVerdict: modelVerdict,
        fusedProb: ensemble.weightedScore,
        vlmProb: vlmProbForPolicy,
        vlmUncertainty,
      },
    };
    providerLog.push({ stage: 'adjudicate', provider_used: 'openai', latency_ms: stream.latencyMs, fell_back: stream.fellBack });
    run.modelVersions['adjudicator_used'] = stream.modelUsed;
    emit({ type: 'adjudicate_done', result: adjudication });
    emit({ type: 'stage_status', stage: 'adjudicate', status: 'done' });
  } catch (err) {
    const message = (err as Error).message;
    // Even adjudication failure resolves to a safe abstain rather than a crash.
    adjudication = {
      verdict: 'abstain', confidence: 0, rationale: 'Adjudication call failed; defaulting to abstain.',
      abstain_reason: message, auto_abstained: true, auto_abstain_reasons: [`adjudicator error: ${message}`],
      perception_unavailable: noPerception,
    };
    emit({ type: 'error', stage: 'adjudicate', message });
    emit({ type: 'stage_status', stage: 'adjudicate', status: 'error' });
    emit({ type: 'adjudicate_done', result: adjudication });
  }
  run.adjudication = adjudication;

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

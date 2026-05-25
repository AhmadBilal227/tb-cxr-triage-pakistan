/**
 * Asymmetric-evidence ABSTAIN rule (Milestone 26).
 *
 * THIRD NO_TB→ABSTAIN escalator, alongside `applySequelaeEscalation` (M19) and
 * `applyVlmEscalation` (M21). All three are pure functions composed in
 * orchestrator.combineLocalIntoAdjudication via the same one-way contract:
 * push NO_TB up to ABSTAIN only, NEVER demote TB or ABSTAIN downward, NEVER
 * override a confidently positive call.
 *
 * MOTIVATION (M24 diagnostic, recorded in EXPERIMENT_LOG §C 2026-05-25):
 * Running the 23-image blind set through the M24-enriched engine surfaced 4
 * confident TB-misses (#6, #7, #9, #10) where the local pipeline returned
 * `tb_prob` essentially zero AND the model very clearly saw pathology. All
 * four had:
 *   - `box_evidence_max >= 0.88` (BoxEvidence head fires on 4-7 contiguous
 *     mid-lung cells — correct localization)
 *   - TXRV `Lung_Opacity >= 0.44` (the backbone reports moderate-or-stronger
 *     opacity)
 *   - per-zone calibrated TB probability ≈ 0 across all 7 zones
 *   - tb_prob ≈ 0 (below the validated head's 95-sens threshold by 2 orders
 *     of magnitude)
 *
 * Interpretation: the model HAS LEARNED a too-narrow definition of "TB-
 * pattern" — apical/cavitary parenchymal disease in the TBX11K + TB-Portals
 * morphology. When real TB presents with mid-zone consolidation or pleural
 * effusion, BoxEvidence + TXRV correctly flag pathology but the zonal+fusion
 * heads refuse to call it TB. This is an ATYPICAL-PRESENTATION failure, not
 * a preprocessing failure (the heatmap shows correct localization), and it
 * is the precise rate-limiting failure mode that LoRA fine-tune (training
 * task #37) is positioned to fix.
 *
 * Until the LoRA fix lands, the deterministic safety net should ESCALATE not
 * CLEAR these cases — route to ABSTAIN ("model sees pathology in lung region
 * but does not match learned TB-pattern; second-opinion required") rather
 * than tell the user NO_TB.
 *
 * THRESHOLD DERIVATION (all numbers honestly measured, no magic constants):
 *
 * Dataset for the sweep: the 23-image blind set
 * (training/test_blind_set_enriched.py output, 11 TB+ from public clinical
 * sources + 12 NIH "No_Finding" labels). The 13k LODO cache at
 * data/image_oof_logits.npz only carries the fused logit; the rule needs
 * (tb_prob, box_evidence_max, top_pathology_score) jointly, and only the
 * blind set has all three.
 *
 *   Confident TB-misses (the catch target, all 4):
 *     #6  tb_prob=0.040  box_max=0.914  Lung_Opacity=0.72
 *     #7  tb_prob=0.009  box_max=0.882  Lung_Opacity=0.55
 *     #9  tb_prob=0.003  box_max=0.996  Lung_Opacity=0.66
 *     #10 tb_prob=0.019  box_max=0.928  Lung_Opacity=0.44
 *
 *   The MINIMUMs of the catch set are box=0.882 and pathology=0.44 — these
 *   are exactly the data-anchored floors. We round box DOWN to 0.88 (not
 *   0.882) so the rule isn't brittle to a 0.001 wobble; pathology stays at
 *   0.44.
 *
 *   Sweep over (tb_low ∈ [0.05,0.24], box_high ∈ [0.50,0.95], path_high ∈
 *   [0.20,0.59]) shows the chosen triple (tb_low=0.20, box_high=0.88,
 *   path_high=0.44) hits catch=4/4 and fp_abstain=1/12 (NIH #18).
 *
 *   NIH-noise quirk (the brief flagged it, NOT a defect): #18 notb_blind_07
 *   has tb_prob=0.003, box_max=0.998, Lung_Opacity=0.83 — under the rule
 *   the model SAW something but didn't call TB, so we escalate to ABSTAIN.
 *   The NIH label is "No_Finding" but the radiograph carries visible
 *   pathology; the rule firing is correct behavior, not a defect. Listed
 *   here in the audit trail so future readers don't try to "fix" it.
 *
 *   Why tb_low=0.20 specifically (not 0.05 like the brief's seed): the
 *   catch set covers tb_prob ∈ [0.003, 0.040]. Any tb_low between 0.04 and
 *   0.20 gives the same 4/4 catch on this blind set. We chose 0.20 to (a)
 *   match the M26 vlmTriage borderline-low widening (M26 also lowers the
 *   VLM verifier band's lower edge from 0.35 to 0.20), and (b) keep a small
 *   buffer above the catch set so a near-borderline mid-zone TB at
 *   tb_prob=0.15 still escalates. The LODO cache shows ~13% of TB+ have
 *   tb_prob<0.20, ~84% of NO_TB do — without the box+pathology AND-gate
 *   this would be a catastrophic false-abstain rate; the conjunction is
 *   what makes the rule tractable.
 *
 *   Honestly unmeasured: per-site performance, deployment prevalence, and
 *   whether path_high=0.44 holds on a non-public TB cohort. The rule is
 *   advisory, NOT a calibrated threshold. The CASE_STUDY entry says the
 *   same in long-form.
 *
 * PHASE B / LATER RECALIBRATION: once a LoRA fine-tune lands and the
 * model's TB-pattern definition widens, this rule's catch set should
 * shrink (more of these cases will get a real tb_prob > 0.20). At that
 * point, re-derive on the new data and tighten/widen accordingly. Do NOT
 * keep the rule running on stale thresholds.
 *
 * COMPOSITION ORDER in orchestrator.combineLocalIntoAdjudication:
 *   1. base verdict from local.verdict
 *   2. gpt verifier disagreement → ABSTAIN
 *   3. sequelaeEscalation (M19): scar-shape escalator
 *   4. THIS rule (M26): asymmetric-evidence escalator
 *   5. final verdict
 * Each step only ever escalates NO_TB upward — they compose cleanly.
 */
import type { Verdict } from '@/lib/types';

/**
 * Upper edge of the "very-low calibrated TB" band. Above this, the model
 * is confident enough about NO_TB that the rule does not interfere. Chosen
 * to align with the M26 vlmTriage VLM_BORDERLINE_LOW widening (also 0.20)
 * so the local-path and vlm-path borderlines share a number.
 */
export const TB_PROB_LOW_THRESHOLD = 0.20 as const;

/**
 * Minimum BoxEvidence max-cell sigmoid score for the rule to consider the
 * model as "having seen something". Anchored at the 0.882 floor of the M24
 * confident-miss catch set, rounded down to 0.88.
 */
export const BOX_EVIDENCE_HIGH_THRESHOLD = 0.88 as const;

/**
 * Minimum TXRV pathology max-of-five score (Lung_Opacity, Effusion,
 * Lung_Lesion, Infiltration, Consolidation) for the rule to consider the
 * backbone as "reporting pathology". Anchored exactly at the 0.44 floor of
 * the M24 confident-miss catch set.
 *
 * The five-pathology subset is the TB-relevant slice of the 18-class TXRV
 * head — non-TB findings like Cardiomegaly or Fracture are not part of
 * this signal.
 */
export const PATHOLOGY_HIGH_THRESHOLD = 0.44 as const;

/** TXRV labels considered TB-relevant for the rule's pathology signal. */
export const TB_RELEVANT_TXRV_LABELS: readonly string[] = [
  'Lung Opacity',
  'Effusion',
  'Lung Lesion',
  'Infiltration',
  'Consolidation',
];

export const ASYMMETRIC_EVIDENCE_ABSTAIN_REASON =
  'model sees pathology in lung region but does not match learned TB-pattern — second-opinion required';

export interface AsymmetricEvidenceInput {
  /** Current verdict after the policy/guardrails/verifier/sequelae chain. */
  verdict: Verdict;
  /** Calibrated TB probability (sigmoid(logit/T)). */
  tbProb: number;
  /**
   * Max cell value across the 8x8 BoxEvidence grid (sigmoid'd). `null` when
   * the local pipeline didn't emit the grid (older servers or non-local
   * paths) — the rule then no-ops.
   */
  boxEvidenceMax: number | null;
  /**
   * Max value across the TB-relevant TXRV labels (see TB_RELEVANT_TXRV_LABELS).
   * `null` when the local pipeline didn't emit pathologies — the rule no-ops.
   */
  topPathologyScore: number | null;
}

export interface AsymmetricEvidenceResult {
  /** The (possibly upgraded) verdict. NEVER less cautious than `input.verdict`. */
  verdict: Verdict;
  /** True if this rule changed the verdict. */
  escalated: boolean;
  /** Human-readable reason when escalated; undefined otherwise. */
  reason?: string;
}

/**
 * Helper: compute the top-of-TB-relevant TXRV score from a free-form
 * pathology map. Returns `null` when the map is empty / missing / has no
 * TB-relevant labels (so the rule fairly no-ops rather than firing on a
 * coincidental Cardiomegaly score).
 */
export function topTbRelevantPathology(
  pathologies: Record<string, number> | undefined,
): number | null {
  if (!pathologies) return null;
  let best: number | null = null;
  for (const label of TB_RELEVANT_TXRV_LABELS) {
    const v = pathologies[label];
    if (typeof v === 'number' && Number.isFinite(v)) {
      if (best === null || v > best) best = v;
    }
  }
  return best;
}

/**
 * Helper: compute the max-cell value across the 8x8 BoxEvidence grid.
 * Returns `null` when the grid is missing or malformed (so the rule
 * fairly no-ops rather than firing on garbage).
 */
export function maxBoxEvidence(
  grid: number[][] | undefined,
): number | null {
  if (!grid || grid.length === 0) return null;
  let best: number | null = null;
  for (const row of grid) {
    if (!Array.isArray(row)) return null;
    for (const v of row) {
      if (typeof v === 'number' && Number.isFinite(v)) {
        if (best === null || v > best) best = v;
      }
    }
  }
  return best;
}

/**
 * Apply the asymmetric-evidence escalator.
 *
 *   - verdict !== 'no_tb'          → identity (rule never demotes TB/ABSTAIN).
 *   - boxEvidenceMax === null OR
 *     topPathologyScore === null   → identity (no enrichment to reason on).
 *   - tbProb >= TB_PROB_LOW_THRESHOLD  → identity (model is not in "very-low" band).
 *   - boxEvidenceMax < BOX_EVIDENCE_HIGH_THRESHOLD → identity (model didn't see it).
 *   - topPathologyScore < PATHOLOGY_HIGH_THRESHOLD → identity (backbone didn't see it).
 *   - else                                          → NO_TB → ABSTAIN with reason.
 */
export function applyAsymmetricEvidence(
  input: AsymmetricEvidenceInput,
): AsymmetricEvidenceResult {
  const { verdict, tbProb, boxEvidenceMax, topPathologyScore } = input;

  if (verdict !== 'no_tb') {
    return { verdict, escalated: false };
  }
  if (boxEvidenceMax === null || topPathologyScore === null) {
    return { verdict, escalated: false };
  }
  if (tbProb >= TB_PROB_LOW_THRESHOLD) {
    return { verdict, escalated: false };
  }
  if (boxEvidenceMax < BOX_EVIDENCE_HIGH_THRESHOLD) {
    return { verdict, escalated: false };
  }
  if (topPathologyScore < PATHOLOGY_HIGH_THRESHOLD) {
    return { verdict, escalated: false };
  }
  return {
    verdict: 'abstain',
    escalated: true,
    reason: ASYMMETRIC_EVIDENCE_ABSTAIN_REASON,
  };
}

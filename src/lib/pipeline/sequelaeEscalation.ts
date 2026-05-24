/**
 * Sequelae-head safety net (Milestone 19, scoped to interface — see CASE_STUDY M19).
 *
 * Wires `s_inactive` (the inactive/sequelae-pattern probability from the deployed
 * InactiveSequelaeHead) into the orchestrator's safety-net combine as an ESCALATE-
 * NOT-CLEAR feature. It can only push a borderline NO_TB up to ABSTAIN — never the
 * other direction. A high tb_prob with high s_inactive stays TB (the inactive head
 * does NOT override a strong active signal; reactivation hides in scar).
 *
 * Derivation of S_INACTIVE_ESCALATE_THRESHOLD = 0.7126 (the 0.7-recall threshold on
 * the sequelae head's own scar probe, training/.venv/bin/python ad-hoc run on the
 * 139-row features_sequelae.npz):
 *
 *     temperature: 1.1313  (data/tb_inactive_meta.json)
 *     s_inactive on the 139 scar examples:
 *       min 0.0315, median 0.7740, max 0.9725, mean 0.7523
 *     0.7-recall quantile = q30 of probs = 0.7126
 *     actual recall at 0.7126 = 0.6978 (closest discrete point, ~70%)
 *
 * In plain English: ~70% of confirmed scar images score s_inactive >= 0.7126. We use
 * THIS scar-distribution threshold (not the SEQ_HIGH_DEFAULT=0.5 used in
 * training/train_tb.activity_verdict) because the safety-net role here is the
 * "definitely scar-shaped" call, not the "any inactive signal" call: a false-positive
 * abstain on a clean lung costs an extra read; a false-negative escalate on a real
 * scar costs the very FPR disaster we're trying to fix.
 *
 * BORDERLINE BAND: [0.35, tau_high) where tau_high comes from the fitted conformal
 * upper threshold when calibration is loaded, else falls back to the deployed
 * thr_at_95sens=0.6105 (data/tb_threshold_t2.json) — capped to keep the contract
 * single-meaning even if a site re-fits the conformal layer.
 *
 * BROWSER-SIDE FEATURE PATHWAY (Phase B gap, documented in CASE_STUDY M19): the
 * Rad-DINO + TXRV feature vectors that the sequelae ONNX consumes are NOT yet built
 * by the live browser code path. Until they are, the orchestrator passes
 * `s_inactive: null` and this function is an identity. The interface is locked down
 * here so the wiring is a single one-line change once those features land.
 */
import type { Verdict } from '@/lib/types';

/** Lower edge of the borderline band — probabilities below this are decisively NO_TB. */
export const BORDERLINE_LOW = 0.35 as const;

/**
 * Default upper edge of the borderline band (= the deployed 0.95-sensitivity threshold,
 * data/tb_threshold_t2.json: 0.6105). When a fitted conformal threshold is present, the
 * caller may pass it through `borderlineHigh` to override.
 */
export const DEFAULT_BORDERLINE_HIGH = 0.6105 as const;

/**
 * The 0.7-scar-recall threshold on the sequelae head's own probe (derivation above).
 * Above this is "definitely scar-shaped" — escalate a borderline NO_TB to ABSTAIN.
 */
export const S_INACTIVE_ESCALATE_THRESHOLD = 0.7126 as const;

export const SCAR_ABSTAIN_REASON =
  'scar-shape pattern flagged for re-read (high s_inactive in the borderline tb_prob band)';

export interface SequelaeEscalationInput {
  /** The current best verdict after policy + guardrails + LLM combine. */
  verdict: Verdict;
  /** Fused TB probability used by the screening policy (calibrated or raw, consistent with caller). */
  tbProb: number;
  /**
   * Sequelae head's calibrated `s_inactive` probability, or `null` when the browser-side
   * feature pathway isn't producing it yet. `null` => no-op (pass through unchanged).
   */
  sInactive: number | null;
  /**
   * Upper edge of the borderline band. If omitted, defaults to DEFAULT_BORDERLINE_HIGH.
   * Pass the fitted conformal upper threshold when calibration is active.
   */
  borderlineHigh?: number;
}

export interface SequelaeEscalationResult {
  /** The (possibly upgraded) verdict. NEVER less cautious than `input.verdict`. */
  verdict: Verdict;
  /** True if this rule changed the verdict. */
  escalated: boolean;
  /** Human-readable reason when escalated; undefined otherwise. */
  reason?: string;
}

/**
 * Escalate-not-clear rule. Pure, deterministic, easy to unit-test.
 *
 *   - sInactive is null/undefined   => identity (Phase B browser pathway not live).
 *   - verdict !== 'no_tb'           => identity (rule never demotes TB or ABSTAIN to clear).
 *   - tbProb in [BORDERLINE_LOW, borderlineHigh) AND sInactive >= threshold
 *                                   => upgrade NO_TB -> ABSTAIN with SCAR_ABSTAIN_REASON.
 *   - otherwise                     => identity.
 */
export function applySequelaeEscalation(
  input: SequelaeEscalationInput,
): SequelaeEscalationResult {
  const { verdict, tbProb, sInactive } = input;
  const high = input.borderlineHigh ?? DEFAULT_BORDERLINE_HIGH;

  if (sInactive === null || sInactive === undefined) {
    return { verdict, escalated: false };
  }
  if (verdict !== 'no_tb') {
    return { verdict, escalated: false };
  }
  const inBand = tbProb >= BORDERLINE_LOW && tbProb < high;
  if (!inBand) {
    return { verdict, escalated: false };
  }
  if (sInactive < S_INACTIVE_ESCALATE_THRESHOLD) {
    return { verdict, escalated: false };
  }
  return { verdict: 'abstain', escalated: true, reason: SCAR_ABSTAIN_REASON };
}

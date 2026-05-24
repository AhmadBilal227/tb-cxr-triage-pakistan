/**
 * VLM-path safety-net escalation (Milestone 21, Task C).
 *
 * Mirrors the structural shape of `applySequelaeEscalation` from M19 but is a
 * SEPARATE function with SEPARATE constants and a SEPARATE threshold. The two
 * functions look similar on purpose — both implement "escalate-not-clear on
 * borderline TB + scar evidence" — but they read from incompatible sources:
 *
 *   - `applySequelaeEscalation`  → consumes `s_inactive` from the local ONNX
 *     `InactiveSequelaeHead`. The threshold (0.7126) was derived from the head's
 *     own scar probe (q30 of 139 confirmed scars under T_sequelae=1.1313). This
 *     pathway DOES NOT EXECUTE in the browser today (Phase B gap).
 *
 *   - `applyVlmEscalation` (this file) → consumes `scar_shape_score_uncalibrated`
 *     from gpt-5.5 vision. This is a DIFFERENT model with a DIFFERENT score
 *     distribution. The 0.7126 threshold is NOT valid here; we deliberately use
 *     a softer 0.5 starting point and label it as a heuristic, because the VLM
 *     heuristic has not been validated against labeled data.
 *
 * The conservative bias is also softer than the ONNX path. We escalate when:
 *   - the VLM picked `screen_negative` AND
 *   - `tb_score_uncalibrated > 0.25` (slightly looser than the ONNX path's
 *     [0.35, tau_high) band — a VLM-positive without a calibrated band still
 *     deserves a second look on a scar-shape flag), AND
 *   - either the scar-shape score is >= the heuristic threshold OR the mimic
 *     features list includes any of the SCAR_TRIGGER_TAGS.
 *
 * Same never-clear contract as the ONNX path: NO_TB → ABSTAIN only; never
 * demotes ABSTAIN to NO_TB; never overrides screen_positive.
 *
 * The escalation reason is explicit about being uncalibrated so the UI / audit
 * trail can render the right disclosure language.
 */
import type { Verdict } from '@/lib/types';
import { SCAR_TRIGGER_TAGS, hasScarMimicFlag } from './vlmTriage';

/**
 * Heuristic threshold on the VLM's `scar_shape_score_uncalibrated`. Starts at
 * 0.5 because the score has not been validated against labels; tighten only
 * after a labeled scar holdout exists (none today — recorded in CASE_STUDY).
 */
export const VLM_SCAR_HEURISTIC_THRESHOLD = 0.5 as const;

/**
 * Lower edge of the VLM borderline band. Softer than the ONNX path's
 * BORDERLINE_LOW=0.35 because the VLM has no calibrated upper threshold for us
 * to anchor against — better to err on the cautious side.
 */
export const VLM_TB_SCORE_ESCALATE_FLOOR = 0.25 as const;

export const VLM_SCAR_ABSTAIN_REASON =
  'VLM scar/mimic heuristic flagged for re-read — uncalibrated';

export interface VlmEscalationInput {
  /** Current best verdict after policy + the model's screen_result. */
  verdict: Verdict;
  /** The VLM's uncalibrated TB score, 0..1. */
  tbScoreUncalibrated: number;
  /** The VLM's uncalibrated scar-shape heuristic score, 0..1. */
  scarShapeScoreUncalibrated: number;
  /** The VLM's mimic_features_present list (free-text tags). */
  mimicFeatures: string[];
}

export interface VlmEscalationResult {
  /** The (possibly upgraded) verdict. NEVER less cautious than `input.verdict`. */
  verdict: Verdict;
  /** True if this rule changed the verdict. */
  escalated: boolean;
  /** Human-readable reason when escalated; undefined otherwise. */
  reason?: string;
}

/**
 * Pure, deterministic, easy to unit-test.
 *
 *   - verdict !== 'no_tb'                       => identity (never demotes).
 *   - tbScoreUncalibrated <= floor              => identity (clean lung tier).
 *   - no scar trigger (neither score nor tags)  => identity.
 *   - else                                      => NO_TB -> ABSTAIN with
 *                                                 VLM_SCAR_ABSTAIN_REASON.
 */
export function applyVlmEscalation(input: VlmEscalationInput): VlmEscalationResult {
  const { verdict, tbScoreUncalibrated, scarShapeScoreUncalibrated, mimicFeatures } = input;

  if (verdict !== 'no_tb') {
    return { verdict, escalated: false };
  }
  if (tbScoreUncalibrated <= VLM_TB_SCORE_ESCALATE_FLOOR) {
    return { verdict, escalated: false };
  }

  const scarScoreTriggered = scarShapeScoreUncalibrated >= VLM_SCAR_HEURISTIC_THRESHOLD;
  const scarTagsTriggered = hasScarMimicFlag(mimicFeatures);

  if (!scarScoreTriggered && !scarTagsTriggered) {
    return { verdict, escalated: false };
  }

  return { verdict: 'abstain', escalated: true, reason: VLM_SCAR_ABSTAIN_REASON };
}

// re-export for tests + transparency
export { SCAR_TRIGGER_TAGS };

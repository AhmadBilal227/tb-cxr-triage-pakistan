/**
 * Unit tests for the VLM-path safety-net escalation rule (Milestone 21, Task C).
 *
 * The three required cases from the M21 brief:
 *   1. high-conf no-mimic   -> screen_negative passes through.
 *   2. borderline + scar    -> ABSTAIN with the VLM scar/mimic reason.
 *   3. high-conf positive   -> passes through (never overrides screen_positive).
 *
 * Plus boundary tests pinning the never-clear contract: the rule mirrors
 * `applySequelaeEscalation` but uses a SEPARATE threshold (0.5 heuristic, not
 * 0.7126), separate score field (`scarShapeScoreUncalibrated`, NOT `s_inactive`),
 * and softer borderline band (floor 0.25, no upper). Those differences are
 * load-bearing — pin them explicitly.
 */
import { describe, it, expect } from 'vitest';
import {
  VLM_SCAR_HEURISTIC_THRESHOLD,
  VLM_TB_SCORE_ESCALATE_FLOOR,
  VLM_SCAR_ABSTAIN_REASON,
  applyVlmEscalation,
} from './vlmEscalation';

describe('applyVlmEscalation — the three required cases', () => {
  it('(1) high-conf no_tb without mimic features: passes through unchanged', () => {
    const r = applyVlmEscalation({
      verdict: 'no_tb',
      tbScoreUncalibrated: 0.05, // clean lung
      scarShapeScoreUncalibrated: 0.1,
      mimicFeatures: [],
    });
    expect(r.verdict).toBe('no_tb');
    expect(r.escalated).toBe(false);
    expect(r.reason).toBeUndefined();
  });

  it('(2) borderline tb-score + scar mimic flag: NO_TB -> ABSTAIN with VLM scar reason', () => {
    const r = applyVlmEscalation({
      verdict: 'no_tb',
      tbScoreUncalibrated: 0.4, // > 0.25 floor
      scarShapeScoreUncalibrated: 0.7, // >= 0.5 threshold
      mimicFeatures: ['apical_fibrosis'],
    });
    expect(r.verdict).toBe('abstain');
    expect(r.escalated).toBe(true);
    expect(r.reason).toBe(VLM_SCAR_ABSTAIN_REASON);
  });

  it('(3) high-conf positive: passes through unchanged (rule never overrides screen_positive)', () => {
    const r = applyVlmEscalation({
      verdict: 'tb',
      tbScoreUncalibrated: 0.92,
      scarShapeScoreUncalibrated: 0.8, // would otherwise trigger
      mimicFeatures: ['apical_fibrosis', 'pleural_thickening'],
    });
    expect(r.verdict).toBe('tb');
    expect(r.escalated).toBe(false);
  });
});

describe('applyVlmEscalation — escalate-not-clear contract', () => {
  it('never demotes ABSTAIN to NO_TB even with clean inputs', () => {
    const r = applyVlmEscalation({
      verdict: 'abstain',
      tbScoreUncalibrated: 0.01,
      scarShapeScoreUncalibrated: 0.0,
      mimicFeatures: [],
    });
    expect(r.verdict).toBe('abstain');
    expect(r.escalated).toBe(false);
  });

  it('below tb-score floor: no escalation even with high scar score (clean lung tier)', () => {
    const r = applyVlmEscalation({
      verdict: 'no_tb',
      tbScoreUncalibrated: 0.1, // <= 0.25 floor
      scarShapeScoreUncalibrated: 0.99,
      mimicFeatures: ['pleural_thickening'],
    });
    expect(r.escalated).toBe(false);
    expect(r.verdict).toBe('no_tb');
  });

  it('above tb-score floor + scar tag only (low scar score): still escalates on the tag', () => {
    const r = applyVlmEscalation({
      verdict: 'no_tb',
      tbScoreUncalibrated: 0.3,
      scarShapeScoreUncalibrated: 0.1, // < 0.5 threshold
      mimicFeatures: ['healed_scar'], // tag carries the signal
    });
    expect(r.verdict).toBe('abstain');
    expect(r.escalated).toBe(true);
  });

  it('above tb-score floor + scar score only (no tags): still escalates on the score', () => {
    const r = applyVlmEscalation({
      verdict: 'no_tb',
      tbScoreUncalibrated: 0.3,
      scarShapeScoreUncalibrated: 0.6, // >= 0.5
      mimicFeatures: [],
    });
    expect(r.verdict).toBe('abstain');
    expect(r.escalated).toBe(true);
  });

  it('above tb-score floor + neither scar signal: no escalation', () => {
    const r = applyVlmEscalation({
      verdict: 'no_tb',
      tbScoreUncalibrated: 0.4,
      scarShapeScoreUncalibrated: 0.1,
      mimicFeatures: ['consolidation'], // not a scar tag
    });
    expect(r.escalated).toBe(false);
  });

  it('case-insensitive tag matching: Fibrosis (capitalized) triggers the rule', () => {
    const r = applyVlmEscalation({
      verdict: 'no_tb',
      tbScoreUncalibrated: 0.3,
      scarShapeScoreUncalibrated: 0.0,
      mimicFeatures: ['Fibrosis'],
    });
    expect(r.escalated).toBe(true);
  });
});

describe('applyVlmEscalation — constants are sane and DIFFERENT from the ONNX path', () => {
  it('heuristic threshold is 0.5 (intentionally below the ONNX 0.7126)', () => {
    expect(VLM_SCAR_HEURISTIC_THRESHOLD).toBe(0.5);
    // The DIFFERENCE from the ONNX path is the load-bearing contract. If anyone
    // ever copies 0.7126 in here, this test fails — and the comment + reason
    // string will tell them why.
    expect(VLM_SCAR_HEURISTIC_THRESHOLD).toBeLessThan(0.7126);
  });

  it('tb-score floor is 0.25 (softer than the ONNX BORDERLINE_LOW=0.35)', () => {
    expect(VLM_TB_SCORE_ESCALATE_FLOOR).toBe(0.25);
    expect(VLM_TB_SCORE_ESCALATE_FLOOR).toBeLessThan(0.35);
  });

  it('the escalation reason names "uncalibrated" so the UI knows which disclosure to render', () => {
    expect(VLM_SCAR_ABSTAIN_REASON).toMatch(/uncalibrated/i);
  });
});

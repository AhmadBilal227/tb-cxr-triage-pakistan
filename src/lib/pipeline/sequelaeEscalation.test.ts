/**
 * Unit tests for the sequelae-head safety-net escalation rule (Milestone 19, Task D).
 *
 * The three required cases:
 *   1. high tb_prob + null s_inactive          -> TB unchanged (rule never demotes).
 *   2. borderline tb_prob + high s_inactive    -> ABSTAIN with the new scar reason.
 *   3. borderline tb_prob + low s_inactive     -> NO_TB unchanged.
 *
 * Plus a few boundary tests that pin the band edges and the never-clear contract.
 */
import { describe, it, expect } from 'vitest';
import {
  BORDERLINE_LOW,
  DEFAULT_BORDERLINE_HIGH,
  S_INACTIVE_ESCALATE_THRESHOLD,
  SCAR_ABSTAIN_REASON,
  applySequelaeEscalation,
} from './sequelaeEscalation';

describe('applySequelaeEscalation — the three required cases', () => {
  it('(1) high tb_prob + null s_inactive: TB unchanged', () => {
    const r = applySequelaeEscalation({ verdict: 'tb', tbProb: 0.92, sInactive: null });
    expect(r.verdict).toBe('tb');
    expect(r.escalated).toBe(false);
    expect(r.reason).toBeUndefined();
  });

  it('(2) borderline tb_prob + high s_inactive: NO_TB -> ABSTAIN with scar reason', () => {
    // Inside [BORDERLINE_LOW, DEFAULT_BORDERLINE_HIGH) and above the 0.7-scar-recall threshold.
    const r = applySequelaeEscalation({
      verdict: 'no_tb',
      tbProb: 0.5, // 0.35 <= 0.5 < 0.6105
      sInactive: 0.85, // >= 0.7126
    });
    expect(r.verdict).toBe('abstain');
    expect(r.escalated).toBe(true);
    expect(r.reason).toBe(SCAR_ABSTAIN_REASON);
  });

  it('(3) borderline tb_prob + low s_inactive: NO_TB unchanged', () => {
    const r = applySequelaeEscalation({
      verdict: 'no_tb',
      tbProb: 0.5,
      sInactive: 0.2, // < 0.7126
    });
    expect(r.verdict).toBe('no_tb');
    expect(r.escalated).toBe(false);
  });
});

describe('applySequelaeEscalation — escalate-not-clear contract', () => {
  it('never demotes TB to ABSTAIN even with high s_inactive', () => {
    const r = applySequelaeEscalation({
      verdict: 'tb',
      tbProb: 0.5,
      sInactive: 0.99,
    });
    expect(r.verdict).toBe('tb');
    expect(r.escalated).toBe(false);
  });

  it('never demotes ABSTAIN to NO_TB with low s_inactive', () => {
    const r = applySequelaeEscalation({
      verdict: 'abstain',
      tbProb: 0.5,
      sInactive: 0.0,
    });
    expect(r.verdict).toBe('abstain');
    expect(r.escalated).toBe(false);
  });

  it('above borderline_high: no escalation even with high s_inactive (the policy already flagged it)', () => {
    const r = applySequelaeEscalation({
      verdict: 'no_tb', // unusual combo but the rule must be pure
      tbProb: 0.9, // >= 0.6105
      sInactive: 0.9,
    });
    expect(r.escalated).toBe(false);
    expect(r.verdict).toBe('no_tb');
  });

  it('below borderline_low: no escalation even with high s_inactive (clean lung tier)', () => {
    const r = applySequelaeEscalation({
      verdict: 'no_tb',
      tbProb: 0.1,
      sInactive: 0.99,
    });
    expect(r.escalated).toBe(false);
    expect(r.verdict).toBe('no_tb');
  });
});

describe('applySequelaeEscalation — constants are sane', () => {
  it('borderline band is non-empty and below 1', () => {
    expect(BORDERLINE_LOW).toBeGreaterThan(0);
    expect(BORDERLINE_LOW).toBeLessThan(DEFAULT_BORDERLINE_HIGH);
    expect(DEFAULT_BORDERLINE_HIGH).toBeLessThan(1);
  });

  it('the scar-recall threshold is derived in [0,1] and tighter than SEQ_HIGH_DEFAULT 0.5', () => {
    expect(S_INACTIVE_ESCALATE_THRESHOLD).toBeGreaterThan(0);
    expect(S_INACTIVE_ESCALATE_THRESHOLD).toBeLessThan(1);
    expect(S_INACTIVE_ESCALATE_THRESHOLD).toBeGreaterThan(0.5);
  });

  it('caller can override borderlineHigh (for fitted conformal calibration)', () => {
    // With a tighter upper edge of 0.45, tbProb=0.5 is OUTSIDE the band -> no escalation.
    const r = applySequelaeEscalation({
      verdict: 'no_tb',
      tbProb: 0.5,
      sInactive: 0.9,
      borderlineHigh: 0.45,
    });
    expect(r.escalated).toBe(false);
  });
});

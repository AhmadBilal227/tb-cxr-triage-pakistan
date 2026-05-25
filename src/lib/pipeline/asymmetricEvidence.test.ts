/**
 * asymmetricEvidence rule tests (Milestone 26).
 *
 * Pins:
 *   - Identity on every "not in band" path (high tb_prob, already TB or ABSTAIN,
 *     missing intermediates, single-evidence-only).
 *   - Escalates ONLY when all three conditions concur:
 *       tb_prob < TB_PROB_LOW_THRESHOLD
 *       boxEvidenceMax >= BOX_EVIDENCE_HIGH_THRESHOLD
 *       topPathologyScore >= PATHOLOGY_HIGH_THRESHOLD
 *   - The 4 M24 confident-miss cases (the catch set) all trip the rule with
 *     their real measured values.
 *   - The NIH #18 label-noise case (`notb_blind_07`) ALSO trips — documented
 *     in the rule's header as expected behavior, NOT a defect.
 */
import { describe, it, expect } from 'vitest';
import {
  applyAsymmetricEvidence,
  topTbRelevantPathology,
  maxBoxEvidence,
  ASYMMETRIC_EVIDENCE_ABSTAIN_REASON,
  TB_PROB_LOW_THRESHOLD,
  BOX_EVIDENCE_HIGH_THRESHOLD,
  PATHOLOGY_HIGH_THRESHOLD,
} from './asymmetricEvidence';

describe('applyAsymmetricEvidence — pass-through paths (identity)', () => {
  it('passes through a high-tb_prob TB verdict unchanged (catch set is for NO_TB only)', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'tb',
      tbProb: 0.93,
      boxEvidenceMax: 0.95,
      topPathologyScore: 0.77,
    });
    expect(r).toEqual({ verdict: 'tb', escalated: false });
  });

  it('passes through an already-ABSTAIN verdict (no double-escalation)', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'abstain',
      tbProb: 0.05,
      boxEvidenceMax: 0.95,
      topPathologyScore: 0.6,
    });
    expect(r).toEqual({ verdict: 'abstain', escalated: false });
  });

  it('passes through a confident NO_TB with low box-evidence (clean lung)', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'no_tb',
      tbProb: 0.001,
      boxEvidenceMax: 0.42,
      topPathologyScore: 0.07,
    });
    expect(r.escalated).toBe(false);
    expect(r.verdict).toBe('no_tb');
  });

  it('passes through when tb_prob sits above TB_PROB_LOW_THRESHOLD (model not in very-low band)', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'no_tb',
      tbProb: TB_PROB_LOW_THRESHOLD + 0.01, // 0.21
      boxEvidenceMax: 0.95,
      topPathologyScore: 0.6,
    });
    expect(r.escalated).toBe(false);
  });

  it('passes through when only ONE signal is high (box alone is not enough)', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'no_tb',
      tbProb: 0.01,
      boxEvidenceMax: 0.95,
      topPathologyScore: 0.1, // backbone sees nothing
    });
    expect(r.escalated).toBe(false);
  });

  it('passes through when only the pathology signal is high (TXRV alone is not enough)', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'no_tb',
      tbProb: 0.01,
      boxEvidenceMax: 0.5, // box didn't fire
      topPathologyScore: 0.7,
    });
    expect(r.escalated).toBe(false);
  });

  it('passes through when enrichment fields are null (older server / non-local path)', () => {
    expect(
      applyAsymmetricEvidence({
        verdict: 'no_tb',
        tbProb: 0.01,
        boxEvidenceMax: null,
        topPathologyScore: 0.7,
      }).escalated,
    ).toBe(false);
    expect(
      applyAsymmetricEvidence({
        verdict: 'no_tb',
        tbProb: 0.01,
        boxEvidenceMax: 0.95,
        topPathologyScore: null,
      }).escalated,
    ).toBe(false);
  });
});

describe('applyAsymmetricEvidence — the catch (M24 confident misses)', () => {
  // Real measured values from training/test_blind_set_enriched.py, 2026-05-25.
  const CATCH_SET = [
    { id: 6,  tb_prob: 0.0398, box_max: 0.914, top_path: 0.72 },
    { id: 7,  tb_prob: 0.0090, box_max: 0.882, top_path: 0.55 },
    { id: 9,  tb_prob: 0.0029, box_max: 0.996, top_path: 0.66 },
    { id: 10, tb_prob: 0.0194, box_max: 0.928, top_path: 0.44 },
  ];

  for (const m of CATCH_SET) {
    it(`escalates NO_TB → ABSTAIN for confident-miss #${m.id} (tb=${m.tb_prob} box=${m.box_max} path=${m.top_path})`, () => {
      const r = applyAsymmetricEvidence({
        verdict: 'no_tb',
        tbProb: m.tb_prob,
        boxEvidenceMax: m.box_max,
        topPathologyScore: m.top_path,
      });
      expect(r.verdict).toBe('abstain');
      expect(r.escalated).toBe(true);
      expect(r.reason).toBe(ASYMMETRIC_EVIDENCE_ABSTAIN_REASON);
    });
  }
});

describe('applyAsymmetricEvidence — NIH label-noise case (expected behavior, not a defect)', () => {
  it('fires on notb_blind_07 (NIH #18) — model sees pathology, label says No_Finding', () => {
    // Documented in the header comment: NIH No_Finding label is incorrect for
    // this radiograph; the rule firing is design ("model sees something, get a
    // second opinion") rather than a special case to suppress.
    const r = applyAsymmetricEvidence({
      verdict: 'no_tb',
      tbProb: 0.0028,
      boxEvidenceMax: 0.998,
      topPathologyScore: 0.83, // Lung_Opacity from the diagnostic
    });
    expect(r.escalated).toBe(true);
    expect(r.verdict).toBe('abstain');
  });
});

describe('applyAsymmetricEvidence — exact-threshold boundary', () => {
  it('escalates at exactly the box_high and path_high floors with tb_prob just below the band', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'no_tb',
      tbProb: TB_PROB_LOW_THRESHOLD - 0.001,
      boxEvidenceMax: BOX_EVIDENCE_HIGH_THRESHOLD,
      topPathologyScore: PATHOLOGY_HIGH_THRESHOLD,
    });
    expect(r.escalated).toBe(true);
  });

  it('does NOT escalate when box is 0.001 below the floor', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'no_tb',
      tbProb: 0.01,
      boxEvidenceMax: BOX_EVIDENCE_HIGH_THRESHOLD - 0.001,
      topPathologyScore: 0.7,
    });
    expect(r.escalated).toBe(false);
  });

  it('does NOT escalate when pathology is 0.001 below the floor', () => {
    const r = applyAsymmetricEvidence({
      verdict: 'no_tb',
      tbProb: 0.01,
      boxEvidenceMax: 0.95,
      topPathologyScore: PATHOLOGY_HIGH_THRESHOLD - 0.001,
    });
    expect(r.escalated).toBe(false);
  });
});

describe('topTbRelevantPathology helper', () => {
  it('returns the max of the 5 TB-relevant TXRV labels (ignoring other findings)', () => {
    const path = {
      'Lung Opacity': 0.44,
      Effusion: 0.10,
      Cardiomegaly: 0.85, // not TB-relevant — ignored
      Pneumonia: 0.30, // not in our 5
    };
    expect(topTbRelevantPathology(path)).toBe(0.44);
  });

  it('returns null when no TB-relevant label is present', () => {
    expect(topTbRelevantPathology({ Cardiomegaly: 0.9, Fracture: 0.4 })).toBeNull();
  });

  it('returns null for undefined / empty maps', () => {
    expect(topTbRelevantPathology(undefined)).toBeNull();
    expect(topTbRelevantPathology({})).toBeNull();
  });
});

describe('maxBoxEvidence helper', () => {
  it('returns the max cell of an 8x8 grid', () => {
    const grid = Array.from({ length: 8 }, (_, r) =>
      Array.from({ length: 8 }, (_, c) => (r === 3 && c === 4 ? 0.997 : 0.05)),
    );
    expect(maxBoxEvidence(grid)).toBeCloseTo(0.997, 3);
  });

  it('returns null for undefined / empty grids', () => {
    expect(maxBoxEvidence(undefined)).toBeNull();
    expect(maxBoxEvidence([])).toBeNull();
  });
});

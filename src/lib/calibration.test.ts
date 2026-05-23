import { describe, it, expect } from 'vitest';
import { logit, sigmoid, clampProb } from './calibration';
import { fitTemperature, applyCalibration } from './calibration';
import { fitPlatt } from './calibration';
import { fuseLogOdds, effectiveWeights, fitFusionWeights } from './calibration';
import { fitConformalThresholds } from './calibration';
import { fitCalibration } from './calibration';
import type { CalibrationSample } from './types';

describe('math primitives', () => {
  it('sigmoid(logit(p)) round-trips', () => {
    for (const p of [0.01, 0.3, 0.5, 0.87, 0.999]) {
      expect(sigmoid(logit(p))).toBeCloseTo(p, 6);
    }
  });
  it('clampProb keeps values off 0/1', () => {
    expect(clampProb(0)).toBeGreaterThan(0);
    expect(clampProb(1)).toBeLessThan(1);
    expect(Number.isFinite(logit(0))).toBe(true);
    expect(Number.isFinite(logit(1))).toBe(true);
  });
});

describe('temperature scaling', () => {
  it('recovers T>1 on overconfident data', () => {
    // Build overconfident probs: true rate 0.5 region but probs pushed to extremes.
    const probs: number[] = [];
    const labels: (0 | 1)[] = [];
    for (let i = 0; i < 200; i++) {
      const y = (i % 2) as 0 | 1;
      // overconfident: positives ~0.95, negatives ~0.05, but 30% are wrong
      const correct = i % 10 >= 3;
      const p = y === 1 ? (correct ? 0.95 : 0.05) : correct ? 0.05 : 0.95;
      probs.push(p);
      labels.push(y);
    }
    const T = fitTemperature(probs, labels);
    expect(T).toBeGreaterThan(1); // softening needed
  });
  it('applyCalibration with T=1 is identity', () => {
    const c = { method: 'temperature' as const, T: 1, A: 1, B: 0, nllRaw: 0, nllCal: 0 };
    expect(applyCalibration(0.8, c)).toBeCloseTo(0.8, 6);
  });
});

describe('platt scaling', () => {
  it('recovers a downward shift (B < 0) on upward-biased data', () => {
    // Deterministic (no RNG): balanced labels, but BOTH classes report probs
    // biased upward (pos 0.7, neg 0.6). A calibrator should pull the logits
    // down toward the true base rate, i.e. recover B < 0.
    const probs: number[] = [];
    const labels: (0 | 1)[] = [];
    for (let i = 0; i < 300; i++) {
      const y = (i % 2) as 0 | 1;
      probs.push(y === 1 ? 0.7 : 0.6);
      labels.push(y);
    }
    const { A, B } = fitPlatt(probs, labels);
    expect(Number.isFinite(A)).toBe(true);
    expect(B).toBeLessThan(0); // downward correction
  });
});

describe('log-odds fusion', () => {
  it('single renormalized member is identity', () => {
    const w = effectiveWeights(['vlm'], { tb: 0.7, general: 0.1, vlm: 0.2 }, 'fixed');
    expect(w.vlm).toBeCloseTo(1, 6);
    const fused = fuseLogOdds([{ id: 'vlm', prob: 0.83 }], w, 0);
    expect(fused).toBeCloseTo(0.83, 5);
  });
  it('two agreeing-high members push fused above each', () => {
    const w = effectiveWeights(['tb', 'vlm'], { tb: 1, general: 1, vlm: 1 }, 'fitted');
    const fused = fuseLogOdds([{ id: 'tb', prob: 0.7 }, { id: 'vlm', prob: 0.7 }], w, 0);
    expect(fused).toBeGreaterThan(0.7);
  });
  it('fitFusionWeights learns to separate', () => {
    const X: number[][] = [];
    const y: (0 | 1)[] = [];
    for (let i = 0; i < 200; i++) {
      const label = (i % 2) as 0 | 1;
      const lt = logit(label === 1 ? 0.8 : 0.2);
      X.push([lt, 0, lt]); // tb + vlm informative, general absent (0)
      y.push(label);
    }
    const { weights } = fitFusionWeights(X, y);
    expect(weights.tb).toBeGreaterThan(0);
  });
});

describe('conformal thresholds', () => {
  it('tauLow guarantees >=92% sensitivity on the calibration set', () => {
    // 100 positives with scores spread 0.2..0.95, 100 negatives 0.05..0.6
    const scores: number[] = [];
    const labels: (0 | 1)[] = [];
    for (let i = 0; i < 100; i++) { scores.push(0.2 + 0.0075 * i); labels.push(1); }
    for (let i = 0; i < 100; i++) { scores.push(0.05 + 0.0055 * i); labels.push(0); }
    const { tauLow, tauHigh } = fitConformalThresholds(scores, labels, {
      alphaSens: 0.92, gammaSpec: 0.1, minPerClass: 20,
    });
    const caught = labels.filter((y, i) => y === 1 && (scores[i] ?? 0) >= tauLow).length;
    expect(caught / 100).toBeGreaterThanOrEqual(0.92);
    expect(tauHigh).toBeGreaterThanOrEqual(tauLow); // never inverted
  });
  it('no positives -> tauLow 0 and incomplete', () => {
    const r = fitConformalThresholds([0.1, 0.2, 0.3], [0, 0, 0]);
    expect(r.tauLow).toBe(0);
    expect(r.incomplete).toBe(true);
  });
});

describe('fitCalibration', () => {
  it('produces fitted params from samples with both classes', () => {
    const samples: CalibrationSample[] = [];
    for (let i = 0; i < 60; i++) {
      const y = (i % 2) as 0 | 1;
      samples.push({
        filename: `f${i}`,
        label: y,
        memberProbs: { tb: y === 1 ? 0.75 : 0.2, vlm: y === 1 ? 0.7 : 0.25 },
        vlmUncertainty: 0.05,
      });
    }
    const p = fitCalibration(samples);
    expect(p.source).toBe('fitted');
    expect(p.conformal.tauHigh).toBeGreaterThanOrEqual(p.conformal.tauLow);
    expect(p.perModel.tb).toBeDefined();
  });
});

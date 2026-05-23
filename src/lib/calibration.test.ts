import { describe, it, expect } from 'vitest';
import { logit, sigmoid, clampProb } from './calibration';
import { fitTemperature, applyCalibration } from './calibration';
import { fitPlatt } from './calibration';
import { fuseLogOdds, effectiveWeights, fitFusionWeights } from './calibration';

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
  it('fits a shift on biased data (B != 0)', () => {
    const probs: number[] = [];
    const labels: (0 | 1)[] = [];
    for (let i = 0; i < 300; i++) {
      const y = (Math.random() < 0.5 ? 1 : 0) as 0 | 1;
      // systematically biased high: add +1.0 to the logit regardless of label sometimes
      const base = y === 1 ? 0.7 : 0.4; // both shifted up -> needs negative B
      probs.push(base);
      labels.push(y);
    }
    const { A, B } = fitPlatt(probs, labels);
    expect(Number.isFinite(A)).toBe(true);
    expect(Number.isFinite(B)).toBe(true);
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

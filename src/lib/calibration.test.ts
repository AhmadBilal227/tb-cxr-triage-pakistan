import { describe, it, expect } from 'vitest';
import { logit, sigmoid, clampProb } from './calibration';
import { fitTemperature, applyCalibration } from './calibration';

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

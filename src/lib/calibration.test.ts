import { describe, it, expect } from 'vitest';
import { logit, sigmoid, clampProb } from './calibration';

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

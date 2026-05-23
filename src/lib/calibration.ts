import { clamp } from './utils';

const EPS = 1e-6;

export function clampProb(p: number): number {
  return clamp(p, EPS, 1 - EPS);
}

export function logit(p: number): number {
  const pc = clampProb(p);
  return Math.log(pc / (1 - pc));
}

export function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

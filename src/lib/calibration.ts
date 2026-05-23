import { clamp } from './utils';
import type { MemberCalibration } from './types';

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

function goldenSectionMin(
  f: (x: number) => number,
  a: number,
  b: number,
  tol: number,
  maxIter: number,
): number {
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = b - gr * (b - a);
  let d = a + gr * (b - a);
  let fc = f(c);
  let fd = f(d);
  for (let k = 0; k < maxIter && b - a > tol; k++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - gr * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + gr * (b - a);
      fd = f(d);
    }
  }
  return (a + b) / 2;
}

function meanNLL(z: number[], labels: (0 | 1)[], transform: (zi: number) => number): number {
  let s = 0;
  for (let i = 0; i < z.length; i++) {
    const p = clampProb(transform(z[i] ?? 0));
    s += labels[i] === 1 ? -Math.log(p) : -Math.log(1 - p);
  }
  return s / z.length;
}

export function fitTemperature(probs: number[], labels: (0 | 1)[]): number {
  const z = probs.map(logit);
  const nll = (T: number): number => meanNLL(z, labels, (zi) => sigmoid(zi / T));
  return goldenSectionMin(nll, 0.05, 20, 1e-4, 100);
}

export function applyCalibration(p: number, c: MemberCalibration): number {
  return c.method === 'platt'
    ? sigmoid(c.A * logit(p) + c.B)
    : sigmoid(logit(p) / c.T);
}

/** Mean binary NLL of a calibration applied to raw probs (for method selection). */
export function calibratedNLL(probs: number[], labels: (0 | 1)[], c: MemberCalibration): number {
  return meanNLL(probs.map(logit), labels, (zi) =>
    c.method === 'platt' ? sigmoid(c.A * zi + c.B) : sigmoid(zi / c.T),
  );
}

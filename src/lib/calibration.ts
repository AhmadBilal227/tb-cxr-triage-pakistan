import { clamp } from './utils';
import type { MemberCalibration, EnsembleMemberId } from './types';

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

export function fitPlatt(
  probs: number[],
  labels: (0 | 1)[],
  iters = 2000,
  lr = 0.1,
): { A: number; B: number } {
  const z = probs.map(logit);
  const n = z.length;
  // Prior-corrected targets for small samples (Platt 1999 / King-Zeng).
  const nPos = labels.filter((y) => y === 1).length;
  const nNeg = n - nPos;
  const tPos = n < 200 ? (nPos + 1) / (nPos + 2) : 1;
  const tNeg = n < 200 ? 1 / (nNeg + 2) : 0;
  let A = 1;
  let B = 0;
  for (let t = 0; t < iters; t++) {
    let gA = 0;
    let gB = 0;
    for (let i = 0; i < n; i++) {
      const q = sigmoid(A * (z[i] ?? 0) + B);
      const target = labels[i] === 1 ? tPos : tNeg;
      const e = q - target;
      gA += e * (z[i] ?? 0);
      gB += e;
    }
    A -= (lr * gA) / n;
    B -= (lr * gB) / n;
  }
  return { A, B };
}

export function fuseLogOdds(
  members: { id: EnsembleMemberId; prob: number }[],
  weights: Record<EnsembleMemberId, number>,
  bias = 0,
): number {
  if (members.length === 0) return 0.5;
  let z = bias;
  for (const m of members) z += (weights[m.id] ?? 0) * logit(m.prob);
  return sigmoid(z);
}

export function effectiveWeights(
  present: EnsembleMemberId[],
  weights: Record<EnsembleMemberId, number>,
  mode: 'fixed' | 'fitted',
): Record<EnsembleMemberId, number> {
  if (mode === 'fitted') return weights; // keep magnitudes + bias
  const sum = present.reduce((a, id) => a + (weights[id] ?? 0), 0) || 1;
  const out = { tb: 0, general: 0, vlm: 0 } as Record<EnsembleMemberId, number>;
  for (const id of present) out[id] = (weights[id] ?? 0) / sum;
  return out;
}

export function fitFusionWeights(
  X: number[][], // rows of [logit(p̂_tb), logit(p̂_general), logit(p̂_vlm)]; absent member -> 0
  y: (0 | 1)[],
  opts = { iters: 3000, lr: 0.05, l2: 1e-2 },
): { weights: Record<EnsembleMemberId, number>; bias: number } {
  const d = 3;
  const w = new Array<number>(d).fill(0);
  let b = 0;
  const n = X.length;
  for (let t = 0; t < opts.iters; t++) {
    const gw = new Array<number>(d).fill(0);
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const row = X[i] ?? [0, 0, 0];
      let z = b;
      for (let j = 0; j < d; j++) z += (w[j] ?? 0) * (row[j] ?? 0);
      const e = sigmoid(z) - (y[i] ?? 0);
      for (let j = 0; j < d; j++) gw[j] = (gw[j] ?? 0) + e * (row[j] ?? 0);
      gb += e;
    }
    for (let j = 0; j < d; j++) w[j] = (w[j] ?? 0) - opts.lr * ((gw[j] ?? 0) / n + opts.l2 * (w[j] ?? 0));
    b -= (opts.lr * gb) / n;
  }
  return { weights: { tb: w[0] ?? 0, general: w[1] ?? 0, vlm: w[2] ?? 0 }, bias: b };
}

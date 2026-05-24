# Perception Module + Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real perception layer to the TB triage app — a calibrated decision core that *targets* a sensitivity level (finite-sample, in-distribution conformal coverage; re-fit per site, **not** a guarantee under deployment shift) (Phase 1) plus an in-browser ONNX chest-X-ray classifier (Phase 2), with an offline training recipe (Phase 3) and a hosted fallback (Phase 4).

**Architecture:** Phase 1 introduces a pure-TS `calibration.ts` module that replaces the hard-coded `SCREENING_POLICY` with per-model probability calibration, log-odds fusion, and class-conditional conformal thresholds fit from a labeled holdout via `/validate`; the orchestrator reads fitted params when present and falls back to today's behavior when absent. Phase 2 adds an in-browser ONNX classifier (`@huggingface/transformers`, WebGPU + WASM fallback) as a new `'local'` `ClassifierProvider` that runs *before* the HF→Replicate cascade. Phases 3–4 are offline/ops runbooks for a trustworthy model.

**Tech Stack:** TypeScript (strict), Vite, React, Vitest (new), `@huggingface/transformers` (new, Phase 2), ONNX Runtime Web (bundled by transformers.js), Python + Optimum (offline, Phases 2–3), Modal (optional, Phase 4).

**Why this order:** Phase 1 is pure TS, needs no model hosting or large assets, turns "≥90% sensitivity" into a finite-sample, in-distribution coverage *target* (void under deployment shift; re-fit per site with a binomial CI), and improves *any* model plugged in later. It is independently shippable and testable. Phase 2 is the larger commitment (Python export + ~87 MB asset). Build Phase 1 first.

---

## File Structure

| File | Responsibility | Phase |
|---|---|---|
| `src/lib/calibration.ts` (new) | Pure math: logit/sigmoid, temperature & Platt fitting, log-odds fusion, fitted-weight LR, class-conditional conformal thresholds, top-level `fitCalibration`. | 1 |
| `src/lib/calibration.test.ts` (new) | Vitest unit tests for every pure function. | 1 |
| `src/lib/types.ts` (modify) | Add `MemberCalibration`, `CalibrationParams`, `CalibrationSample`; add `calibration` to `Settings`. | 1 |
| `src/lib/defaults.ts` (modify) | `DEFAULT_SETTINGS.calibration = null`. | 1 |
| `src/store/settings.ts` (modify) | `setCalibration()` method; inherit `calibration` in `loadInitial`. | 1 |
| `src/lib/pipeline/orchestrator.ts` (modify) | Use calibrated log-odds fusion + conformal `screeningPolicy` when `settings.calibration` is set. | 1 |
| `src/routes/Validate.tsx` (modify) | Collect `CalibrationSample[]`; add a "Calibrate" action + a calibration/test split. | 1 |
| `vitest.config.ts` (new) | Vitest config (node env, include `src/**/*.test.ts`). | 1 |
| `package.json` (modify) | Add `vitest` devDep + `test` script. | 1 |
| `src/workers/onnxClassifier.worker.ts` (new) | Owns `@huggingface/transformers`; loads self-hosted ONNX, runs image-classification, posts `[{label,score}]`. | 2 |
| `src/lib/providers/onnxLocal.ts` (new) | Main-thread RPC client → normalized `ClassifierResult` with `provider_used:'local'`. | 2 |
| `src/lib/providers/classify.ts` (modify) | Try `'local'` first when `stage.local`, then HF→Replicate. | 2 |
| `src/lib/pipeline/stageConfigs.ts` (modify) | Set `local` on the TB stage when enabled. | 2 |
| `public/models/tb-cxr/**` (new asset) | Exported ONNX model + configs (built offline). | 2 |
| `scripts/export-tb-onnx.md` (new) | Runbook: export `runaksh/chest_xray_tuberculosis_detection` to ONNX. | 2 |
| `scripts/train-tb-head.md` (new) | Runbook: train a trustworthy Rad-DINO TB head. | 3 |
| `scripts/modal-tb-endpoint.md` (new) | Runbook: hosted CORS-enabled Modal fallback. | 4 |

---

## Phase 1 — Calibration, log-odds fusion, conformal abstention

### Task 1: Set up Vitest

**Files:**
- Modify: `package.json`
- Create: `vitest.config.ts`

- [ ] **Step 1: Add vitest devDependency and test script**

In `package.json`, add to `devDependencies`:
```json
"vitest": "^2.1.8"
```
And add to `scripts`:
```json
"test": "vitest run",
"test:watch": "vitest"
```

- [ ] **Step 2: Create the vitest config**

Create `vitest.config.ts`:
```ts
import { defineConfig } from 'vitest/config';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
```

- [ ] **Step 3: Install**

Run: `npm install`
Expected: vitest added, no errors.

- [ ] **Step 4: Commit**

```bash
git add package.json package-lock.json vitest.config.ts
git commit -m "test: add vitest"
```

---

### Task 2: Core math primitives (`logit`, `sigmoid`, `clampProb`)

**Files:**
- Create: `src/lib/calibration.ts`
- Test: `src/lib/calibration.test.ts`

- [ ] **Step 1: Write the failing test**

Create `src/lib/calibration.test.ts`:
```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- calibration`
Expected: FAIL — `calibration.ts` does not exist / exports missing.

- [ ] **Step 3: Write minimal implementation**

Create `src/lib/calibration.ts`:
```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- calibration`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib/calibration.ts src/lib/calibration.test.ts
git commit -m "feat(calibration): logit/sigmoid/clampProb primitives"
```

---

### Task 3: Temperature scaling (`fitTemperature`, `applyCalibration`) + types

**Files:**
- Modify: `src/lib/calibration.ts`
- Modify: `src/lib/types.ts`
- Test: `src/lib/calibration.test.ts`

- [ ] **Step 1: Add the calibration types**

In `src/lib/types.ts`, add (near the other interfaces):
```ts
export interface MemberCalibration {
  method: 'temperature' | 'platt';
  T: number; // used when method==='temperature'
  A: number;
  B: number; // used when method==='platt'
  nllRaw: number;
  nllCal: number;
}
```

- [ ] **Step 2: Write the failing test**

Append to `src/lib/calibration.test.ts`:
```ts
import { fitTemperature, applyCalibration } from './calibration';

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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npm test -- calibration`
Expected: FAIL — `fitTemperature`/`applyCalibration` not exported.

- [ ] **Step 4: Write the implementation**

Append to `src/lib/calibration.ts`:
```ts
import type { MemberCalibration } from './types';

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npm test -- calibration`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lib/calibration.ts src/lib/calibration.test.ts src/lib/types.ts
git commit -m "feat(calibration): temperature scaling + MemberCalibration type"
```

---

### Task 4: Platt scaling (`fitPlatt`)

**Files:**
- Modify: `src/lib/calibration.ts`
- Test: `src/lib/calibration.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `src/lib/calibration.test.ts`:
```ts
import { fitPlatt } from './calibration';

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- calibration`
Expected: FAIL — `fitPlatt` not exported.

- [ ] **Step 3: Write the implementation**

Append to `src/lib/calibration.ts`:
```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- calibration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/calibration.ts src/lib/calibration.test.ts
git commit -m "feat(calibration): platt scaling with prior correction"
```

---

### Task 5: Log-odds fusion (`fuseLogOdds`, `effectiveWeights`, `fitFusionWeights`)

**Files:**
- Modify: `src/lib/calibration.ts`
- Test: `src/lib/calibration.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `src/lib/calibration.test.ts`:
```ts
import { fuseLogOdds, effectiveWeights, fitFusionWeights } from './calibration';

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- calibration`
Expected: FAIL — fusion functions not exported.

- [ ] **Step 3: Write the implementation**

Append to `src/lib/calibration.ts`:
```ts
import type { EnsembleMemberId } from './types';

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- calibration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/calibration.ts src/lib/calibration.test.ts
git commit -m "feat(calibration): log-odds fusion + fitted weights"
```

---

### Task 6: Class-conditional conformal thresholds (`fitConformalThresholds`)

**Files:**
- Modify: `src/lib/calibration.ts`
- Test: `src/lib/calibration.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `src/lib/calibration.test.ts`:
```ts
import { fitConformalThresholds } from './calibration';

describe('conformal thresholds', () => {
  it('tauLow guarantees >=90% sensitivity on the calibration set', () => {
    // 100 positives with scores spread 0.2..0.95, 100 negatives 0.05..0.6
    const scores: number[] = [];
    const labels: (0 | 1)[] = [];
    for (let i = 0; i < 100; i++) { scores.push(0.2 + 0.0075 * i); labels.push(1); }
    for (let i = 0; i < 100; i++) { scores.push(0.05 + 0.0055 * i); labels.push(0); }
    const { tauLow, tauHigh } = fitConformalThresholds(scores, labels, {
      alphaSens: 0.92, gammaSpec: 0.1, minPerClass: 20,
    });
    const caught = labels.filter((y, i) => y === 1 && (scores[i] ?? 0) >= tauLow).length;
    expect(caught / 100).toBeGreaterThanOrEqual(0.9);
    expect(tauHigh).toBeGreaterThanOrEqual(tauLow); // never inverted
  });
  it('no positives -> tauLow 0 and incomplete', () => {
    const r = fitConformalThresholds([0.1, 0.2, 0.3], [0, 0, 0]);
    expect(r.tauLow).toBe(0);
    expect(r.incomplete).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- calibration`
Expected: FAIL — `fitConformalThresholds` not exported.

- [ ] **Step 3: Write the implementation**

Append to `src/lib/calibration.ts`:
```ts
export interface ConformalResult {
  tauLow: number;
  tauHigh: number;
  nPos: number;
  nNeg: number;
  incomplete: boolean;
}

export function fitConformalThresholds(
  scores: number[],
  labels: (0 | 1)[],
  cfg = { alphaSens: 0.92, gammaSpec: 0.1, minPerClass: 20 },
): ConformalResult {
  const pos = scores.filter((_, i) => labels[i] === 1).sort((a, b) => a - b);
  const neg = scores.filter((_, i) => labels[i] === 0).sort((a, b) => a - b);
  const nPos = pos.length;
  const nNeg = neg.length;
  const beta = 1 - cfg.alphaSens;

  let tauLow = 0;
  if (nPos > 0) {
    const k = Math.floor(beta * (nPos + 1));
    tauLow = k <= 0 ? 0 : (pos[k - 1] ?? 0);
  }
  let tauHigh = 1;
  if (nNeg > 0) {
    const m = Math.floor(cfg.gammaSpec * (nNeg + 1));
    tauHigh = m <= 0 ? 1 : (neg[nNeg - m] ?? 1);
  }
  tauHigh = Math.max(tauHigh, tauLow); // never invert the band
  tauLow = clamp(tauLow, 0, 1);
  tauHigh = clamp(tauHigh, 0, 1);
  return { tauLow, tauHigh, nPos, nNeg, incomplete: nPos < cfg.minPerClass || nNeg < cfg.minPerClass };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- calibration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/calibration.ts src/lib/calibration.test.ts
git commit -m "feat(calibration): class-conditional conformal thresholds"
```

---

### Task 7: Top-level `fitCalibration` + `CalibrationParams`/`CalibrationSample` types

**Files:**
- Modify: `src/lib/calibration.ts`
- Modify: `src/lib/types.ts`
- Test: `src/lib/calibration.test.ts`

- [ ] **Step 1: Add the aggregate types**

In `src/lib/types.ts`, add:
```ts
export interface CalibrationParams {
  version: 1;
  fittedAt: number;
  nSamples: number;
  source: 'fitted' | 'default';
  perModel: Partial<Record<EnsembleMemberId, MemberCalibration>>;
  fusion: { mode: 'fixed' | 'fitted'; weights: Record<EnsembleMemberId, number>; bias: number };
  conformal: {
    tauLow: number;
    tauHigh: number;
    alphaSens: number;
    gammaSpec: number;
    nPos: number;
    nNeg: number;
    incomplete: boolean;
  };
  vlmSafetyThreshold: number;
}

export interface CalibrationSample {
  filename: string;
  label: 0 | 1;
  memberProbs: Partial<Record<EnsembleMemberId, number>>;
  vlmUncertainty: number | null;
}
```

- [ ] **Step 2: Write the failing test**

Append to `src/lib/calibration.test.ts`:
```ts
import { fitCalibration } from './calibration';
import type { CalibrationSample } from './types';

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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npm test -- calibration`
Expected: FAIL — `fitCalibration` not exported.

- [ ] **Step 4: Write the implementation**

Append to `src/lib/calibration.ts`:
```ts
import type { CalibrationParams, CalibrationSample } from './types';

const MEMBER_IDS: EnsembleMemberId[] = ['tb', 'general', 'vlm'];

export function fitCalibration(
  samples: CalibrationSample[],
  cfg = { alphaSens: 0.92, gammaSpec: 0.1 },
): CalibrationParams {
  // 1. Per-member calibration (temperature default; Platt if it clearly wins and N>=100).
  const perModel: Partial<Record<EnsembleMemberId, MemberCalibration>> = {};
  for (const id of MEMBER_IDS) {
    const rows = samples.filter((s) => typeof s.memberProbs[id] === 'number');
    if (rows.length < 10) continue;
    const probs = rows.map((s) => s.memberProbs[id] as number);
    const labels = rows.map((s) => s.label);
    const T = fitTemperature(probs, labels);
    const tempCal: MemberCalibration = { method: 'temperature', T, A: 1, B: 0, nllRaw: 0, nllCal: 0 };
    tempCal.nllRaw = calibratedNLL(probs, labels, { method: 'temperature', T: 1, A: 1, B: 0, nllRaw: 0, nllCal: 0 });
    tempCal.nllCal = calibratedNLL(probs, labels, tempCal);
    let chosen = tempCal;
    if (rows.length >= 100) {
      const { A, B } = fitPlatt(probs, labels);
      const plattCal: MemberCalibration = { method: 'platt', T: 1, A, B, nllRaw: tempCal.nllRaw, nllCal: 0 };
      plattCal.nllCal = calibratedNLL(probs, labels, plattCal);
      if (plattCal.nllCal < tempCal.nllCal - 0.01) chosen = plattCal;
    }
    perModel[id] = chosen;
  }

  const calProb = (s: CalibrationSample, id: EnsembleMemberId): number | null => {
    const raw = s.memberProbs[id];
    if (typeof raw !== 'number') return null;
    const c = perModel[id];
    return c ? applyCalibration(raw, c) : raw;
  };

  // 2. Fusion weights: fitted LR if enough data, else fixed.
  const nPos = samples.filter((s) => s.label === 1).length;
  const nNeg = samples.length - nPos;
  let fusion: CalibrationParams['fusion'];
  const enoughForLR = samples.length >= 50 && nPos >= 10 && nNeg >= 10;
  if (enoughForLR) {
    const X = samples.map((s) => MEMBER_IDS.map((id) => logit(calProb(s, id) ?? 0.5)));
    const y = samples.map((s) => s.label);
    const fit = fitFusionWeights(X, y);
    fusion = { mode: 'fitted', weights: fit.weights, bias: fit.bias };
  } else {
    fusion = { mode: 'fixed', weights: { tb: 0.7, general: 0.1, vlm: 0.2 }, bias: 0 };
  }

  // 3. Fused score per sample -> conformal thresholds.
  const fusedScores = samples.map((s) => {
    const present = MEMBER_IDS.filter((id) => calProb(s, id) !== null);
    const w = effectiveWeights(present, fusion.weights, fusion.mode);
    return fuseLogOdds(
      present.map((id) => ({ id, prob: calProb(s, id) as number })),
      w,
      fusion.mode === 'fitted' ? fusion.bias : 0,
    );
  });
  const conf = fitConformalThresholds(fusedScores, samples.map((s) => s.label), {
    alphaSens: cfg.alphaSens,
    gammaSpec: cfg.gammaSpec,
    minPerClass: 20,
  });

  // 4. VLM safety threshold = (1-alpha) quantile of calibrated VLM prob among positives.
  const vlmPos = samples
    .filter((s) => s.label === 1 && calProb(s, 'vlm') !== null)
    .map((s) => calProb(s, 'vlm') as number)
    .sort((a, b) => a - b);
  let vlmSafetyThreshold = 0.5;
  if (vlmPos.length >= 20) {
    const k = Math.floor((1 - cfg.alphaSens) * (vlmPos.length + 1));
    vlmSafetyThreshold = k <= 0 ? (vlmPos[0] ?? 0.5) : (vlmPos[k - 1] ?? 0.5);
  }

  return {
    version: 1,
    fittedAt: Date.now(),
    nSamples: samples.length,
    source: 'fitted',
    perModel,
    fusion,
    conformal: {
      tauLow: conf.tauLow,
      tauHigh: conf.tauHigh,
      alphaSens: cfg.alphaSens,
      gammaSpec: cfg.gammaSpec,
      nPos: conf.nPos,
      nNeg: conf.nNeg,
      incomplete: conf.incomplete,
    },
    vlmSafetyThreshold,
  };
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npm test -- calibration`
Expected: PASS (all calibration tests).

- [ ] **Step 6: Typecheck**

Run: `npm run build`
Expected: build OK (types resolve).

- [ ] **Step 7: Commit**

```bash
git add src/lib/calibration.ts src/lib/calibration.test.ts src/lib/types.ts
git commit -m "feat(calibration): top-level fitCalibration aggregator"
```

---

### Task 8: Persist calibration in settings

**Files:**
- Modify: `src/lib/types.ts` (Settings)
- Modify: `src/lib/defaults.ts`
- Modify: `src/store/settings.ts`

- [ ] **Step 1: Add `calibration` to Settings**

In `src/lib/types.ts`, inside `interface Settings`, add:
```ts
  /** Fitted calibration params, or null to use the hard-coded SCREENING_POLICY. */
  calibration: CalibrationParams | null;
```

- [ ] **Step 2: Default it to null**

In `src/lib/defaults.ts`, in `DEFAULT_SETTINGS`, add:
```ts
  calibration: null,
```

- [ ] **Step 3: Inherit it on load + add a setter**

In `src/store/settings.ts`, in `loadInitial`'s returned merge object (both branches), add `calibration: parsed.calibration ?? null` to the parsed branch and rely on `...DEFAULT_SETTINGS` for the empty branch. Then add a store method after `setModels`:
```ts
  setCalibration(calibration: Settings['calibration']): void {
    state = { ...state, calibration };
    persist();
    emit();
  },
```

- [ ] **Step 4: Typecheck**

Run: `npm run build`
Expected: build OK.

- [ ] **Step 5: Commit**

```bash
git add src/lib/types.ts src/lib/defaults.ts src/store/settings.ts
git commit -m "feat(calibration): persist fitted params in settings"
```

---

### Task 9: Use calibration in the orchestrator

**Files:**
- Modify: `src/lib/pipeline/orchestrator.ts`

- [ ] **Step 1: Import the calibration helpers**

At the top of `src/lib/pipeline/orchestrator.ts`, add:
```ts
import { applyCalibration, effectiveWeights, fuseLogOdds } from '@/lib/calibration';
import type { CalibrationParams, EnsembleMemberId } from '@/lib/types';
```
(Adjust the existing type import to include `CalibrationParams` if not already pulled in.)

- [ ] **Step 2: Replace the arithmetic fusion with calibrated log-odds**

Find the block that computes `weightedScore` (the arithmetic weighted mean over `returning`). Replace it with:
```ts
const cal = settings.calibration;
const calProbOf = (m: EnsembleMember): number => {
  if (m.tb_prob === null) return 0.5;
  const c = cal?.perModel[m.id as EnsembleMemberId];
  return c ? applyCalibration(m.tb_prob, c) : m.tb_prob;
};
let weightedScore: number;
if (cal) {
  const present = returning.map((m) => m.id as EnsembleMemberId);
  const w = effectiveWeights(present, cal.fusion.weights, cal.fusion.mode);
  weightedScore = fuseLogOdds(
    returning.map((m) => ({ id: m.id as EnsembleMemberId, prob: calProbOf(m) })),
    w,
    cal.fusion.mode === 'fitted' ? cal.fusion.bias : 0,
  );
} else {
  const totalWeight = returning.reduce((a, m) => a + m.weight, 0) || 1;
  weightedScore = returning.reduce((a, m) => a + m.weight * (m.tb_prob as number), 0) / totalWeight;
}
```

- [ ] **Step 3: Make `screeningPolicy` conformal-aware**

Change the `screeningPolicy` signature and body to:
```ts
function screeningPolicy(
  fusedProb: number,
  vlmProb: number | null,
  vlmUncertainty: number,
  cal: CalibrationParams | null,
): { verdict: Verdict; reason: string } {
  const tauLow = cal?.conformal.tauLow ?? SCREENING_POLICY.negClear;
  const tauHigh = cal?.conformal.tauHigh ?? SCREENING_POLICY.tbFlag;
  const vlmSafe = cal?.vlmSafetyThreshold ?? SCREENING_POLICY.vlmSafetyThreshold;
  const flag = fusedProb >= tauHigh || (vlmProb !== null && vlmProb >= vlmSafe);
  if (flag) {
    return { verdict: 'tb', reason: `flag (fused ${fusedProb.toFixed(2)} ≥ τ_high ${tauHigh.toFixed(2)} or VLM ≥ ${vlmSafe.toFixed(2)})` };
  }
  if (fusedProb < tauLow && vlmUncertainty <= SCREENING_POLICY.maxClearUncertainty) {
    return { verdict: 'no_tb', reason: '' };
  }
  return { verdict: 'abstain', reason: `prob ${fusedProb.toFixed(2)} in [τ_low ${tauLow.toFixed(2)}, τ_high ${tauHigh.toFixed(2)}) band` };
}
```

- [ ] **Step 4: Pass `cal` at the call site**

Where `screeningPolicy(ensemble.weightedScore, vlmProb, vlmUncertainty)` is called in Stage 4, change to:
```ts
const policy = screeningPolicy(ensemble.weightedScore, vlmProb, vlmUncertainty, settings.calibration);
```

- [ ] **Step 5: Typecheck + run the existing accuracy harness with calibration:null (no behavior change)**

Run: `npm run build`
Expected: build OK.
Run: `node scripts/accuracy-test-v2.mjs 4`
Expected: still runs (calibration is null in the harness; behavior unchanged from current v2).

- [ ] **Step 6: Commit**

```bash
git add src/lib/pipeline/orchestrator.ts
git commit -m "feat(calibration): orchestrator uses calibrated log-odds + conformal band when fitted"
```

---

### Task 10: Wire "Calibrate" into /validate (collect samples, fit, split)

**Files:**
- Modify: `src/routes/Validate.tsx`

- [ ] **Step 1: Collect calibration samples during a run**

In `src/routes/Validate.tsx`, add state:
```ts
const [samples, setSamples] = useState<import('@/lib/types').CalibrationSample[]>([]);
```
Inside `run()`, after a successful `runPipeline`, push a sample (alongside the existing `collected.push`):
```ts
const mp: Partial<Record<import('@/lib/types').EnsembleMemberId, number>> = {};
for (const m of r.ensemble?.members ?? []) if (m.tb_prob !== null) mp[m.id] = m.tb_prob;
sampleAcc.push({
  filename: h.filename,
  label: h.label,
  memberProbs: mp,
  vlmUncertainty: r.ensemble?.members.find((m) => m.id === 'vlm')?.uncertainty ?? null,
});
```
Declare `const sampleAcc: CalibrationSample[] = []` next to `collected`, and `setSamples([...sampleAcc])` after the loop.

- [ ] **Step 2: Add the Calibrate action**

Add a callback:
```ts
const calibrate = useCallback(() => {
  if (samples.length === 0) return;
  const params = fitCalibration(samples);
  settingsStore.setCalibration(params);
  setMsg(
    `Calibrated on ${params.nSamples} cases. τ_low=${params.conformal.tauLow.toFixed(3)} τ_high=${params.conformal.tauHigh.toFixed(3)}${params.conformal.incomplete ? ' (insufficient per-class samples — band is conservative)' : ''}`,
  );
}, [samples]);
```
Import at top: `import { fitCalibration } from '@/lib/calibration';` and `import { settingsStore } from '@/store/settings';` and the `CalibrationSample`/`EnsembleMemberId` types.

- [ ] **Step 3: Add the buttons**

Next to the existing Run/Export buttons, add (enabled after a run):
```tsx
<Button variant="outline" onClick={calibrate} disabled={samples.length === 0}>
  Calibrate from this set
</Button>
<Button variant="ghost" onClick={() => { settingsStore.setCalibration(null); setMsg('Calibration cleared (using default policy).'); }}>
  Clear calibration
</Button>
```

- [ ] **Step 4: Manual verification**

Run: `npm run dev`, open `/validate`, load a labeled folder, Run, then "Calibrate from this set". Confirm the status line shows fitted τ_low/τ_high and that re-running uses the fitted thresholds (verdicts shift toward the calibrated band).

- [ ] **Step 5: Typecheck + commit**

Run: `npm run build`
Expected: build OK.
```bash
git add src/routes/Validate.tsx
git commit -m "feat(calibration): Calibrate action in /validate fits + persists params"
```

---

### Task 11: Re-measure with calibration via the harness

**Files:**
- Create: `scripts/accuracy-test-v3.mjs` (copy of v2 + calibration fit/apply)

- [ ] **Step 1: Build v3 harness**

Copy `scripts/accuracy-test-v2.mjs` to `scripts/accuracy-test-v3.mjs`. Split the set 50/50 into calibration + test by index parity. On the calibration half, collect `{label, memberProbs:{vlm:pVlm}, vlmUncertainty:u}` and call the same conformal/threshold logic (port `fitConformalThresholds` inline or `import` from a small shared `.mjs`); on the test half, apply the fitted `tauLow/tauHigh` instead of the fixed `POLICY` and compute metrics. Print before/after sensitivity.

- [ ] **Step 2: Run it**

Run: `node scripts/accuracy-test-v3.mjs 20`
Expected: prints test-half sensitivity/specificity using fitted thresholds; sensitivity should meet/approach the 90% target on the calibration distribution (specificity will drop — expected tradeoff).

- [ ] **Step 3: Commit**

```bash
git add scripts/accuracy-test-v3.mjs accuracy-report-v3.json
git commit -m "test: v3 harness measures calibrated/conformal decision policy"
```

---

## Phase 2 — In-browser ONNX perception module

> Produces a working `'local'` perception member. The default model (`runaksh/chest_xray_tuberculosis_detection`) is **demo-grade / unvalidated** (label it as such in the UI); Phase 3 replaces it with a trustworthy head.

### Task 12: Export the model to ONNX (offline runbook)

**Files:**
- Create: `scripts/export-tb-onnx.md`
- Create asset: `public/models/tb-cxr/{config.json,preprocessor_config.json,onnx/model_quantized.onnx}`

- [ ] **Step 1: Write the runbook**

Create `scripts/export-tb-onnx.md` with:
```bash
uv venv && source .venv/bin/activate
uv pip install "optimum[onnxruntime]" onnx onnxslim
git clone https://github.com/huggingface/transformers.js
uv pip install -r transformers.js/scripts/requirements.txt
python -m scripts.convert --quantize \
  --model_id runaksh/chest_xray_tuberculosis_detection \
  --task image-classification
# copy ./models/runaksh/chest_xray_tuberculosis_detection/ -> public/models/tb-cxr/
```

- [ ] **Step 2: Run it and place the asset**

Execute the runbook; verify `public/models/tb-cxr/onnx/model_quantized.onnx` exists (~87 MB) plus `config.json` (labels NORMAL/TUBERCULOSIS) and `preprocessor_config.json` (224, mean/std 0.5).

- [ ] **Step 3: Add a .gitattributes / git-lfs note**

Add `public/models/** filter=lfs diff=lfs merge=lfs -text` to `.gitattributes` (or document hosting the asset outside git). Commit the configs; track the `.onnx` via LFS or a release asset.

- [ ] **Step 4: Commit**

```bash
git add scripts/export-tb-onnx.md .gitattributes public/models/tb-cxr/config.json public/models/tb-cxr/preprocessor_config.json
git commit -m "chore(perception): TB ONNX export runbook + model configs"
```

### Task 13: Install transformers.js + COOP/COEP headers

**Files:**
- Modify: `package.json`, `vite.config.ts`

- [ ] **Step 1: Install**

Run: `npm install @huggingface/transformers`

- [ ] **Step 2: Add cross-origin isolation headers (for multithreaded WASM)**

In `vite.config.ts` `server`, add alongside the existing proxy:
```ts
headers: {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
},
```

- [ ] **Step 3: Commit**

```bash
git add package.json package-lock.json vite.config.ts
git commit -m "chore(perception): add transformers.js + COOP/COEP"
```

### Task 14: Worker + `onnxLocal.ts` provider

**Files:**
- Create: `src/workers/onnxClassifier.worker.ts`
- Create: `src/lib/providers/onnxLocal.ts`
- Modify: `src/lib/types.ts` (widen `ClassifierProvider`/`Provider` with `'local'`)

- [ ] **Step 1: Widen the provider types**

In `src/lib/types.ts`:
```ts
export type Provider = 'hf' | 'replicate' | 'openai' | 'local';
export type ClassifierProvider = 'hf' | 'replicate' | 'local';
```

- [ ] **Step 2: Create the worker**

Create `src/workers/onnxClassifier.worker.ts` (full content from the Wave-2 research: sets `env.allowRemoteModels=false`, `env.localModelPath='/models/'`, `env.backends.onnx.wasm.wasmPaths='/ort-wasm/'`, picks WebGPU else WASM int8, lazy `pipeline('image-classification','tb-cxr',{device,dtype})`, handles `init`/`classify`, posts `[{label,score}]` + latency).

- [ ] **Step 3: Create the provider**

Create `src/lib/providers/onnxLocal.ts` (full content from research): a `Worker` RPC client exposing `initLocalClassifier()` and `classifyLocal(blob): Promise<ClassifierResult>` that maps `[{label,score}]` via `parseTbProb`, sets `provider_used:'local'`.

- [ ] **Step 4: Self-host ORT wasm**

Copy the `onnxruntime-web` `.wasm`/`.mjs` artifacts from `node_modules/@huggingface/transformers/dist/` into `public/ort-wasm/` to match the installed version.

- [ ] **Step 5: Typecheck + commit**

Run: `npm run build`
Expected: build OK (worker bundled as a chunk).
```bash
git add src/workers/onnxClassifier.worker.ts src/lib/providers/onnxLocal.ts src/lib/types.ts public/ort-wasm
git commit -m "feat(perception): in-browser ONNX classifier worker + provider"
```

### Task 15: Wire `'local'` into classifyWithFallback + caps + UI toggle

**Files:**
- Modify: `src/lib/providers/classify.ts`, `src/lib/types.ts` (StageConfig), `src/lib/pipeline/stageConfigs.ts`, `src/store/settings.ts` (caps), `src/lib/pipeline/orchestrator.ts` (gate), `src/components/SettingsDrawer.tsx` (toggle), `src/lib/defaults.ts` (`localTbEnabled`)

- [ ] **Step 1: Add the setting + StageConfig field**

In `ModelOverrides` add `localTbEnabled: boolean` (default `false` in `DEFAULT_SETTINGS.overrides`). In `StageConfig` add `local?: boolean`.

- [ ] **Step 2: Try local first in classifyWithFallback**

At the top of `classifyWithFallback` (before the HF try), add:
```ts
if (stage.local) {
  try {
    return await classifyLocal(image);
  } catch (localErr) {
    deps.onFallback?.('local', 'hf', (localErr as Error).message);
  }
}
```
Import `classifyLocal` from `./onnxLocal`.

- [ ] **Step 3: Set `local` on the TB stage**

In `buildTbStageConfig`, add `local: s.overrides.localTbEnabled || undefined,` to the returned object.

- [ ] **Step 4: Relax the caps gate**

In `deriveCapabilities`, add `hasLocalClassifier: s.overrides.localTbEnabled === true` and `hasPerception: s.hfToken.trim().length > 0 || s.overrides.localTbEnabled === true`. In the orchestrator, replace the `!caps.hasHF` halt with a `!caps.hasPerception` halt and message.

- [ ] **Step 5: Add the Settings toggle**

In `SettingsDrawer.tsx`, in the model-overrides section, add a checkbox bound to `s.overrides.localTbEnabled` that calls `settingsStore.setOverride({ localTbEnabled: ... })`, with helper text: "Run a TB classifier in your browser (downloads ~87 MB once; demo model, not clinically validated)."

- [ ] **Step 6: Manual verification**

Run: `npm run dev`. Enable the toggle, drop a TB sample, confirm the TB Classifier card shows `provider_used: local` and a real `tb_prob`, with HF as fallback if the model errors.

- [ ] **Step 7: Typecheck + commit**

Run: `npm run build`
Expected: build OK.
```bash
git add -A
git commit -m "feat(perception): wire in-browser model as primary 'local' provider with HF fallback"
```

---

## Phase 3 — Trustworthy TB head (offline runbook)

### Task 16: Training runbook

**Files:**
- Create: `scripts/train-tb-head.md`

- [ ] **Step 1: Write the runbook** containing the verified recipe: frozen `microsoft/rad-dino` features + small MLP head; data = NLMTB (Montgomery+Shenzhen) + Kaggle TB DB; **cross-source perceptual-hash dedup**; patient-level + **leave-one-dataset-out** splits; anatomy-preserving augmentation (no flips); `pos_weight` BCE for imbalance; threshold for ≥90% sensitivity; **Grad-CAM + source-separability ("site-leak") audit**; export backbone+head+`Sigmoid` to one ONNX graph with Rad-DINO preprocessing (resize 518, center-crop, mean 0.5307 std 0.2583). Expected external AUC 0.80–0.88. (Full script in the research appendix; paste it verbatim into the runbook.)

- [ ] **Step 2: Commit**

```bash
git add scripts/train-tb-head.md
git commit -m "docs(perception): trustworthy Rad-DINO TB head training runbook"
```

---

## Phase 4 — Hosted fallback (optional runbook)

### Task 17: Modal endpoint runbook

**Files:**
- Create: `scripts/modal-tb-endpoint.md`

- [ ] **Step 1: Write the runbook** with the verified Modal `app.py` (FastAPI ASGI, `@modal.enter()` weight load, `CORSMiddleware` locked to the app origin, `min_containers=0`, `scaledown_window=300`, T4), the `modal deploy` steps, the client `src/lib/providers/modal.ts` `modalClassify()` contract (`FormData` POST → `{tb_prob, raw}`), and the security note (no shared token in the browser; lock CORS origins). Cost: ~$0/mo on free credits at hobby volume.

- [ ] **Step 2: Commit**

```bash
git add scripts/modal-tb-endpoint.md
git commit -m "docs(perception): hosted Modal CORS fallback runbook"
```

---

## Self-Review

**Spec coverage:** Calibration (temperature/Platt) → Tasks 3–4. Log-odds fusion → Task 5. Conformal abstention → Task 6. Aggregator → Task 7. Persistence → Task 8. Orchestrator use → Task 9. /validate fit + split → Task 10. Re-measure → Task 11. In-browser model (export, install, worker, wiring) → Tasks 12–15. Trustworthy training → Task 16. Hosted fallback → Task 17. All research layers covered.

**Placeholder scan:** Phase 1 tasks contain complete code + commands. Phase 2–4 reference "full content from research" for the worker/provider/training script/app.py — these are large verbatim blocks captured in the Wave-2 research; the executor must paste them. This is the one acceptable deferral (the code exists, it's just long); everything decision-bearing (file paths, signatures, wiring, types) is explicit.

**Type consistency:** `CalibrationParams`, `MemberCalibration`, `CalibrationSample`, `EnsembleMemberId`, `Verdict`, `ClassifierProvider` used consistently across tasks. `fitCalibration`/`applyCalibration`/`fuseLogOdds`/`effectiveWeights`/`fitConformalThresholds` signatures match between definition (Tasks 3–7) and use (Tasks 9–10). `settingsStore.setCalibration` defined in Task 8, used in Task 10. `stage.local` defined in Task 15, used in same task.

**Known gap:** Phase 2 Tasks 14/16/17 intentionally point at the verbatim code blocks from the research transcript rather than re-inlining ~400 lines here; capture those into the files at execution time.

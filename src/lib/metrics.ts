import type { Verdict } from './types';

export interface ValItem {
  filename: string;
  trueLabel: 0 | 1;
  verdict: Verdict | null;
  score: number | null; // ensemble weighted score, used for AUC
  abstained: boolean;
  halted: boolean;
  error?: string;
}

export interface Confusion {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface Metrics {
  total: number;
  nDecided: number; // non-abstain, non-halted
  nAbstain: number;
  nHalted: number;
  confusion: Confusion;
  accuracy: number;
  sensitivity: number; // recall for TB
  specificity: number;
  auc: number; // NaN if a class is absent
}

/** Mann–Whitney U / rank-based ROC AUC over items with a numeric score. */
export function computeAUC(items: { score: number; label: 0 | 1 }[]): number {
  const pos = items.filter((i) => i.label === 1);
  const neg = items.filter((i) => i.label === 0);
  if (pos.length === 0 || neg.length === 0) return NaN;

  const sorted = [...items].sort((a, b) => a.score - b.score);
  // Assign average ranks (1-based), handling ties.
  const ranks = new Array<number>(sorted.length);
  let i = 0;
  while (i < sorted.length) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1]!.score === sorted[i]!.score) j++;
    const avgRank = (i + j) / 2 + 1; // average of (i+1..j+1)
    for (let k = i; k <= j; k++) ranks[k] = avgRank;
    i = j + 1;
  }
  let sumRanksPos = 0;
  sorted.forEach((it, idx) => {
    if (it.label === 1) sumRanksPos += ranks[idx]!;
  });
  const nPos = pos.length;
  const nNeg = neg.length;
  return (sumRanksPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

export function computeMetrics(items: ValItem[]): Metrics {
  const confusion: Confusion = { tp: 0, fp: 0, tn: 0, fn: 0 };
  let nAbstain = 0;
  let nHalted = 0;

  for (const it of items) {
    if (it.halted) {
      nHalted++;
      continue;
    }
    if (it.verdict === 'abstain' || it.verdict === null) {
      nAbstain++;
      continue;
    }
    const pred = it.verdict === 'tb' ? 1 : 0;
    if (pred === 1 && it.trueLabel === 1) confusion.tp++;
    else if (pred === 1 && it.trueLabel === 0) confusion.fp++;
    else if (pred === 0 && it.trueLabel === 0) confusion.tn++;
    else confusion.fn++;
  }

  const { tp, fp, tn, fn } = confusion;
  const nDecided = tp + fp + tn + fn;
  const accuracy = nDecided ? (tp + tn) / nDecided : NaN;
  const sensitivity = tp + fn ? tp / (tp + fn) : NaN;
  const specificity = tn + fp ? tn / (tn + fp) : NaN;

  const aucItems = items
    .filter((it) => it.score !== null && !it.halted)
    .map((it) => ({ score: it.score as number, label: it.trueLabel }));
  const auc = computeAUC(aucItems);

  return {
    total: items.length,
    nDecided,
    nAbstain,
    nHalted,
    confusion,
    accuracy,
    sensitivity,
    specificity,
    auc,
  };
}

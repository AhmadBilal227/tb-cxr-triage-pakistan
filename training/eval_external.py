"""External blind evaluation on the Kiran/Jabeen Pakistani cohort.

A genuinely-new cohort (NOT in data/index.csv: the model trained LODO on
montgomery/qatar/shenzhen/tbx11k only). Labels by folder:
  "TB Chest X-rays"     -> 1
  "Normal Chest X-rays" -> 0

Runs the deployed validated verdict path (TriageEngine.run) per image,
bypassing the browser quality gate. Records calibrated tb_prob + verdict +
safety_net for each image. Writes results incrementally so a crash keeps
progress. Computes the confusion matrix + sensitivity/specificity/PPV/NPV
with Wilson 95% CIs at the deployed threshold (0.6105), AUROC, and PPV at a
realistic screening prevalence.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
    training/.venv/bin/python training/eval_external.py [--limit-per-class N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

from triage_core import get_engine  # noqa: E402

COHORT = REPO / "data" / "raw" / "Kiran:Jabeen"
OUT = REPO / "data" / "eval_kiran_jabeen.json"
FOLDERS = {"TB Chest X-rays": 1, "Normal Chest X-rays": 0}
IMG_EXT = {".png", ".jpg", ".jpeg"}


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Point estimate + Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def collect_images(limit_per_class: int | None) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for folder, label in FOLDERS.items():
        paths = sorted(p for p in (COHORT / folder).iterdir() if p.suffix.lower() in IMG_EXT)
        if limit_per_class is not None:
            paths = paths[:limit_per_class]
        items.extend((p, label) for p in paths)
    return items


def run_eval(limit_per_class: int | None) -> list[dict]:
    eng = get_engine()
    items = collect_images(limit_per_class)
    print(f"[eval] {len(items)} images "
          f"({sum(1 for _, l in items if l == 1)} TB+, {sum(1 for _, l in items if l == 0)} normal)",
          flush=True)

    results: list[dict] = []
    t0 = time.perf_counter()
    for i, (path, label) in enumerate(items):
        try:
            res = eng.run(path.read_bytes())
            results.append({
                "file": path.name,
                "label": label,
                "tb_prob": float(res.tb_prob),
                "s_inactive": float(res.s_inactive),
                "verdict": res.verdict,
                "safety_net_applied": res.safety_net_applied,
                "decided_at_threshold": float(res.decided_at_threshold),
            })
        except Exception as e:  # noqa: BLE001 — record + continue
            results.append({"file": path.name, "label": label, "error": str(e)[:160]})
        if (i + 1) % 50 == 0:
            rate = (time.perf_counter() - t0) / (i + 1)
            print(f"[eval] {i + 1}/{len(items)}  {rate:.2f}s/img  eta {rate * (len(items) - i - 1) / 60:.1f}min",
                  flush=True)
            OUT.write_text(json.dumps(results))
    OUT.write_text(json.dumps(results))
    print(f"[eval] done in {(time.perf_counter() - t0) / 60:.1f}min -> {OUT}", flush=True)
    return results


def metrics(results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    errs = len(results) - len(ok)
    y = np.array([r["label"] for r in ok])
    p = np.array([r["tb_prob"] for r in ok])
    thr = ok[0]["decided_at_threshold"] if ok else 0.6105
    pred = (p >= thr).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())

    sens, sl, sh = wilson(tp, n_pos)
    spec, pl, ph = wilson(tn, n_neg)
    ppv, vl, vh = wilson(tp, tp + fp) if (tp + fp) else (float("nan"),) * 3
    npv, nl, nh = wilson(tn, tn + fn) if (tn + fn) else (float("nan"),) * 3

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p)) if n_pos and n_neg else float("nan")
    except Exception:
        auc = float("nan")

    # Verdict distribution (escalators turn some no_tb into abstain).
    verdicts: dict[str, int] = {}
    for r in ok:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

    print("\n" + "=" * 64)
    print("EXTERNAL BLIND EVAL — Kiran/Jabeen (Pakistani cohort)")
    print("=" * 64)
    print(f"images scored: {len(ok)}  (errors: {errs})   prevalence: {n_pos}/{len(ok)} = {n_pos/len(ok):.3f}")
    print(f"threshold (calibrated tb_prob >= ): {thr:.4f}")
    print("\nConfusion matrix (binary @ threshold):")
    print(f"               pred TB   pred no-TB")
    print(f"  actual TB      {tp:5d}      {fn:5d}   (n={n_pos})")
    print(f"  actual normal  {fp:5d}      {tn:5d}   (n={n_neg})")
    print("\nMetrics [95% Wilson CI]:")
    print(f"  sensitivity (recall)  {sens:.3f}  [{sl:.3f}, {sh:.3f}]")
    print(f"  specificity           {spec:.3f}  [{pl:.3f}, {ph:.3f}]")
    print(f"  PPV (this prevalence) {ppv:.3f}  [{vl:.3f}, {vh:.3f}]")
    print(f"  NPV (this prevalence) {npv:.3f}  [{nl:.3f}, {nh:.3f}]")
    print(f"  accuracy              {(tp+tn)/len(ok):.3f}")
    print(f"  AUROC                 {auc:.3f}")
    # PPV recomputed at screening prevalences using THIS sens/spec.
    print("\nPPV / NPV at realistic screening prevalence (using measured sens/spec):")
    for prev in (0.01, 0.05, 0.10):
        tpr, fpr = sens, 1 - spec
        ppv_p = (tpr * prev) / (tpr * prev + fpr * (1 - prev)) if (tpr * prev + fpr * (1 - prev)) else float("nan")
        npv_p = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev)
        print(f"  prevalence {prev:>4.0%}:  PPV {ppv_p:.3f}   NPV {npv_p:.3f}")
    print("\nDeployed verdict distribution (incl. safety-net abstains):")
    for k, v in sorted(verdicts.items()):
        print(f"  {k:8s}: {v}")
    print("=" * 64, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--metrics-only", action="store_true", help="recompute metrics from existing JSON")
    args = ap.parse_args()
    if args.metrics_only and OUT.exists():
        metrics(json.loads(OUT.read_text()))
        return 0
    results = run_eval(args.limit_per_class)
    metrics(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

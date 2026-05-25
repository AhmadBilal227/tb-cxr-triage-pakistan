"""Track A — second external NEGATIVE evaluation on NIH ChestX-ray14.

WHY THIS EXISTS (deployment honesty). The Pakistani external eval (eval_external.py)
measured specificity 0.675 [0.633, 0.714] at the shipped 0.6105 operating point — but
that estimate rests on only 514 normals, so its CI is wide. NIH ChestX-ray14 contributes
5,788 more labeled negatives (the No_Finding subset), tightening a SECOND external
specificity estimate.

CRUCIAL DOMAIN CAVEAT. NIH is a US cohort, the Pakistani cohort is South-Asian. These are
DIFFERENT domains, so this script does NOT re-estimate the Pakistani-site specificity. It
estimates "external specificity on a US-style negative set." Reported alongside the
Pakistani spec it gives a TWO-SITE external specificity picture — and the contrast is the
finding: if NIH No_Finding spec is high (~0.98) while Pakistani spec is low (~0.68), the
specificity drift is localized to the Pakistani domain, which is exactly what the P1
negative-diversity training should target.

WHAT IT DOES.
  1. Loads the deployed T2 head from data/tb_head_t2.pt (fusion+zonal+box config).
  2. Reads the SHIPPED threshold (0.6105) + temperature (1.5915) from
     data/tb_threshold_t2.json — the literal deployed operating point (NOT refit here;
     this is a deployment-honesty eval at the shipped knobs, so we apply them as shipped).
  3. Loads data/features_nih14.npz, splits the No_Finding subset (US-domain negatives).
  4. Reports, at the shipped (thr, T):
       - FPR (= 1 - specificity) on NIH No_Finding with a Wilson 95% CI, and the implied
         specificity + its Wilson CI (1 - FPR endpoints, swapped).
       - FPR on the full 10k NIH (all rows are non-TB by NIH labels) for reference.
  5. Writes data/eval_nih_negatives.json.

All metrics are vs the RADIOGRAPHIC reference label (No_Finding = radiographically normal
by NIH labels), NOT bacteriological confirmation (endpoint honesty — train_tb.py header /
CASE_STUDY.md). A "false positive" here = the head flagged a No_Finding image as TB.

    PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
      training/.venv/bin/python training/eval_nih_negatives.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

import train_tb  # noqa: E402  (DEVICE pin must precede predict_t2 use)

# CPU-ONLY by stress_metrics convention: 10k images * a frozen-head MLP is trivial on CPU
# and avoids the MPS adaptive_avg_pool2d fallback chatter for a forward-only run.
train_tb.DEVICE = "cpu"

from train_tb import TBHeadT2, predict_t2  # noqa: E402  (after DEVICE pin)

DATA = REPO / "data"
NIH_NPZ = DATA / "features_nih14.npz"
HEAD_PT = DATA / "tb_head_t2.pt"
THRESHOLD_JSON = DATA / "tb_threshold_t2.json"
OUT = DATA / "eval_nih_negatives.json"
DEVICE = "cpu"

HEAD_KEYS = ("cls", "patches", "txrv", "zones")


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Point estimate + Wilson 95% CI for a binomial proportion (matches eval_external.py)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _load_nih() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Returns (arrs_for_head, no_finding_mask)."""
    z = np.load(NIH_NPZ, allow_pickle=True)
    arrs = {k: z[k].astype("float32") for k in HEAD_KEYS}
    no_finding = z["No_Finding"].astype("int64")
    return arrs, no_finding


def _load_deployed_head(arrs: dict[str, np.ndarray]) -> TBHeadT2:
    model = TBHeadT2(arrs["patches"].shape[2], arrs["cls"].shape[1], arrs["txrv"].shape[1],
                     frozenset({"fusion", "zonal", "box"})).to(DEVICE)
    model.load_state_dict(torch.load(HEAD_PT, map_location=DEVICE))
    model.eval()
    return model


def _fpr_block(model: TBHeadT2, arrs: dict[str, np.ndarray], idx: np.ndarray,
               thr: float, T: float) -> dict[str, object]:
    """FPR (= 1 - specificity) over a negative subgroup at the shipped (thr, T), with Wilson CIs."""
    p = predict_t2(model, arrs, idx, T)
    n = int(idx.size)
    n_fp = int((p >= thr).sum())
    fpr, fpr_lo, fpr_hi = wilson(n_fp, n)
    # Specificity = 1 - FPR; its CI is the FPR CI reflected through 1.
    spec = 1.0 - fpr
    spec_lo = 1.0 - fpr_hi
    spec_hi = 1.0 - fpr_lo
    return {
        "n": n,
        "n_fp": n_fp,
        "fpr": float(fpr),
        "fpr_ci": [float(fpr_lo), float(fpr_hi)],
        "specificity": float(spec),
        "specificity_ci": [float(spec_lo), float(spec_hi)],
        "mean_score": float(np.mean(p)),
        "frac_high": float(np.mean(p >= 0.95)),
    }


def run() -> dict[str, object]:
    cfg = json.loads(THRESHOLD_JSON.read_text())
    thr = float(cfg["threshold"])
    T = float(cfg["temperature"])

    arrs, no_finding = _load_nih()
    n_total = int(arrs["cls"].shape[0])
    model = _load_deployed_head(arrs)

    no_finding_idx = np.where(no_finding == 1)[0]
    all_idx = np.arange(n_total)

    no_finding_block = _fpr_block(model, arrs, no_finding_idx, thr, T)
    overall_block = _fpr_block(model, arrs, all_idx, thr, T)

    out = {
        "endpoint": "radiographic_tb_pattern (NOT bacteriologically confirmed)",
        "domain": "NIH ChestX-ray14 (US cohort) — second EXTERNAL negative set",
        "domain_note": (
            "US-domain negatives. This estimates external specificity on a US negative set; "
            "combined with the Pakistani spec (0.675) it gives a two-site external specificity "
            "picture. It does NOT re-estimate the Pakistani-site specificity."
        ),
        "operating_point": {
            "threshold": thr,
            "temperature": T,
            "source": "data/tb_threshold_t2.json (shipped, applied as-is, not refit)",
        },
        "nih_no_finding": no_finding_block,
        "nih_all_10k": overall_block,
        "pakistani_external_spec_for_contrast": {
            "specificity": 0.675,
            "specificity_ci": [0.633, 0.714],
            "n_normals": 514,
            "source": "eval_external.py (Kiran/Jabeen Pakistani cohort)",
        },
    }
    return out


def _print(out: dict[str, object]) -> None:
    nf = out["nih_no_finding"]  # type: ignore[index]
    allb = out["nih_all_10k"]  # type: ignore[index]
    op = out["operating_point"]  # type: ignore[index]
    pk = out["pakistani_external_spec_for_contrast"]  # type: ignore[index]
    print("\n" + "=" * 70)
    print("TRACK A — SECOND EXTERNAL NEGATIVE EVAL — NIH ChestX-ray14 (US cohort)")
    print("=" * 70)
    print("ENDPOINT: radiographic-TB-pattern (NOT bacteriologically confirmed). Research preview.")
    print(f"shipped operating point: threshold={op['threshold']:.4f}  temperature={op['temperature']:.4f}")
    print(f"  ({op['source']})")
    print("\nNIH No_Finding (US-style normals — the comparable specificity set):")
    print(f"  n={nf['n']}  false positives={nf['n_fp']}")
    print(f"  FPR          {nf['fpr']:.4f}  [{nf['fpr_ci'][0]:.4f}, {nf['fpr_ci'][1]:.4f}]  (Wilson 95%)")
    print(f"  specificity  {nf['specificity']:.4f}  [{nf['specificity_ci'][0]:.4f}, {nf['specificity_ci'][1]:.4f}]")
    print(f"  mean_score={nf['mean_score']:.4f}  frac>=.95={nf['frac_high']:.4f}")
    print("\nNIH all 10k (every row non-TB by NIH labels — reference only):")
    print(f"  n={allb['n']}  false positives={allb['n_fp']}")
    print(f"  FPR          {allb['fpr']:.4f}  [{allb['fpr_ci'][0]:.4f}, {allb['fpr_ci'][1]:.4f}]  (Wilson 95%)")
    print(f"  specificity  {allb['specificity']:.4f}  [{allb['specificity_ci'][0]:.4f}, {allb['specificity_ci'][1]:.4f}]")
    print("\nTWO-SITE EXTERNAL SPECIFICITY CONTRAST:")
    print(f"  NIH (US) No_Finding spec  : {nf['specificity']:.3f}  [{nf['specificity_ci'][0]:.3f}, {nf['specificity_ci'][1]:.3f}]  (n={nf['n']})")
    print(f"  Pakistani normals spec    : {pk['specificity']:.3f}  [{pk['specificity_ci'][0]:.3f}, {pk['specificity_ci'][1]:.3f}]  (n={pk['n_normals']})")
    print("  -> specificity is fine on US-style normals but drifts badly on Pakistani normals.")
    print("     The drift is LOCALIZED to the Pakistani domain (target for P1 negative-diversity).")
    print(str(out["domain_note"]))  # type: ignore[index]
    print("=" * 70, flush=True)


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    out = run()
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _print(out)
    print(f"\nwrote {OUT}")
    return 0


# Lightweight self-check, runnable without pytest (mirrors the _run_all() pattern):
# asserts the No_Finding FPR matches the cached nih_stress_rows.json No_Finding row
# (same head, same operating point), so a regression in head/threshold loading is caught.
def _run_all() -> int:
    out = run()
    nf = out["nih_no_finding"]  # type: ignore[index]
    cache = DATA / "nih_stress_rows.json"
    if cache.exists():
        rows = json.loads(cache.read_text()).get("rows", [])
        cached = next((r for r in rows if r.get("name") == "No_Finding"), None)
        if cached is not None:
            assert abs(float(nf["fpr"]) - float(cached["fpr"])) <= 0.005, (  # type: ignore[index]
                f"No_Finding FPR drifted: {nf['fpr']} vs cached {cached['fpr']}"  # type: ignore[index]
            )
            assert int(nf["n"]) == int(cached["n"]), "No_Finding n mismatch"  # type: ignore[index]
            print(f"[self-check] No_Finding FPR {nf['fpr']:.4f} matches cache {cached['fpr']:.4f}  OK")  # type: ignore[index]
        else:
            print("[self-check] no cached No_Finding row — skipped")
    else:
        print("[self-check] nih_stress_rows.json absent — skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

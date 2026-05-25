"""How many of 10 truly-unseen labeled images does the validated model classify correctly?

Sample: 5 No_Finding + 2 Fibrosis + 2 Nodule + 1 Consolidation from NIH ChestX-ray14,
which the model has never seen (NIH Clinical Center patients, fully disjoint from our
4 training sources). All true label = NO_TB (NIH has no TB-labeled images).

Expected result, given the M18 stress test numbers:
  - No_Finding: FPR 0.016 (per-image), so ~5/5 correct expected
  - Fibrosis:    FPR 0.102 -> ~1.8 / 2 correct expected
  - Nodule:      FPR 0.063 -> ~1.9 / 2 correct expected
  - Consolidation: FPR 0.049 -> ~0.95 / 1 correct expected
  Total expected ballpark: ~9-10 / 10

Run: training/.venv/bin/python training/test_10_unseen_nih.py
"""
from __future__ import annotations
import csv
import random
from pathlib import Path

from triage_core import TriageEngine  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent.parent
NIH = ROOT / "data" / "raw" / "nih14"
SEED = 42  # distinct from training SEED=0 — these picks are independent
N_NO_FINDING = 5
N_FIBROSIS = 2
N_NODULE = 2
N_CONSOLIDATION = 1


def pick_one_finding(rows: list[dict[str, str]], col: str, k: int, seed_offset: int) -> list[dict[str, str]]:
    rng = random.Random(SEED + seed_offset)
    matches = [r for r in rows if r.get(col) == "1"]
    return rng.sample(matches, k)


def main() -> None:
    csv_path = NIH / "nih14_findings.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows):,} NIH labels.")

    picks = []
    picks += pick_one_finding(rows, "No_Finding", N_NO_FINDING, 0)
    picks += pick_one_finding(rows, "Fibrosis", N_FIBROSIS, 1)
    picks += pick_one_finding(rows, "Nodule", N_NODULE, 2)
    picks += pick_one_finding(rows, "Consolidation", N_CONSOLIDATION, 3)
    assert len(picks) == 10, len(picks)

    print("Loading validated model (Rad-DINO + TXRV + TBHeadT2 + sequelae)…")
    engine = TriageEngine()

    print()
    print(f"{'#':<3}{'filename':<26}{'findings':<34}{'pred':<10}{'tb_prob':>10}{'s_inactive':>12}  correct?")
    print("-" * 110)

    correct = 0
    fp_easy = 0
    fp_mimic = 0
    for i, r in enumerate(picks, 1):
        img_path = NIH / r["filename"]
        assert img_path.exists(), img_path
        findings = (r.get("findings") or "No Finding").strip() or "No Finding"
        with img_path.open("rb") as f:
            result = engine.run(f.read())
        pred = result.verdict.upper()
        ok = pred == "NO_TB"
        if ok:
            correct += 1
        elif r["No_Finding"] == "1":
            fp_easy += 1
        else:
            fp_mimic += 1
        marker = "✓" if ok else "✗"
        print(f"{i:<3}{r['filename']:<26}{findings[:32]:<34}{pred:<10}{result.tb_prob:>10.4f}{result.s_inactive:>12.4f}  {marker}")

    print("-" * 110)
    print()
    print(f"RESULT: {correct}/10 correct (all true label = NO_TB)")
    print(f"  FPs on No_Finding (5 easy negatives):     {fp_easy}/{N_NO_FINDING}")
    print(f"  FPs on TB mimics (Fibrosis/Nodule/Consolidation, 5 hard cases): {fp_mimic}/{N_FIBROSIS + N_NODULE + N_CONSOLIDATION}")
    print()
    print("Endpoint: radiographic TB pattern. NIH provenance: NIH Clinical Center,")
    print("disjoint from Montgomery/Shenzhen/Qatar/TBX11K training sources.")
    print("This measures specificity on negatives — not sensitivity (no TB-positives in NIH).")


if __name__ == "__main__":
    main()

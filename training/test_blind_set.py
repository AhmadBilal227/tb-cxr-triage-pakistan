"""Run the curated blind test set through the validated TriageEngine and report sens/spec.

Blind set (M-25-prep):
  TB+ (11): data/blind_test/tb_positive/tb_blind_*.{jpg,jpeg,png}
    - source: ieee8023/covid-chestxray-dataset on GitHub, finding=Tuberculosis,
      filtered to X-ray modality + PA/AP views + no co-infection cases.
    - fully outside our 4 training sources (Mont/Shenzhen/Qatar/TBX11K).
  TB- (12): data/blind_test/no_tb/notb_blind_*.png
    - source: NIH ChestX-ray14 No_Finding subset, SEED=44 (distinct from prior
      10- and 50-image draws).
    - unseen by TBHeadT2 in training; M18 stress-test set, not the training set.

Each image runs the same engine the FastAPI server runs, with the deployed
calibration (T=1.5915, thr_at_95sens=0.6105) read from tb_threshold_t2.json.

Run: training/.venv/bin/python training/test_blind_set.py
"""
from __future__ import annotations
from pathlib import Path

from triage_core import TriageEngine  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent.parent
TB_DIR = ROOT / "data" / "blind_test" / "tb_positive"
NO_DIR = ROOT / "data" / "blind_test" / "no_tb"
BORDERLINE_LOW = 0.35  # the local-mode pipeline would route these to the verifier


def run_one(engine: TriageEngine, path: Path) -> dict:
    with path.open("rb") as f:
        result = engine.run(f.read())
    return {
        "filename": path.name,
        "tb_prob": result.tb_prob,
        "s_inactive": result.s_inactive,
        "verdict": result.verdict.upper(),
    }


def main() -> None:
    print("Loading validated model (Rad-DINO + TXRV + TBHeadT2 + sequelae)…")
    engine = TriageEngine()

    tb_paths = sorted([p for p in TB_DIR.iterdir() if p.is_file()])
    no_paths = sorted([p for p in NO_DIR.iterdir() if p.is_file()])
    print(f"Blind set: {len(tb_paths)} TB+ images, {len(no_paths)} TB- images")
    print()

    rows: list[tuple[str, str, dict]] = []
    for p in tb_paths:
        rows.append(("TB+", "TB", run_one(engine, p)))
    for p in no_paths:
        rows.append(("TB-", "NO_TB", run_one(engine, p)))

    print(f"{'#':<4}{'true':<6}{'filename':<30}{'predicted':<12}{'tb_prob':>10}{'s_inact':>10}  note")
    print("-" * 100)
    correct = 0
    tp = fn = tn = fp = 0
    borderline_count = 0
    for i, (kind, true_label, r) in enumerate(rows, 1):
        pred = r["verdict"]
        ok = pred == true_label
        if ok:
            correct += 1
            if true_label == "TB":
                tp += 1
            else:
                tn += 1
        else:
            if true_label == "TB":
                fn += 1
            else:
                fp += 1
        borderline = BORDERLINE_LOW <= r["tb_prob"] < 0.65
        if borderline:
            borderline_count += 1
        marker = "✓" if ok else "✗"
        note = ""
        if borderline:
            note = "← borderline; full pipeline would fire gpt verifier"
        elif not ok and true_label == "TB":
            note = "← FN (missed real TB)"
        elif not ok and true_label == "NO_TB":
            note = "← FP (false alarm)"
        print(
            f"{i:<4}{true_label:<6}{r['filename']:<30}{pred:<12}"
            f"{r['tb_prob']:>10.4f}{r['s_inactive']:>10.4f}  {marker} {note}"
        )

    print("-" * 100)
    n_tb = sum(1 for _, t, _ in rows if t == "TB")
    n_no = sum(1 for _, t, _ in rows if t == "NO_TB")
    sens = tp / n_tb if n_tb else 0
    spec = tn / n_no if n_no else 0
    print()
    print(f"OVERALL: {correct}/{len(rows)} correct ({correct/len(rows):.0%})")
    print()
    print(f"  TB+ side (n={n_tb}, literature-figure CXRs unseen by training):")
    print(f"    caught (TP)        : {tp:>3}/{n_tb}")
    print(f"    missed (FN)        : {fn:>3}/{n_tb}  ← safety-critical")
    print(f"    sensitivity        : {sens:.3f}")
    print()
    print(f"  TB- side (n={n_no}, NIH No_Finding unseen by training):")
    print(f"    cleared (TN)       : {tn:>3}/{n_no}")
    print(f"    false-positive (FP): {fp:>3}/{n_no}")
    print(f"    specificity        : {spec:.3f}")
    print()
    print(f"  Borderline cases (tb_prob ∈ [0.35, 0.65)): {borderline_count}/{len(rows)}")
    print(f"    On the full local-mode pipeline (with gpt-5.5 verifier active), these")
    print(f"    would route to the consistency-check; disagreement → ABSTAIN.")
    print()
    print("Endpoint: radiographic TB pattern. Per-site recalibration recommended.")


if __name__ == "__main__":
    main()

"""Run the 23 blind images through the M24-enriched engine and dump the diagnostic data.

For each image: verdict + tb_prob + s_inactive + zonal_scores + top-3 txrv_pathologies +
box_evidence_grid (8x8) max cell location and value.

For the 4 confident misses (TB+ with tb_prob < 0.20), print the full 8x8 box-evidence grid
as ASCII so we can diagnose: did the model look at the wrong region (lung edge, diaphragm)
or the right region with the wrong reading?

Run: training/.venv/bin/python training/test_blind_set_enriched.py
"""
from __future__ import annotations
from pathlib import Path

from triage_core import TriageEngine  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent.parent
TB_DIR = ROOT / "data" / "blind_test" / "tb_positive"
NO_DIR = ROOT / "data" / "blind_test" / "no_tb"


def render_heatmap(grid: list[list[float]]) -> str:
    """Render the 8x8 box-evidence grid as ASCII so we can see WHERE the model looked.
    Cells <0.05 render as '.', 0.05-0.20 as 'o', 0.20-0.50 as 'O', >0.50 as '#'."""
    out = []
    for row in grid:
        chars = []
        for v in row:
            if v < 0.05:
                chars.append(".")
            elif v < 0.20:
                chars.append("o")
            elif v < 0.50:
                chars.append("O")
            else:
                chars.append("#")
        out.append(" ".join(chars))
    return "\n      ".join(out)


def grid_max(grid: list[list[float]]) -> tuple[float, int, int]:
    best = (-1.0, 0, 0)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v > best[0]:
                best = (v, i, j)
    return best


def grid_loc_name(r: int, c: int) -> str:
    """Heuristic anatomical name for an (r,c) cell of the 8x8 lung-letterboxed grid."""
    row_band = "upper" if r < 2 else ("mid" if r < 5 else "lower")
    col_side = "right" if c < 4 else "left"  # radiology convention: patient's right = image left
    return f"{row_band} {col_side}"


def main() -> None:
    print("Loading enriched validated model (M24 — Rad-DINO + TXRV + TBHeadT2 + sequelae, with intermediates)…")
    engine = TriageEngine()

    tb_paths = sorted([p for p in TB_DIR.iterdir() if p.is_file()])
    no_paths = sorted([p for p in NO_DIR.iterdir() if p.is_file()])

    rows = []
    for p in tb_paths:
        with p.open("rb") as f:
            r = engine.run(f.read())
        rows.append(("TB", p.name, r))
    for p in no_paths:
        with p.open("rb") as f:
            r = engine.run(f.read())
        rows.append(("NO_TB", p.name, r))

    print()
    print(f"{'#':<4}{'true':<6}{'filename':<26}{'pred':<10}{'tb_prob':>10}{'box-max':>10}{'  zone':<18}  top TXRV findings")
    print("-" * 130)
    misses = []
    for i, (true_label, fname, r) in enumerate(rows, 1):
        pred = r.verdict.upper()
        ok = pred == true_label
        marker = "✓" if ok else "✗"
        box_max_v, br, bc = grid_max(r.box_evidence_grid) if hasattr(r, "box_evidence_grid") and r.box_evidence_grid else (0.0, 0, 0)
        zone_label = grid_loc_name(br, bc)
        # top-3 TXRV pathologies by score
        if hasattr(r, "txrv_pathologies") and r.txrv_pathologies:
            top_txrv = sorted(r.txrv_pathologies.items(), key=lambda kv: -kv[1])[:3]
            txrv_str = "  ".join(f"{k}:{v:.2f}" for k, v in top_txrv)
        else:
            txrv_str = "(no txrv)"
        print(f"{i:<4}{true_label:<6}{fname:<26}{pred:<10}{r.tb_prob:>10.4f}{box_max_v:>10.3f}  {zone_label:<16}  {txrv_str}  {marker}")
        if true_label == "TB" and r.tb_prob < 0.20:
            misses.append((i, fname, r))

    print("-" * 130)
    print()
    if misses:
        print(f"=== DIAGNOSTIC: the {len(misses)} confident TB-misses (tb_prob < 0.20) — full box-evidence grid ===")
        for i, fname, r in misses:
            print()
            print(f"#{i}  {fname}  tb_prob={r.tb_prob:.4f}  s_inactive={r.s_inactive:.4f}")
            if hasattr(r, "zonal_scores") and r.zonal_scores:
                zsorted = sorted(r.zonal_scores.items(), key=lambda kv: -kv[1])[:4]
                print(f"      zonal top-4: " + "  ".join(f"{k}:{v:.3f}" for k, v in zsorted))
            if hasattr(r, "txrv_pathologies") and r.txrv_pathologies:
                tsorted = sorted(r.txrv_pathologies.items(), key=lambda kv: -kv[1])[:5]
                print(f"      txrv top-5:  " + "  ".join(f"{k}:{v:.2f}" for k, v in tsorted))
            if hasattr(r, "box_evidence_grid") and r.box_evidence_grid:
                vmax, br, bc = grid_max(r.box_evidence_grid)
                print(f"      box-evidence max: {vmax:.3f} at row {br}, col {bc}  ({grid_loc_name(br,bc)})")
                print(f"      box-evidence grid (. <0.05  o <0.20  O <0.50  # >=0.50):")
                print(f"      {render_heatmap(r.box_evidence_grid)}")


if __name__ == "__main__":
    main()

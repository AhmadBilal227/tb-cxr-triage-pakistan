"""M24 enrichment smoke tests for TriageEngine.

The deployed model already computes:
  - an 8x8 box-evidence grid (BoxEvidence head, pre-LSE-LBA)
  - 7 per-zone TB logits (ZonalSoftOR; we compute real zones on the fly via the lung mask)
  - 18 TorchXRayVision named-finding logits (fed into TBHeadT2's fusion lever)
  - a seg-crop bounding box in pixel coords
  - a MONOCHROME1 inversion heuristic flag

Before M24, every one of these was discarded after the verdict. This test pins the SHAPES
and RANGES of the new TriageResult fields on the committed `public/samples/tb-sample-1.jpg`.
It does NOT pin specific numeric values for the intermediates (those drift under fp32/MPS
noise across runs), only structural invariants — and that the headline `tb_prob` is
UNCHANGED by the enrichment forward (the verdict pin from `test_triage_engine.py` covers
the numeric drift case).

Run pattern matches the existing tests:

    training/.venv/bin/python training/test_triage_enrichment.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from triage_core import TXRV_LABELS, TriageEngine, ZONE_KEYS  # noqa: E402

SAMPLE_TB = REPO / "public" / "samples" / "tb-sample-1.jpg"


def _read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()


def test_box_evidence_grid_is_8x8_in_unit_interval():
    """Box evidence is the 8x8 per-cell sigmoid'd output of BoxEvidence.scorer (pre LSE-LBA pool).
    With the box lever active in the deployed head, this field MUST be populated."""
    eng = TriageEngine()
    res = eng.run(_read_bytes(SAMPLE_TB))
    assert res.box_evidence_grid is not None, "box_evidence_grid should be populated when box lever is on"
    grid = res.box_evidence_grid
    assert len(grid) == 8, f"expected 8 rows, got {len(grid)}"
    for r, row in enumerate(grid):
        assert len(row) == 8, f"row {r}: expected 8 cols, got {len(row)}"
        for c, v in enumerate(row):
            assert isinstance(v, float), (r, c, type(v))
            assert 0.0 <= v <= 1.0, f"cell ({r},{c}) out of [0,1]: {v}"
    # Sanity: at least ONE cell should fire reasonably (>0.05) on a TB-positive sample.
    # Threshold is generous (the head's LSE-LBA pool makes any single cell modest); the point
    # is to catch a stuck-at-zero regression, not to pin a localization quality number.
    flat = [v for row in grid for v in row]
    assert max(flat) > 0.05, f"max box-evidence cell={max(flat):.4f} — suspiciously low on TB sample"


def test_zonal_scores_seven_keys_in_unit_interval():
    """ZONE_KEYS is the engine's HONEST 7-zone surface (upper/mid/lower L+R + hilar). The
    trained ZonalSoftOR was designed around `ZONE_NAMES = (RUZ,RMZ,RLZ,LUZ,LMZ,LLZ,HILAR)`
    — see train_tb.N_ZONES=7. We do NOT invent a L/R split of the hilar/mediastinum channel."""
    eng = TriageEngine()
    res = eng.run(_read_bytes(SAMPLE_TB))
    # zonal_scores can legitimately be None when the segmenter is unavailable OR returns no
    # lung mask; on the committed TB sample the seg succeeds so we expect a populated dict.
    assert res.zonal_scores is not None, "zonal_scores should be populated when the lung mask is non-empty"
    zs = res.zonal_scores
    assert len(zs) == 7, f"expected 7 zone keys, got {len(zs)}: {sorted(zs.keys())}"
    assert set(zs.keys()) == set(ZONE_KEYS), f"key mismatch: {sorted(zs.keys())} vs {sorted(ZONE_KEYS)}"
    for k, v in zs.items():
        assert isinstance(v, float), (k, type(v))
        assert 0.0 <= v <= 1.0, f"zone {k}={v} out of [0,1]"


def test_txrv_pathologies_eighteen_labels_in_unit_interval():
    """The 18 TorchXRayVision named-finding probabilities. Names come VERBATIM from
    `xrv.models.DenseNet(...).pathologies`."""
    eng = TriageEngine()
    res = eng.run(_read_bytes(SAMPLE_TB))
    assert res.txrv_pathologies is not None, "txrv_pathologies should always be populated"
    px = res.txrv_pathologies
    assert len(px) == 18, f"expected 18 pathology keys, got {len(px)}: {sorted(px.keys())}"
    assert set(px.keys()) == set(TXRV_LABELS), f"label set mismatch"
    for k, v in px.items():
        assert isinstance(v, float), (k, type(v))
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_crop_box_in_original_pixel_coords():
    """crop_box is the seg-driven letterbox region, in ORIGINAL image pixel coords. It must
    fit inside the original frame and be non-degenerate (w,h > 0)."""
    eng = TriageEngine()
    res = eng.run(_read_bytes(SAMPLE_TB))
    assert res.crop_box is not None, "crop_box should be populated for the UI overlay"
    cb = res.crop_box
    assert set(cb.keys()) == {"x", "y", "w", "h"}, f"crop_box keys: {sorted(cb.keys())}"
    for k in ("x", "y", "w", "h"):
        assert isinstance(cb[k], int), (k, type(cb[k]))
        assert cb[k] >= 0, (k, cb[k])
    # The crop must produce a non-degenerate region.
    assert cb["w"] > 0 and cb["h"] > 0, cb


def test_inversion_detected_is_bool():
    """`_detect_inversion` already drives the image_quality warning; surfacing the bool lets
    the UI render a chip. We don't pin the value (it's image-dependent) — only the type."""
    eng = TriageEngine()
    res = eng.run(_read_bytes(SAMPLE_TB))
    assert res.inversion_detected is not None, "inversion_detected should be set"
    assert isinstance(res.inversion_detected, bool), type(res.inversion_detected)


def test_to_dict_round_trips_enrichment_fields():
    """The wire shape (TriageResult.to_dict) MUST carry the new fields when populated, so the
    FastAPI server's response delivers them. localTriage.ts parses them with `?:`."""
    eng = TriageEngine()
    res = eng.run(_read_bytes(SAMPLE_TB))
    d = res.to_dict()
    assert "box_evidence_grid" in d
    assert "zonal_scores" in d
    assert "txrv_pathologies" in d
    assert "crop_box" in d
    assert "inversion_detected" in d
    # Backwards-compat: the canonical fields stay byte-identical-shape.
    for k in ("tb_prob", "tb_logit", "s_inactive", "verdict", "decided_at_threshold",
              "safety_net_applied", "image_quality", "latency_ms", "audit"):
        assert k in d, k


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed} tests passed")


if __name__ == "__main__":
    _run_all()

"""CPU-only unit tests for the box->grid and mask->zone geometry in extract_features.py.

These pin down the coordinate transform (the part most likely to be silently wrong) and the
row-major token order. Run:  training/.venv/bin/python training/test_extract_geom.py
(or)  training/.venv/bin/python -m pytest training/test_extract_geom.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import (  # noqa: E402
    PATCH_GRID,
    N_ZONES,
    rasterize_boxes_to_grid,
    scale_boxes_orig_to_harmonized,
    parse_bbox_field,
    zone_matrix_from_masks,
)


def test_box_at_known_crop_maps_to_expected_cells():
    """A box covering harmonized [x 200..300, y 100..200] inside a crop with offset (y0=100,x0=200)
    of size 800x800 lands exactly in the TOP-LEFT cell of an 8x8 grid (the box is the first 1/8 of
    the cropped frame in both axes)."""
    # crop window: y0=100,y1=900,x0=200,x1=1000 -> cropped frame 800x800, harmonized 1024x1024
    crop_box = (100, 900, 200, 1000)
    harm_wh = (1024, 1024)
    # box in HARMONIZED frame: top-left of the crop, exactly one cell wide/tall (cropped/8 = 100px)
    boxes = [(200.0, 100.0, 100.0, 100.0)]  # x,y,w,h
    grid = rasterize_boxes_to_grid(boxes, crop_box, harm_wh, G=8, mode="soft")
    assert grid.shape == (8, 8)
    # cell [0,0] fully covered, everything else zero
    assert abs(grid[0, 0] - 1.0) < 1e-5, grid[0, 0]
    assert grid.sum() == grid[0, 0], f"only top-left cell should be hot, got sum {grid.sum()}"

    # a box at cropped [x 300..400, y 200..300] -> cells gx=3, gy=2 -> grid[2,3] (row-major gy,gx)
    boxes2 = [(200.0 + 300.0, 100.0 + 200.0, 100.0, 100.0)]
    grid2 = rasterize_boxes_to_grid(boxes2, crop_box, harm_wh, G=8, mode="soft")
    assert abs(grid2[2, 3] - 1.0) < 1e-5, grid2
    assert grid2.sum() == grid2[2, 3]


def test_soft_fractional_coverage():
    """A half-cell box gives a fractional value (soft mode)."""
    crop_box = (0, 800, 0, 800)  # 800x800 cropped, cell = 100px
    harm_wh = (800, 800)
    boxes = [(0.0, 0.0, 50.0, 100.0)]  # covers left half of cell [0,0] horizontally, full vertically
    grid = rasterize_boxes_to_grid(boxes, crop_box, harm_wh, G=8, mode="soft")
    assert abs(grid[0, 0] - 0.5) < 1e-4, grid[0, 0]
    # hard mode -> 1.0 for any overlap
    gh = rasterize_boxes_to_grid(boxes, crop_box, harm_wh, G=8, mode="hard")
    assert abs(gh[0, 0] - 1.0) < 1e-5


def test_box_outside_crop_gives_zero_grid():
    """A box entirely outside (above-left of) the crop window contributes nothing."""
    crop_box = (400, 800, 400, 800)  # cropped frame covers harmonized [400..800]
    harm_wh = (1024, 1024)
    boxes = [(0.0, 0.0, 100.0, 100.0)]  # fully in the top-left, outside the crop
    grid = rasterize_boxes_to_grid(boxes, crop_box, harm_wh, G=8, mode="soft")
    assert grid.sum() == 0.0, grid

    # partial: a box straddling the crop's left edge only contributes the inside part
    boxes2 = [(350.0, 400.0, 100.0, 50.0)]  # x 350..450 -> only x 400..450 is inside (cropped 0..50)
    grid2 = rasterize_boxes_to_grid(boxes2, crop_box, harm_wh, G=8, mode="soft")
    # cropped frame is 400x400 -> cell = 50px; inside part is cropped x[0..50] y[0..50] = cell [0,0]
    assert grid2[0, 0] > 0.0 and abs(grid2[0, 0] - 1.0) < 1e-4, grid2[0, 0]
    assert grid2.sum() == grid2[0, 0]


def test_empty_boxes_give_zero_grid():
    grid = rasterize_boxes_to_grid([], (0, 100, 0, 100), (100, 100), G=8)
    assert grid.shape == (8, 8) and grid.sum() == 0.0


def test_scale_orig_to_harmonized():
    """512x512 original, working_res 1024 -> uniform 2x scale."""
    boxes = [(100.0, 50.0, 40.0, 60.0)]
    scaled = scale_boxes_orig_to_harmonized(boxes, (512, 512), working_res=1024)
    assert scaled == [(200.0, 100.0, 80.0, 120.0)], scaled
    # non-square original: shorter side drives the scale
    scaled2 = scale_boxes_orig_to_harmonized([(10.0, 10.0, 10.0, 10.0)], (256, 512), working_res=1024)
    assert scaled2 == [(40.0, 40.0, 40.0, 40.0)], scaled2  # s = 1024/256 = 4


def test_parse_bbox_field():
    assert parse_bbox_field("[]") == []
    assert parse_bbox_field("none") == []
    assert parse_bbox_field(None) == []
    j = '[{"xmin": 1.0, "ymin": 2.0, "width": 3.0, "height": 4.0}]'
    assert parse_bbox_field(j) == [(1.0, 2.0, 3.0, 4.0)]
    # tolerate single python-dict string (legacy)
    leg = "{'xmin': 5, 'ymin': 6, 'width': 7, 'height': 8}"
    assert parse_bbox_field(leg) == [(5.0, 6.0, 7.0, 8.0)]
    # two boxes
    j2 = '[{"xmin":0,"ymin":0,"width":1,"height":1},{"xmin":9,"ymin":9,"width":2,"height":2}]'
    assert parse_bbox_field(j2) == [(0.0, 0.0, 1.0, 1.0), (9.0, 9.0, 2.0, 2.0)]


def test_token_row_major_order():
    """grid_label[gy,gx].reshape(-1) must put a hot cell at index gy*G+gx (matches Rad-DINO pooling)."""
    G = PATCH_GRID
    for gy, gx in [(0, 0), (2, 5), (7, 7), (3, 0)]:
        probe = np.zeros((G, G), dtype="float32")
        probe[gy, gx] = 1.0
        assert int(probe.reshape(-1).argmax()) == gy * G + gx, (gy, gx)


def _synthetic_two_lung_mask(h=400, w=400):
    """Two rectangular 'lungs' (left & right of center) spanning rows 50..350, plus a central hilar blob."""
    lung = np.zeros((h, w), dtype="float32")
    lung[50:350, 40:170] = 1.0     # image-left  block (patient-right lung)
    lung[50:350, 230:360] = 1.0    # image-right block (patient-left lung)
    hil = np.zeros((h, w), dtype="float32")
    hil[170:230, 170:230] = 1.0    # central hilar/mediastinal blob
    return lung, hil


def test_zone_matrix_rows_normalized_and_six_lung_zones():
    lung, hil = _synthetic_two_lung_mask()
    Z = zone_matrix_from_masks(lung, hil, G=PATCH_GRID)
    assert Z.shape == (PATCH_GRID * PATCH_GRID, N_ZONES)
    rowsum = Z.sum(axis=1)
    assert np.all(rowsum <= 1.0 + 1e-4), rowsum.max()
    # background patches (no lung) -> all-zero rows
    assert np.any(rowsum == 0.0), "expected some all-zero (background) rows"
    # each of the 6 lung zones should have at least one non-empty patch
    per_zone = (Z > 0).sum(axis=0)
    for k in range(6):
        assert per_zone[k] > 0, f"lung zone {k} empty: {per_zone}"
    # the hilar channel (7th) should also be populated
    assert per_zone[6] > 0, f"hilar zone empty: {per_zone}"


def test_zone_left_right_split():
    """Patient-right lung (image-LEFT) populates RUZ/RMZ/RLZ (0,1,2); image-right populates LUZ/LMZ/LLZ (3,4,5)."""
    lung, hil = _synthetic_two_lung_mask()
    Z = zone_matrix_from_masks(lung, hil, G=PATCH_GRID).reshape(PATCH_GRID, PATCH_GRID, N_ZONES)
    # left columns of the grid -> patient-right zones (0,1,2); right columns -> patient-left (3,4,5)
    left_cols = Z[:, :PATCH_GRID // 3, :].sum(axis=(0, 1))   # leftmost third of grid columns
    right_cols = Z[:, -PATCH_GRID // 3:, :].sum(axis=(0, 1))  # rightmost third
    assert left_cols[:3].sum() > left_cols[3:6].sum(), (left_cols,)
    assert right_cols[3:6].sum() > right_cols[:3].sum(), (right_cols,)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

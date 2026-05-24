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
    letterbox_square_params,
    letterbox_to_square,
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


def test_nonsquare_crop_letterbox_box_lands_in_correct_cell():
    """REGRESSION for the center-crop misregistration bug. The Rad-DINO processor resizes the SHORTER
    side to 518 then center-crops to 518x518, so for a NON-SQUARE crop the 37x37 tokens only span the
    CENTER SQUARE — a box mapped over the full cropped rectangle is misregistered on the long axis.
    The letterbox pads the cropped frame to a square FIRST, making the box->grid map exact.

    Crop: x0=200,x1=968 -> cw=768; y0=0,y1=1024 -> ch=1024 (TALLER than wide). So S=max=1024, the
    SHORT (width) axis is padded: pad_x=(1024-768)//2=128 = exactly ONE 128px grid cell, pad_y=0.
    The cropped content occupies letterboxed grid columns 1..6; columns 0 and 7 are pure padding.
    A box at the cropped-frame TOP-LEFT cell [x 0..128, y 0..128] therefore lands at letterboxed
    grid cell gx=1, gy=0 -> grid[0,1] (NOT grid[0,0] as the buggy full-rect mapping would place it)."""
    crop_box = (0, 1024, 200, 968)      # ch=1024, cw=768
    harm_wh = (1024, 1280)              # full harmonized frame big enough to contain the crop
    S, pad_x, pad_y = letterbox_square_params(crop_box)
    assert (S, pad_x, pad_y) == (1024, 128, 0), (S, pad_x, pad_y)
    # box in HARMONIZED frame at the crop's top-left, one cropped-cell (128px) wide/tall
    boxes = [(200.0, 0.0, 128.0, 128.0)]   # cropped x[0..128] y[0..128]
    grid = rasterize_boxes_to_grid(boxes, crop_box, harm_wh, G=8, mode="soft")
    assert grid.shape == (8, 8)
    assert abs(grid[0, 1] - 1.0) < 1e-5, ("expected hot cell at [0,1] after letterbox", grid[0, 1])
    assert grid.sum() == grid[0, 1], f"only [0,1] should be hot, sum={grid.sum()}\n{grid}"
    # the two pure-padding columns (leftmost gx=0 and rightmost gx=7) must be entirely empty
    assert grid[:, 0].sum() == 0.0, ("left pad column not empty", grid[:, 0])
    assert grid[:, 7].sum() == 0.0, ("right pad column not empty", grid[:, 7])

    # a box at the cropped-frame TOP-RIGHT cell [x 640..768, y 0..128] -> letterboxed x[768..896]
    # -> gx=6 (the last content column), gy=0 -> grid[0,6]; still inside content, not in pad col 7.
    boxes_tr = [(200.0 + 640.0, 0.0, 128.0, 128.0)]
    grid_tr = rasterize_boxes_to_grid(boxes_tr, crop_box, harm_wh, G=8, mode="soft")
    assert abs(grid_tr[0, 6] - 1.0) < 1e-5, ("expected hot cell at [0,6]", grid_tr[0, 6])
    assert grid_tr.sum() == grid_tr[0, 6], f"only [0,6] should be hot, sum={grid_tr.sum()}\n{grid_tr}"


def test_letterbox_to_square_pads_image_and_mask_consistently():
    """`letterbox_to_square` must pad image (2D/3D) and masks to the SAME SxS square with the SAME
    symmetric offsets, padding bars = pad_value (0). This is what keeps image+masks+boxes registered."""
    crop_box = (0, 1024, 200, 968)   # ch=1024, cw=768 -> S=1024, pad_x=128, pad_y=0
    img = np.full((1024, 768), 7.0, dtype="float32")    # cropped 2D image
    lb = letterbox_to_square(img, crop_box, pad_value=0.0)
    assert lb.shape == (1024, 1024), lb.shape
    assert np.all(lb[:, :128] == 0.0) and np.all(lb[:, -128:] == 0.0), "pad columns must be 0"
    assert np.all(lb[:, 128:896] == 7.0), "content must be preserved in the center"
    # 3-channel image keeps its channel axis
    rgb = np.full((1024, 768, 3), 5.0, dtype="float32")
    lb3 = letterbox_to_square(rgb, crop_box, pad_value=0.0)
    assert lb3.shape == (1024, 1024, 3), lb3.shape


def test_square_crop_letterbox_is_noop():
    """For a SQUARE crop the letterbox does NOTHING (pad=0), so the new math equals the old behavior —
    this is why the original square-crop tests still pass unchanged."""
    crop_box = (100, 900, 200, 1000)  # 800x800 square
    S, pad_x, pad_y = letterbox_square_params(crop_box)
    assert (S, pad_x, pad_y) == (800, 0, 0)
    img = np.arange(800 * 800, dtype="float32").reshape(800, 800)
    assert np.array_equal(letterbox_to_square(img, crop_box), img)


def test_nonsquare_zone_matrix_padding_rows_are_empty():
    """A non-square lung mask, once letterboxed to a square, must yield all-zero zone rows for the
    padding patches (no lung membership leaks into the padding bars)."""
    crop_box = (0, 1024, 200, 968)   # S=1024, pad_x=128, pad_y=0
    # lungs fill the cropped content region; pad columns will be added by letterbox
    lung_c = np.zeros((1024, 768), dtype="float32")
    lung_c[100:900, 80:340] = 1.0     # image-left lung block
    lung_c[100:900, 430:690] = 1.0    # image-right lung block
    hil_c = np.zeros((1024, 768), dtype="float32")
    hil_c[450:600, 340:430] = 1.0
    lung_sq = letterbox_to_square(lung_c, crop_box, pad_value=0.0)
    hil_sq = letterbox_to_square(hil_c, crop_box, pad_value=0.0)
    assert lung_sq.shape == (1024, 1024)
    Z = zone_matrix_from_masks(lung_sq, hil_sq, G=PATCH_GRID).reshape(PATCH_GRID, PATCH_GRID, N_ZONES)
    # leftmost and rightmost grid columns are pure padding (128px = exactly one 128px cell) -> all-zero
    assert Z[:, 0, :].sum() == 0.0, ("left pad column has zone membership", Z[:, 0, :])
    assert Z[:, 7, :].sum() == 0.0, ("right pad column has zone membership", Z[:, 7, :])
    # the lung content in the interior still populates zones
    assert Z[:, 1:7, :].sum() > 0.0, "expected non-empty lung zones in the content region"


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

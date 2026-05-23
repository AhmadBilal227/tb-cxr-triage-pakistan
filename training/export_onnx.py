"""ONNX export for the in-browser perception slot.

DEFERRED / DESIGN NOTE (T2): the trained model is now a DUAL-BACKBONE fused model —
Rad-DINO (CLS + patch tokens) + TorchXRayVision DenseNet (features + 18 logits) → gated-attention
MLP head (see train_tb.py `TBHead`). A single clean ONNX graph is no longer trivial because:
  - Two backbones with DIFFERENT preprocessing (Rad-DINO: CLAHE+soft-mask, 518, mean .5307/std .2583;
    TorchXRayVision: xrv-normalize [-1024,1024], 224) must both run at inference.
  - The browser would need to run both encoders + the attention head, or we distill.

Two export options to choose in T2 (in-browser wiring):
  A. Ship TWO ONNX models (Rad-DINO, TXRV-DenseNet) + replicate the gated-attention MLP head in
     onnxruntime-web/JS, with both preprocessing paths reproduced bit-exactly. Most faithful, heaviest.
  B. DISTILL the fused teacher into a SINGLE-backbone student (Rad-DINO-only head trained to match the
     fused model's probabilities), then export that one clean ONNX. Lighter in-browser, small accuracy cost.

This script intentionally does not emit a mismatched single-backbone ONNX. Pick A or B in T2, then
implement here. The trained weights live at data/tb_head.pt; the head class is train_tb.TBHead.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from train_tb import TBHead  # noqa: E402,F401  (the head to export once the option is chosen)


def main() -> None:
    raise SystemExit(
        "export_onnx is deferred to T2: choose option A (two ONNX + JS attention head) or "
        "B (distill to a single-backbone student), then implement. See module docstring."
    )


if __name__ == "__main__":
    main()

"""Export frozen Rad-DINO backbone + trained head + Sigmoid as a single ONNX graph for the
in-browser perception slot, then int8-quantize for a browser-sized asset.

    python training/export_onnx.py

Output: public/models/tb-cxr/{model.onnx, model_quantized.onnx, config.json}
The browser preprocessing (T2) MUST replicate: CLAHE -> resize 518 -> center-crop 518 ->
/255 -> normalize(mean 0.5307, std 0.2583) per channel, to match training bit-for-bit.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from train_tb import Head  # noqa: E402

DATA = REPO / "data"
OUT = REPO / "public" / "models" / "tb-cxr"
OUT.mkdir(parents=True, exist_ok=True)


class TBModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/rad-dino")
        self.head = Head(768)
        self.head.load_state_dict(torch.load(DATA / "tb_head.pt", map_location="cpu"))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cls = self.backbone(pixel_values=pixel_values).pooler_output
        return torch.sigmoid(self.head(cls))


def main() -> None:
    model = TBModel().eval()
    dummy = torch.randn(1, 3, 518, 518)
    fp32 = OUT / "model.onnx"
    torch.onnx.export(
        model,
        (dummy,),
        str(fp32),
        input_names=["pixel_values"],
        output_names=["tb_prob"],
        dynamic_axes={"pixel_values": {0: "batch"}, "tb_prob": {0: "batch"}},
        opset_version=17,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(fp32))
    out = sess.run(None, {"pixel_values": dummy.numpy()})
    print("fp32 ONNX sanity output:", np.asarray(out[0]).ravel()[:4])

    # int8 dynamic quantization -> browser-sized asset
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(str(fp32), str(OUT / "model_quantized.onnx"), weight_type=QuantType.QInt8)

    thr = json.load(open(DATA / "tb_threshold.json")) if (DATA / "tb_threshold.json").exists() else {"threshold": 0.5}
    json.dump(
        {
            "id2label": {"0": "NORMAL", "1": "TUBERCULOSIS"},
            "preprocess": {
                "clahe": True,
                "resize_shortest_edge": 518,
                "center_crop": 518,
                "mean": [0.5307, 0.5307, 0.5307],
                "std": [0.2583, 0.2583, 0.2583],
            },
            "threshold": thr["threshold"],
        },
        open(OUT / "config.json", "w"),
        indent=2,
    )
    fp32_mb = fp32.stat().st_size / 1e6
    q_mb = (OUT / "model_quantized.onnx").stat().st_size / 1e6
    print(f"exported {fp32} ({fp32_mb:.0f} MB) + model_quantized.onnx ({q_mb:.0f} MB) + config.json")


if __name__ == "__main__":
    main()

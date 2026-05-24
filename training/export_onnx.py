"""ONNX export for the TB-triage perception heads (Milestone 19, scoped to export+parity).

This script exports BOTH deployed heads to onnxruntime-web-compatible ONNX graphs and writes a
calibration sidecar with the deployed temperature/threshold. It does NOT change weights, retrain,
or wire the browser pathway — that remains Phase B (see CASE_STUDY M19 + EXPERIMENT_LOG §C).

WHAT IS EXPORTED (and why this is shippable now):
  1. Main T2 head (`TBHeadT2`, levers = {fusion, zonal, box}) -> public/models/tb_head_t2.onnx
     Inputs:  cls       [B, 768]  Rad-DINO CLS token
              patches   [B, 64, 768]  Rad-DINO patch tokens (8x8 grid)
              txrv      [B, 1042]  TorchXRayVision DenseNet features (1024) + 18 named-finding logits
              zones     [B, 64, 7]   Soft zone membership (RUZ, RMZ, RLZ, LUZ, LMZ, LLZ, HILAR)
     Outputs: tb_logit  [B]       Pre-temperature image logit. Browser applies
                                  prob = sigmoid(tb_logit / T), with T from head_calibration.json.

  2. Sequelae head (`InactiveSequelaeHead`) -> public/models/sequelae_head.onnx
     Inputs:  cls       [B, 768]
              patches   [B, 64, 768]
              txrv      [B, 1042]
     Outputs: s_inactive [B]     Temperature-scaled probability of an inactive/sequelae pattern.
                                 Used as an ESCALATE-NOT-CLEAR feature in the orchestrator (a high
                                 score forces a borderline NO_TB to ABSTAIN; it can NEVER clear a
                                 flagged film).

The export wraps `TBHeadT2.forward` so `zones` is an explicit forward arg (the trained module reads
it from `self._zones`, which ONNX cannot trace). Weights are NOT modified — the wrapper just plumbs
the tensor through. Sequelae head exports as-is (no zones input).

Opset 17, dynamic batch axis, CPU-traced. Parity asserted by training/test_onnx_parity.py.

Usage:
    training/.venv/bin/python training/export_onnx.py
"""
from __future__ import annotations
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from train_tb import (  # noqa: E402
    InactiveSequelaeHead,
    TBHeadT2,
)

DATA = REPO / "data"
OUT = REPO / "public" / "models"
OPSET = 18  # onnxruntime-web supports up to 21+; torch dynamo exporter min-supports 18, and bumping
# avoids a downgrade pass that emits attributes the older op-set's verifier rejects (we saw the
# `noop_with_empty_axes` ReduceLogSumExp attribute at opset 17). 18 is still broadly compatible.
SEED = 0


class TBHeadT2Exportable(nn.Module):
    """Thin wrapper that promotes `zones` from a side-channel attribute to an explicit forward
    arg and returns only the scalar `tb_logit` per item (ONNX prefers tensor outputs).

    Weight-identical to the trained `TBHeadT2`: this module owns the same submodules by reference,
    so loading the cached state dict into the wrapped `TBHeadT2` is sufficient.
    """

    def __init__(self, core: TBHeadT2) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        cls: torch.Tensor,
        patches: torch.Tensor,
        txrv: torch.Tensor,
        zones: torch.Tensor,
    ) -> torch.Tensor:
        self.core._zones = zones
        out = self.core(cls, patches, txrv)
        return out["logit"]


class InactiveSequelaeExportable(nn.Module):
    """Wraps the sequelae head to emit the TEMPERATURE-SCALED probability directly.

    The deployed sequelae head's calibration temperature T_seq lives in data/tb_inactive_meta.json
    (1.1312...). Bake it into the ONNX graph so the browser side gets `s_inactive` as a probability
    in [0,1] without needing to know about T_seq. Note: the MAIN head's temperature is NOT baked in
    (browser controls thresholding via head_calibration.json) — only the sequelae head's, because
    s_inactive is consumed as a probability inside the orchestrator's escalate logic.
    """

    def __init__(self, core: InactiveSequelaeHead, temperature: float) -> None:
        super().__init__()
        self.core = core
        self.temperature = float(temperature)

    def forward(
        self,
        cls: torch.Tensor,
        patches: torch.Tensor,
        txrv: torch.Tensor,
    ) -> torch.Tensor:
        logit = self.core(cls, patches, txrv)
        return torch.sigmoid(logit / self.temperature)


def _load_main_head() -> TBHeadT2:
    """Reload the deployed T2 head with the same configuration used at training/serve time."""
    # Use feature shapes from the cached features file rather than hard-coding.
    z = np.load(DATA / "features.npz", allow_pickle=True)
    d_tok = int(z["patches"].shape[2])
    d_cls = int(z["cls"].shape[1])
    d_txrv = int(z["txrv"].shape[1])
    model = TBHeadT2(d_tok, d_cls, d_txrv, frozenset({"fusion", "zonal", "box"}))
    model.load_state_dict(torch.load(DATA / "tb_head_t2.pt", map_location="cpu"))
    model.eval()
    return model


def _load_seq_head() -> tuple[InactiveSequelaeHead, float]:
    z = np.load(DATA / "features_sequelae.npz", allow_pickle=True)
    d_tok = int(z["patches"].shape[2])
    d_cls = int(z["cls"].shape[1])
    model = InactiveSequelaeHead(d_tok, d_cls)
    model.load_state_dict(torch.load(DATA / "tb_head_inactive.pt", map_location="cpu"))
    model.eval()
    meta = json.loads((DATA / "tb_inactive_meta.json").read_text())
    return model, float(meta["temperature"])


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO))
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _total_size(out_path: Path) -> int:
    """Sum the ONNX graph file + its sibling external-data file (torch dynamo emits both)."""
    total = out_path.stat().st_size
    sidecar = out_path.with_suffix(out_path.suffix + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total


def _export_main(out_path: Path) -> int:
    core = _load_main_head()
    wrapper = TBHeadT2Exportable(core).eval()
    # Dummy inputs (batch=2 to exercise the batch axis during tracing).
    cls = torch.randn(2, 768, dtype=torch.float32)
    patches = torch.randn(2, 64, 768, dtype=torch.float32)
    txrv = torch.randn(2, 1042, dtype=torch.float32)
    zones = torch.softmax(torch.randn(2, 64, 7, dtype=torch.float32), dim=2)
    torch.onnx.export(
        wrapper,
        (cls, patches, txrv, zones),
        str(out_path),
        input_names=["cls", "patches", "txrv", "zones"],
        output_names=["tb_logit"],
        dynamic_axes={
            "cls": {0: "B"},
            "patches": {0: "B"},
            "txrv": {0: "B"},
            "zones": {0: "B"},
            "tb_logit": {0: "B"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    return _total_size(out_path)


def _export_seq(out_path: Path) -> tuple[int, float]:
    core, T_seq = _load_seq_head()
    wrapper = InactiveSequelaeExportable(core, T_seq).eval()
    cls = torch.randn(2, 768, dtype=torch.float32)
    patches = torch.randn(2, 64, 768, dtype=torch.float32)
    txrv = torch.randn(2, 1042, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (cls, patches, txrv),
        str(out_path),
        input_names=["cls", "patches", "txrv"],
        output_names=["s_inactive"],
        dynamic_axes={
            "cls": {0: "B"},
            "patches": {0: "B"},
            "txrv": {0: "B"},
            "s_inactive": {0: "B"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    return _total_size(out_path), T_seq


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    main_path = OUT / "tb_head_t2.onnx"
    seq_path = OUT / "sequelae_head.onnx"

    main_size = _export_main(main_path)
    seq_size, T_seq = _export_seq(seq_path)
    print(f"exported {main_path}  ({main_size} bytes)")
    print(f"exported {seq_path}  ({seq_size} bytes)")

    cfg = json.loads((DATA / "tb_threshold_t2.json").read_text())
    sidecar = {
        "version": 1,
        "T": float(cfg["temperature"]),
        "thr_at_95sens": float(cfg["threshold"]),
        "T_sequelae": T_seq,
        "seed": SEED,
        "git_sha": _git_sha(),
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "head_input_schema": {
            "tb_head_t2": {
                "inputs": {
                    "cls": {"shape": ["B", 768], "dtype": "float32",
                            "doc": "Rad-DINO CLS token"},
                    "patches": {"shape": ["B", 64, 768], "dtype": "float32",
                                "doc": "Rad-DINO patch tokens, row-major 8x8 grid"},
                    "txrv": {"shape": ["B", 1042], "dtype": "float32",
                             "doc": "TXRV DenseNet pooled features [0:1024] + 18 named-finding logits [1024:]"},
                    "zones": {"shape": ["B", 64, 7], "dtype": "float32",
                              "doc": "Soft zone membership (RUZ, RMZ, RLZ, LUZ, LMZ, LLZ, HILAR); per-patch rows sum<=1"},
                },
                "outputs": {
                    "tb_logit": {"shape": ["B"], "dtype": "float32",
                                 "doc": "Pre-temperature image logit; browser applies sigmoid((logit)/T) using top-level T."},
                },
            },
            "sequelae_head": {
                "inputs": {
                    "cls": {"shape": ["B", 768], "dtype": "float32"},
                    "patches": {"shape": ["B", 64, 768], "dtype": "float32"},
                    "txrv": {"shape": ["B", 1042], "dtype": "float32"},
                },
                "outputs": {
                    "s_inactive": {"shape": ["B"], "dtype": "float32",
                                   "doc": "Temperature-scaled probability of an inactive/sequelae pattern. Consume as an ESCALATE-not-clear feature."},
                },
            },
        },
        "notes": (
            "tb_head_t2: output is a pre-temperature logit; T (top-level) is the calibration "
            "temperature, thr_at_95sens is the 0.95-sensitivity threshold on the calibrated probability. "
            "sequelae_head: output already has T_sequelae baked in (probability ready to consume). "
            "Each .onnx is paired with an .onnx.data external-tensor file in the same directory; "
            "both must be served together for onnxruntime-web to load the model."
        ),
    }
    sidecar_path = OUT / "head_calibration.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {sidecar_path}")


if __name__ == "__main__":
    main()

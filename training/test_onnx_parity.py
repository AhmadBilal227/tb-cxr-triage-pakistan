"""Parity assertion: the ONNX exports of the deployed TB heads match their PyTorch teachers.

LOAD-BEARING: this is the gate that says we can serve the ONNX in the browser without
silently changing predictions. Both heads (TBHeadT2 main + InactiveSequelaeHead) are run
on a deterministic 50-row slice of features.npz/features_sequelae.npz.

PARITY METRIC: max-abs diff on the consumer-facing PROBABILITY (post-temperature, post-
sigmoid) must be < 1e-4. We do not assert on the raw logit because float32 accumulation
across a [B,64,768] gated-attention pool + an LSE-LBA over 64 cells produces relative-
1e-5-level deltas (~5e-4 absolute on logits up to ~30) that completely vanish under the
calibrated sigmoid. The downstream consumer (the orchestrator's safety net) only ever
compares the probability to a threshold; that's what we lock down.

Convention: mirror training/test_stress_metrics.py — no pytest, a `_run_all()` plus
PASS/FAIL prints. Run with:

    training/.venv/bin/python training/test_onnx_parity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MODELS = REPO / "public" / "models"
sys.path.insert(0, str(REPO / "training"))
from train_tb import InactiveSequelaeHead, TBHeadT2  # noqa: E402

SEED = 0
N_SLICE = 50
TOL = 1e-4


def _load_main() -> TBHeadT2:
    z = np.load(DATA / "features.npz", allow_pickle=True)
    d_tok = int(z["patches"].shape[2])
    d_cls = int(z["cls"].shape[1])
    d_txrv = int(z["txrv"].shape[1])
    m = TBHeadT2(d_tok, d_cls, d_txrv, frozenset({"fusion", "zonal", "box"}))
    m.load_state_dict(torch.load(DATA / "tb_head_t2.pt", map_location="cpu"))
    m.eval()
    return m


def _load_seq() -> tuple[InactiveSequelaeHead, float]:
    z = np.load(DATA / "features_sequelae.npz", allow_pickle=True)
    m = InactiveSequelaeHead(int(z["patches"].shape[2]), int(z["cls"].shape[1]))
    m.load_state_dict(torch.load(DATA / "tb_head_inactive.pt", map_location="cpu"))
    m.eval()
    meta = json.loads((DATA / "tb_inactive_meta.json").read_text())
    return m, float(meta["temperature"])


def _slice_indices(n: int, k: int = N_SLICE) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.choice(n, size=min(k, n), replace=False)


# --------------------------------------------------------------------------- main head parity
def test_main_head_parity() -> None:
    """Compare PROBABILITIES (post-T, post-sigmoid) — see module docstring."""
    z = np.load(DATA / "features.npz", allow_pickle=True)
    idx = _slice_indices(int(z["cls"].shape[0]))
    cls = z["cls"][idx].astype("float32")
    patches = z["patches"][idx].astype("float32")
    txrv = z["txrv"][idx].astype("float32")
    zones = z["zones"][idx].astype("float32")

    cfg = json.loads((DATA / "tb_threshold_t2.json").read_text())
    T = float(cfg["temperature"])

    model = _load_main()
    with torch.no_grad():
        model._zones = torch.from_numpy(zones)
        out = model(torch.from_numpy(cls), torch.from_numpy(patches), torch.from_numpy(txrv))
        torch_logit = out["logit"].cpu().numpy()
    torch_prob = (1.0 / (1.0 + np.exp(-torch_logit / T))).astype("float32")

    sess = ort.InferenceSession(str(MODELS / "tb_head_t2.onnx"), providers=["CPUExecutionProvider"])
    onnx_logit = sess.run(
        ["tb_logit"],
        {"cls": cls, "patches": patches, "txrv": txrv, "zones": zones},
    )[0]
    onnx_prob = (1.0 / (1.0 + np.exp(-onnx_logit / T))).astype("float32")

    logit_diff = float(np.max(np.abs(torch_logit - onnx_logit)))
    prob_diff = float(np.max(np.abs(torch_prob - onnx_prob)))
    print(f"  main head: logit max-abs-diff = {logit_diff:.3e} on range [{torch_logit.min():.2f},"
          f"{torch_logit.max():.2f}]; prob max-abs-diff = {prob_diff:.3e}  (tol {TOL:.1e})")
    assert prob_diff < TOL, f"main-head probability parity {prob_diff:.3e} >= {TOL:.1e}"


# --------------------------------------------------------------------------- sequelae head parity
def test_sequelae_head_parity() -> None:
    z = np.load(DATA / "features_sequelae.npz", allow_pickle=True)
    idx = _slice_indices(int(z["cls"].shape[0]))
    cls = z["cls"][idx].astype("float32")
    patches = z["patches"][idx].astype("float32")
    txrv = z["txrv"][idx].astype("float32")

    model, T = _load_seq()
    with torch.no_grad():
        logits = model(torch.from_numpy(cls), torch.from_numpy(patches), torch.from_numpy(txrv))
        torch_probs = (1.0 / (1.0 + np.exp(-logits.cpu().numpy() / T))).astype("float32")

    sess = ort.InferenceSession(str(MODELS / "sequelae_head.onnx"), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["s_inactive"], {"cls": cls, "patches": patches, "txrv": txrv})[0]

    diff = float(np.max(np.abs(torch_probs - onnx_out)))
    print(f"  sequelae head: s_inactive range [{onnx_out.min():.3f},{onnx_out.max():.3f}], "
          f"max-abs-diff = {diff:.3e}  (tol {TOL:.1e})")
    assert diff < TOL, f"sequelae-head ONNX/torch parity {diff:.3e} >= {TOL:.1e}"


# --------------------------------------------------------------------------- batch axis is honored
def test_dynamic_batch_axis() -> None:
    """The dynamic-batch axis must actually accept varying B."""
    z = np.load(DATA / "features.npz", allow_pickle=True)
    sess = ort.InferenceSession(str(MODELS / "tb_head_t2.onnx"), providers=["CPUExecutionProvider"])
    for B in (1, 3, 7):
        cls = z["cls"][:B].astype("float32")
        patches = z["patches"][:B].astype("float32")
        txrv = z["txrv"][:B].astype("float32")
        zones = z["zones"][:B].astype("float32")
        out = sess.run(["tb_logit"], {"cls": cls, "patches": patches, "txrv": txrv, "zones": zones})[0]
        assert out.shape == (B,), f"expected shape ({B},), got {out.shape}"
    print(f"  dynamic batch axis honored for B in (1,3,7)")


def _run_all() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed} tests passed")


if __name__ == "__main__":
    _run_all()

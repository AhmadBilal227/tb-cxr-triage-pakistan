"""CHEAP go/no-go probe: does BONE SUPPRESSION help our FROZEN Rad-DINO encoder?

WHY THIS EXISTS (the concern). The +5-11 AUROC bone-suppression gains in the literature were
measured on classifiers FINE-TUNED on bone-suppressed (soft-tissue) images. Rad-DINO is FROZEN
and was pretrained on bone-PRESENT CXRs (Pérez-García et al. 2024). A soft-tissue image is
therefore potentially OFF-DISTRIBUTION for it — the same failure mode CLAHE hit (non-monotonic
contrast pushed the frozen encoder off its pretraining manifold and HURT external AUROC). This
probe tests cheaply whether bone-suppressed CLS features even carry as much signal as the
original ones for the frozen encoder, BEFORE we pay for a ~35-min full re-extraction.

WHAT IT IS (and is NOT — honest caveat). This is a CLS-ONLY, in-/cross-source LINEAR PROBE
(LogisticRegression on the 768-d Rad-DINO CLS), leave-one-source-out. It is NOT the full T2-head
LODO and does NOT use the lung-seg crop / letterbox / patch tokens / TorchXRayVision channels of
the production pipeline. It is a DIRECTIONAL go/no-go: if bone-supp CLS features do not match
original CLS features on a held-out source under a linear probe, the off-distribution concern is
confirmed and the full re-extraction is not worth it. The arms are identical except for the
bone-suppression step, so any delta is attributable to bone suppression alone.

BONE-SUPPRESSION MODEL. Gusarev et al. 2017 (6-layer CNN autoencoder, BatchNorm variant) from
danielnflam/Deep-Learning-Models-for-bone-suppression-in-chest-radiographs. The pretrained
PyTorch checkpoint (network_intermediate_4.tar, 400 epochs / 72k real pairs shown, trained on
JSRT 256x256 soft-tissue pairs) is vendored to training/bonesupp_weights/. The model arch is
re-declared locally so we do not drag in the repo's matplotlib/scipy module-level imports.

ARMS (identical except the bone-supp step; CLS-only, no seg/letterbox — same for both = apples-to-apples):
  A ORIGINAL  : _read_gray -> _harmonize(uint8) -> Rad-DINO RGB -> CLS(768)
  B BONE-SUPP : _read_gray -> _harmonize(uint8) -> resize 256 + [0,1] -> Gusarev -> soft-tissue 256
                -> upscale to Rad-DINO RGB -> CLS(768)

DECISION CRITERION. Bone-supp HELPS the frozen encoder iff its external (held-out-source) probe
AUROC is >= original within noise. If it is LOWER, that confirms the off-distribution concern ->
DO NOT do the full bone-supp re-extraction; recommend skipping.

    PYTORCH_ENABLE_MPS_FALLBACK=1 training/.venv/bin/python training/bonesupp_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
# Reuse the EXACT production read + harmonize so the only difference vs baseline is bone-supp.
from extract_features import _read_gray, _harmonize  # noqa: E402

DATA = REPO / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RAD_ID = "microsoft/rad-dino"
BS_CKPT = REPO / "training" / "bonesupp_weights" / "gusarev_network_intermediate_4.tar"
BS_RES = 256              # Gusarev was trained on 256x256
SUBSET_TARGET = 2500      # tractable probe size
QATAR_PER_LABEL = 700     # balanced qatar slice (montgomery + shenzhen taken in full)
SEED = 17
BATCH = 16


# ---------------------------------------------------------------------------------------------
# Gusarev 6-layer CNN (BatchNorm variant), re-declared locally to avoid the upstream module's
# matplotlib/scipy top-level imports. Verified to load the vendored state_dict strict=True.
# ---------------------------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    def __init__(self, ci: int, co: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 5, 1, 2, bias=True)
        self.norm = nn.BatchNorm2d(co)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class GusarevBoneSupp(nn.Module):
    """conv(5x5)->BN->ReLU x5 (16,32,64,128,256) then 1x1-channel conv_output -> Sigmoid soft-tissue."""

    def __init__(self, in_nc: int = 1) -> None:
        super().__init__()
        o = 16
        self.layer1 = _ConvBlock(in_nc, o)
        self.layer2 = _ConvBlock(o, o * 2)
        self.layer3 = _ConvBlock(o * 2, o * 4)
        self.layer4 = _ConvBlock(o * 4, o * 8)
        self.layer5 = _ConvBlock(o * 8, o * 16)
        self.conv_output = nn.Conv2d(o * 16, in_nc, 5, 1, 2, bias=True)
        self.output_layer = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.output_layer(self.conv_output(x))


def load_bonesupp() -> GusarevBoneSupp:
    if not BS_CKPT.exists():
        raise SystemExit(f"missing bone-supp checkpoint {BS_CKPT}")
    net = GusarevBoneSupp().to(DEVICE).eval()
    ck = torch.load(BS_CKPT, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model_state_dict"], strict=True)  # strict: confirms arch matches
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def build_subset() -> pd.DataFrame:
    """Tractable balanced-ish subset: montgomery + shenzhen in full (clean external CXR folds) +
    a label-balanced slice of qatar. ~2.2k images across 3 sources."""
    df = pd.read_csv(DATA / "index_dedup.csv")
    rng = np.random.RandomState(SEED)
    parts: list[pd.DataFrame] = []
    parts.append(df[df["source"] == "montgomery"])
    parts.append(df[df["source"] == "shenzhen"])
    q = df[df["source"] == "qatar"]
    for lab in (0, 1):
        ql = q[q["label"] == lab]
        n = min(QATAR_PER_LABEL, len(ql))
        parts.append(ql.sample(n=n, random_state=rng))
    sub = pd.concat(parts).reset_index(drop=True)
    if len(sub) > SUBSET_TARGET:  # keep tractable while preserving per-source label balance
        sub = sub.groupby("source", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), int(SUBSET_TARGET * len(g) / len(sub))), random_state=rng)
        ).reset_index(drop=True)
    return sub


PATCH_GRID = 8  # same 8x8 pooled grid as the production extractor (extract_features.PATCH_GRID)


@torch.no_grad()
def _radino_cls_and_grid(rad: AutoModel, rin: dict) -> tuple[np.ndarray, np.ndarray]:
    """One Rad-DINO pass -> (CLS [b,768], patch_grid [b,8,8,768]).

    Patch pooling mirrors extract_features.py EXACTLY: tokens (no register tokens) -> [b,D,g,g] ->
    adaptive_avg_pool2d to 8x8. adaptive_avg_pool2d with non-divisible sizes (37->8) is unsupported
    on MPS, so the pool is done on CPU. Grid is returned [b, gy, gx, D] (row-major gy*8+gx)."""
    out = rad(**rin)
    cls = out.pooler_output.float().cpu().numpy()              # [b,768]
    tok = out.last_hidden_state[:, 1:, :]                      # drop CLS; Rad-DINO has no register tokens
    g = int(round(tok.shape[1] ** 0.5))
    assert tok.shape[1] == g * g, f"unexpected Rad-DINO token count {tok.shape[1]}"
    grid = tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], g, g).float().cpu()  # [b,D,g,g]
    pooled = F.adaptive_avg_pool2d(grid, (PATCH_GRID, PATCH_GRID))                       # [b,D,8,8]
    pooled = pooled.permute(0, 2, 3, 1).numpy()  # [b, gy, gx, D] — row-major (gy down, gx across)
    return cls, pooled.astype("float32")


def _harm_to_rgb_inputs(proc: AutoImageProcessor, gray_u8_list: list[np.ndarray]) -> dict:
    rgbs = [Image.fromarray(cv2.cvtColor(h, cv2.COLOR_GRAY2RGB)) for h in gray_u8_list]
    return proc(images=rgbs, return_tensors="pt").to(DEVICE)


@torch.no_grad()
def arm_original(rad: AutoModel, proc: AutoImageProcessor, harm_u8_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ORIGINAL arm: harmonized uint8 gray -> RGB -> Rad-DINO -> (CLS, 8x8 patch grid)."""
    return _radino_cls_and_grid(rad, _harm_to_rgb_inputs(proc, harm_u8_list))


@torch.no_grad()
def arm_bonesupp(
    rad: AutoModel, proc: AutoImageProcessor, bs: GusarevBoneSupp, harm_u8_list: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """BONE-SUPP arm: harmonized uint8 -> 256x256 [0,1] -> Gusarev soft-tissue -> upscale to RGB ->
    Rad-DINO -> (CLS, 8x8 patch grid). Gusarev input convention matches the upstream notebook
    (ToTensor -> [0,1], 256x256 grayscale)."""
    inp = np.stack([
        cv2.resize(h, (BS_RES, BS_RES), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        for h in harm_u8_list
    ])  # [b,256,256] in [0,1]
    t = torch.from_numpy(inp)[:, None, ...].to(DEVICE)  # [b,1,256,256]
    soft = bs(t).clamp(0.0, 1.0)                        # [b,1,256,256] soft-tissue in [0,1]
    soft_u8 = (soft[:, 0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return _radino_cls_and_grid(rad, _harm_to_rgb_inputs(proc, list(soft_u8)))


def mil_mean_pool(grid: np.ndarray, apical_only: bool) -> np.ndarray:
    """Mean-MIL pool an [N, gy, gx, D] 8x8 patch grid -> [N, D].

    apical_only=False -> mean over ALL 64 patches (full-patch MIL).
    apical_only=True  -> mean over the TOP 1/3 rows only (apical zones: rows 0..2 of 8 = upper lung,
                         where rib/clavicle overlap hides TB most and a soft-tissue benefit should
                         concentrate). Mean (not attention) MIL keeps the probe cheap and stable on
                         a few-hundred-image held-out fold; an attention head would add fitted params
                         that overfit the small folds and muddy the directional read.

    NOTE on registration: this is CLS-only-pipeline grid pooling — NO lung-seg crop / letterbox. So
    'top 1/3 rows' is the top of the WHOLE harmonized image, not of a seg-cropped lung field. For most
    PA CXRs the apices sit in roughly the top third of the frame, so this is a reasonable cheap proxy;
    it is NOT the production zone matrix (which uses the seg masks). Same crop convention for both arms,
    so the ORIGINAL-vs-BONE-SUPP delta is still apples-to-apples."""
    rows = 3 if apical_only else PATCH_GRID  # top 3 of 8 rows ~= upper third
    return grid[:, :rows, :, :].reshape(grid.shape[0], -1, grid.shape[3]).mean(axis=1)


def lodo_probe(X: np.ndarray, y: np.ndarray, src: np.ndarray) -> dict[str, float]:
    """Leave-one-source-out linear probe. Train LogisticRegression on the other sources, test AUROC
    on the held-out source. Returns per-source held-out AUROC + a sample-weighted mean."""
    out: dict[str, float] = {}
    sources = sorted(set(src.tolist()))
    n_total = 0
    weighted = 0.0
    for held in sources:
        tr = src != held
        te = src == held
        if len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2:
            out[held] = float("nan")
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), y[tr])
        prob = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        auc = float(roc_auc_score(y[te], prob))
        out[held] = auc
        n = int(te.sum())
        weighted += auc * n
        n_total += n
    out["_mean_weighted"] = weighted / max(1, n_total)
    return out


CACHE = DATA / "bonesupp_probe_features.npz"


def extract_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (CLS, 8x8 grid) for BOTH arms over the subset; cache to npz so probe re-runs are
    instant. Returns (cls_o, grid_o, cls_b, grid_b, y, src)."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        print(f"loaded cached features {CACHE.name}  (delete to force re-extract)")
        return d["cls_o"], d["grid_o"], d["cls_b"], d["grid_b"], d["y"], d["src"]

    t0 = time.time()
    sub = build_subset()
    print(f"subset: {len(sub)} images | "
          + " ".join(f"{s}={int((sub['source'] == s).sum())}(pos {int(((sub['source'] == s) & (sub['label'] == 1)).sum())})"
                     for s in sorted(set(sub['source']))))
    proc = AutoImageProcessor.from_pretrained(RAD_ID)
    rad = AutoModel.from_pretrained(RAD_ID).to(DEVICE).eval()
    for p in rad.parameters():
        p.requires_grad_(False)
    bs = load_bonesupp()
    print(f"models loaded ({time.time() - t0:.0f}s)")

    paths = [str(p) if str(p).startswith("/") else str(REPO / p) for p in sub["path"]]
    y_all = sub["label"].to_numpy().astype("int64")
    src_all = sub["source"].to_numpy().astype(str)

    cls_o, grid_o, cls_b, grid_b = [], [], [], []
    y_keep: list[int] = []
    src_keep: list[str] = []
    skipped = 0
    n = len(paths)
    for s in range(0, n, BATCH):
        rows = list(range(s, min(s + BATCH, n)))
        harm_list: list[np.ndarray] = []
        ok: list[int] = []
        for i in rows:
            try:
                harm_list.append(_harmonize(_read_gray(paths[i])))
                ok.append(i)
            except Exception as e:  # fail-visible: count, never hide
                print("skip", paths[i], repr(e)[:80])
                skipped += 1
        if not harm_list:
            continue
        co, go = arm_original(rad, proc, harm_list)
        cb, gb = arm_bonesupp(rad, proc, bs, harm_list)
        for j, i in enumerate(ok):
            cls_o.append(co[j]); grid_o.append(go[j])
            cls_b.append(cb[j]); grid_b.append(gb[j])
            y_keep.append(int(y_all[i])); src_keep.append(str(src_all[i]))
        done = min(s + BATCH, n)
        if done % (BATCH * 10) == 0 or done == n:
            el = time.time() - t0
            print(f"  {done}/{n} ({100 * done // n}%)  {el:.0f}s  {el / max(1, done):.3f} s/img")

    if skipped:
        print(f"SKIPPED {skipped}/{n} unreadable/harmonize-fail images (fail-visible)")

    cls_o = np.stack(cls_o).astype("float32"); grid_o = np.stack(grid_o).astype("float32")
    cls_b = np.stack(cls_b).astype("float32"); grid_b = np.stack(grid_b).astype("float32")
    y = np.asarray(y_keep, dtype="int64"); src = np.asarray(src_keep, dtype=object)
    np.savez_compressed(CACHE, cls_o=cls_o, grid_o=grid_o, cls_b=cls_b, grid_b=grid_b, y=y, src=src)
    print(f"cached -> {CACHE.name}  ({time.time() - t0:.0f}s)")
    return cls_o, grid_o, cls_b, grid_b, y, src


def _print_arm(name: str, res_o: dict[str, float], res_b: dict[str, float]) -> float:
    print(f"\n--- {name} ---")
    print(f"{'held-out source':<18}{'ORIGINAL':>12}{'BONE-SUPP':>12}{'delta(BS-ORIG)':>16}")
    for s in sorted(k for k in res_o if not k.startswith("_")):
        print(f"{s:<18}{res_o[s]:>12.4f}{res_b[s]:>12.4f}{res_b[s] - res_o[s]:>+16.4f}")
    mo, mb = res_o["_mean_weighted"], res_b["_mean_weighted"]
    print("-" * 58)
    print(f"{'MEAN (weighted)':<18}{mo:>12.4f}{mb:>12.4f}{mb - mo:>+16.4f}")
    return mb - mo


def main() -> None:
    t0 = time.time()
    print(f"device={DEVICE}  bone-supp model=Gusarev MultilayerCNN (vendored {BS_CKPT.name})")
    cls_o, grid_o, cls_b, grid_b, y, src = extract_features()
    print(f"\nfeatures: CLS {cls_o.shape} grid {grid_o.shape}  pos {int(y.sum())} neg {int((y == 0).sum())}")

    # THREE arms, identical split (leave-one-source-out), ORIGINAL vs BONE-SUPP:
    #   CLS         — global pooler vector (a local visibility benefit can be invisible here)
    #   full-patch  — mean-MIL over all 64 patches (local benefit shows up)
    #   apical      — mean-MIL over the top 1/3 rows only (apices = where ribs/clavicles hide TB most)
    arms = {
        "CLS (global)": (cls_o, cls_b),
        "FULL-PATCH MIL (mean over 64)": (mil_mean_pool(grid_o, False), mil_mean_pool(grid_b, False)),
        "APICAL-PATCH MIL (top 1/3 rows)": (mil_mean_pool(grid_o, True), mil_mean_pool(grid_b, True)),
    }
    print("\n========== LEAVE-ONE-SOURCE-OUT LINEAR PROBE (CHEAP; CLS-pipeline, no seg/letterbox) ==========")
    deltas: dict[str, float] = {}
    res_by_arm: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for name, (Xo, Xb) in arms.items():
        ro, rb = lodo_probe(Xo, y, src), lodo_probe(Xb, y, src)
        deltas[name] = _print_arm(name, ro, rb)
        res_by_arm[name] = (ro, rb)

    # Decision (updated per GPT review). Bone-supp is a LOCAL visibility intervention, so the
    # PATCH/APICAL arms are the ones that can reveal benefit; a flat/lower CLS alone is NOT a no-go.
    # Worth a full re-extraction only if a PATCH or APICAL arm IMPROVES on a held-out source AND does
    # not tank the others (no per-source regression worse than -NOISE on the arm that improved).
    NOISE = 0.01  # ~AUROC noise band for hundreds-of-images held-out folds
    print("\n================= GO / NO-GO =================")
    print(f"criterion (updated): bone-supp is worth the full re-extraction only if the FULL-PATCH or "
          f"APICAL MIL probe improves on a held-out source by > +{NOISE} AND does not regress another "
          f"source by worse than -{NOISE} on that same arm. CLS alone is not decisive (local benefit "
          f"can be invisible to a global vector).")
    local_arms = ["FULL-PATCH MIL (mean over 64)", "APICAL-PATCH MIL (top 1/3 rows)"]
    go = False
    go_reason = ""
    for name in local_arms:
        ro, rb = res_by_arm[name]
        per = {s: rb[s] - ro[s] for s in ro if not s.startswith("_")}
        improved = [s for s, d in per.items() if d > NOISE]
        regressed = [s for s, d in per.items() if d < -NOISE]
        if improved and not regressed:
            go = True
            go_reason = (f"{name}: improved on {improved} (deltas "
                         f"{ {s: round(per[s], 4) for s in improved} }) with no source regressing > -{NOISE}.")
            break
    print("\nper-arm weighted-mean delta (BS - ORIG):")
    for name, d in deltas.items():
        print(f"  {name:<34}{d:>+8.4f}")
    if go:
        verdict = ("GO (weak/directional): a LOCAL (patch/apical) probe shows a held-out-source gain without "
                   "tanking others. Worth confirming with the REAL seg+letterbox+patch T2-head LODO before "
                   "committing the 35-min full re-extraction. " + go_reason)
    else:
        verdict = ("NO-GO: NO local (patch/apical) arm shows a held-out-source improvement beyond noise "
                   "without regressing another source; CLS is also flat/lower. This confirms the off-"
                   "distribution concern — soft-tissue is OFF-distribution for the bone-present-pretrained "
                   "FROZEN Rad-DINO (the CLAHE failure mode), and the local-visibility benefit that helps "
                   "FINE-TUNED models does not transfer to frozen features. DO NOT do the full re-extraction.")
    print(f"\nVERDICT: {verdict}")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

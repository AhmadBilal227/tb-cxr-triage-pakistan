# T1 — TB Classifier Training Implementation Plan

> **For agentic workers:** This is an offline Python data-science pipeline (runs on the user's Apple M4, not the browser app). Steps are concrete commands + code. Reviewable per-task; not unit-TDD-shaped like the app code.

**Goal:** Train a best-quality, generalization-validated **multi-task** chest-X-ray perception model and export it to ONNX for the in-browser perception slot. Primary output is a calibrated TB probability (`ensemble.tb`, targeting WHO triage: sensitivity ≥90% / specificity ≥70%), plus richer outputs: TB activity subtype, pathology multi-label, lesion localization, and uncertainty.

**Architecture:** **Feature fusion of two frozen experts** — `microsoft/rad-dino` (ViT-B, MIT, self-supervised on 838k CXRs → 768-d CLS) ⊕ `torchxrayvision` DenseNet121 (supervised multi-pathology, pretrained → 1024-d + 18 logits) — concatenated (~1.8k-d), feeding **small multi-task heads**: (1) TB activity 4-class (TBX11K: healthy/sick-non-TB/active-TB/latent-TB), (2) binary TB probability (primary), (3) pathology multi-label (TorchXRayVision passthrough), (4) lesion localization (Rad-DINO patch grid, box-supervised on TBX11K). Both backbones are **frozen** (inference only); features cached once → heads train in seconds. Multi-task supervision regularizes the primary TB head and forces pathology-grounded (not site-shortcut) features. Honest performance by **leave-one-dataset-out (LODO)** per head; shipped model trained on all open data; later phase fine-tunes on the user's own in-domain data. Calibration + Mondrian (class-conditional) conformal + TTA + log-odds ensemble on top.

**Tech Stack:** Python 3.11 (via `uv`), PyTorch + MPS (Apple Metal) backend, `transformers` (Rad-DINO), `torchxrayvision` (baseline + lung utils), `segmentation-models-pytorch`/U-Net or a pretrained lung-mask model, `scikit-learn`, `scikit-image`, `imagehash`, `opencv` (CLAHE), `pytorch-grad-cam`, `onnx`/`onnxruntime`.

**Hardware:** Apple MacBook Pro M4, 24 GB unified memory. Frozen-backbone approach fits comfortably (ViT-B inference at batch ≤32 uses a few GB). `device="mps"`, `PYTORCH_ENABLE_MPS_FALLBACK=1`.

---

## How practitioners actually do it (expert workflow + tips)

Synthesis of the research swarm + practitioner/competition literature. These are the techniques that separate a real model from a leaderboard-overfit one:

1. **Lung-field segmentation + cropping is the highest-value preprocessing.** Segment lungs (U-Net trained on the 704-image lung-mask set, or DeepLabv3+; IoU ~95% on Montgomery) and zero/crop everything outside the lung field. This *removes* the extra-pulmonary regions where shortcut signals live (burned-in markers, collimation borders, scanner text, body habitus). ([MDPI 2024](https://www.mdpi.com/2075-4418/14/9/952))
2. **CLAHE contrast normalization** (Contrast-Limited Adaptive Histogram Equalization) standardizes contrast across scanners/sites, suppressing a known site shortcut and surfacing texture (cavitation, miliary nodules). ([ResearchGate](https://www.researchgate.net/publication/328457997))
3. **Shortcut/site bias is THE failure mode.** A CNN can read age/sex/BMI/scanner at AUROC≥0.90 from one CXR; if those correlate with the TB label per-site, the model "cheats" and collapses on new data. ([shortcut study arXiv:2009.10132](https://arxiv.org/pdf/2009.10132), [debias arXiv:2203.09860](https://arxiv.org/pdf/2203.09860)). Defenses baked into this plan: cross-source hash dedup, lung-masking, CLAHE, patient-level + LODO splits, a site-leak canary, and Grad-CAM verification.
4. **Anatomy-preserving augmentation only:** rotation ±7–10°, small translation/scale, mild brightness/contrast. **No horizontal flip** (breaks cardiac silhouette/situs — a real signal). No heavy elastic/cutout over lungs.
5. **Frozen domain-pretrained backbone + small head generalizes better than full fine-tune** on small TB sets (fewer params to overfit to site noise). Rad-DINO was SSL-pretrained on ~838k CXRs and never saw these TB labels.
6. **Ensembling + test-time augmentation (TTA)** lift AUC a few points (average predictions over a few augmented views; combine Rad-DINO head with a TXRV-DenseNet head via log-odds). Use sparingly — bigger wins are upstream.
7. **Report LODO / external numbers, never in-distribution.** In-dist AUC reads ~0.99 and is meaningless; the honest number is train-on-two-sources, test-on-the-third (~0.80–0.88).
8. **Calibrate the operating threshold for sensitivity**, don't use 0.5. (This is exactly the Phase-1 calibration we already built — feed this model's probabilities into `/validate` → Calibrate.)

---

## Phased strategy

- **Phase A (this plan): open data.** Build the pipeline, measure honest LODO, train the final head on all open data, export ONNX, wire into the app, validate.
- **Phase B (later): your own data.** Re-extract features for your in-domain labeled images and continue/refit the head on (open + own) features. In-domain data is the single best lever for closing the deployment-population gap. Frozen backbone keeps this cheap on the M4.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/fetch_tb_data.sh` (done) | Download datasets into `data/raw/` (disk-guarded). |
| `training/requirements.txt` | Pinned Python deps. |
| `training/build_index.py` | Walk `data/raw/`, emit a unified `data/index.csv` (path,label,source,patient_id). |
| `training/dedup.py` | Cross-source perceptual+exact hash dedup → `data/index_dedup.csv`. |
| `training/preprocess.py` | Lung segmentation/crop + CLAHE; cache processed images or do it inline in extraction. |
| `training/extract_features.py` | Frozen Rad-DINO on MPS over all images → `data/features.npz` (X, labels, source, patient_id). |
| `training/train_tb.py` | LODO eval + head training + threshold-for-90%-sens; saves `tb_head.pt` + `tb_threshold.json`. |
| `training/audit.py` | Site-leak canary + Grad-CAM/attention audit. |
| `training/export_onnx.py` | Backbone+head+Sigmoid → `public/models/tb-cxr/onnx/model.onnx` (+ preprocessing config). |
| `data/` (gitignored) | Raw + processed data, feature caches. |

---

## Tasks

### Task 1: Python environment (M4 / MPS)

- [ ] Create `training/requirements.txt`:
```
torch>=2.4
torchvision>=0.19
torchxrayvision>=1.2.0
transformers>=4.44
scikit-learn>=1.5
scikit-image>=0.24
opencv-python-headless>=4.10
imagehash>=4.3
pillow>=10.4
pandas>=2.2
numpy>=1.26
onnx>=1.16
onnxruntime>=1.19
grad-cam>=1.5
segmentation-models-pytorch>=0.3.4
tqdm
```
- [ ] Set up + verify MPS:
```bash
cd "/Users/ahmadbilal/Downloads/hobby/TB detector"
uv venv training/.venv --python 3.11
source training/.venv/bin/activate
uv pip install -r training/requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 python -c "import torch; print('mps available:', torch.backends.mps.is_available())"
```
Expected: `mps available: True`.

> Use Python **3.11** (not the system 3.14) — torch/torchxrayvision wheels target ≤3.12.

### Task 2: Build unified index + dedup

- [ ] `training/build_index.py`: walk each `data/raw/<source>/` folder, infer label from folder/filename convention per source (Qatar: `Normal/` vs `Tuberculosis/`; Montgomery/Shenzhen: filename `*_0`=normal/`*_1`=TB; TBX11K: its CSV), infer `patient_id` (filename stem sans suffix; for sets without IDs use the filename). Emit `data/index.csv` with columns `path,label,source,patient_id`. Print per-source class counts.
- [ ] `training/dedup.py`: for every image compute `average_hash` (near-dup) + md5 (exact); drop cross-source duplicates (keep first); emit `data/index_dedup.csv`. Print how many dups removed (expect overlap between Qatar and Montgomery/Shenzhen).

### Task 3: Lung segmentation/crop + CLAHE preprocessing

- [ ] `training/preprocess.py`: a `preprocess(path) -> PIL.Image` that (1) loads grayscale, (2) applies **CLAHE** (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`), (3) segments the lung field (use the `data/raw/lungseg` masks to train a small U-Net via `segmentation-models-pytorch`, OR use a pretrained lung-mask model; fall back to a center-crop if segmentation unavailable), (4) zeroes pixels outside the lung mask + crops to the lung bounding box, (5) returns RGB (3-ch replicate). This is shared by extraction, the ONNX preprocessing, and the eventual browser preprocessing — keep it bit-exact between train and inference.
- [ ] Sanity-check: save 10 before/after thumbnails to `training/_debug/` and eyeball that lungs are isolated.

### Task 4: Feature extraction (Rad-DINO, MPS, cached)

- [ ] `training/extract_features.py`: load `microsoft/rad-dino` + its `AutoImageProcessor` (resize 518, center-crop, mean 0.5307 std 0.2583), `device="mps"`, `eval()`, `requires_grad_(False)`. For each row in `index_dedup.csv`: `preprocess()` → processor → `model(**x).pooler_output` (768-d CLS). Batch (≤32). Cache to `data/features.npz` with arrays `X [N,768]`, `y [N]`, `source [N]`, `patient_id [N]`. Use `tqdm`. **One-time ~30–60 min on M4.**
- [ ] Verify: `python -c "import numpy as np; d=np.load('data/features.npz', allow_pickle=True); print(d['X'].shape, d['y'].sum(), set(d['source']))"`.

### Task 5: LODO evaluation + head training + threshold calibration

- [ ] `training/train_tb.py`: define `Head` (`LayerNorm(768)→Dropout(0.3)→Linear(768,256)→GELU→Dropout(0.3)→Linear(256,1)`).
  - **LODO loop** (honest scoreboard): for each `source` as held-out test, train the head on the other sources' cached features (`pos_weight` BCE for imbalance, AdamW lr 1e-3, ~60 epochs, early-stop on val AUPRC), evaluate on the held-out source. Print per-fold AUC + sensitivity@90%-spec-threshold and the resulting specificity.
  - **Threshold for ≥90% sensitivity** via `roc_curve` on a validation split; save to `tb_threshold.json` (target 0.92 for CI headroom).
  - **Final model**: retrain the head on **all** features; save `tb_head.pt`.
  - Print the headline: mean LODO AUC, and "expected external sensitivity/specificity".

### Task 6: Anti-shortcut audits (gate before trusting the model)

- [ ] `training/audit.py`:
  - **Site-leak canary:** train a logistic regression to predict `source` from `X` (5-fold CV accuracy). If ≈1.0, the backbone trivially separates sites → high shortcut risk; report loudly.
  - **Grad-CAM / attention:** on held-out TB cases, verify activation lands on lung parenchyma (apices for TB), not corners/markers. Save overlays to `training/_audit/`.
- [ ] Decision gate: if LODO AUC is high but site-leak ≈1.0 and Grad-CAM is off-lung, the result is not trustworthy — strengthen masking/dedup before shipping.

### Task 7: Final model + ONNX export

- [ ] `training/export_onnx.py`: wrap frozen Rad-DINO backbone + trained `Head` + final `Sigmoid` into one `nn.Module`; `torch.onnx.export` (opset 17, dynamic batch axis, input `pixel_values [B,3,518,518]` already-normalized, output `tb_prob [B]`). Verify with `onnxruntime`. Quantize to int8 (`optimum-cli onnxruntime quantize`) for a browser-sized asset (~87 MB). Place at `public/models/tb-cxr/onnx/model.onnx` (+ `config.json` with labels, `preprocessor_config.json`, `tb_threshold.json`).
- [ ] **The browser preprocessing in T2 MUST replicate Task 3 + Rad-DINO normalization bit-for-bit**, or the in-browser scores will silently diverge from training.

### Task 8: Wire into the app (T2) + validate

- [ ] Implement the in-browser `'local'` provider (plan `2026-05-24-perception-module.md` Phase 2, Tasks 12–15): `onnxClassifier.worker.ts` + `onnxLocal.ts`, the lung-crop+CLAHE preprocessing in JS, wired as the `ensemble.tb` primary with HF→Replicate fallback.
- [ ] Run `/validate` on a held-out labeled set → **Calibrate** (Phase 1) to fit the conformal threshold for ≥90% sensitivity → confirm sens/spec with CIs.

### Task 9 (Phase B, later): Fine-tune on your own data

- [ ] Re-run `extract_features.py` on your labeled in-domain images; append to the feature cache with a new `source` tag; retrain the head on (open + own). Re-export ONNX. Re-validate. This is the step that realistically pushes past 90% on *your* population.

---

## Expected results & success criteria

- **LODO AUC ~0.80–0.88**, sensitivity ~80–88% at 60–70% specificity on open data — a clear upgrade over the ~42% VLM-only, borderline on WHO 90/70.
- **Success gate to ship as `ensemble.tb`:** LODO AUC ≥ 0.82, site-leak canary not ≈1.0, Grad-CAM on-lung, and `/validate` sensitivity CI lower-bound ≥ ~85% on a held-out set.
- **Path to ≥90%:** Phase B in-domain fine-tune + ensemble (Rad-DINO head ⊕ TXRV-DenseNet head) + calibrated threshold.

## Biggest risk
Site/shortcut bias inflating LODO-adjacent numbers. The audit gate (Task 6) is mandatory, not optional — a model that looks great in-distribution but fails the site-leak/Grad-CAM checks must not ship.

---

## Expert-panel validation — corrections to claims & method (2026-05-24)

A six-lens validation panel (ML methodologist, radiologist, TB epidemiologist/patient advocate, steelman, literature, red-team) reviewed this plan. Consensus: the architecture is evidence-supported and the honesty discipline (LODO, audits, calibration) is real, but several **claims are not earned as written** and several **methodology gaps let the model pass its own gate while still shortcut-driven.** The literature lens independently confirmed our expected LODO AUC band (~0.80–0.88) matches commercial CAD external numbers — so the engineering is sound; the claims and gate must be corrected. The following SUPERSEDE the optimistic language above.

### Claims honesty (correct the wording everywhere)
- **Conformal does NOT "guarantee ≥90% sensitivity."** Its coverage holds only under exchangeability, which cross-site/population deployment shift breaks; fitting calibration+fusion+conformal on one set is also circular, and at ~20 positives the band is high-variance. Reframe as: *"a calibrated operating point with in-distribution, finite-sample coverage that MUST be re-fit on labeled data from each deployment site, reported with a binomial CI."* The honest sensitivity mechanism is empirical per-site threshold + CI, not the conformal guarantee.
- **Metrics are against RADIOGRAPHIC/program labels, not bacteriological confirmation.** The WHO TPP bar is defined against culture/NAAT. Never map our radiographic-label sensitivity onto WHO 90/70 without a microbiologically-confirmed test set. State this in the model card and every reported number.
- **WHO 90/70 is the TPP *minimum* (optimal 95/80), not a target.** State the honest LODO band next to the goal; at 90% sensitivity expect external specificity ~55–70% (matches commercial CAD), not both-high.
- **Multi-task "raises primary AUC" is unproven** — present it as regularization/anti-shortcut, and only claim an AUC lift if a single-task-vs-multi-task LODO ablation shows it.
- **Sample size:** no ≥90% sensitivity claim without ≥~150 held-out TB positives (≥250 for ±5%); report AUC (tighter CIs) as primary until then.

### Clinical corrections
- **Drop the user-facing "active vs latent" output.** Latent TB is radiographically silent by definition; TBX11K's "latent" = old/healed sequelae. Relabel the 4th class **"TB sequelae (old/healed)"**, keep the 4-class head INTERNAL-only (regularizer), never surface "latent" to a user.
- **Don't hard-crop to a tight lung bbox** — it amputates hilar/mediastinal lymphadenopathy, pleural effusion, and apical extent (all TB-relevant, several are our own targets). Use a **dilated/soft mask (+15–20%) on an anatomical frame** (retain hila, costophrenic angles, apices), or a 2-channel image+mask input. Verify apices/CP-angles retained in the debug check, not just "lungs isolated."
- **Finding vocabulary:** remove **tree-in-bud** (a CT sign, not CXR); add primary-vs-post-primary pattern cue; explicitly note HIV/pediatric TB is often lower-zone/adenopathy-predominant/normal-film so those don't falsely lower TB probability.
- **Hard-gate pediatric (<15y) out of scope** (WHO does not endorse CAD there); name **HIV, pediatric, prior-TB scarring, field-equipment** as known failure populations in the model card.

### Method corrections (the eventual code must satisfy these)
- **Collapse Qatar + Montgomery + Shenzhen overlaps:** Qatar aggregates NLM, so treat overlapping sources as ONE LODO group; dedup with **pHash + embedding-NN (cosine on Rad-DINO features)**, threshold chosen on a labeled dup/non-dup set, not a fixed aHash-5. Reconcile patient identity from each dataset's manifest; where absent, state that "patient-level" split is actually image-level.
- **Nested threshold:** in each LODO fold, pick the operating threshold on a **train-only validation split**, never on the held-out test fold (current `train_tb.py` fits it on the test fold → optimistic specificity).
- **Add the highest-value experiment to the gate:** fit conformal/threshold on sources {A,B}, then report **realized sensitivity at that FROZEN threshold on held-out source C** (rotate). If it swings across folds, exchangeability is violated and the "guarantee" is void — this is the number that predicts deployment.
- **Reframe the site-leak canary as diagnostic, not pass/fail** (Rad-DINO will separate sites ~1.0 regardless). Gate instead on: LODO AUC surviving lung-masking + frozen-threshold cross-source sensitivity + site-swap/CLAHE-match counterfactual probability stability.
- **Labels from official manifests, not path substrings** (`build_index.py` heuristics can mislabel); assert per-source counts equal published totals.
- **Report at deployment prevalence:** add a prevalence → PPV/NPV → confirmatory-tests-per-case panel to `/validate`; never quote accuracy/PPV on balanced sets.

### Revised success gate (replaces the earlier one)
Ship to `ensemble.tb` only if: (1) **frozen-threshold cross-source (LODO) sensitivity** holds with a binomial CI lower bound ≥85% on ≥~150 positives; (2) LODO AUC survives lung-masking; (3) site-swap counterfactual probability is stable; (4) Grad-CAM on-lung; (5) all numbers stated against their reference standard (radiographic vs bacteriological) with the PPV-at-prevalence panel; (6) "research preview, not a medical device," pediatric hard-gated, failure populations named. The conformal band is re-fit per deployment site (Phase B), never trusted from open-data calibration alone.

### Remediation — converting three "caveats" into fixes
These three were initially logged as caveats; they are actually fixable, and the plan now requires the fix rather than the disclaimer:

- **Labels → WHO bar (was: caveat) → FIX: add a bacteriologically-confirmed validation/calibration tier.** Train representation on the open sets (Qatar/Montgomery/Shenzhen/TBX11K), but acquire **TB Portals (NIAID, https://tbportals.niaid.nih.gov/download-data)** via its online DUA — culture/Xpert/DST-confirmed TB + imaging across 14 countries — and use a held-out CXR slice of it as the **WHO-comparable (bacteriological) eval + conformal calibration tier**. Optionally use Shenzhen's micro-confirmed subset. Report sensitivity/specificity on this confirmed tier as the WHO-comparable number. *Residual caveats to state:* TB Portals is a referral/DR-enriched spectrum (not screening prevalence), and a fully clean screening claim still needs a prospective screening cohort (Phase B). Requires the user to sign the DUA. New task: `T1-6 fetch+filter TB Portals CXR → confirmed eval tier`.

- **Multi-task raises AUC (was: unproven) → FIX: run the ablation.** Train single-task (binary head only) vs multi-task (binary + activity/pathology/localization heads) on the *same cached features* under LODO; report both AUCs. Claim the lift only as the measured delta; keep the heads for explainability regardless. Near-free (heads train in seconds). Folded into `train_tb`/`train_multitask`.

- **n=30 eval / PPV (was: insufficient) → FIX: it was a paid-VLM cost cap, not a real limit.** The trained CNN evaluates for free per image, so evaluate on the **full held-out source** (≥150 TB positives per LODO fold from TBX11K/Shenzhen/Qatar) and report Clopper-Pearson CIs. Add the prevalence→PPV/NPV→confirmatory-tests-per-case panel to `/validate`. No sample-size compromise once off the paid VLM path.

### Geographic diversity + data provenance (regional dataset audit, 2026-05-24)
A regional audit found **hidden re-mix leakage in the current mix** and the diversity sources worth pursuing.

**Re-mix leakage (correct the LODO):**
- **Qatar is a re-mix** of NIAID/Belarus (~3k) + Montgomery + RSNA — it already contains Belarus/NIAID images.
- **TBX11K is a re-mix** of DA+DB (India, NITRD) + Montgomery + Shenzhen — it already contains the Indian DA/DB images.
- So our 4 "sources" ≈ 3 primary (Montgomery, Shenzhen, India-DA/DB) + a Belarus slice, mutually overlapping → LODO is partly in-distribution. **Rule:** treat overlapping sources as ONE LODO group; pHash+embedding dedup ACROSS all sources; do **NOT** add DA/DB (leaks TBX11K) or a standalone Belarus set (leaks Qatar); avoid generic Kaggle "TB CXR" sets (re-mixes).

**Diversity sources to pursue (geographic + HIV-endemic gap):**
- **Training diversity:** TB Portals (NIAID) minus its Belarus subset (E. Europe/Central Asia/DR-TB + some India/Nigeria; bacteriological+genomic; CC0/DUA); **India National TB Prevalence Survey / IN-CXR (ICMR-NIRT)** (genuine Indian community pop, independent of DA/DB — but verify whether the public release is radiographic-only or micro-linked).
- **Culture-confirmed validation:** TB Portals confirmed cases + outreach for HIV-endemic African NAAT-confirmed data (CIDRZ Zambia, Sibanye South Africa, TREATS) — access-restricted, validation-only; the populations the model is currently blind to.
- **Avoid:** Pakistan Mendeley set (undocumented labels — weak training at most, never validation), standalone Belarus (leaks Qatar), DA/DB (leaks TBX11K), generic Kaggle re-mixes.

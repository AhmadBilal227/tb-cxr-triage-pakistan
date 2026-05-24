# TB Feature-Sharpening Blueprint (Pakistan/India focus)

> Consolidation of a 10-agent literature swarm (Waves 1+2, 2026-05-24). Direction (user): **do NOT
> domain-generalize/blur features for invariance — make them SHARP and SPECIFIC for active TB.**
> Sharp pathology features generalize because the pathology is real; the honest test is external
> LODO sensitivity + lesion-localized evidence, not the site-leak canary (which stays a *diagnostic*).

**Endpoint:** radiographic-TB-pattern (NOT bacteriologically-confirmed active TB). Research preview, not a device.

---

## 0. Corrections the swarm surfaced (act on these first)
1. **TBX11K "latent TB" is a misnomer.** True immunologic LTBI is radiographically silent (CXR sens ~15%). What TBX11K boxed as "latent" is **old/inactive/sequelae** pattern. → Rename `data/index_tbx_latent.csv` semantics to an **inactive-sequelae specificity probe**. It has duplicate rows: **169 unique images**, not 239 — use 169 as the probe denominator.
2. **TBX11K ships bounding boxes** (COCO xywh, on 512×512; the `bbox` column in our `data.csv` carries them for active-TB images, "none" otherwise). → Box-supervised localization is buildable. Supervise on **active-TB boxes only** (exclude latent/sequelae boxes — ethos).
3. **TorchXRayVision has no `Calcification` logit** (18 labels confirmed). The most specific *inactive* sign is absent from the 18-d block → keep Rad-DINO patch tokens in any active-vs-healed head.
4. **The patch grid is 8×8 (pooled from Rad-DINO's native 37×37).** 8×8 is coarse for small TB foci; if localization fidelity matters, the real fix is caching the **37×37** grid. Note as a known ceiling.

## 1. Population "don't-miss" priorities (Pakistan/India) — reshapes the model
- **Lower-zone & lower-lobe cavities** — the **diabetes** signature (Karachi: 74.6% of diabetic TB cavitated; 17% non-upper-zone). DM is highly prevalent in South Asia. **Do NOT hard-anchor upper-lobe.** [PMC3099518, PubMed 37095759]
- **Hilar/mediastinal lymphadenopathy + miliary** — HIV (83.8% atypical), pediatric (92% adenopathy), primary pattern. [PMC3359433, PubMed 1727316]
- **Subtle/minimal abnormality** — ~40–50% of prevalent TB is subclinical; CXR sens drops to ~56% there. Tune sensitivity-first. [PMC8326537]
- **Advanced/bilateral** disease (malnutrition, late presentation).
- Operating point: **WHO 90/70 floor**; deployed qXR/CAD4TB ~0.93 sens at **9–15% presumptive prevalence**, AUC 0.88–0.94, PPV 20–40%; **thresholds non-transferable → re-fit per site**. [Lancet Dig Health 2020/2021, PMC11339183]

---

## 2. Build sequence (ranked; each phase gated before the next)

### Phase A — Validation harness FIRST (so every change is measurable & honest)
New `training/localize.py` + `training/sanity.py`, plus additions to `train_tb.py`:
- **Localization faithfulness** vs TBX11K boxes rasterized to the grid: **pointing-game hit-rate + mIoU**, 1000-bootstrap CIs, vs a random-attention floor. Evidence map = the **ABMIL attention weights** (faithful — they are the pooling weights), NOT post-hoc Grad-CAM (Adebayo sanity checks; Saporta CheXlocalize). Grad-CAM only as a negative control.
- **Anti-shortcut (Adebayo):** (a) **label-randomization** — retrain on permuted `y`; LODO AUC must collapse <0.60 (else site/scanner shortcut — couples to the M12 site-leak canary); (b) **model-randomization** — attn Spearman vs a random head <0.3.
- **Stratified sensitivity** by radiographic-burden + zone *proxies* (small-box = subclinical proxy; lower-zone = atypical proxy) — disclosed proxies, no per-patient metadata in open data.
- **Deployment-prevalence PPV/NPV** at **9–15%** (presumptive band) in addition to 1–2%; lead with NPV for rule-out; tests-per-flagged-case.
- **Threshold-transfer penalty** = recalibrated − cold-start sensitivity (already have both arms).
- **`feature_sharpness` diagnostic tuple** logged per change: {loc_mIoU↑, loc_hit↑, site_leak_bacc↓/flat, perm_label_auc≈0.5, worst_fold_external_sens↑}. **Reject any change where sensitivity rises WITH rising site-leak or falling mIoU** (shortcut).

### Phase B — Box-supervised spatial TB-evidence head (the #1 lever)
Turn the 64 patch tokens into an 8×8 **evidence map** via a shared 1×1 scorer → **LSE-LBA pooling** (learnable sharpness, numerically stable, calibrated [0,1]) to the image logit. Keep CLS+TXRV as a **parallel fusion branch** with a learned log-odds blend.
- Loss = image BCE on all 13k + **masked per-cell BCE on active-TB boxes only** (don't supervise the map to all-zero on negatives — the SymFormer "excessive background" trap) + tiny total-variation prior.
- Cache: `grid_label[N,8,8]` + `has_box[N]` (rasterize active boxes; verify token row-major order matches `extract_features.py`).
- Two-stage schedule (warm on boxed positives, then all images). Per-cell `pos_weight`.
- Validate: localization hit/mIoU ↑ AND LODO sensitivity preserved/↑ AND Adebayo checks pass.

### Phase C — Lower-zone-floored zonal pooling + hilar/miliary channels
- Build a **mask-driven** zone matrix `zones[N,64,7]` (6 lung zones via per-hemithorax 1/3–2/3 lung-height cuts using the PSPNet lung mask we already run, + 1 hilar/mediastinal from the hilus/mediastinum channels). Cut by mask, not fixed grid rows (our crop varies).
- Per-zone attention pooling → **learned-but-floored zone gate** (`g_min≈0.25`, init mild upper prior) → **`logsumexp` soft-OR** so any single zone (incl. lower) can escalate the image logit alone. Counters the documented lower-zone-cavity 0%-sensitivity collapse (Sci Rep 2023).
- **HilarLymphHead** (hilar zone vs lung background + TXRV Enlarged-Cardiomediastinum/Mass logits) and **MiliaryHead** (low inter-zone variance + uniform nodularity + TXRV Nodule logit) — structurally motivated; **unsupervised on open data → no miliary/adenopathy metric claimed without held-out positives.**

### Phase D — Pathology grounding + auxiliary distillation
- Route the **18 TXRV logits** through a named-finding embedding into the head (equal footing, not diluted in the 1042-d block).
- **Auxiliary distillation head** re-predicts the 18 TXRV logits from the Rad-DINO representation (MSE-on-logits, T≈2; TXRV is multi-label) — anchors the representation in named findings vs dataset identity (cited to raise specificity / cut shortcuts). Re-run the site-leak canary to confirm distillation doesn't re-introduce site signal.

### Phase E — Inactive-sequelae (active-vs-healed) head + probe metric
- Aux head predicting an **inactive-sequelae score** trained on active (neg) vs the 169-image sequelae probe (pos). Gate: high inactive-score + weak active signal → **`activity: indeterminate` → escalate-not-clear** (never clear a flagged film as "old scar"; reactivation hides in scar).
- **Sequelae-probe metric:** SeqFPR = fraction of the 169 sequelae images scored ≥ threshold (Clopper-Pearson CI) + active-vs-sequelae AUROC (honest ceiling ~0.82–0.88; >0.9 ⇒ suspect leakage). Keep the probe out of TB-head training/eval.

---

## 3. Acceptance gate (every phase)
`npm run build` clean · tests green · **permuted-label LODO AUC < 0.60** · site-leak balanced-acc not rising · lead with **worst-fold external recalibrated sensitivity + Clopper-Pearson CI** · localization hit/mIoU with CI vs random floor · cold-start vs recalibrated both reported · PPV/NPV at 9–15% with radiographic-endpoint caveat · **no ≥90% sensitivity CLAIM without ≥150 held-out positives** · append dated CASE_STUDY.md entry with measured numbers.

## 4. Honest caveats (carry into CASE_STUDY.md)
- 8×8 grid is coarse → mIoU is a *relative* tracking number vs a random floor, not an absolute localization-accuracy claim. 37×37 native grid is the real fix.
- Miliary/lymphadenopathy heads are **unsupervised** on open data — evidence channels only, no claimed metric without labels.
- Subclinical/atypical sensitivity uses **radiographic-burden/zone proxies** (no symptom/HIV/DM metadata).
- Endpoint is **radiographic, not bacteriological**; activity is often **unprovable on one film** (destroyed lung) → escalate-not-clear; needs bacteriology/serial imaging.
- This sharpens features; it does **not** make the system medical-grade (needs microbiological labels + locked deployment + clinical validation + QMS).

## 5. Key sources
Radiographic signs/active-vs-healed: Radiology Assistant TB; PMC8743064; RadioGraphics 2017; PMC10323207 (DL activity); Radiology 2021 (10.1148/radiol.2021210063). Boxes/SymFormer: arXiv 2307.02848; CVPR 2020. Pooling: Li et al. 1803.07703 (LSE-LBA); Ilse et al. ABMIL; Kolesnikov 1603.06098 (GWRP). Zones: Sci Rep s41598-023-28079-0; PMC9845381. Pathology grounding/distillation: arXiv 1906.00768, 2411.08937, s41598-025-16669-z. Validation: Saporta CheXlocalize s42256-022-00536-x; Adebayo NeurIPS 2018. Population: PMC3099518 (DM), PMC3359433 (HIV), PubMed 1727316 (peds), PMC8326537 (subclinical). Deployment/thresholds: Lancet Dig Health 2020 (33328086)/2021; PMC11339183 (threshold non-transferability); WHO TPP.

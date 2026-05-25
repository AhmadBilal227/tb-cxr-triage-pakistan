# TB Data Sources — for the two unmet needs (activity labels + multimodal fusion)

Researched 2026-05-24. We currently have only TB-vs-normal *radiographic* labels (Montgomery/Shenzhen/
Qatar/TBX11K), no activity dimension, no clinical fields (CRP/symptoms/HIV/diabetes), no bacteriological
reference at the image level. This file ranks what could fill those gaps + how to get it.

**Blunt landscape:** the *open* releases are either tabular (CAD score + CRP + Xpert + culture, no pixels)
or radiographic-only (no clinical). The **image-level + clinical + bacteriological pairing almost always
needs a DUA or outreach.** So the multimodal/activity data we actually want lives behind 3 doors.

## Integrated / registered (P0.5, 2026-05-25)

### 0a. Mendeley Pakistani TB CXR (Kiran/Jabeen 2024, CC-BY-4.0) — EXTERNAL VALIDATION HOLDOUT
2,494 TB+ + 514 normal (3,008 total) from a Pakistani hospital. Single-site, preprocessed.
- DOI: https://doi.org/10.17632/8j2g3csprk.2
- On disk: `data/raw/Kiran:Jabeen/{TB Chest X-rays, Normal Chest X-rays}/`
- Source id in build_index: `mendeley_pk` (tagged `split='external_holdout'`)
- **NOT training data.** The external blind eval on this cohort (AUROC 0.781 external vs 0.922 LODO,
  specificity 0.675 — 1-in-3 normals false-flagged) established it as our only well-powered external
  TB+ set. It is the standing external eval the GO gate measures generalization against (a site the
  model never trained on). `build_index.py` routes it to `data/index_external_holdout.csv`, NOT
  `data/index.csv`. Re-confirm zero overlap with the training index via the dedup audit before each
  eval (see `data/dedup_audit.log`).

### 0b. PadChest TB-7-label union (BIMCV, Spain, ~150+ TB+) — REQUESTED via BIMCV form, BUILDER REGISTERED
TB-positive filter is the union of: tuberculosis + sequelae tuberculosis + cavitation + calcified
adenopathy + granuloma + calcified granuloma + apical pleural thickening (see PMC11843218 for the
published harvest protocol).
- Access: BIMCV form request at https://bimcv.cipf.es/bimcv-projects/padchest/
- Source id in build_index: `padchest_tb` (TRAINING source — atypical-TB richness addresses M24)
- When the credentialed download lands: place files under `data/raw/padchest_tb/{tb,normal}/` and
  re-run `build_index.py`. The builder is a no-op (returns 0 rows) until then.
- Status: DUA form to be submitted (manual, out-of-band). Expected delivery: days.

### 0c. NIH ChestX-ray14 `No_Finding` normals — TRAINING NEGATIVES (specificity-drift lever)
5,788 diverse US normals, already extracted in `data/features_nih14.npz` (cls/patches/txrv/zones
present, patches confirmed). Pulled as label-0 training rows via `train_tb.py --sources nih14_normals`.
Lever against the specificity drift the external Pakistani eval surfaced. NOTE: the raw
`data/raw/nih14/` dir stays EXCLUDED from `index.csv` (it is also the locked per-finding FPR stress
set); only the `No_Finding` slice of the pre-extracted feature cache feeds the negative pool.

## Do this week (open / quick)
1. **TB Portals (NIAID)** — TOP PICK. ~3,400+ imaging-annotated cases, frontal CXR + CT, **bacteriologically
   confirmed** (culture/smear/DST), **serial films** (5 wk–>2 yr → activity-by-change), linkable clinical/lab.
   Geography: ex-Soviet/global, **not South Asia**. Access: Data Access Request + DUA (institutional
   Authorizing Representative); TWO requests (clinical portal + imaging form); free; Aspera/API. → gives us
   the **WHO-comparable confirmed-label tier** + serial activity. (This is the source the user skipped earlier
   — it's the #1 ask.) https://tbportals.niaid.nih.gov/access-data
2. **TBX11K active/latent** — OPEN, already have it. A coarse active-vs-inactive activity head v0; relabel
   "latent"→"inactive/healed-appearing" (latent is radiographically silent — ethos). NOTE: our sequelae head
   already tested this → active-vs-sequelae AUROC **0.72** = the honest ceiling on these labels with frozen
   features. So TBX11K activity is ~exhausted; the real gain needs better labels/features.
3. **R2D2 TB Network cohort** — OPEN **tabular** (CAD4TB score + point-of-care **CRP** + Xpert-HR + culture +
   symptoms; Philippines/Vietnam/India/SA/Uganda, n=1,392, 22% culture+). Images via outreach only. → use the
   tabular triples to **prototype + validate the fusion head's logic** + cite the CRP evidence.
   https://doi.org/10.11588/data/KGVQ4T · https://academic.oup.com/cid/advance-article/doi/10.1093/cid/ciae549/7885258
4. **VinDr-CXR** — PhysioNet credentialed (CITI training, days). Vietnam, single "Tuberculosis" label (no
   activity), radiologist-only. Asian augmentation, not activity/multimodal. Start CITI now (unblocks several).
5. **IN-CXR (ICMR-NIRT, India survey)** — South-Asia PA CXR, normal/abnormal, on-request/DUA. The bacteriological
   + symptom linkage is NOT in the open image release (that's the gated microdata, #6). https://nirt.res.in/html/xray.html

## Gated / outreach (the South-Asia, image-level, multimodal prizes)
6. **India National TB Prevalence Survey microdata (ICMR-NIRT)** — CXR + symptom checklist + Xpert/culture at
   population scale; image+clinical linkage needs a formal MoU/application. Highest-value South-Asia multimodal target.
7. **Pakistani CAD4TB cohorts (Indus Health Network, Karachi/Peshawar/Balochistan)** — operational CXR + CAD +
   Xpert at screening prevalence, our exact geography. No public release → research collaboration/MoU (Indus = most plausible partner).
8. **Korean active-vs-inactive sets** (Boramae/SNU; bacteriology+CT+temporal-stability reference, ~95% acc) —
   **private**. Value = copy their active/inactive **label criteria** + cite as benchmark; don't request.
9. **MIMIC-CXR / PadChest** — no TB-activity or TB bacteriological linkage → skip for our needs (architecture reference only).

## Multimodal (CXR + CRP/symptoms) — the honest effect size
- **R2D2 (prospective, culture reference):** at fixed **90% sensitivity**, specificity rose CAD4TB-alone **70.3%
  → CRP→CAD 75.9% (+5.6 pts) → Xpert-HR→CAD 79.6% (+9.3 pts)**. A cheap inflammation marker (CRP) buys real,
  modest specificity at the safety-critical operating point.
- **Crucial caveat:** fusion gains **shrink as the image model gets stronger** — a multimodal transformer added
  +0.07 AUC on a weak model (0.70→0.77) but only **+0.01 on a strong one (0.83→0.84)**. Our LODO AUROC ~0.92 is
  strong, so **expect a single-digit specificity gain, not a transformation.**
- **Defensible framing:** multimodal "buys specificity (fewer confirmatory Xperts) at the WHO 90% sensitivity
  floor" — NOT "makes the screen accurate." And CRP is exactly the active-vs-scar tie-breaker our features lack.
  Sources: R2D2 (above) · Lesotho/SA CRP+CAD PMC11581699 · DeepGB-TB arxiv 2508.02741 · fusion-shrinks Radiology 10.1148/radiol.230806

## Bottom line / recommended sequence
- **Free + immediate:** start **CITI credentialing** (unblocks PhysioNet/VinDr); confirm TBX11K activity is exhausted (0.72).
- **#1 ask (DUA):** **TB Portals** DARs — the confirmed-label tier + serial-film activity, no South-Asia but the only obtainable bacteriological imaging.
- **Fusion prototype (open tabular):** R2D2 — build/validate the CXR+CRP+symptom fusion *logic* on real triples now; get images by outreach.
- **Outreach in parallel (the real South-Asia multimodal data):** ICMR-NIRT (India survey microdata), Indus Health Network (Pakistan), R2D2 network (images).
- **Reality:** image-level multimodal/activity TRAINING data needs DUA/outreach — confirming this is a *data-acquisition* effort, not a frozen-feature tweak.

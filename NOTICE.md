# NOTICE

## ⚠️ Not a medical device

This software and the accompanying model weights are a **research preview, not a medical device**,
and are not cleared or approved by any regulatory body (FDA, CE, etc.). They are **not** intended for
clinical use, diagnosis, or treatment decisions. Reported performance is against **radiographic**
labels on open datasets, **not bacteriologically confirmed** tuberculosis, and **does not transfer
unchanged to new sites**. A real screening deployment requires on-site re-validation and
re-calibration on locally labelled data. Use for research and education only.

## Upstream models & data — respect their licenses

The project's code and the trained heads in `public/models/` are released under the
[MIT License](./LICENSE). They are the authors' own artifacts, but they **depend on** and were
**trained with** third-party components that carry their own terms:

- **Rad-DINO** (Microsoft Research) — feature backbone, obtained separately under its own license.
- **TorchXRayVision** (Cohen et al.) — pathology feature extractor.
- **Datasets** (training): U.S. NLM Montgomery & Shenzhen, Qatar TB-CXR (Kaggle, CC-BY), TBX11K,
  NIH ChestX-ray14. **Evaluation-only / held-out:** the Mendeley Pakistani TB-CXR cohort
  (Kiran & Jabeen, 2024, CC-BY-4.0) — never trained on.

When using or redistributing this work, cite the upstream models and datasets and honour their
individual licenses. No raw patient images or dataset-derived feature caches are distributed in this
repository.

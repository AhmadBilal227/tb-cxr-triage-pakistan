# Bone-suppression weights (not committed)

Pretrained PyTorch checkpoint for the Gusarev 6-layer CNN (BatchNorm variant) used by
`training/bonesupp_probe.py`. The `.tar` itself is gitignored (large, reproducible).

## Reproduce

```bash
git clone --depth 1 https://github.com/danielnflam/Deep-Learning-Models-for-bone-suppression-in-chest-radiographs.git /tmp/gusarev
cp /tmp/gusarev/network_intermediate_4.tar training/bonesupp_weights/gusarev_network_intermediate_4.tar
cp /tmp/gusarev/LICENSE training/bonesupp_weights/GUSAREV_LICENSE
```

## Provenance

- Repo: `danielnflam/Deep-Learning-Models-for-bone-suppression-in-chest-radiographs`
- Architecture: `MultilayerCNN` (conv5x5 -> BatchNorm -> ReLU x5, channels 16/32/64/128/256, then
  1-channel `conv_output` + Sigmoid). Re-declared locally in `bonesupp_probe.py:_ConvBlock` /
  `GusarevBoneSupp` so loading does not pull in the upstream module's matplotlib/scipy top-level
  imports. Strict state-dict load against `network_intermediate_4.tar` succeeds.
- Training data: JSRT (Shiraishi et al.) + bone-suppressed pairs (Juhász et al.), 256x256 PNG.
- Pretrained metadata in the checkpoint: epochs_completed=400, reals_shown=72000.
- Original reference: Gusarev et al. 2017 (https://doi.org/10.1109/CIBCB.2017.8058543); BatchNorm
  variant per the upstream README.

## License

Upstream is GPL-3.0 (see `GUSAREV_LICENSE` after the copy step above). The checkpoint is used here
only for the offline research probe — not shipped to clients, not embedded in the production app.

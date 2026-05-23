"""CXR preprocessing shared by training feature-extraction and (later) the ONNX/browser path.

Phase-A: CLAHE contrast normalization + grayscale->RGB. Rad-DINO's own AutoImageProcessor
handles resize-518 + center-crop + mean/std normalization downstream, so this only needs to
deliver a contrast-normalized RGB image. Lung-field segmentation/cropping is a Phase-A.2
enhancement (see plan) layered on top of this same function.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_image(path: str) -> Image.Image:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:  # some formats (e.g. odd PNGs) fail cv2.imread
        img = np.array(Image.open(path).convert("L"))
    img = _CLAHE.apply(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

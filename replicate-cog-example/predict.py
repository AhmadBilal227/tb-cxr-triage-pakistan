"""
Minimal Cog predictor wrapping a public TB chest-X-ray CNN.

This returns the SAME shape the app's tolerant parser understands:
a list of {"label": str, "score": float}. The app's `parseTbProb` looks for a
label matching /tub|tb|positive/i, so emit a "Tuberculosis" class.

Replace `_load_model()` with your actual weights. The example below shows a
torchvision ResNet adapted to 2 classes (Normal / Tuberculosis) — point it at a
checkpoint trained on e.g. the Montgomery + Shenzhen + TBX11K sets.
"""

from typing import List, Dict
import io

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
from cog import BasePredictor, Input, Path

LABELS = ["Normal", "Tuberculosis"]


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model().to(self.device).eval()
        self.tf = T.Compose(
            [
                T.Grayscale(num_output_channels=3),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self) -> nn.Module:
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        # Load your trained weights here:
        # model.load_state_dict(torch.load("weights/tb_resnet50.pt", map_location="cpu"))
        return model

    def predict(
        self,
        image: Path = Input(description="Chest X-ray image"),
    ) -> List[Dict[str, float]]:
        img = Image.open(io.BytesIO(Path(image).read_bytes())).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].tolist()
        return [
            {"label": LABELS[i], "score": float(probs[i])} for i in range(len(LABELS))
        ]

"""CPU-only MobileNet v2 inference from explicit local artifacts."""

import hashlib
import io
import json

import torch
from PIL import Image

from galaxeye.contracts import CLASSES
from galaxeye.data import ROOT
from galaxeye.versions.v2_mobilenet.train import make_model, transform


class Predictor:
    def __init__(self) -> None:
        folder = ROOT / "artifacts" / "v2"
        self.manifest = json.loads((folder / "manifest.json").read_text())
        weights = folder / self.manifest["weights_file"]
        if (
            hashlib.sha256(weights.read_bytes()).hexdigest()
            != self.manifest["weights_sha256"]
        ):
            raise ValueError("v2 weights digest mismatch")
        if tuple(self.manifest["classes"]) != CLASSES:
            raise ValueError("v2 class manifest mismatch")
        self.model = make_model(pretrained=False)
        self.model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        self.preprocessing = transform(False)

    def predict(self, raw: bytes) -> list[float]:
        with Image.open(io.BytesIO(raw)) as image:
            tensor = self.preprocessing(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(tensor)
            return [float(value) for value in logits.softmax(dim=1)[0].tolist()]

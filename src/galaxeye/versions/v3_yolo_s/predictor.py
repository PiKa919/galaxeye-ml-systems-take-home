"""CPU-only YOLO v3 inference from explicit local artifacts."""

import hashlib
import io
import json

from PIL import Image
from ultralytics import YOLO

from galaxeye.contracts import CLASSES
from galaxeye.data import ROOT


class Predictor:
    def __init__(self) -> None:
        folder = ROOT / "artifacts" / "v3"
        self.manifest = json.loads((folder / "manifest.json").read_text())
        weights = folder / self.manifest["weights_file"]
        if (
            hashlib.sha256(weights.read_bytes()).hexdigest()
            != self.manifest["weights_sha256"]
        ):
            raise ValueError("v3 weights digest mismatch")
        if tuple(self.manifest["classes"]) != CLASSES:
            raise ValueError("v3 class manifest mismatch")
        self.model = YOLO(str(weights))
        names = tuple(self.model.names[i] for i in range(len(self.model.names)))
        if names != CLASSES:
            raise ValueError(f"v3 checkpoint class order mismatch: {names}")

    def predict(self, raw: bytes) -> list[float]:
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            result = self.model.predict(
                source=rgb,
                imgsz=self.manifest["image_size"],
                device="cpu",
                verbose=False,
            )[0]
        return [float(value) for value in result.probs.data.cpu().tolist()]

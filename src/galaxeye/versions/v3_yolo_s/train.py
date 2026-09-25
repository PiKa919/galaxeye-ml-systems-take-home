"""Fine-tune a local YOLO26s classification checkpoint on candidate tiles."""

import hashlib
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from galaxeye.contracts import CLASSES
from galaxeye.data import ROOT, SEED, SPLITS

ARTIFACTS = ROOT / "artifacts"
PRETRAINED = ARTIFACTS / "pretrained" / "yolo26s-cls.pt"
OUTPUT = ARTIFACTS / "v3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train() -> Path:
    split = json.loads((SPLITS / "manifest.json").read_text())
    PRETRAINED.parent.mkdir(parents=True, exist_ok=True)
    if not PRETRAINED.exists():
        YOLO("yolo26s-cls.pt")
        shutil.copy2(ROOT / "yolo26s-cls.pt", PRETRAINED)
    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(SPLITS),
        epochs=20,
        patience=5,
        imgsz=128,
        batch=32,
        device="mps",
        workers=0,
        seed=SEED,
        deterministic=True,
        amp=False,
        project=str(ARTIFACTS / "runs"),
        name="v3_yolo_s",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    best = ARTIFACTS / "runs" / "v3_yolo_s" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"YOLO training did not produce {best}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "model.pt"
    shutil.copy2(best, target)
    trained = YOLO(str(target))
    names = tuple(trained.names[i] for i in range(len(trained.names)))
    if names != CLASSES:
        raise ValueError(f"Unexpected YOLO class order: {names}")
    manifest = {
        "version": "v3",
        "architecture": "YOLO26s-cls",
        "weights_file": target.name,
        "weights_sha256": sha256(target),
        "base_weights_sha256": sha256(PRETRAINED),
        "classes": list(CLASSES),
        "split_digest": split["digest"],
        "image_size": 128,
        "preprocess_id": "ultralytics-classify-default-128",
        "review_score_threshold": 0.6,
        "review_threshold_status": "provisional; score is not calibrated confidence",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target

"""Deterministic, stratified data preparation."""

import hashlib
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image

from galaxeye.contracts import CLASSES

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    os.getenv("GALAXEYE_TRAINING_IMAGES", ROOT / "data" / "training_images")
).expanduser()
SPLITS = ROOT / "data" / "splits"
SEED = 20260924
ALLOCATION = {"train": 105, "test": 30, "val": 15}


def prepare_data() -> dict:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Candidate dataset missing: {SOURCE}")
    records = []
    for class_name in CLASSES:
        source_files = sorted((SOURCE / class_name).glob("*.png"))
        if len(source_files) != 150:
            raise ValueError(
                f"Expected 150 {class_name} tiles, found {len(source_files)}"
            )
        rng = random.Random(f"{SEED}:{class_name}")
        rng.shuffle(source_files)
        offset = 0
        for split, count in ALLOCATION.items():
            for source in source_files[offset : offset + count]:
                with Image.open(source) as image:
                    if (
                        image.format != "PNG"
                        or image.size != (64, 64)
                        or image.mode != "RGB"
                    ):
                        raise ValueError(f"Unexpected tile format: {source}")
                    image.verify()
                raw = source.read_bytes()
                records.append(
                    {
                        "split": split,
                        "class": class_name,
                        "source": str(source.relative_to(SOURCE)),
                        "file": f"{split}/{class_name}/{source.name}",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            offset += count

    hashes = [record["sha256"] for record in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate tile bytes across candidate splits")
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    manifest = {
        "seed": SEED,
        "allocation_per_class": ALLOCATION,
        "digest": digest,
        "records": records,
    }

    if SPLITS.exists():
        shutil.rmtree(SPLITS)
    for record in records:
        target = SPLITS / record["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / record["source"], target)
    (SPLITS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

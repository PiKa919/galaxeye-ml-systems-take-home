"""Model identity shared between the API and inference processes."""

import hashlib
import json

from galaxeye.contracts import CLASSES, MODEL_VERSIONS
from galaxeye.settings import ARTIFACTS


def load_manifest(version: str) -> dict:
    if version not in MODEL_VERSIONS:
        raise ValueError(f"Unknown model version: {version}")
    path = ARTIFACTS / version / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["version"] != version or tuple(manifest["classes"]) != CLASSES:
        raise ValueError(f"Bad {version} model manifest")
    weights = path.parent / manifest["weights_file"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError(f"{version} checkpoint hash mismatch")
    identity = {
        "version": version,
        "weights_sha256": manifest["weights_sha256"],
        "preprocess_id": manifest["preprocess_id"],
        "classes": manifest["classes"],
        "review_score_threshold": manifest["review_score_threshold"],
    }
    manifest["pipeline_digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return manifest

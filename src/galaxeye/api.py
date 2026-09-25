"""Public local API: accept tiles, call CPU inference, commit and query results."""

import hashlib
import json
import math
import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from galaxeye.contracts import CLASSES, ModelVersion
from galaxeye.files import reconcile_orphans, stage_png, stored_path, validate_png
from galaxeye.model_manifest import load_manifest
from galaxeye.settings import ARTIFACTS, INFERENCE_URL, MAX_TILE_BYTES
from galaxeye.storage import (
    get_existing,
    get_prediction,
    init_db,
    known_image_hashes,
    list_predictions,
    save_prediction,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    reconcile_orphans(known_image_hashes())
    app.state.manifests = {
        version: load_manifest(version) for version in ("v1", "v2", "v3")
    }
    yield


app = FastAPI(title="GalaxEye tile classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    try:
        worker = httpx.get(f"{INFERENCE_URL}/health", timeout=2)
        worker.raise_for_status()
        status = worker.json()
    except httpx.HTTPError:
        status = {"ready": False}
    return {"api_ready": True, "inference": status}


@app.get("/models")
def models() -> dict:
    return {
        version: {
            "architecture": manifest["architecture"],
            "pipeline_digest": manifest["pipeline_digest"],
            "classes": manifest["classes"],
            "review_score_threshold": manifest["review_score_threshold"],
        }
        for version, manifest in app.state.manifests.items()
    }


@app.get("/reports")
def reports() -> dict:
    current = {}
    for version in ("v1", "v2", "v3"):
        path = ARTIFACTS / version / "metrics.json"
        if path.is_file():
            report = json.loads(path.read_text())
            if (
                report["weights_sha256"]
                == app.state.manifests[version]["weights_sha256"]
            ):
                current[version] = report
    return current


@app.post("/classify")
def classify(
    file: UploadFile, version: Annotated[ModelVersion, Query()] = "v1"
) -> dict:
    raw = file.file.read(MAX_TILE_BYTES + 1)
    try:
        validate_png(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    manifest = app.state.manifests[version]
    try:
        sha256, relative_path = stage_png(raw)
        existing = get_existing(sha256, manifest["pipeline_digest"])
        if existing is not None:
            return existing
        response = httpx.post(
            f"{INFERENCE_URL}/infer",
            params={"version": version},
            content=raw,
            headers={"content-type": "image/png"},
            timeout=120,
        )
        response.raise_for_status()
        inference = response.json()
        scores = inference["scores"]
        if (
            inference["version"] != version
            or inference["pipeline_digest"] != manifest["pipeline_digest"]
            or tuple(inference["classes"]) != CLASSES
            or len(scores) != len(CLASSES)
            or not all(
                isinstance(score, (int, float)) and math.isfinite(score)
                for score in scores
            )
            or abs(sum(scores) - 1) > 0.05
        ):
            raise HTTPException(
                status_code=502, detail="Inference response identity or scores mismatch"
            )
        top_index = max(range(len(scores)), key=scores.__getitem__)
        review_required = scores[top_index] < manifest["review_score_threshold"]
        return save_prediction(
            sha256=sha256,
            relative_path=relative_path,
            byte_count=len(raw),
            version=version,
            pipeline_digest=manifest["pipeline_digest"],
            predicted_class=CLASSES[top_index],
            scores=dict(zip(CLASSES, scores)),
            review_required=review_required,
            review_reason="score below provisional threshold"
            if review_required
            else None,
            inference_ms=float(inference["inference_ms"]),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Local inference unavailable: {exc}"
        ) from exc
    except sqlite3.DatabaseError as exc:
        raise HTTPException(
            status_code=503, detail=f"Could not persist prediction: {exc}"
        ) from exc


@app.get("/predictions")
def predictions(
    version: ModelVersion | None = None,
    predicted_class: str | None = None,
    review_required: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict]:
    if predicted_class is not None and predicted_class not in CLASSES:
        raise HTTPException(status_code=422, detail="Unknown class")
    return list_predictions(
        version=version,
        predicted_class=predicted_class,
        review_required=review_required,
        limit=limit,
    )


@app.get("/predictions/{prediction_id}")
def prediction(prediction_id: int) -> dict:
    result = get_prediction(prediction_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result


@app.get("/images/{sha256}")
def image(sha256: str) -> FileResponse:
    try:
        path = stored_path(sha256)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise HTTPException(status_code=404, detail="Stored image not found")
    return FileResponse(path, media_type="image/png")

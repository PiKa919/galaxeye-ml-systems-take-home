"""Private local inference service; owns resident CPU models only."""

import math
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import Body, FastAPI, HTTPException, Query

from galaxeye.contracts import CLASSES, ModelVersion
from galaxeye.files import validate_png
from galaxeye.model_manifest import load_manifest
from galaxeye.versions.v1_yolo.predictor import Predictor as YoloPredictor
from galaxeye.versions.v2_mobilenet.predictor import Predictor as MobileNetPredictor
from galaxeye.versions.v3_yolo_s.predictor import Predictor as YoloSmallPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Explicit local paths make missing weights fail startup instead of downloading.
    torch.set_num_threads(4)
    app.state.predictors = {
        "v1": YoloPredictor(),
        "v2": MobileNetPredictor(),
        "v3": YoloSmallPredictor(),
    }
    app.state.manifests = {
        version: load_manifest(version) for version in ("v1", "v2", "v3")
    }
    app.state.lock = threading.Lock()
    yield


app = FastAPI(title="GalaxEye local inference", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "ready": True,
        "device": "cpu",
        "models": {
            version: manifest["pipeline_digest"]
            for version, manifest in app.state.manifests.items()
        },
    }


@app.post("/infer")
def infer(
    raw: Annotated[bytes, Body(media_type="image/png")],
    version: Annotated[ModelVersion, Query()],
) -> dict:
    try:
        validate_png(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with app.state.lock:
        start = time.perf_counter()
        scores = app.state.predictors[version].predict(raw)
        elapsed_ms = (time.perf_counter() - start) * 1000
    if len(scores) != len(CLASSES) or not all(math.isfinite(score) for score in scores):
        raise HTTPException(
            status_code=500, detail="Classifier returned invalid scores"
        )
    return {
        "version": version,
        "scores": scores,
        "classes": CLASSES,
        "pipeline_digest": app.state.manifests[version]["pipeline_digest"],
        "inference_ms": elapsed_ms,
    }

"""Paths and local service settings."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB_PATH = DATA / "results.sqlite3"
UPLOAD_ROOT = DATA / "uploads"
ARTIFACTS = ROOT / "artifacts"
INFERENCE_URL = os.getenv("GALAXEYE_INFERENCE_URL", "http://127.0.0.1:8001")
API_URL = os.getenv("GALAXEYE_API_URL", "http://127.0.0.1:8000")
MAX_TILE_BYTES = 2 * 1024 * 1024

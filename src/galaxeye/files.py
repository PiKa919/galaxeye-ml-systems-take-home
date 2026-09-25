"""Validate and store immutable original PNG tiles by content hash."""

import hashlib
import io
import os
import re
import tempfile
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from galaxeye.settings import MAX_TILE_BYTES, UPLOAD_ROOT

SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_png(raw: bytes) -> None:
    if not raw or len(raw) > MAX_TILE_BYTES:
        raise ValueError(f"PNG must be between 1 and {MAX_TILE_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.size != (64, 64) or image.mode != "RGB":
                raise ValueError("Expected a 64x64 RGB PNG tile")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid PNG tile") from exc


def stored_path(sha256: str) -> Path:
    if not SHA_PATTERN.fullmatch(sha256):
        raise ValueError("Invalid SHA-256")
    return UPLOAD_ROOT / sha256[:2] / f"{sha256}.png"


def stage_png(raw: bytes) -> tuple[str, str]:
    sha256 = hashlib.sha256(raw).hexdigest()
    target = stored_path(sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
            raise OSError(f"Stored tile hash mismatch: {target}")
    else:
        fd, temp_name = tempfile.mkstemp(
            prefix=".tile-", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return sha256, str(target.relative_to(UPLOAD_ROOT))


def reconcile_orphans(known_hashes: set[str], min_age_seconds: int = 3600) -> int:
    """Remove finalized files never committed to SQLite after in-flight work expires."""
    if not UPLOAD_ROOT.exists():
        return 0
    removed = 0
    cutoff = time.time() - min_age_seconds
    for path in UPLOAD_ROOT.glob("*/*.png"):
        if path.stem not in known_hashes and path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    return removed

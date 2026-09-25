"""SQLite result persistence and analyst queries."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from galaxeye.settings import DB_PATH


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(DB_PATH, timeout=5)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys=ON")
    try:
        yield database
        database.commit()
    except Exception:
        database.rollback()
        raise
    finally:
        database.close()


def init_db() -> None:
    with connect() as database:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                sha256 TEXT PRIMARY KEY,
                relative_path TEXT NOT NULL,
                byte_count INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                ingested_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_sha256 TEXT NOT NULL REFERENCES images(sha256),
                version TEXT NOT NULL,
                pipeline_digest TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                review_required INTEGER NOT NULL,
                review_reason TEXT,
                inference_ms REAL NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE(image_sha256, pipeline_digest)
            );
            CREATE INDEX IF NOT EXISTS prediction_filters
                ON predictions(version, predicted_class, review_required, created_at_utc);
            """
        )


def known_image_hashes() -> set[str]:
    with connect() as database:
        return {row[0] for row in database.execute("SELECT sha256 FROM images")}


def _prediction(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    result["scores"] = json.loads(result.pop("scores_json"))
    result["review_required"] = bool(result["review_required"])
    return result


def get_existing(sha256: str, pipeline_digest: str) -> dict | None:
    with connect() as database:
        row = database.execute(
            """SELECT p.*, i.relative_path FROM predictions p JOIN images i
               ON p.image_sha256=i.sha256
               WHERE p.image_sha256=? AND p.pipeline_digest=?""",
            (sha256, pipeline_digest),
        ).fetchone()
    return _prediction(row)


def save_prediction(
    *,
    sha256: str,
    relative_path: str,
    byte_count: int,
    version: str,
    pipeline_digest: str,
    predicted_class: str,
    scores: dict[str, float],
    review_required: bool,
    review_reason: str | None,
    inference_ms: float,
) -> dict:
    now = datetime.now(UTC).isoformat()
    with connect() as database:
        database.execute(
            """INSERT OR IGNORE INTO images VALUES (?, ?, ?, 64, 64, ?)""",
            (sha256, relative_path, byte_count, now),
        )
        database.execute(
            """INSERT OR IGNORE INTO predictions
               (image_sha256,version,pipeline_digest,predicted_class,scores_json,
                review_required,review_reason,inference_ms,created_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                sha256,
                version,
                pipeline_digest,
                predicted_class,
                json.dumps(scores, separators=(",", ":")),
                int(review_required),
                review_reason,
                inference_ms,
                now,
            ),
        )
    result = get_existing(sha256, pipeline_digest)
    if result is None:
        raise RuntimeError("Prediction commit was not readable")
    return result


def list_predictions(
    *,
    version: str | None = None,
    predicted_class: str | None = None,
    review_required: bool | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses = []
    arguments = []
    if version is not None:
        clauses.append("p.version=?")
        arguments.append(version)
    if predicted_class is not None:
        clauses.append("p.predicted_class=?")
        arguments.append(predicted_class)
    if review_required is not None:
        clauses.append("p.review_required=?")
        arguments.append(int(review_required))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    arguments.append(limit)
    with connect() as database:
        rows = database.execute(
            f"""SELECT p.*, i.relative_path FROM predictions p JOIN images i
                 ON p.image_sha256=i.sha256 {where}
                 ORDER BY p.id DESC LIMIT ?""",
            arguments,
        ).fetchall()
    return [_prediction(row) for row in rows]


def get_prediction(prediction_id: int) -> dict | None:
    with connect() as database:
        row = database.execute(
            """SELECT p.*, i.relative_path FROM predictions p JOIN images i
               ON p.image_sha256=i.sha256 WHERE p.id=?""",
            (prediction_id,),
        ).fetchone()
    return _prediction(row)

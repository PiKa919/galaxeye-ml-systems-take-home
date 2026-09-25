"""Frozen-split CPU evaluation for all versioned classifiers."""

import csv
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from galaxeye.contracts import CLASSES
from galaxeye.data import ROOT, SOURCE, SPLITS
from galaxeye.versions.v1_yolo.predictor import Predictor as YoloPredictor
from galaxeye.versions.v2_mobilenet.predictor import Predictor as MobileNetPredictor
from galaxeye.versions.v3_yolo_s.predictor import Predictor as YoloSmallPredictor


def candidate_rows(split: str) -> list[tuple[Path, str]]:
    manifest = json.loads((SPLITS / "manifest.json").read_text())
    return [
        (candidate_source(record["source"]), record["class"])
        for record in manifest["records"]
        if record["split"] == split
    ]


def candidate_source(source: str) -> Path:
    relative = Path(source)
    configured = SOURCE / relative
    if configured.is_file():
        return configured
    previous_manifest_path = ROOT / relative
    if previous_manifest_path.is_file():
        return previous_manifest_path
    return configured


def supplied_rows() -> list[tuple[Path, str]]:
    images = Path(
        os.getenv("GALAXEYE_EVAL_IMAGES", ROOT / "data" / "evaluation_images")
    ).expanduser()
    labels_path = Path(
        os.getenv("GALAXEYE_EVAL_LABELS", ROOT / "data" / "evaluation_labels.csv")
    ).expanduser()
    with labels_path.open(newline="") as stream:
        labels = list(csv.DictReader(stream))
    if len(labels) != 210 or any(row["true_label"] not in CLASSES for row in labels):
        raise ValueError("Unexpected supplied evaluation labels")
    return [
        (images / row["filename"], row["true_label"]) for row in labels
    ]


def run_split(predictor, rows: list[tuple[Path, str]]) -> tuple[dict, list[dict]]:
    observed = []
    latencies = []
    for path, true_label in rows:
        started = time.perf_counter()
        scores = predictor.predict(path.read_bytes())
        latencies.append((time.perf_counter() - started) * 1000)
        if len(scores) != len(CLASSES) or not all(
            math.isfinite(value) for value in scores
        ):
            raise ValueError(f"Invalid scores for {path}")
        top_index = max(range(len(scores)), key=scores.__getitem__)
        observed.append(
            {
                "image": path.name,
                "true_label": true_label,
                "predicted_label": CLASSES[top_index],
                "top_score": scores[top_index],
                "scores": dict(zip(CLASSES, scores)),
            }
        )
    y_true = [row["true_label"] for row in observed]
    y_pred = [row["predicted_label"] for row in observed]
    ordered_latencies = sorted(latencies)
    report = {
        "count": len(observed),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro"),
        "by_class": classification_report(
            y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "class_order": CLASSES,
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": ordered_latencies[
            min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
        ],
    }
    return report, observed


def choose_review_threshold(rows: list[dict]) -> dict:
    """Maximize validation coverage while keeping observed accepted error <= 10%."""
    candidates = [round(value / 100, 2) for value in range(50, 100, 5)]
    eligible = []
    for threshold in candidates:
        accepted = [row for row in rows if row["top_score"] >= threshold]
        if not accepted:
            continue
        error = sum(
            row["predicted_label"] != row["true_label"] for row in accepted
        ) / len(accepted)
        if error <= 0.10:
            eligible.append((len(accepted), -threshold, threshold, error))
    if eligible:
        count, _, threshold, error = max(eligible)
    else:
        count, threshold, error = 0, 1.0, None
    return {
        "threshold": threshold,
        "accepted_count": count,
        "validation_count": len(rows),
        "coverage": count / len(rows),
        "observed_error_among_accepted": error,
        "selection_rule": "maximum validation coverage with observed accepted error <= 10%; threshold grid 0.50..0.95",
    }


def evaluate_all() -> dict:
    torch.set_num_threads(4)
    split_manifest = json.loads((SPLITS / "manifest.json").read_text())
    outcomes = {}
    for version, predictor_type in (
        ("v1", YoloPredictor),
        ("v2", MobileNetPredictor),
        ("v3", YoloSmallPredictor),
    ):
        started = time.perf_counter()
        predictor = predictor_type()
        load_ms = (time.perf_counter() - started) * 1000
        validation, validation_rows = run_split(predictor, candidate_rows("val"))
        policy = choose_review_threshold(validation_rows)
        manifest_path = ROOT / "artifacts" / version / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["review_score_threshold"] = policy["threshold"]
        manifest["review_threshold_status"] = (
            "chosen on candidate validation only; not calibrated"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        # These sets are examined only after the checkpoint and policy are frozen.
        internal_test, test_rows = run_split(predictor, candidate_rows("test"))
        supplied_eval, supplied_predictions = run_split(predictor, supplied_rows())
        report = {
            "version": version,
            "architecture": manifest["architecture"],
            "split_digest": split_manifest["digest"],
            "weights_sha256": manifest["weights_sha256"],
            "platform": platform.platform(),
            "processor": platform.machine(),
            "torch_version": torch.__version__,
            "load_ms": load_ms,
            "rss_mb_after_evaluation": psutil.Process().memory_info().rss
            / (1024 * 1024),
            "validation": validation,
            "review_policy": policy,
            "internal_test": internal_test,
            "supplied_eval": supplied_eval,
            "caveats": [
                "Per-class estimates have only 15 validation and 30 test/evaluation images.",
                "No source-scene or geographic identifiers are available to prove independent scenes.",
                "Model scores are not calibrated probabilities of correctness.",
            ],
        }
        folder = manifest_path.parent
        (folder / "metrics.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        for name, prediction_rows in (
            ("internal_test", test_rows),
            ("supplied_eval", supplied_predictions),
        ):
            with (folder / f"{name}_predictions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "image",
                        "true_label",
                        "predicted_label",
                        "top_score",
                        *CLASSES,
                    ],
                )
                writer.writeheader()
                for row in prediction_rows:
                    writer.writerow(
                        {
                            "image": row["image"],
                            "true_label": row["true_label"],
                            "predicted_label": row["predicted_label"],
                            "top_score": row["top_score"],
                            **row["scores"],
                        }
                    )
        outcomes[version] = report
        print(
            f"{version}: validation={validation['accuracy']:.3f}, "
            f"internal_test={internal_test['accuracy']:.3f}, "
            f"supplied_eval={supplied_eval['accuracy']:.3f}, "
            f"CPU p50={internal_test['latency_ms_p50']:.1f} ms",
            flush=True,
        )
    return outcomes

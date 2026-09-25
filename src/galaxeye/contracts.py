"""Shared, explicit model and dataset identifiers."""

from typing import Literal

CLASSES = (
    "AnnualCrop",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake",
)

ModelVersion = Literal["v1", "v2", "v3"]
MODEL_VERSIONS = ("v1", "v2", "v3")

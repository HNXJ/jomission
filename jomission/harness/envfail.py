"""Classify failure text against manifests/harness/environment_failures.json signatures."""

from __future__ import annotations

import json
import re
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "manifests" / "harness" / "environment_failures.json"


def classify(text: str) -> list[str]:
    """Return ids of known environment failures whose `match` regex occurs in text."""
    failures = json.load(open(MANIFEST, encoding="utf-8"))["failures"]
    return [f["id"] for f in failures if f.get("match") and re.search(f["match"], text)]

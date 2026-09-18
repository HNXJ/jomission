"""Write guard for registered sealed artifacts.

A bare `pytest tests/` runs the tier-3 batteries, and those batteries rewrite `results/*.json`
when they run. On 2026-09-17 that overwrote 55 sealed result files in one command. The rule in
manifests/sealed_artifacts.json said "rerun only under an authorized lineage"; nothing enforced it.

This module enforces it: a path registered in manifests/sealed_artifacts.json cannot be opened for
writing, and the attempt raises before the file is touched. An authorized lineage names the paths
it is allowed to rewrite in JOMISSION_ALLOW_SEAL_WRITE (comma-separated, repo-relative, or the
single token "all" for a deliberate wholesale reseal). Unsealed and new paths are unaffected.
"""

from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "manifests" / "sealed_artifacts.json"
ENV = "JOMISSION_ALLOW_SEAL_WRITE"
WRITE_MODES = ("w", "a", "x", "+")

_original_open = None


class SealedArtifactWriteError(RuntimeError):
    """Raised instead of overwriting a registered sealed artifact."""


def sealed_paths() -> set[str]:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {e["path"] for e in reg["artifacts"]}


def authorized() -> set[str]:
    raw = os.environ.get(ENV, "").strip()
    return {p.strip().replace("\\", "/") for p in raw.split(",") if p.strip()}


def _relative(path) -> str | None:
    try:
        p = Path(os.fspath(path)).resolve()
    except (TypeError, ValueError, OSError):
        return None
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return None


def check_write(path) -> None:
    """Raise if `path` is a registered sealed artifact and this process is not authorized to rewrite it."""
    rel = _relative(path)
    if rel is None or rel not in sealed_paths():
        return
    allow = authorized()
    if "all" in allow or rel in allow:
        return
    raise SealedArtifactWriteError(
        f"{rel} is a registered sealed artifact (manifests/sealed_artifacts.json) and this process is "
        f"not authorized to rewrite it. Sealed evidence is rerun only under an authorized lineage: set "
        f"{ENV}={rel} for that run, or write to a new path.")


def _guarded_open(file, mode="r", *args, **kwargs):
    if isinstance(mode, str) and any(m in mode for m in WRITE_MODES):
        check_write(file)
    return _original_open(file, mode, *args, **kwargs)


def install() -> bool:
    """Route every open() through the guard. Idempotent; returns True when it installs."""
    global _original_open
    if _original_open is not None:
        return False
    _original_open = builtins.open
    builtins.open = _guarded_open
    io.open = _guarded_open
    return True


def uninstall() -> bool:
    global _original_open
    if _original_open is None:
        return False
    builtins.open = _original_open
    io.open = _original_open
    _original_open = None
    return True


def active() -> bool:
    return _original_open is not None

"""Atomic / incremental phase persistence for production workers.

Save-path engineering fix (NO scientific change): the previous terminal save was a
single monolithic `save()` bottleneck — Executors B/D crashed there after ~2.5 h of
compute, losing result.json / EvidenceRef / post observations.

Each phase is now persisted independently and durably BEFORE the next phase proceeds:

    simulate phase -> write temp -> fsync -> sha256 -> atomic os.replace -> manifest receipt

For every phase (pre, exposure, post, recovery) we write spikes/rate/field/H/Theta
per-trial and per-phase snapshots; a crash mid-run can therefore never corrupt or
lose an already-completed phase. This module only adds save machinery; it does not
touch any frozen scientific config/hash/dt/trial or estimators.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from datetime import datetime, timezone

PHASES: tuple[str, ...] = ("pre", "exposure", "post", "recovery")

SCHEMA = "jomission.phase_snapshot.v1"

DEFAULT_MANIFEST_NAME = "save_manifest.jsonl"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_dir(dir_path: pathlib.Path) -> None:
    """fsync a directory so the rename is durable (best-effort on platforms that allow it)."""
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_temp_and_replace_bytes(final_path: pathlib.Path, data: bytes) -> str:
    """Write bytes to a temp file in the same directory, fsync, hash, atomic os.replace.

    Returns the sha256 hexdigest of ``data``. On any failure the temp file is removed
    and the final artifact is left untouched (never a partial file).
    """
    final_path = pathlib.Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(final_path.parent), prefix=f"{final_path.name}.", suffix=".tmp"
    )
    tmp_path = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        digest = hashlib.sha256(data).hexdigest()
        os.replace(tmp_path, final_path)
        _fsync_dir(final_path.parent)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return digest


def atomic_write_json(final_path: pathlib.Path | str, payload: dict) -> str:
    """Atomically persist ``payload`` as JSON at ``final_path``; return sha256 hexdigest."""
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return _write_temp_and_replace_bytes(pathlib.Path(final_path), data)


def append_manifest_receipt(
    manifest_path: pathlib.Path | str, receipt: dict
) -> None:
    """Append ``receipt`` (one JSON object per line) atomically.

    The whole manifest is re-written via temp->fsync->replace so a crash never leaves
    a truncated or half-appended receipt line.
    """
    manifest_path = pathlib.Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if manifest_path.exists():
        text = manifest_path.read_text()
        if text.strip():
            lines = text.rstrip("\n").splitlines()
    lines.append(json.dumps(receipt, sort_keys=True))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    _write_temp_and_replace_bytes(manifest_path, data)


def read_manifest_receipts(manifest_path: pathlib.Path | str) -> list[dict]:
    """Parse a save manifest into a list of receipt dicts."""
    manifest_path = pathlib.Path(manifest_path)
    if not manifest_path.exists():
        return []
    receipts: list[dict] = []
    for line_no, line in enumerate(manifest_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            receipts.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed manifest line {line_no + 1}: {exc}") from exc
    return receipts


def persist_phase_snapshot(
    results_dir: pathlib.Path | str,
    phase: str,
    trial_range: tuple[int, int],
    payload: dict,
    manifest_path: pathlib.Path | str | None = None,
) -> dict:
    """Durably persist one phase's observable snapshot and commit a manifest receipt.

    Writes ``{phase}_snapshot.json`` atomically (temp -> fsync -> sha256 -> os.replace)
    then appends a receipt with phase, trial range, artifact hash and timestamp.
    Returns the receipt.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    start, end = trial_range
    if start < 0 or end < start:
        raise ValueError(f"invalid trial_range {trial_range}")
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = results_dir / f"{phase}_snapshot.json"
    snapshot = {
        "schema": SCHEMA,
        "phase": phase,
        "trial_range": {"start": int(start), "end": int(end)},
        "timestamp": _iso_now(),
        "payload": payload,
    }
    digest = atomic_write_json(snapshot_path, snapshot)
    receipt = {
        "schema": SCHEMA,
        "phase": phase,
        "trial_range": {"start": int(start), "end": int(end)},
        "artifact": snapshot_path.name,
        "sha256": digest,
        "bytes": snapshot_path.stat().st_size,
        "timestamp": _iso_now(),
    }
    manifest = (
        pathlib.Path(manifest_path)
        if manifest_path is not None
        else results_dir / DEFAULT_MANIFEST_NAME
    )
    append_manifest_receipt(manifest, receipt)
    return receipt


def append_trial_snapshot(
    results_dir: pathlib.Path | str,
    phase: str,
    trial_index: int,
    record: dict,
) -> None:
    """Durably append one trial's snapshot to ``{phase}_trials.jsonl`` (fsync per line).

    Incremental durability within a phase: each trial's observable record is durable
    before the next trial runs. The per-phase atomic snapshot remains the integrity
    authority at the phase boundary.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "schema": SCHEMA,
            "phase": phase,
            "trial_index": int(trial_index),
            "timestamp": _iso_now(),
            "record": record,
        },
        sort_keys=True,
    )
    with open(results_dir / f"{phase}_trials.jsonl", "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def verify_artifacts_readable(
    results_dir: pathlib.Path | str,
    phases: tuple[str, ...] | list[str] | None = None,
    manifest_path: pathlib.Path | str | None = None,
) -> dict:
    """Re-read and re-hash every phase snapshot; compare against manifest receipts.

    Read-only verification — never mutates artifacts.
    """
    results_dir = pathlib.Path(results_dir)
    phases = list(phases or PHASES)
    manifest = (
        pathlib.Path(manifest_path)
        if manifest_path is not None
        else results_dir / DEFAULT_MANIFEST_NAME
    )
    try:
        receipts = {r["phase"]: r for r in read_manifest_receipts(manifest)}
    except ValueError as exc:
        receipts = {}
        manifest_error = str(exc)
    else:
        manifest_error = None

    readable: dict[str, bool] = {}
    hashes_match: dict[str, bool] = {}
    issues: list[str] = []
    for phase in phases:
        path = results_dir / f"{phase}_snapshot.json"
        if not path.exists():
            readable[phase] = False
            hashes_match[phase] = False
            issues.append(f"{phase}: snapshot artifact missing")
            continue
        try:
            json.loads(path.read_text())  # re-read as JSON — proves readability
        except Exception as exc:  # noqa: BLE001
            readable[phase] = False
            hashes_match[phase] = False
            issues.append(f"{phase}: unreadable snapshot: {exc}")
            continue
        readable[phase] = True
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = receipts.get(phase, {}).get("sha256")
        if expected is None:
            hashes_match[phase] = False
            issues.append(f"{phase}: no manifest receipt")
        else:
            hashes_match[phase] = actual == expected
            if actual != expected:
                issues.append(f"{phase}: hash mismatch {actual[:16]} != {expected[:16]}")
    if manifest_error is not None:
        issues.append(f"manifest: {manifest_error}")
    all_readable = all(readable.values()) if readable else False
    all_hashes_match = all(hashes_match.values()) if hashes_match else False
    return {
        "readable": readable,
        "hashes_match": hashes_match,
        "all_readable": all_readable,
        "all_hashes_match": all_hashes_match,
        "verified": all_readable and all_hashes_match,
        "issues": issues,
    }


def completion_predicate(result: dict) -> dict:
    """Worker terminal predicate: COMPLETED only when all five gates hold.

    COMPLETED = simulation_terminal ∧ observations_persisted ∧ artifacts_readable
                ∧ hashes_verified ∧ manifest_committed

    ``result`` must carry ``terminal_predicate`` / ``total_steps`` (simulation
    terminality) and ``observations`` = {dir, phases, manifest} (persistence). Returns
    a dict with the five boolean flags plus ``all`` and ``details``.
    """
    tp = result.get("terminal_predicate") or {}
    simulation_terminal = bool(tp.get("terminated_by_schedule")) and int(
        result.get("total_steps", 0)
    ) > 0

    obs = result.get("observations") or {}
    phases = list(obs.get("phases") or [])
    results_dir = obs.get("dir")
    manifest = obs.get("manifest")

    observations_persisted = False
    artifacts_readable = False
    hashes_verified = False
    manifest_committed = False
    details: dict = {}

    if results_dir:
        results_dir = pathlib.Path(results_dir)
        manifest = (
            pathlib.Path(manifest)
            if manifest
            else results_dir / DEFAULT_MANIFEST_NAME
        )
        observations_persisted = bool(phases) and all(
            (results_dir / f"{phase}_snapshot.json").exists() for phase in phases
        )
        report = verify_artifacts_readable(
            results_dir, phases=phases, manifest_path=manifest
        )
        artifacts_readable = report["all_readable"]
        hashes_verified = report["all_hashes_match"]
        try:
            receipts = read_manifest_receipts(manifest)
        except ValueError:
            receipts = []
        manifest_committed = bool(receipts) and all(
            any(r.get("phase") == phase for r in receipts) for phase in phases
        )
        details = {"verify": report, "n_receipts": len(receipts)}

    flags = {
        "simulation_terminal": simulation_terminal,
        "observations_persisted": observations_persisted,
        "artifacts_readable": artifacts_readable,
        "hashes_verified": hashes_verified,
        "manifest_committed": manifest_committed,
    }
    return {**flags, "all": all(flags.values()), "details": details}

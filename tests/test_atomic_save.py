"""Atomic / incremental phase persistence — save-path engineering (NO scientific change)."""

import hashlib
import json

import pytest

import jomission.simulation.atomic_save as atomic_save
from jomission.simulation.atomic_save import (
    PHASES,
    append_manifest_receipt,
    append_trial_snapshot,
    atomic_write_json,
    completion_predicate,
    persist_phase_snapshot,
    read_manifest_receipts,
    verify_artifacts_readable,
)


def _sample_payload(phase: str) -> dict:
    return {
        "rate_hz_mean": 11.8,
        "area_rates_hz": {"V1": 12.1, "V4": 11.4, "FEF": 11.9, "PFC": 11.6},
        "h_summary": {"min": 1.028, "max": 1.087},
        "theta_summary": {"min": 0.01, "max": 10.0},
        "field_present": True,
    }


def test_atomic_rename_leaves_no_partial_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "phase_snapshot.json"

    def boom(src, dst, **kw):
        raise OSError("simulated crash mid-replace")

    monkeypatch.setattr(atomic_save.os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write_json(target, {"phase": "pre"})

    assert not target.exists(), "no partial artifact may exist after a failed replace"
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], f"temp files must be cleaned up, got {leftovers}"


def test_atomic_rename_preserves_prior_artifact_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "phase_snapshot.json"
    target.write_text('{"old": true}')
    old_bytes = target.read_bytes()

    def boom(src, dst, **kw):
        raise OSError("simulated crash mid-replace")

    monkeypatch.setattr(atomic_save.os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write_json(target, {"phase": "pre"})

    assert target.read_bytes() == old_bytes, "prior artifact must be untouched"


def test_per_phase_artifacts_independently_durable_and_hash_verified(tmp_path):
    manifest = tmp_path / "save_manifest.jsonl"

    # Persist pre + exposure only (simulating a run that crashes mid-lifecycle).
    rec_pre = persist_phase_snapshot(
        tmp_path, "pre", (0, 95), _sample_payload("pre"), manifest_path=manifest
    )
    rec_exp = persist_phase_snapshot(
        tmp_path, "exposure", (0, 259), _sample_payload("exposure"), manifest_path=manifest
    )

    # Each phase is independently durable: pre artifact exists, hashes, receipts.
    assert (tmp_path / "pre_snapshot.json").exists()
    assert (tmp_path / "exposure_snapshot.json").exists()
    assert not (tmp_path / "post_snapshot.json").exists()

    report = verify_artifacts_readable(tmp_path, phases=["pre", "exposure"], manifest_path=manifest)
    assert report["all_readable"] is True
    assert report["all_hashes_match"] is True
    assert report["verified"] is True

    # Receipt hashes match the artifacts on disk exactly.
    for phase, receipt in (("pre", rec_pre), ("exposure", rec_exp)):
        path = tmp_path / f"{phase}_snapshot.json"
        assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert receipt["trial_range"] == {"start": 0, "end": 95 if phase == "pre" else 259}
        assert receipt["artifact"] == f"{phase}_snapshot.json"

    receipts = read_manifest_receipts(manifest)
    assert [r["phase"] for r in receipts] == ["pre", "exposure"]
    assert len(receipts) == 2

    # Complete the lifecycle: post + recovery persist independently and all verify.
    persist_phase_snapshot(tmp_path, "post", (0, 95), _sample_payload("post"), manifest_path=manifest)
    persist_phase_snapshot(tmp_path, "recovery", (0, 5), _sample_payload("recovery"), manifest_path=manifest)
    full = verify_artifacts_readable(tmp_path, manifest_path=manifest)
    assert full["all_readable"] is True
    assert full["all_hashes_match"] is True
    assert sorted(r["phase"] for r in read_manifest_receipts(manifest)) == sorted(PHASES)


def test_hash_mismatch_detected(tmp_path):
    manifest = tmp_path / "save_manifest.jsonl"
    persist_phase_snapshot(tmp_path, "pre", (0, 95), _sample_payload("pre"), manifest_path=manifest)
    (tmp_path / "pre_snapshot.json").write_text('{"tampered": true}')

    report = verify_artifacts_readable(tmp_path, phases=["pre"], manifest_path=manifest)
    assert report["all_readable"] is True
    assert report["all_hashes_match"] is False
    assert report["verified"] is False
    assert any("hash mismatch" in i for i in report["issues"])


def test_per_trial_snapshot_appends_durably(tmp_path):
    for i in range(3):
        append_trial_snapshot(tmp_path, "exposure", i, {"rate_hz_mean": 11.8 + i, "trial": i})
    lines = (tmp_path / "exposure_trials.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["phase"] == "exposure"
        assert rec["trial_index"] == i


def test_manifest_receipt_append_is_atomic_and_ordered(tmp_path):
    manifest = tmp_path / "save_manifest.jsonl"
    append_manifest_receipt(manifest, {"phase": "pre", "sha256": "a" * 64, "trial_range": {"start": 0, "end": 95}})
    append_manifest_receipt(manifest, {"phase": "post", "sha256": "b" * 64, "trial_range": {"start": 0, "end": 95}})
    receipts = read_manifest_receipts(manifest)
    assert [r["phase"] for r in receipts] == ["pre", "post"]


def test_completion_predicate_false_when_phase_artifact_missing(tmp_path):
    manifest = tmp_path / "save_manifest.jsonl"
    result = {
        "terminal_predicate": {"terminated_by_schedule": True},
        "total_steps": 100,
        "observations": {
            "dir": str(tmp_path),
            "phases": ["pre", "exposure", "post", "recovery"],
            "manifest": str(manifest),
        },
    }

    # Only three of four phases persisted -> not COMPLETED.
    for phase in ("pre", "exposure", "post"):
        persist_phase_snapshot(tmp_path, phase, (0, 95), _sample_payload(phase), manifest_path=manifest)
    incomplete = completion_predicate(result)
    assert incomplete["all"] is False
    assert incomplete["simulation_terminal"] is True
    assert incomplete["observations_persisted"] is False
    assert incomplete["artifacts_readable"] is False
    assert incomplete["hashes_verified"] is False
    assert incomplete["manifest_committed"] is False


def test_completion_predicate_true_when_all_present(tmp_path):
    manifest = tmp_path / "save_manifest.jsonl"
    for idx, phase in enumerate(PHASES):
        end = 95 if phase in ("pre", "post") else (259 if phase == "exposure" else 5)
        persist_phase_snapshot(tmp_path, phase, (0, end), _sample_payload(phase), manifest_path=manifest)
    result = {
        "terminal_predicate": {"terminated_by_schedule": True},
        "total_steps": 100,
        "observations": {
            "dir": str(tmp_path),
            "phases": list(PHASES),
            "manifest": str(manifest),
        },
    }
    complete = completion_predicate(result)
    assert complete["all"] is True
    for flag in ("simulation_terminal", "observations_persisted", "artifacts_readable", "hashes_verified", "manifest_committed"):
        assert complete[flag] is True, flag


def test_completion_predicate_simulation_terminal_gate(tmp_path):
    manifest = tmp_path / "save_manifest.jsonl"
    for phase in PHASES:
        persist_phase_snapshot(tmp_path, phase, (0, 95), _sample_payload(phase), manifest_path=manifest)
    result = {
        "terminal_predicate": {"terminated_by_schedule": False},  # crashed early
        "total_steps": 100,
        "observations": {
            "dir": str(tmp_path),
            "phases": list(PHASES),
            "manifest": str(manifest),
        },
    }
    comp = completion_predicate(result)
    assert comp["simulation_terminal"] is False
    assert comp["all"] is False

"""Tier 0: drive additivity, estimator metadata, sealed artifacts, lineage schema, skills."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jomission.harness import validate as v
from jomission.harness.drive import DriveAdditivityError, check_additive_schedule

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"E": 300, "PV": 40, "SST": 32, "VIP": 28}


def _cls():
    return np.concatenate([np.full(n, c) for c, n in COUNTS.items()])


def test_drive_additivity_mean_controlled_shot_passes_and_tonic_copy_fails():
    spec = importlib.util.spec_from_file_location("_v21b", ROOT / "tests" / "test_v21b_operation.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    vec = json.load(open(ROOT / "results" / "v21_vectors.json"))["vectors"]["V_HIGH"]
    cls = _cls()
    drive = np.array([vec[c] for c in cls])
    chunks, _ = b.shot_chunks(drive, 1000.0, 2.0, 0.1, seed=5, n_chunk=1, steps=20000)
    sched = next(chunks)
    out = check_additive_schedule(drive, sched, cls, expected_schedule_mean={c: 0.0 for c in COUNTS})
    assert abs(out["E"]["executed_mean"] - vec["E"]) < 0.05 * vec["E"]
    with pytest.raises(DriveAdditivityError, match="duplicates tonic"):
        check_additive_schedule(drive, np.tile(drive, (100, 1)), cls)
    with pytest.raises(DriveAdditivityError, match="expected"):
        check_additive_schedule(drive, sched + 0.5 * drive, cls, expected_schedule_mean={c: 0.0 for c in COUNTS})
    check_additive_schedule(drive, np.zeros((10, len(cls))), cls)


def test_estimator_metadata_required():
    ok = {"estimator": {"bin_width_ms": 10, "aggregation": "trial mean", "baseline": "pre 200 ms", "sign": "peak"}}
    assert v.validate_estimator(ok) == []
    errs = v.validate_estimator({"estimator": {"bin_width_ms": 10, "sign": ""}})
    assert {e.split(".")[-1] for e in errs} == {"aggregation", "baseline", "sign"}


def test_sealed_artifacts_unchanged():
    reg = v.load("manifests/sealed_artifacts.json")
    assert v.validate_sealed_registry(reg) == []
    bad = copy.deepcopy(reg)
    bad["artifacts"][0]["blob"] = "0" * 40
    assert any("registry blob" in e for e in v.validate_sealed_registry(bad))
    import subprocess
    old_handoff = subprocess.run(["git", "rev-parse", "a488348:docs/HANDOFF_CURRENT.md"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
    edited = {"artifacts": [{"path": "docs/HANDOFF_CURRENT.md", "blob": old_handoff, "sealed_in": "a488348"}]}
    assert any("working tree differs" in e for e in v.validate_sealed_registry(edited))


def test_lineage_registry_valid_and_preexecution_sealed():
    reg = v.load("manifests/lineage_registry.json")
    for rec in reg["lineages"]:
        assert v.validate_lineage(rec) == [], rec["lineage_id"]


def test_lineage_validator_rejects_defects():
    rec = copy.deepcopy(v.load("manifests/lineage_registry.json")["lineages"][2])
    rec["preexecution_spec"]["commit"], rec["commit"] = rec["commit"], rec["preexecution_spec"]["commit"]
    assert any("does not strictly precede" in e for e in v.validate_lineage(rec))
    rec = copy.deepcopy(v.load("manifests/lineage_registry.json")["lineages"][2])
    rec["observations"].append({"claim": "promoted", "evidence_class": "INFERRED"})
    rec["observations"].append({"claim": "no receipt", "evidence_class": "OBSERVED"})
    rec["inferred"].append({"claim": "x", "evidence_class": "OBSERVED"})
    rec["observations"].append({"claim": "peak", "evidence_class": "OBSERVED",
                                "receipt": "results/v21b_lineage.json", "kind": "transient", "estimator": {}})
    rec["qualification"].pop("realized")
    rec["preexecution_spec_backup"] = rec.pop("preexecution_spec")
    errs = v.validate_lineage(rec)
    assert any("missing preexecution_spec" in e for e in errs)
    rec["preexecution_spec"] = rec.pop("preexecution_spec_backup")
    errs = v.validate_lineage(rec)
    for needle in ("labelled INFERRED", "without resolvable receipt", "labelled OBSERVED",
                   "estimator.bin_width_ms", "missing stage realized"):
        assert any(needle in e for e in errs), (needle, errs)
    rec3 = copy.deepcopy(rec)
    rec3.pop("qualification")
    assert any("without configured/realized/executed/effective" in e for e in v.validate_lineage(rec3))
    rec2 = copy.deepcopy(rec)
    rec2["preexecution_spec"] = None
    assert any("without pre-execution spec" in e for e in v.validate_lineage(rec2))


def test_lineage_observation_checks_value_match():
    rec = next(r for r in v.load("manifests/lineage_registry.json")["lineages"]
               if r["lineage_id"] == "SCI-WS-OSC-2")
    obs = next(o for o in rec["observations"] if "check" in o)
    assert v._observation_check(rec, obs) == []

    planted = copy.deepcopy(obs)
    planted["check"] = dict(obs["check"], expected=obs["check"]["expected"] + 1.0)
    assert any("contradicts expected" in e for e in v._observation_check(rec, planted))

    bad_ptr = dict(obs["check"], json_pointer="/no/such/path")
    assert any("does not resolve" in e for e in v._observation_check(rec, {"check": bad_ptr}))

    unreadable = dict(obs["check"], path="results/does_not_exist.json")
    assert any("unreadable" in e for e in v._observation_check(rec, {"check": unreadable}))

    assert v._observation_check(rec, {"claim": "x"}) == []
    incomplete = {"check": {"path": "a", "json_pointer": "/b", "expected": 1.0}}
    assert any("check missing" in e for e in v._observation_check(rec, incomplete))


def test_project_skills_valid(tmp_path):
    assert v.validate_skills() == []
    sk = tmp_path / "bad-skill"
    (sk / "references").mkdir(parents=True)
    (sk / "SKILL.md").write_text("---\nname: other\ndescription: short\n---\nsee (references/x.md) at 07d3966\n")
    errs = v.validate_skills(tmp_path)
    for needle in ("name mismatch", "too short", "project state", "missing reference"):
        assert any(needle in e for e in errs), (needle, errs)


def test_environment_failure_classifier():
    from jomission.harness.envfail import classify
    text = "PermissionError: [WinError 32] The process cannot access the file because it is being used by another process"
    assert classify(text) == ["ENV-WIN-FILELOCK"]
    assert classify("AssertionError: rate 12.0 > 10.0") == []


# Result vs test semantics: a battery test checks that the procedure ran and wrote a
# verdict; a scientific FAIL must not turn pytest red. Allowlisted asserts are
# technical preconditions, not scientific verdicts.
PASS_ASSERT_ALLOWLIST = {
    "tests/test_t2_readiness.py": "numerical stability precondition for exposure (finite state), not a scientific gate",
    "tests/test_v2_operation_prospective.py": "known-answer verdict label on synthetic windows; runs no simulation and reads no result verdict",
    "tests/test_regime_atlas.py": "atlas-integrity, not a gate: it asserts the rendered page reproduces the sealed pass value, whatever that value is. A scientific FAIL keeps it green; a rendering that disagreed with the seal would not",
}


def test_scientific_verdicts_do_not_fail_pytest():
    import re
    pat = re.compile(r"assert[^#\n]*(\[[\"']pass[\"']\]|cell_pass|==\s*[\"'][A-Z0-9_]+_PASS[\"'])")
    offenders = []
    for f in sorted((ROOT / "tests").glob("*.py")):
        rel = f.relative_to(ROOT).as_posix()
        if rel in PASS_ASSERT_ALLOWLIST or rel == "tests/test_harness_guards.py":
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{rel}:{n}")
    assert offenders == [], offenders
    assert pat.search('    assert res["pass"]') and pat.search('assert verdict == "V2_LOCAL_OPERATION_PASS"')


def test_new_run_continuation_call_sites_use_drive_guard():
    import re
    import subprocess
    hist = {e["path"]: e["blob"] for e in v.load("manifests/harness/historical_callsites.json")["run_continuation"]}
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    call = re.compile(r"run_continuation\(")
    offenders = []
    for rel in files:
        f = ROOT / rel
        if not f.is_file() or rel.startswith("jomission/harness/"):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if not call.search(text) or "jomission.harness.drive" in text:
            continue
        blob = subprocess.run(["git", "hash-object", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if hist.get(rel) != blob:
            offenders.append(rel)
    assert offenders == [], f"new or edited run_continuation call sites must use jomission.harness.drive: {offenders}"

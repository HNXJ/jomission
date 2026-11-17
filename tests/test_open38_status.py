"""OPEN-38 status source on synthetic receipts: a launched or unfinished run is never PASS; STOP is BLOCKED; a stale
log is UNKNOWN; PASS needs DONE plus every named checker receipt."""
import importlib.util
import json
import os
import time
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "st", Path(__file__).resolve().parents[1] / "scripts" / "open38_status.py")
st = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(st)

T = "2026-11-17T00:00:00Z"


def _log(repo: Path, name: str, lines: list[str], age_min: float = 0.0):
    p = repo / st.R / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{T} {x}\n" for x in lines), encoding="utf-8")
    t = time.time() - age_min * 60
    os.utime(p, (t, t))


def _rec(repo: Path, name: str, ok: bool):
    (repo / st.R).mkdir(parents=True, exist_ok=True)
    (repo / st.R / name).write_text(json.dumps({"pass": ok}), encoding="utf-8")


SMOKE_HEAD = ["mode smoke", "HEAD abc", "construction invariance: PASS", "start o38s_v1_n4000_s1_T8 L exec=x"]


def test_launched_run_is_running_never_pass(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD)
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "RUNNING" and s["runs_exit0"] == 0


def test_stale_running_log_is_unknown(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD, age_min=45)
    assert st.stage2(tmp_path, 30)["status"] == "UNKNOWN"


def test_stop_is_blocked_and_attempts_listed(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", ["mode smoke", "HEAD abc", "construction invariance: FAIL",
                                         "STOP: construction invariance failed; no run started"])
    _log(tmp_path, "o38s_s2_attempt1_stage.log", ["mode smoke", "HEAD old", "construction invariance: FAIL", "STOP: x"])
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "BLOCKED" and s["phases"][0]["status"] == "BLOCKED" and "STOP" in s["reason"]
    assert s["attempts"] == [{"attempt": 1, "mode": "smoke", "head": "old", "construction": "FAIL", "stop": "STOP: x"}]


def test_smoke_pass_without_rest_is_blocked_not_pass(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=0", "smoke integrity: PASS", "DONE smoke"])
    s = st.stage2(tmp_path, 30)
    assert s["phases"][0]["status"] == "PASS" and s["status"] == "BLOCKED" and s["runs_exit0"] == 1


def test_rest_pass_needs_integrity_receipt(tmp_path):
    lines = SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=0", "smoke integrity: PASS", "DONE smoke", "mode rest", "HEAD abc",
                          "start o38s_mt_n4000_s1_T8 L exec=y", "o38s_mt_n4000_s1_T8 rc=0", "stage-2 integrity: PASS",
                          "DONE rest"]
    _log(tmp_path, "o38s_s2_stage.log", lines)
    assert st.stage2(tmp_path, 30)["status"] == "UNKNOWN"
    _rec(tmp_path, "o38s_s2_integrity.json", True)
    assert st.stage2(tmp_path, 30)["status"] == "PASS"
    _rec(tmp_path, "o38s_s2_integrity.json", False)
    assert st.stage2(tmp_path, 30)["status"] != "PASS"


def test_nonzero_exit_fails_and_missing_exit_is_unknown(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=1", "smoke integrity: FAIL", "DONE smoke"])
    assert st.stage2(tmp_path, 30)["status"] == "FAIL"
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["smoke integrity: PASS", "DONE smoke"])
    assert st.stage2(tmp_path, 30)["phases"][0]["status"] == "UNKNOWN"


def test_stage1_pass_needs_every_receipt(tmp_path):
    _log(tmp_path, "o38s_pc_stage.log", ["HEAD abc", "start o38s_pc_ref_n4000_s1_T8 L", "o38s_pc_ref_n4000_s1_T8 rc=0",
                                         "stage check: PASS", "DONE"])
    for f in ("o38s_pc_stage1_check.json", "o38s_pc_integrity.json", "o38s_pc_construction.json"):
        _rec(tmp_path, f, True)
    assert st.stage1(tmp_path, 30)["status"] == "UNKNOWN"  # ruling-B construction receipt missing
    _rec(tmp_path, "o38s_pc_construction_rulingB.json", True)
    assert st.stage1(tmp_path, 30)["status"] == "PASS"
    _rec(tmp_path, "o38s_pc_integrity.json", False)
    assert st.stage1(tmp_path, 30)["status"] == "FAIL"

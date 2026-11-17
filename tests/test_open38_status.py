"""OPEN-38 status source on synthetic receipts: a launched or unfinished run is never PASS; STOP is BLOCKED; a stale
log is UNKNOWN; PASS needs DONE plus every named checker receipt, all from one source revision (or a verified
compatibility entry); the public view shows a status only from committed receipts."""
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "st", Path(__file__).resolve().parents[1] / "scripts" / "open38_status.py")
st = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(st)

T = "2026-11-17T00:00:00Z"
EXE = "a" * 40


def _log(repo: Path, name: str, lines: list[str], age_min: float = 0.0):
    p = repo / st.R / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{T} {x}\n" for x in lines), encoding="utf-8")
    t = time.time() - age_min * 60
    os.utime(p, (t, t))


def _rec(repo: Path, name: str, ok: bool, **kw):
    (repo / st.R).mkdir(parents=True, exist_ok=True)
    (repo / st.R / name).write_text(json.dumps({"pass": ok, **kw}), encoding="utf-8")


def _git(repo: Path, *a) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True,
                          stdin=subprocess.DEVNULL).stdout.strip()


def _repo(tmp: Path) -> Path:
    _git(tmp, "init", "-q")
    _git(tmp, "config", "user.email", "t@t")
    _git(tmp, "config", "user.name", "t")
    (tmp / "jomission").mkdir()
    (tmp / "jomission" / "model.py").write_text("x = 1\n")
    _git(tmp, "add", "-A")
    _git(tmp, "commit", "-qm", "exe")
    return tmp


def _commit_all(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


SMOKE_HEAD = ["mode smoke", f"HEAD {EXE}", "construction invariance: PASS", "start o38s_v1_n4000_s1_T8 L exec=x"]


def _stage1(repo: Path, exe: str, integ_head: str | None = None, run_sha: str | None = None):
    _log(repo, "o38s_pc_stage.log", [f"HEAD {exe}", "start o38s_pc_ref_n4000_s1_T8 L", "o38s_pc_ref_n4000_s1_T8 rc=0",
                                      "stage check: PASS", "DONE"])
    _rec(repo, "o38s_pc_stage1_check.json", True, runs={"ref": {"1": {"git_sha": run_sha or exe}}})
    _rec(repo, "o38s_pc_integrity.json", True, head=integ_head or exe)
    _rec(repo, "o38s_pc_construction.json", True)
    _rec(repo, "o38s_pc_construction_rulingB.json", True)


def _compat(repo: Path, exe: str, src: str, paths=("jomission",)):
    (repo / st.R / "o38s_receipt_compat.json").write_text(json.dumps({"schema": "o38s_compat/1", "entries": [
        {"receipt": f"{st.R}/o38s_pc_construction_rulingB.json", "execution_sha": exe, "source_sha": src,
         "identical_paths": list(paths), "why": "test"}]}), encoding="utf-8")


def test_launched_run_is_running_never_pass(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD)
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "RUNNING" and s["runs_exit0"] == 0 and s["public_status"] == "UNKNOWN"


def test_stale_running_log_is_unknown(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD, age_min=45)
    assert st.stage2(tmp_path, 30)["status"] == "UNKNOWN"


def test_stop_is_blocked_and_attempts_listed(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", ["mode smoke", f"HEAD {EXE}", "construction invariance: FAIL",
                                         "STOP: construction invariance failed; no run started"])
    _log(tmp_path, "o38s_s2_attempt1_stage.log", ["mode smoke", "HEAD old", "construction invariance: FAIL", "STOP: x"])
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "BLOCKED" and s["phases"][0]["status"] == "BLOCKED" and "STOP" in s["reason"]
    assert s["attempts"] == [{"attempt": 1, "mode": "smoke", "head": "old", "construction": "FAIL", "stop": "STOP: x"}]


def test_smoke_pass_without_rest_is_blocked_not_pass(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=0", "smoke integrity: PASS", "DONE smoke"])
    s = st.stage2(tmp_path, 30)
    assert s["phases"][0]["status"] == "PASS" and s["status"] == "BLOCKED" and s["runs_exit0"] == 1


REST = SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=0", "smoke integrity: PASS", "DONE smoke", "mode rest", f"HEAD {EXE}",
                     "start o38s_mt_n4000_s1_T8 L exec=y", "o38s_mt_n4000_s1_T8 rc=0", "stage-2 integrity: PASS",
                     "DONE rest"]


def test_rest_pass_needs_integrity_receipt(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", REST)
    assert st.stage2(tmp_path, 30)["status"] == "UNKNOWN"
    _rec(tmp_path, "o38s_s2_integrity.json", True, head=EXE)
    assert st.stage2(tmp_path, 30)["status"] == "PASS"
    _rec(tmp_path, "o38s_s2_integrity.json", False, head=EXE)
    assert st.stage2(tmp_path, 30)["status"] != "PASS"


def test_stage2_receipts_from_another_revision_are_not_pass(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", REST)
    _rec(tmp_path, "o38s_s2_smoke_integrity.json", True, head=EXE)
    _rec(tmp_path, "o38s_s2_integrity.json", True, head="b" * 40)  # integrity re-run at another commit
    s = st.stage2(tmp_path, 30)
    assert s["status"] == "UNKNOWN" and "revisions differ" in s["reason"]


def test_nonzero_exit_fails_and_missing_exit_is_unknown(tmp_path):
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["o38s_v1_n4000_s1_T8 rc=1", "smoke integrity: FAIL", "DONE smoke"])
    assert st.stage2(tmp_path, 30)["status"] == "FAIL"
    _log(tmp_path, "o38s_s2_stage.log", SMOKE_HEAD + ["smoke integrity: PASS", "DONE smoke"])
    assert st.stage2(tmp_path, 30)["phases"][0]["status"] == "UNKNOWN"


def test_stage1_pass_needs_every_receipt(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _stage1(repo, exe)
    (repo / st.R / "o38s_pc_construction_rulingB.json").unlink()
    assert st.stage1(repo, 30)["status"] == "UNKNOWN"
    _rec(repo, "o38s_pc_construction_rulingB.json", True)
    _compat(repo, exe, exe)
    assert st.stage1(repo, 30)["status"] == "PASS"
    _rec(repo, "o38s_pc_integrity.json", False, head=exe)
    assert st.stage1(repo, 30)["status"] == "FAIL"


def test_stage1_mixed_revisions_are_not_pass(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _stage1(repo, exe, integ_head="c" * 40)
    _compat(repo, exe, exe)
    s = st.stage1(repo, 30)
    assert s["status"] == "UNKNOWN" and "revisions differ" in s["reason"]
    _stage1(repo, exe, run_sha="d" * 40)
    assert st.stage1(repo, 30)["status"] == "UNKNOWN"


def test_foreign_receipt_needs_verified_compat_entry(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "check.py").write_text("level = 2\n")  # the re-check's own code changed
    src_same = _commit_all(repo, "checker only")
    _stage1(repo, exe)
    s = st.stage1(repo, 30)
    assert s["status"] == "UNKNOWN" and "no compatibility entry" in s["reason"]
    _compat(repo, exe, src_same)
    assert st.stage1(repo, 30)["status"] == "PASS"
    (repo / "jomission" / "model.py").write_text("x = 2\n")  # model code changed after execution
    src_diff = _commit_all(repo, "model change")
    _compat(repo, exe, src_diff)
    s = st.stage1(repo, 30)
    assert s["status"] == "UNKNOWN" and "declared paths differ" in s["reason"]


def test_public_status_needs_committed_receipts(tmp_path):
    repo = _repo(tmp_path)
    exe = _git(repo, "rev-parse", "HEAD")
    _stage1(repo, exe)
    _compat(repo, exe, exe)
    s = st.stage1(repo, 30)
    assert s["status"] == "PASS" and s["public_status"] == "UNKNOWN"  # receipts untracked
    _commit_all(repo, "receipts")
    assert st.stage1(repo, 30)["public_status"] == "PASS"
    _rec(repo, "o38s_pc_integrity.json", True, head=exe, note="edited")  # modified after commit
    assert st.stage1(repo, 30)["public_status"] == "UNKNOWN"

"""Tests for scripts/capture_env.py on a throwaway git repo. The script resolves ROOT from its own path, so each test
copies it into tmp_path/repo/scripts/ and runs the copy there. OUT lives outside the repo. Each test asserts its
fixture precondition."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capture_env.py"
GIT_ID = ["-c", "user.name=capture-env-test", "-c", "user.email=capture-env-test@example.invalid"]
KEYS = {
    "python", "platform", "machine", "jax_version", "jaxlib_version", "numpy_version", "scipy_version",
    "jaxfne_version", "git_sha", "git_dirty", "generated_utc",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / "capture_env.py")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "README.md", "scripts/capture_env.py")
    subprocess.run(["git", "-C", str(repo), *GIT_ID, "commit", "-q", "-m", "fixture"], check=True, capture_output=True)
    return repo


def run_capture(repo: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(repo / "scripts" / "capture_env.py"), str(out)],
                          capture_output=True, text=True, cwd=str(repo))


def test_clean_tree_reports_not_dirty_and_head_sha(tmp_path):
    repo = make_repo(tmp_path)
    head = git(repo, "rev-parse", "HEAD").strip()
    assert len(head) == 40 and git(repo, "status", "--porcelain").strip() == "", "fixture must be a clean committed tree"
    out = tmp_path / "out" / "x.json"
    assert not out.is_relative_to(repo), "OUT must sit outside the repo so it does not dirty the tree"
    r = run_capture(repo, out)
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["git_dirty"] is False, rec
    assert rec["git_sha"] == head, (rec["git_sha"], head)


def test_untracked_file_reports_dirty(tmp_path):
    repo = make_repo(tmp_path)
    stray = repo / "stray.txt"
    stray.write_text("untracked\n", encoding="utf-8")
    assert stray.exists() and git(repo, "status", "--porcelain").startswith("?? stray.txt"), \
        "fixture: untracked file must show in git status"
    out = tmp_path / "out" / "x.json"
    r = run_capture(repo, out)
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["git_dirty"] is True, rec


def test_modified_tracked_file_reports_dirty(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    assert git(repo, "status", "--porcelain").startswith(" M README.md"), "fixture: tracked change must show in status"
    out = tmp_path / "out" / "x.json"
    r = run_capture(repo, out)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["git_dirty"] is True


def test_existing_out_is_refused_and_left_unchanged(tmp_path):
    repo = make_repo(tmp_path)
    out = tmp_path / "out" / "x.json"
    out.parent.mkdir(parents=True)
    sentinel = b"sentinel: must survive\n"
    out.write_bytes(sentinel)
    assert out.exists() and out.read_bytes() == sentinel, "fixture: OUT must already exist with sentinel bytes"
    r = run_capture(repo, out)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "REFUSING" in r.stderr, r.stderr
    assert out.read_bytes() == sentinel, "an existing OUT must not be overwritten"


def test_record_has_all_keys(tmp_path):
    repo = make_repo(tmp_path)
    out = tmp_path / "out" / "x.json"
    r = run_capture(repo, out)
    assert r.returncode == 0, r.stderr
    assert out.exists(), "fixture: first run must write OUT"
    rec = json.loads(out.read_text(encoding="utf-8"))
    missing = KEYS - set(rec)
    assert not missing, f"missing keys: {sorted(missing)}"

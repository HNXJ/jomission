"""OPEN-38 provenance manifest: covers every tracked o38s result JSON, matches committed bytes, and fails on a
changed file or an unknown name. Registration only: no file in it is sealed."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "open38_provenance.py"
SPEC = importlib.util.spec_from_file_location("o38p", SCRIPT)
o38p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(o38p)
HELD = json.loads((ROOT / o38p.OUT).read_text(encoding="utf-8"))


def test_manifest_matches_its_source_revision_and_the_tree():
    r = subprocess.run([sys.executable, "-B", str(SCRIPT), "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "O38S_PROVENANCE PASS" in r.stdout, r.stdout + r.stderr


def test_every_tracked_o38s_json_is_listed_once():
    tracked = set(subprocess.run(["git", "ls-files", o38p.PATTERN], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.split())
    listed = [f["path"] for f in HELD["files"]]
    assert tracked, "fixture must find o38s files"
    assert len(listed) == len(set(listed)) == HELD["n_files"]
    assert set(listed) == tracked
    assert sum(g["n_files"] for g in HELD["groups"].values()) == HELD["n_files"]


def test_registration_is_not_a_seal():
    sealed = {e["path"] for e in json.loads((ROOT / "manifests/sealed_artifacts.json").read_text())["artifacts"]}
    assert not sealed & {f["path"] for f in HELD["files"]}


def test_check_fails_on_a_changed_file(tmp_path):
    clone = tmp_path / "c"
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(ROOT), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", HELD["source_rev"]], check=True)
    for rel in (o38p.OUT, "scripts/open38_provenance.py"):
        (clone / rel).parent.mkdir(parents=True, exist_ok=True)
        (clone / rel).write_bytes((ROOT / rel).read_bytes())
    ok = subprocess.run([sys.executable, "-B", str(clone / "scripts/open38_provenance.py"), "--check"],
                        cwd=clone, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    victim = clone / HELD["files"][0]["path"]
    victim.write_bytes(victim.read_bytes() + b" ")
    bad = subprocess.run([sys.executable, "-B", str(clone / "scripts/open38_provenance.py"), "--check"],
                         cwd=clone, capture_output=True, text=True)
    assert bad.returncode == 1 and "working tree differs" in bad.stdout


@pytest.mark.parametrize("field", ["sha256", "blob", "evidence_commit", "group", "execution_sha"])
def test_check_fails_on_a_changed_manifest_field(tmp_path, monkeypatch, field):
    held = json.loads(json.dumps(HELD))
    held["files"][0][field] = "x"
    fake = tmp_path / "m.json"
    fake.write_bytes(json.dumps(held).encode())
    monkeypatch.setattr(o38p, "OUT", str(fake))  # ROOT / absolute path -> the absolute path
    monkeypatch.setattr(sys, "argv", ["open38_provenance.py", "--check"])
    assert o38p.main() == 1


def test_unknown_name_has_no_group():
    with pytest.raises(SystemExit):
        o38p.group("o38s_new_arm_n4000_s1_T8.json")

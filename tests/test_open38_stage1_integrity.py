"""Stage-1 integrity on synthetic runs: a planted complete batch passes; each detached-batch failure mode (absent,
duplicated, incomplete, wrong seed, wrong arm, extra run) fails, and the frozen stage checker's PASS is not enough."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
SPEC = importlib.util.spec_from_file_location("integ", ROOT / "scripts" / "open38_stage1_integrity.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
import test_open38_smoke_check as fx  # noqa: E402

HEAD = "53f8b89fdb1ec4d67e3c972dd24ee4c899a680cc"


def _plant(d: Path):
    for i, arm in enumerate(("ref", "flat", "rev")):
        for s in range(1, 6):
            t = mod.s1.tag(arm, s)
            fx._write(d, t, r_ab=0.84 if arm == "ref" else -0.7, sha=None, counts_bump=s + 10 * i)  # distinct runs
            rec = json.loads((d / f"{t}.json").read_text())
            rec.update(seed=s, layout=mod.LAYOUT[arm], T_s=8.0, git_sha=HEAD, git_dirty=False,
                       jaxfne_file="D:/repos/_work/_temp_jaxfne_351f6048/jaxfne/__init__.py",
                       wall_s={"construct": 26.0 + 0.1 * i, "simulate": 175.0 + s, "total": 210.0 + s + i})
            (d / f"{t}.json").write_text(json.dumps(rec))
            (d / f"{t}_similarity.json").write_text("{}")


def _run(d: Path):
    rc = mod.main(str(d / "integ.json"), d, HEAD)
    return rc, json.loads((d / "integ.json").read_text())["checks"]


def _edit(d, t, **kw):
    rec = json.loads((d / f"{t}.json").read_text())
    rec.update(kw)
    (d / f"{t}.json").write_text(json.dumps(rec))


def test_complete_batch_passes(tmp_path):
    _plant(tmp_path)
    rc, c = _run(tmp_path)
    assert rc == 0, c


def test_absent_run_fails(tmp_path):
    _plant(tmp_path)
    (tmp_path / f"{mod.s1.tag('rev', 3)}_verdict.json").unlink()
    rc, c = _run(tmp_path)
    assert rc == 1 and "incomplete" in c["runs"]["failed"][mod.s1.tag("rev", 3)][0]


def test_identical_outputs_with_distinct_executions_warn_only(tmp_path):
    _plant(tmp_path)
    a, b = mod.s1.tag("flat", 2), mod.s1.tag("flat", 4)
    shutil.copyfile(tmp_path / f"{a}.npz", tmp_path / f"{b}.npz")  # same outputs, separate execution records
    rc, c = _run(tmp_path)
    warn = json.loads((tmp_path / "integ.json").read_text())["warnings"]["identical_outputs"]
    assert rc == 0 and [a, b] in warn["npz_sha256"], "identical outputs alone must not fail a valid batch"


def test_one_execution_reported_twice_fails(tmp_path):
    _plant(tmp_path)
    a, b = mod.s1.tag("flat", 2), mod.s1.tag("flat", 4)
    for suf in (".json", ".npz", "_verdict.json", "_similarity.json"):
        shutil.copyfile(tmp_path / f"{a}{suf}", tmp_path / f"{b}{suf}")
    _edit(tmp_path, b, seed=4)  # relabelled consistently: only the shared execution record reveals it
    rc, c = _run(tmp_path)
    assert rc == 1 and c["runs"]["ok"], "fixture must pass every per-run check"
    assert [a, b] in c["provenance_unique"]["shared_execution_record"]


def test_incomplete_npz_fails(tmp_path):
    _plant(tmp_path)
    t = mod.s1.tag("ref", 5)
    raw = (tmp_path / f"{t}.npz").read_bytes()
    (tmp_path / f"{t}.npz").write_bytes(raw[: len(raw) // 2])  # truncated write
    rc, c = _run(tmp_path)
    assert rc == 1 and "incomplete" in c["runs"]["failed"][t][0]


def test_wrong_seed_and_wrong_arm_fail(tmp_path):
    _plant(tmp_path)
    _edit(tmp_path, mod.s1.tag("ref", 1), seed=2)
    _edit(tmp_path, mod.s1.tag("flat", 3), layout=mod.LAYOUT["rev"])
    rc, c = _run(tmp_path)
    f = c["runs"]["failed"]
    assert rc == 1 and "wrong seed" in f[mod.s1.tag("ref", 1)][0] and "wrong arm" in f[mod.s1.tag("flat", 3)][0]


def test_frozen_stage_checker_alone_misses_a_mislabelled_batch(tmp_path):
    _plant(tmp_path)  # ref r_ab 0.84 (in band), arms -0.7 (ref higher): the frozen checker's gates are met
    a, b = mod.s1.tag("flat", 1), mod.s1.tag("flat", 2)
    shutil.copyfile(tmp_path / f"{a}.npz", tmp_path / f"{b}.npz")
    shutil.copyfile(tmp_path / f"{a}_verdict.json", tmp_path / f"{b}_verdict.json")
    _edit(tmp_path, b, seed=1)  # flat seed 2 is a relabelled copy of seed 1
    assert mod.s1.main(str(tmp_path / "s1.json"), tmp_path) == 0, "precondition: frozen checker passes this batch"
    rc, c = _run(tmp_path)
    assert rc == 1 and "wrong seed" in c["runs"]["failed"][b][0]


def test_extra_run_and_dirty_provenance_fail(tmp_path):
    _plant(tmp_path)
    shutil.copyfile(tmp_path / f"{mod.s1.tag('ref', 1)}.json", tmp_path / "o38s_pc_ref_n4000_s6_T8.json")
    _edit(tmp_path, mod.s1.tag("rev", 2), git_dirty=True)
    rc, c = _run(tmp_path)
    assert rc == 1 and c["no_extra"]["extra"] == ["o38s_pc_ref_n4000_s6_T8"]
    assert "provenance" in c["runs"]["failed"][mod.s1.tag("rev", 2)][0]

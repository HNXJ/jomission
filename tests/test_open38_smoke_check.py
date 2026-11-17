"""OPEN-38 smoke check on synthetic run files: identical ref/nc pass; a changed array, a wrong engine, a ratio off by
more than one cell, an out-of-band r_ab and a silent deep unit each fail their own check."""
import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "smoke", Path(__file__).resolve().parents[1] / "scripts" / "open38_smoke_check.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

SIZES = {"S": {"E": 100, "PV": 15, "SOM": 4}, "M": {"E": 100, "PV": 9, "SOM": 10}, "D": {"E": 100, "PV": 6, "SOM": 13}}
RHO = {"S": 0.25, "M": 1.0, "D": 2.0}
KEYS = [f"{u}_{p}" for u in "SMD" for p in ("E", "PV", "SOM")]


def _write(d: Path, tag: str, r_ab=0.84, deep_spikes=5, sha="351f6048abc", counts_bump=0):
    n = 40000  # 4 s at 0.1 ms
    counts = np.zeros((n, len(KEYS)), dtype=np.int32)
    counts[25000::1000, KEYS.index("S_E")] = 1 + counts_bump
    counts[25000::1000, KEYS.index("M_E")] = 1
    counts[25000::1000, KEYS.index("D_E")] = deep_spikes
    rec = {"sizes": SIZES, "rho": RHO, "edges": {"e_e:S": 10}, "seed": 1, "T_s": 4.0, "dt_ms": 0.1,
           "jaxfne_version": "0.5.9", "jaxfne_git_sha": sha, "jaxfne_source": None, "wall_s": {"total": 1.0},
           "git_sha": "x"}
    (d / f"{tag}.json").write_text(json.dumps(rec), encoding="utf-8")
    (d / f"{tag}_verdict.json").write_text(json.dumps({"r_ab": r_ab, "r_gm": 0.4, "r_proto": 0.3}), encoding="utf-8")
    np.savez_compressed(d / f"{tag}.npz", counts=counts, block_keys=np.array(KEYS), lfp_pA=np.ones((n, 3), np.float32))


def _failed(d):
    return [k for k, c in json.loads((d / "o38s_smoke_check.json").read_text())["checks"].items() if not c["ok"]]


def test_identical_runs_pass(tmp_path):
    _write(tmp_path, "ref")
    _write(tmp_path, "nc")
    assert mod.main("ref", "nc", tmp_path) == 0
    rates = json.loads((tmp_path / "o38s_smoke_check.json").read_text())["checks"]["stop:ref"]["e_rates_hz"]
    assert rates["D"] > mod.DEEP_MIN_HZ, "fixture deep unit must fire above the stop floor"


def test_each_defect_fails_its_check(tmp_path):
    _write(tmp_path, "ref")
    _write(tmp_path, "nc", counts_bump=1)
    assert mod.main("ref", "nc", tmp_path) == 1 and _failed(tmp_path) == ["nc_equals_ref"]
    _write(tmp_path, "nc", sha="deadbeef")
    _write(tmp_path, "ref", sha="deadbeef", r_ab=0.70, deep_spikes=0)
    failed = _failed(tmp_path) if mod.main("ref", "nc", tmp_path) else []
    assert {"engine:ref", "engine:nc", "ref_r_ab_band", "stop:ref"} <= set(failed)


def test_engine_identity_by_verified_export_path(tmp_path):
    for tag in ("ref", "nc"):
        _write(tmp_path, tag, sha=None)
        rec = json.loads((tmp_path / f"{tag}.json").read_text())
        rec["jaxfne_file"] = "D:\\repos/_work/_temp_jaxfne_351f6048\\jaxfne\\__init__.py"
        (tmp_path / f"{tag}.json").write_text(json.dumps(rec))
    assert mod.main("ref", "nc", tmp_path) == 0
    rec["jaxfne_file"] = "D:/tmp/jaxfne_v059/jaxfne/__init__.py"  # unidentified tree
    (tmp_path / "nc.json").write_text(json.dumps(rec))
    assert mod.main("ref", "nc", tmp_path) == 1 and "engine:nc" in _failed(tmp_path)


def test_nonfinite_ratio_and_metadata_fail_through_main(tmp_path):
    _write(tmp_path, "ref")
    _write(tmp_path, "nc")
    rec = json.loads((tmp_path / "nc.json").read_text())
    rec["sizes"] = {**SIZES, "D": {"E": 100, "PV": 9, "SOM": 10}}  # ratio off rho 2 by > 1 cell; metadata differs
    (tmp_path / "nc.json").write_text(json.dumps(rec))
    v = json.loads((tmp_path / "nc_verdict.json").read_text())
    v["r_gm"] = float("nan")
    (tmp_path / "nc_verdict.json").write_text(json.dumps(v))
    assert mod.main("ref", "nc", tmp_path) == 1
    assert {"ratio:nc", "finite:nc", "nc_equals_ref"} <= set(_failed(tmp_path))
    meta = json.loads((tmp_path / "o38s_smoke_check.json").read_text())["checks"]["nc_equals_ref"]
    assert meta["differing_arrays"] == [] and meta["metadata_equal"] is False, "metadata alone must fail nc==ref"


def test_ratio_off_by_more_than_one_cell():
    bad = {**SIZES, "D": {"E": 100, "PV": 9, "SOM": 10}}
    assert not mod.ratio_ok(bad, RHO)["D"]["ok"] and mod.ratio_ok(SIZES, RHO)["D"]["ok"]

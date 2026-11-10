"""Paper block for one E2 run (scripts/e2_paper_block.py) on a tiny synthetic run written by the test.

The fixture has 2 modules, 40 + 30 E cells, and every array the script reads. It writes nothing outside tmp_path.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import e2_paper_block as pb  # noqa: E402
from jomission.laminar import caminos_stack as cs  # noqa: E402

TAG = "synthetic_s1_T4"
UNITS = ["S", "M"]
SIZES = {"S": {"E": 40, "PV": 8, "SOM": 8}, "M": {"E": 30, "PV": 6, "SOM": 6}}
DT_MS = 0.1
T_S = 5.0  # the 2 s discard and 2 s window must fit in the run after the 6 ms RWS delay trim
LOCAL = ("e_e", "e_pv", "e_sst", "pv_e", "pv_pv", "sst_e", "sst_pv")
INTER = ("x_S_M_e_e", "x_S_M_e_pv", "x_S_M_e_som", "x_M_S_e_e", "x_M_S_e_pv", "x_M_S_e_som")


def write_run(root: Path) -> Path:
    rng = np.random.default_rng(1)
    steps = int(round(T_S * 1000 / DT_MS))
    edges = {f"{name}_{m}": int(rng.integers(1000, 5000)) for m in UNITS for name in LOCAL}
    edges.update({key: int(rng.integers(100, 900)) for key in INTER})
    cols = [(m, r) for m in UNITS for r in ("ampa", "gaba", "gaba_sst")]
    rec = {
        "experiment": "synthetic fixture (tests/test_e2_paper_block.py)",
        "layout": "e2_fixture",
        "units": UNITS,
        "depth_rel_L4": {"S": -0.4, "M": 0.3},
        "seed": 1,
        "T_s": T_S,
        "dt_ms": DT_MS,
        "p_inter": None,
        "k_inter": 80.0,
        "rho": {"S": 0.25, "M": 1.0},
        "sizes": SIZES,
        "edges": edges,
        "tau_sst_e_ms": 15.0,
        "modifiers": {"n": 40.0, "kd": 1.0, "sst": 15.0},
        "p_local": {"S": 0.1, "M": 0.1},
        "g_columns": [list(c) for c in cols],
        "g_reversal_mV": [0.0, -80.0, -80.0] * len(UNITS),
        "g_tau_ms": [2.0, 8.0, 15.0] * len(UNITS),
        "jaxfne_version": "fixture",
        "jaxfne_file": "fixture",
        "jax_devices": ["cpu:0"],
        "git_sha": "0" * 40,
        "wall_s": {"total": 0.0},
    }
    block_keys = [f"{m}_{p}" for m in UNITS for p in pb.POPS]
    arrays = {
        "block_keys": np.array(block_keys),
        "counts": np.stack([rng.binomial(SIZES[m][p], 0.002, steps) for m in UNITS for p in pb.POPS],
                           axis=1).astype(np.int32),
        "g_nS": (rng.random((steps, len(cols))) + 0.1).astype(np.float32),
    }
    for m in UNITS:
        for p in pb.POPS:
            n_row = pb._sample_size(p, SIZES[m][p])
            st, cell = np.nonzero(rng.random((steps, n_row)) < 0.002)
            arrays[f"{m}_{p}_t"] = st.astype(np.int32)
            arrays[f"{m}_{p}_c"] = cell.astype(np.int16)
    assert int(arrays["counts"].sum()) > 0, "fixture must spike"
    assert arrays["S_E_t"].size > 0, "fixture must have raster spikes"
    run_dir = root / "run"
    run_dir.mkdir()
    (run_dir / f"{TAG}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    np.savez_compressed(run_dir / f"{TAG}.npz", **arrays)
    return run_dir


@pytest.fixture
def run_dir(tmp_path):
    return write_run(tmp_path)


def test_details_edges_equal_run_json_and_sizes_sum_to_n(run_dir):
    rec, _ = pb.load_run(run_dir, TAG)
    det = pb.build_details(rec, TAG)
    assert det["edges"] == rec["edges"], "table edges differ from the run JSON edges"
    assert len(det["rules"]) == len(rec["edges"]) == 2 * len(LOCAL) + len(INTER) == 20
    n_fixture = sum(sum(SIZES[m].values()) for m in UNITS)
    assert det["N"] == n_fixture == len(cs.neuron_rows(SIZES)), "sizes do not sum to N"
    assert sum(r["edges"] for r in det["rules"]) == sum(rec["edges"].values())


def test_missing_run_edge_is_refused(run_dir):
    rec, _ = pb.load_run(run_dir, TAG)
    bad = copy.deepcopy(rec)
    del bad["edges"][f"e_e_{UNITS[0]}"]
    with pytest.raises(ValueError, match="rule names differ"):
        pb.build_details(bad, TAG)


def test_every_parameter_row_has_a_source(run_dir):
    rec, _ = pb.load_run(run_dir, TAG)
    rows = pb.build_details(rec, TAG)["parameters"]
    assert len(rows) >= 40, f"parameter table has {len(rows)} rows; examined set too small"
    empty = [r["name"] for r in rows if not str(r.get("source", "")).strip()]
    assert not empty, f"rows without a source: {empty}"
    assert any(r["source"].startswith(pb.PORT) for r in rows), "no PORT_SPEC row"
    agent = [r["name"] for r in rows if r["source"].startswith(pb.AGENT)]
    for key in ("kd", "sst", "K_INTER", "sigma", "V_ref", "depth rel L4", "MODULE_BINS"):
        assert any(key in name for name in agent), f"{key} is not labelled agent choice"


def test_figure_and_v0_are_written_and_nonempty(run_dir, tmp_path):
    out = tmp_path / "out"
    rc = pb.main([TAG, "--results-dir", str(run_dir), "--out-dir", str(out), "--discard", "2"])
    assert rc == 0
    for name in (f"{TAG}_block.png", f"{TAG}_V0_architecture.png", f"{TAG}_details.json", f"{TAG}_details.md"):
        path = out / name
        assert path.is_file() and path.stat().st_size > 0, name
    head = (out / f"{TAG}_block.png").read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "block is not a PNG"
    width = int.from_bytes(head[16:20], "big")
    assert width == round(pb.FIG_W * 300), f"block width {width} px is not the 7.2 in final size at 300 dpi"

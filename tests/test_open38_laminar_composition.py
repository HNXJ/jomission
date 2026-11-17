"""OPEN-38 composition: cluster sums, module mapping, missing agranular L4, counting-noise interval."""
import csv
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "open38", Path(__file__).resolve().parents[1] / "scripts" / "open38_laminar_composition.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _write(d: Path, sub: str, n: int, rows: dict) -> None:
    with open(d / f"cell_summation_monkey_f001_{sub}.{n}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "Cell Count"])
        for k, v in rows.items():
            w.writerow([k, v])


def _fixture(d: Path) -> None:
    # area A granular: SST over two clusters, D/S planted at SST/PV 8 deep vs 1 superficial
    _write(d, "SST", 1, {"A-l1": 5, "A-l2": 30, "A-l3": 70, "A-l4": 40, "A-l5": 300, "A-l6": 500})
    _write(d, "SST", 2, {"A-l1": 5, "A-l2": 0, "A-l3": 0, "A-l4": 10, "A-l5": 0, "A-l6": 0})
    _write(d, "PVALB", 1, {"A-l1": 1, "A-l2": 50, "A-l3": 50, "A-l4": 50, "A-l5": 50, "A-l6": 50,
                           "B-l1": 1, "B-l2": 10, "B-l3": 10, "B-l5": 10, "B-l6": 10})
    _write(d, "SST", 3, {"B-l1": 1, "B-l2": 10, "B-l3": 10, "B-l5": 10, "B-l6": 10})
    for s in ("PV_CHC", "VIP", "VIP_RELN"):
        _write(d, s, 1, {"A-l2": 1, "B-l2": 1})


def test_cluster_sum_modules_and_planted_ratio(tmp_path):
    _fixture(tmp_path)
    counts, n = mod.load(tmp_path)
    assert n["SST"] == 3, "fixture must hold three SST clusters"
    rows = {(r["area"], r["module"]): r for a in ("A", "B") for r in mod.module_rows(counts, a)}
    s, m, dd = rows[("A", "S")], rows[("A", "M")], rows[("A", "D")]
    assert (s["SST"], s["PVALB"]) == (100, 100)
    assert (m["SST"], m["PVALB"]) == (50, 50)  # l4 sums both SST clusters
    assert (dd["SST"], dd["PVALB"]) == (800, 100)
    assert s["R_sst_pvalb"] == pytest.approx(1.0, rel=1e-12)
    assert dd["R_sst_pvalb"] == pytest.approx(8.0, rel=1e-12)
    assert dd["R_lo"] < 8.0 < dd["R_hi"]
    assert rows[("A", "L1")]["R_sst_pvalb"] == "", "L1 enters no ratio"


def test_agranular_l4_is_missing_not_zero(tmp_path):
    _fixture(tmp_path)
    counts, _ = mod.load(tmp_path)
    m = {r["module"]: r for r in mod.module_rows(counts, "B")}["M"]
    assert m["missing"] is True and m["SST"] == "" and m["R_sst_pvalb"] == ""


def test_ratio_interval_widens_with_fewer_counts():
    lo1, hi1 = mod.ratio_ci(10, 10)
    lo2, hi2 = mod.ratio_ci(1000, 1000)
    assert lo1 < lo2 < 1.0 < hi2 < hi1


def test_duplicate_region_refused(tmp_path):
    _write(tmp_path, "SST", 1, {"A-l2": 1})
    with open(tmp_path / "cell_summation_monkey_f001_SST.1.csv", "a", encoding="utf-8") as fh:
        fh.write("A-l2,2\n")
    with pytest.raises(ValueError, match="duplicate region"):
        mod.load(tmp_path)

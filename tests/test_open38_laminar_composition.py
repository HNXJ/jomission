"""OPEN-38 composition: cluster sums, module mapping, missing agranular L4, counting-noise interval."""
import csv
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "open38", Path(__file__).resolve().parents[1] / "scripts" / "open38_laminar_composition.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


REGIONS = ["A-l1", "A-l2", "A-l3", "A-l4", "A-l5", "A-l6", "B-l1", "B-l2", "B-l3", "B-l5", "B-l6"]


def _write(d: Path, sub: str, n: int, rows: dict, regions=REGIONS) -> None:
    """Every portal export lists the same regions; regions a cluster lacks carry 0."""
    with open(d / f"cell_summation_monkey_f001_{sub}.{n}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "Cell Count"])
        for k in (regions if regions is not None else rows):
            w.writerow([k, rows.get(k, 0)])


def _fixture(d: Path) -> None:
    # area A granular: SST over two clusters, D/S planted at SST/PV 8 deep vs 1 superficial
    _write(d, "SST", 1, {"A-l1": 5, "A-l2": 30, "A-l3": 70, "A-l4": 40, "A-l5": 300, "A-l6": 500})
    _write(d, "SST", 2, {"A-l1": 5, "A-l2": 0, "A-l3": 0, "A-l4": 10, "A-l5": 0, "A-l6": 0})
    _write(d, "PVALB", 1, {"A-l1": 1, "A-l2": 50, "A-l3": 50, "A-l4": 50, "A-l5": 50, "A-l6": 50,
                           "B-l1": 1, "B-l2": 10, "B-l3": 10, "B-l5": 10, "B-l6": 10})
    _write(d, "SST", 3, {"B-l1": 1, "B-l2": 10, "B-l3": 10, "B-l5": 10, "B-l6": 10})
    for i, s in enumerate(("PV_CHC", "VIP", "VIP_RELN"), 1):
        _write(d, s, 1, {"A-l2": i, "B-l2": 1})


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


def test_denominator_and_fractions(tmp_path):
    _fixture(tmp_path)
    counts, _ = mod.load(tmp_path)
    all_cells = {"A-l1": 100, "A-l2": 400, "A-l3": 600, "A-l4": 1000, "A-l5": 2000, "A-l6": 2000}
    rows = {r["module"]: r for r in mod.module_rows(counts, "A", all_cells)}
    assert rows["S"]["n_all"] == 1000 and rows["D"]["n_all"] == 4000
    assert rows["S"]["frac_SST"] == pytest.approx(0.1, rel=1e-12)
    assert rows["D"]["frac_SST"] == pytest.approx(0.2, rel=1e-12)
    assert rows["D"]["frac_PVALB"] == pytest.approx(0.025, rel=1e-12)
    m = {r["module"]: r for r in mod.module_rows(counts, "B", {"B-l2": 50})}["M"]
    assert m["n_all"] == "" and m["frac_SST"] == "", "missing module keeps a missing denominator"


def test_region_set_mismatch_refused(tmp_path):
    _write(tmp_path, "SST", 1, {"A-l2": 1, "A-l3": 1}, regions=None)
    _write(tmp_path, "SST", 2, {"A-l2": 1}, regions=None)
    with pytest.raises(ValueError, match="region set differs"):
        mod.load(tmp_path)


ALL = {r: 10_000 for r in REGIONS}


def test_verify_passes_fixture_and_reports_coverage(tmp_path):
    _fixture(tmp_path)
    counts, _ = mod.load(tmp_path)
    rec = mod.verify(tmp_path, {**ALL, "A": 60_000, "B": 80_000}, counts)
    assert len(rec["files"]) == 7, "fixture must hold seven exports"
    assert rec["layer_sum_over_area_all_cells"]["A"] == pytest.approx(1.0, rel=1e-12)
    assert rec["layer_sum_over_area_all_cells"]["B"] == pytest.approx(0.625, rel=1e-12)


def test_verify_refuses_duplicate_export(tmp_path):
    _fixture(tmp_path)
    _write(tmp_path, "SST", 4, {"A-l1": 5, "A-l2": 0, "A-l3": 0, "A-l4": 10, "A-l5": 0, "A-l6": 0})  # = SST.2
    counts, _ = mod.load(tmp_path)
    with pytest.raises(ValueError, match="same bytes"):
        mod.verify(tmp_path, ALL, counts)


def test_verify_zero_export_only_when_declared_absent(tmp_path):
    _fixture(tmp_path)
    _write(tmp_path, "SST", 4, {})
    _write(tmp_path, "VIP", 2, {})  # second all-zero file: identical bytes, allowed when both are declared
    counts, _ = mod.load(tmp_path)
    assert mod.verify(tmp_path, ALL, counts, absent_ok={"SST.4", "VIP.2"})["files"], "declared zeros pass"
    with pytest.raises(ValueError, match="all zero is True, declared absent is False"):
        mod.verify(tmp_path, ALL, counts, absent_ok={"SST.4"})
    with pytest.raises(ValueError, match="all zero is False, declared absent is True"):
        mod.verify(tmp_path, ALL, counts, absent_ok={"SST.4", "VIP.2", "SST.1"})


def test_verify_refuses_count_above_all_cells(tmp_path):
    _fixture(tmp_path)
    counts, _ = mod.load(tmp_path)
    with pytest.raises(ValueError, match="above all cells"):
        mod.verify(tmp_path, {**ALL, "A-l6": 400}, counts)


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

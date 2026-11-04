"""SCI-AGSDR-CALIBRATION stage-0 correction: tau-weighted Efrac, hard-gate predicates, records.

Non-simulating: reads sealed stage-0 receipts and the tau-check receipt; writes nothing.
"""
import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import agsdr_stage0_correction as C  # noqa: E402

RECEIPTS = os.path.join(REPO, "docs", "receipts", "agsdr_local")
NAN = float("nan")


def _receipt(cid: str) -> dict:
    with open(os.path.join(RECEIPTS, f"candidate_{cid}_seed0.json"), encoding="utf-8") as f:
        return json.load(f)


def _json(name: str) -> dict:
    with open(os.path.join(RECEIPTS, name), encoding="utf-8") as f:
        return json.load(f)


# --- tau-weighted Efrac -----------------------------------------------------------------

def test_tau_formula_known_value_differs_from_omitted_tau_proxy():
    # Regression: the runner's proxy (tau omitted) returns 0.5 here. The corrected value is 2/7.
    assert C.tau_weighted_efrac(1.0, 1.0, 2.0, 5.0) == pytest.approx(2.0 / 7.0, abs=1e-15)
    assert C.proxy_efrac(1.0, 1.0) == pytest.approx(0.5, abs=1e-15)
    assert C.tau_weighted_efrac(1.0, 1.0, 2.0, 5.0) != pytest.approx(C.proxy_efrac(1.0, 1.0))


def test_equal_tau_reduces_to_proxy():
    # Known negative control: a common tau is a common factor, so the correction is a no-op.
    for i_e, i_i in [(3.0, 1.0), (0.2, 7.5), (806.8, 393.4)]:
        assert C.tau_weighted_efrac(i_e, i_i, 4.0, 4.0) == pytest.approx(
            C.proxy_efrac(i_e, i_i), rel=1e-15)


def test_proxy_reproduces_recorded_candidate_06():
    m = _receipt("AGS_S0_06")["metrics"]
    assert C.proxy_efrac(m["I_E_proxy"], m["I_I_proxy"]) == pytest.approx(
        m["Efrac_proxy_global"], abs=1e-12)


def test_tau_correction_candidate_06_value():
    m = _receipt("AGS_S0_06")["metrics"]
    got = C.tau_weighted_efrac(m["I_E_proxy"], m["I_I_proxy"], 2.0, 5.0)
    assert got == pytest.approx(0.4506, abs=1e-3)


# --- hard-gate predicates: synthetic boundary and failure cases -------------------------

GATE_CASES = [
    ("H01_finite", False, False), ("H01_finite", True, True),
    ("H02_bar_r_open_1_30", 1.0, False), ("H02_bar_r_open_1_30", 30.0, False),
    ("H02_bar_r_open_1_30", 1.0001, True), ("H02_bar_r_open_1_30", NAN, False),
    ("H03_mean_rho_below_0p60", 0.60, False), ("H03_mean_rho_below_0p60", 0.59, True),
    ("H03_mean_rho_below_0p60", NAN, False),
    ("H04_CV_rate_above_0p15", 0.15, False), ("H04_CV_rate_above_0p15", 0.1501, True),
    ("H05_P_CV_E_gt_0p5_at_least_0p15", 0.1499, False),
    ("H05_P_CV_E_gt_0p5_at_least_0p15", 0.15, True),
    ("H06_max_CV_E_at_least_1p5", 1.4999, False), ("H06_max_CV_E_at_least_1p5", 1.5, True),
    ("H09_Efrac_open_0p05_0p95", 0.05, False), ("H09_Efrac_open_0p05_0p95", 0.95, False),
    ("H09_Efrac_open_0p05_0p95", 0.5, True), ("H09_Efrac_open_0p05_0p95", NAN, False),
    ("H10a_class_mean_not_above_80", 80.0, True), ("H10a_class_mean_not_above_80", 80.0001, False),
    ("H10b_E_class_mean_above_0p5", 0.5, False), ("H10b_E_class_mean_above_0p5", 0.5001, True),
    ("H10c_population_mean_not_above_80", 80.0001, False),
    ("H10d_E_population_mean_above_0p5", 0.5, False),
]


@pytest.mark.parametrize("gid,value,expected", GATE_CASES)
def test_gate_predicate_boundary(gid, value, expected):
    values = {g: 0.3 for g in C.GATE_PREDICATES}
    values[gid] = value
    assert C.evaluate_gates(values)[gid] is expected


def test_null_control_all_gates_pass_on_candidate_06_with_tau():
    # Positive control: candidate 06 is hard-feasible under the corrected Efrac.
    r = _receipt("AGS_S0_06")
    values = {g["id"]: g["value"] for g in r["hard_gates"]}
    m = r["metrics"]
    tau = _json("stage0_tau_check.json")["candidates"]["AGS_S0_06"]
    efrac = C.tau_weighted_efrac(m["I_E_proxy"], m["I_I_proxy"],
                                 tau["E_pre"]["tau_ms_values"][0], tau["I_pre"]["tau_ms_values"][0])
    values["H09_Efrac_open_0p05_0p95"] = efrac
    assert all(C.evaluate_gates(values).values())


def test_predicates_reproduce_all_24_stored_verdicts():
    # Guards drift between the predicates and the runner: every stored pass must be reproduced.
    for cid in [f"AGS_S0_{i:02d}" for i in range(24)]:
        r = _receipt(cid)
        values = {g["id"]: g["value"] for g in r["hard_gates"]}
        stored = {g["id"]: g["pass"] for g in r["hard_gates"]}
        assert C.evaluate_gates(values) == stored, cid


# --- tau-check and correction records ---------------------------------------------------

def test_realized_tau_is_class_constant_for_all_24_candidates():
    tc = _json("stage0_tau_check.json")["candidates"]
    assert len(tc) == 24
    for cid, rec in tc.items():
        assert rec["E_pre"]["tau_ms_values"] == [2.0], cid
        assert rec["I_pre"]["tau_ms_values"] == [5.0], cid
        assert rec["E_pre"]["edges"] + rec["I_pre"]["edges"] == rec["n_edges"], cid


def test_correction_record_is_consistent_with_receipts():
    rec = _json("stage0_correction_v1.json")
    rows = rec["per_candidate"]
    assert len(rows) == 24
    for row in rows:
        r = _receipt(row["candidate_id"])
        assert row["I_E_proxy"] == r["metrics"]["I_E_proxy"]
        assert row["stored_gate_verdicts_reproduced"] is True, row["candidate_id"]
        assert 0.0 < row["Efrac_tau_weighted"] < 1.0
    tau_feasible = [x["candidate_id"] for x in rows if x["hard_pass_tau"]]
    assert rec["hard_feasibility"]["tau_feasible_count"] == len(tau_feasible)


def test_sampling_departure_reconstruction_matches_recorded_count():
    dep = _json("stage0_correction_v1.json")["sampling_departure"]
    assert dep["rows_redrawn_recorded"] == 16
    assert dep["rows_redrawn_reconstructed"] == 16
    assert dep["non_background_or_unredrawn_mismatches"] == []
    assert dep["consistent"] is True


def test_selection_is_not_determined_in_record():
    sel = _json("stage0_correction_v1.json")["selection"]
    assert sel["status"] == "NOT DETERMINED"
    assert sel["eight_selected"] is None
    assert any("B3" in b for b in sel["blocked_by"])

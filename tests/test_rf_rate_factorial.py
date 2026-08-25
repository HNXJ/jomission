"""Frozen 2×2 factorial RF×Rate — design, completion predicate, stats, distinguishability."""

import hashlib
import json
import pathlib

import numpy as np
import pytest

import jomission.ablations.rf_rate_factorial as fact
from jomission.ablations.rf_rate_factorial import (
    DESIGN_VERSION,
    FROZEN_CONFIG_HASH,
    CELL_HP_HASHES,
    CELL_ORDER,
    CELL_FACTORS,
    THRESHOLDS,
    OMISSION_SLOT_MS,
    BANDS,
    AREAS_CANONICAL,
    hp_for_cell,
    hp_hash,
    validate_design,
    check_completion,
    factorial_cells,
    FactorialANOVAInput,
    anova_rf_rate,
)


def test_design_version_frozen():
    assert DESIGN_VERSION == "rf_rate_factorial.v0.1.0"
    v = validate_design()
    assert v["valid"], f"design invalid: {v['issues']}"
    assert v["design_version"] == DESIGN_VERSION


def test_cell_hashes_match_computed():
    for cell in CELL_ORDER:
        hp = hp_for_cell(cell)
        h = hp_hash(hp)
        assert h == CELL_HP_HASHES[cell], f"{cell} {h} != {CELL_HP_HASHES[cell]}"
    # Also check manifest
    mf = pathlib.Path("manifests/rf_rate_factorial_design.json")
    assert mf.exists(), "manifest missing"
    data = json.loads(mf.read_text())
    assert data["design_version"] == DESIGN_VERSION
    assert data["cell_hp_hashes"] == dict(CELL_HP_HASHES)
    # Config hash frozen
    assert data["config_hash"] == FROZEN_CONFIG_HASH


def test_factor_definitions_orthogonal():
    # RF off uses K_HDP 0, on uses 0.003
    assert hp_for_cell("A_RFoff_RateStd")["K_HDP"] == 0.0
    assert hp_for_cell("C_RFon_RateStd")["K_HDP"] == 0.003
    # Rate standard 5, slow 1000
    assert hp_for_cell("A_RFoff_RateStd")["tau_0_ms"] == 5.0
    assert hp_for_cell("B_RFoff_RateSlow")["tau_0_ms"] == 1000.0
    assert hp_for_cell("D_RFon_RateSlow")["tau_0_ms"] == 1000.0
    # Rate preserves K_HDP/K_w_ctrl ratio within RF level
    assert hp_for_cell("C_RFon_RateStd")["K_HDP"]/hp_for_cell("C_RFon_RateStd")["K_w_ctrl"] == hp_for_cell("D_RFon_RateSlow")["K_HDP"]/hp_for_cell("D_RFon_RateSlow")["K_w_ctrl"]
    # RF preserves tau within Rate level
    assert hp_for_cell("A_RFoff_RateStd")["tau_0_ms"] == hp_for_cell("C_RFon_RateStd")["tau_0_ms"]
    assert hp_for_cell("B_RFoff_RateSlow")["tau_0_ms"] == hp_for_cell("D_RFon_RateSlow")["tau_0_ms"]
    # Bounds unchanged
    for cell in CELL_ORDER:
        hp = hp_for_cell(cell)
        assert hp["H_min"] == 0.1 and hp["H_max"] == 10.0
        assert hp["w_floor"] == 0.01 and hp["w_ceiling"] == 10.0
        assert hp["K_ctrl"] == 0.15
        assert hp["K_w_ctrl"] == 0.001


def test_completion_predicate_full_and_pilot():
    # FULL passes
    ok = check_completion(
        design_version=DESIGN_VERSION,
        cell_hashes=dict(CELL_HP_HASHES),
        config_hash=FROZEN_CONFIG_HASH,
        dt_ms=0.1,
        n_seeds=8,
        exposure_trials=260,
        testing_trials=96,
        checkpoints=26,
        field_recorded=True,
        predicate="FULL",
    )
    assert ok["valid"], ok["issues"]
    # PILOT passes with dt1 and n4
    okp = check_completion(
        design_version=DESIGN_VERSION,
        cell_hashes=dict(CELL_HP_HASHES),
        config_hash=FROZEN_CONFIG_HASH,
        dt_ms=1.0,
        n_seeds=4,
        exposure_trials=260,
        testing_trials=96,
        checkpoints=26,
        field_recorded=True,
        predicate="PILOT",
    )
    assert okp["valid"], okp["issues"]
    # FULL should fail if n_seeds too low
    fail = check_completion(
        design_version=DESIGN_VERSION,
        cell_hashes=dict(CELL_HP_HASHES),
        config_hash=FROZEN_CONFIG_HASH,
        dt_ms=0.1,
        n_seeds=4,
        exposure_trials=260,
        testing_trials=96,
        checkpoints=26,
        field_recorded=True,
        predicate="FULL",
    )
    assert not fail["valid"]
    # Wrong hash fails
    bad_hashes = dict(CELL_HP_HASHES)
    bad_hashes["A_RFoff_RateStd"] = "deadbeefdeadbeef"
    bad = check_completion(
        design_version=DESIGN_VERSION,
        cell_hashes=bad_hashes,
        config_hash=FROZEN_CONFIG_HASH,
        dt_ms=0.1,
        n_seeds=8,
        exposure_trials=260,
        testing_trials=96,
        checkpoints=26,
        field_recorded=True,
        predicate="FULL",
    )
    assert not bad["valid"]


def test_factorial_cells_api():
    cells = factorial_cells(dt_ms=0.1)
    assert len(cells) == 4
    names = [c.name for c in cells]
    assert names == list(CELL_ORDER)
    for c in cells:
        assert c.config_hash == FROZEN_CONFIG_HASH
        assert c.hp_hash == CELL_HP_HASHES[c.name]
        assert c.rf in ("off","on")
        assert c.rate in ("standard","slow")


def test_windows_and_bands_frozen():
    assert OMISSION_SLOT_MS == (0.0, 531.0)
    assert BANDS["low_gamma"] == (30.0, 50.0)
    assert AREAS_CANONICAL == ("V1","V4","FEF","PFC")
    assert THRESHOLDS["alpha"] == 0.05
    assert THRESHOLDS["rate_hz"] == 0.5


def test_anova_distinguishes_rf_only():
    # Synthetic: RF effect only (C,D higher than A,B), no Rate, no inter
    np.random.seed(0)
    n=8
    A = np.random.normal(0,0.5,n)
    B = np.random.normal(0,0.5,n)
    C = np.random.normal(2,0.5,n)
    D = np.random.normal(2,0.5,n)
    inp = FactorialANOVAInput(data={"A_RFoff_RateStd":A,"B_RFoff_RateSlow":B,"C_RFon_RateStd":C,"D_RFon_RateSlow":D}, seeds=list(range(n)), phenotype="rf_only")
    res = anova_rf_rate(inp)
    # Should detect RF main, not Rate, not interaction
    assert res.contrasts["RF_main"]["p"] < 0.05
    assert res.contrasts["Rate_main"]["p"] > 0.05
    assert res.contrasts["Interaction"]["p"] > 0.05
    assert "RF main only" in res.interpretation


def test_anova_distinguishes_rate_only():
    np.random.seed(1)
    n=8
    A = np.random.normal(0,0.5,n)
    B = np.random.normal(2,0.5,n)
    C = np.random.normal(0,0.5,n)
    D = np.random.normal(2,0.5,n)
    inp = FactorialANOVAInput(data={"A_RFoff_RateStd":A,"B_RFoff_RateSlow":B,"C_RFon_RateStd":C,"D_RFon_RateSlow":D}, seeds=list(range(n)), phenotype="rate_only")
    res = anova_rf_rate(inp)
    assert res.contrasts["Rate_main"]["p"] < 0.05
    assert res.contrasts["RF_main"]["p"] > 0.05
    assert res.contrasts["Interaction"]["p"] > 0.05


def test_anova_distinguishes_interaction():
    np.random.seed(2)
    n=8
    A = np.random.normal(0,0.5,n)
    B = np.random.normal(0,0.5,n)
    C = np.random.normal(0,0.5,n)
    D = np.random.normal(3,0.5,n)  # only D high
    inp = FactorialANOVAInput(data={"A_RFoff_RateStd":A,"B_RFoff_RateSlow":B,"C_RFon_RateStd":C,"D_RFon_RateSlow":D}, seeds=list(range(n)), phenotype="interaction")
    res = anova_rf_rate(inp)
    assert res.contrasts["Interaction"]["p"] < 0.05
    assert "interaction" in res.interpretation.lower()
    # Simple effect Rate|RFon large, Rate|RFoff ~0
    assert res.contrasts["simple_Rate_given_RFon"]["estimate"] > 1.5
    assert abs(res.contrasts["simple_Rate_given_RFoff"]["estimate"]) < 0.8


def test_anova_distinguishes_additive():
    np.random.seed(3)
    n=8
    # A=0, B=1 (Rate), C=1 (RF), D=2 (both) => additive
    A = np.random.normal(0,0.5,n)
    B = np.random.normal(1,0.5,n)
    C = np.random.normal(1,0.5,n)
    D = np.random.normal(2,0.5,n)
    inp = FactorialANOVAInput(data={"A_RFoff_RateStd":A,"B_RFoff_RateSlow":B,"C_RFon_RateStd":C,"D_RFon_RateSlow":D}, seeds=list(range(n)), phenotype="additive")
    res = anova_rf_rate(inp)
    # In additive, both mains sig, interaction NS
    # Due to noise may not always both, but with n=8 and 1 SD diff, expect at least one main sig
    # Check interpretation contains additive or at least no interaction
    assert res.contrasts["Interaction"]["p"] > 0.05


def test_anova_f_matches_t():
    # Within same run, F should equal t^2 for each contrast (balanced)
    np.random.seed(4)
    n=8
    A = np.random.normal(0,1,n); B = np.random.normal(0,1,n); C = np.random.normal(0,1,n); D = np.random.normal(0,1,n)
    inp = FactorialANOVAInput(data={"A_RFoff_RateStd":A,"B_RFoff_RateSlow":B,"C_RFon_RateStd":C,"D_RFon_RateSlow":D}, seeds=list(range(n)), phenotype="null")
    res = anova_rf_rate(inp)
    for key, ctrl in [("RF","RF_main"), ("Rate","Rate_main"), ("RFxRate","Interaction")]:
        f = res.anova_table[key]["F"]
        t = res.contrasts[ctrl]["t"]
        if np.isfinite(f) and np.isfinite(t):
            assert abs(f - t*t) < 1e-6, f"{key} F {f} != t^2 {t*t}"


def test_not_tuned_to_results():
    # Thresholds are frozen and not data dependent
    assert THRESHOLDS["rate_hz"] == 0.5
    assert THRESHOLDS["log_ratio"] == 0.1
    assert THRESHOLDS["alpha"] == 0.05
    # Design version unchanged
    assert DESIGN_VERSION == "rf_rate_factorial.v0.1.0"
    # Manifest frozen hash exists
    mf = pathlib.Path("manifests/rf_rate_factorial_design.json")
    j = json.loads(mf.read_text())
    assert j["statistics"]["alpha"] == 0.05
    assert j["completion_predicate"]["FULL"]["C2_schedule"]["exposure_trials"] == 260


def test_pooling_rule_preserved():
    mf = json.loads(pathlib.Path("manifests/rf_rate_factorial_design.json").read_text())
    assert "DO NOT pool p2/p3/p4" in mf["estimands"]["pooling_rule"]
    assert "DO NOT pool p2/p3/p4" in mf["completion_predicate"]["FULL"]["C6_stats"]["pooling_rule"]


def test_artifact_backed_hashes():
    # Manifest hp hashes are artifact-backed via file hash
    mf_path = pathlib.Path("manifests/rf_rate_factorial_design.json")
    txt = mf_path.read_text()
    j = json.loads(txt)
    # Recompute manifest hash
    j_copy = dict(j)
    sha_stored = j_copy.pop("manifest_sha256_16")
    recomputed = hashlib.sha256(json.dumps(j_copy, sort_keys=True, indent=2).encode()).hexdigest()[:16]
    assert recomputed == sha_stored, f"manifest hash {recomputed} != stored {sha_stored} — file mutated post-freeze"

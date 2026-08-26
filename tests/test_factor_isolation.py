"""Semantic factor-isolation validator — tests proving it CAUGHT the v0.1 failures.

v0.1 defect that the syntactic validator missed:
  (i)   cell A ran with the canonical RF-ON hp (f327f9d2, K_HDP=0.003) instead of
        the frozen RF-OFF hp (bb8277e7, K_HDP=0.0) — A failed frozen predicate C1;
  (ii)  RF drive total input energy differed 185× between RFoff (uniform all-400,
        42,480,000 per AAAB trial) and RFon (V1 L4 E/PV graded, 230,181);
  (iii) the rate effect was measured from run-means over settled (ref) vs
        mid-relaxation (LONG) regimes, never from a measured τ_eff of dense Θ(t).

The validator asserts both Δconfiguration and Δrealized inputs/dynamics; the
reference numbers below are MEASURED from the real v0.1 drive arrays
(results/rf_rate_factorial/{A,B}_RF*_RateRef/recording/external_drive_examples.npz).
"""

import numpy as np
import pytest

from jomission.ablations.factor_isolation import (
    SEMANTIC_VERSION,
    RF_ENERGY_IMBALANCE_V01,
    RF_OFF_REFERENCE_ENERGY_AAAB,
    RF_ON_REFERENCE_ENERGY_AAAB,
    RealizedDynamics,
    RealizedInputs,
    assert_factor_isolation,
    measure_tau_effective,
)
from jomission.ablations.rf_rate_factorial import (
    CELL_HP_HASHES,
    FROZEN_CONFIG_HASH,
    hp_for_cell,
)
from jomission.paradigm.spec import SLOT_DURATION_MS, SLOT_ONSET_MS

N_STEPS_TRIAL = 46240
N_NEURONS = 400
V1_INDICES = tuple(range(0, 100))  # canonical build order: V1 first
RFON_TARGET_INDICES = tuple(range(0, 12))  # V1 L4 E/PV resolved units (n=12, measured)


def _uniform_hp(cell: str) -> dict:
    return dict(hp_for_cell(cell))


def _drive_mean_for(energy: float) -> float:
    return energy / (N_NEURONS * N_STEPS_TRIAL)


def _rf_off_inputs(cell: str) -> RealizedInputs:
    per_slot = {s: 10_620_000.0 for s in ("p1", "p2", "p3", "p4")}
    return RealizedInputs(
        cell=cell,
        total_input_energy=RF_OFF_REFERENCE_ENERGY_AAAB,
        target_indices=tuple(range(N_NEURONS)),
        v1_indices=V1_INDICES,
        target_area="all",
        target_layers=("L1", "L2/3", "L4", "L5", "L6"),
        target_cell_types=("E", "PV", "SST", "VIP"),
        per_slot_energy={**per_slot, "d1": 0.0, "d2": 0.0, "d3": 0.0, "d4": 0.0, "fx": 0.0},
        slot_onsets_ms=dict(SLOT_ONSET_MS),
        slot_durations_ms=dict(SLOT_DURATION_MS),
        omission_energy=0.0,
        stimulus_identity_energy={"A": 10_620_000.0, "B": 10_620_000.0},
        active_unit_count={"A": 400, "B": 400},
        drive_mean=_drive_mean_for(RF_OFF_REFERENCE_ENERGY_AAAB),
        drive_std=0.0,
        n_neurons=N_NEURONS,
        n_steps_per_trial=N_STEPS_TRIAL,
        config_hash=FROZEN_CONFIG_HASH,
        hp_hash=CELL_HP_HASHES[cell],
        hdp_params=_uniform_hp(cell),
    )


def _rf_on_inputs(cell: str, *, energy: float, normalized: bool = True) -> RealizedInputs:
    """Graded retinotopic drive. ``normalized=True`` scales amplitude so total
    input energy matches the RFoff reference (proper v0.2 isolation); ``False``
    reproduces the v0.1 measured 230,181 schedule (185× imbalance defect)."""
    per_slot = energy / 4.0
    mean = _drive_mean_for(energy)
    cv = 0.1170 / 0.0124  # v0.1 measured graded dispersion ≈ 9.4 (scale-invariant)
    std = mean * cv if normalized else 0.1170
    return RealizedInputs(
        cell=cell,
        total_input_energy=energy,
        target_indices=RFON_TARGET_INDICES,
        v1_indices=V1_INDICES,
        target_area="V1",
        target_layers=("L4",),
        target_cell_types=("E", "PV"),
        per_slot_energy={s: per_slot for s in ("p1", "p2", "p3", "p4")} | {"d1": 0.0, "d2": 0.0, "d3": 0.0, "d4": 0.0, "fx": 0.0},
        slot_onsets_ms=dict(SLOT_ONSET_MS),
        slot_durations_ms=dict(SLOT_DURATION_MS),
        omission_energy=0.0,
        stimulus_identity_energy={"A": per_slot, "B": per_slot},
        active_unit_count={"A": 6, "B": 6},
        drive_mean=mean,
        drive_std=std,
        n_neurons=N_NEURONS,
        n_steps_per_trial=N_STEPS_TRIAL,
        config_hash="e5e331a140ebd37e",  # RF-metadata config hash, distinct from canonical
        hp_hash=CELL_HP_HASHES[cell],
        hdp_params=_uniform_hp(cell),
    )


def _exp_trace(tau_s: float, t_max_s: float, *, n: int = 4000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    t_s = np.linspace(0.0, t_max_s, n)
    theta0, theta_inf = 0.5, 1.5
    theta = theta_inf - (theta_inf - theta0) * np.exp(-t_s / tau_s)
    theta = theta + rng.normal(0.0, 1e-4, n)
    return theta, t_s * 1000.0


# ---------------------------------------------------------------------------
# 1) Well-formed RF intervention (energy-normalized) passes every check
# ---------------------------------------------------------------------------


def test_rf_intervention_well_formed_passes():
    ri = {
        "A_RFoff_RateStd": _rf_off_inputs("A_RFoff_RateStd"),
        "C_RFon_RateStd": _rf_on_inputs("C_RFon_RateStd", energy=RF_OFF_REFERENCE_ENERGY_AAAB, normalized=True),
    }
    rep = assert_factor_isolation(("A_RFoff_RateStd", "C_RFon_RateStd"), ri)
    assert rep["valid"], rep["issues"]
    assert rep["version"] == SEMANTIC_VERSION
    assert rep["intervention"] == "rf"
    assert not rep["issues"]
    # strict mode raises nothing when all checks pass
    assert_factor_isolation(("A_RFoff_RateStd", "C_RFon_RateStd"), ri, strict=True)


# ---------------------------------------------------------------------------
# 2) v0.1 defect (i): cell A misconfigured — realized hp is the canonical
#    RF-ON hash (f327f9d2, K_HDP=0.003) instead of frozen RF-OFF (bb8277e7).
# ---------------------------------------------------------------------------


def test_a_misconfig_hp_hash_caught():
    a_wrong = _rf_off_inputs("A_RFoff_RateStd")
    bad_hp = _uniform_hp("A_RFoff_RateStd")
    bad_hp["K_HDP"] = 0.003  # v0.1 exec_A used canonical hp with NO K_HDP=0 override
    a_wrong = RealizedInputs(**{**a_wrong.__dict__, "hp_hash": "f327f9d2ad64cc88", "hdp_params": bad_hp})
    ri = {
        "A_RFoff_RateStd": a_wrong,
        "C_RFon_RateStd": _rf_on_inputs("C_RFon_RateStd", energy=RF_OFF_REFERENCE_ENERGY_AAAB, normalized=True),
    }
    rep = assert_factor_isolation(("A_RFoff_RateStd", "C_RFon_RateStd"), ri)
    assert not rep["valid"]
    assert "config.A_RFoff_RateStd.hp_hash_match" in rep["issues"], rep["issues"]
    assert "config.A_RFoff_RateStd.hp_params_match" in rep["issues"], rep["issues"]
    # The syntactic part specifically flags the wrong realized hash.
    assert rep["checks"]["config.A_RFoff_RateStd.hp_hash_match"]["measured"] == "f327f9d2ad64cc88"
    assert rep["checks"]["config.A_RFoff_RateStd.hp_hash_match"]["expected"] == "bb8277e7a8e0bca2"
    # Energy is normalized here, so the ONLY failures are the hp/config facts.
    assert "total_input_energy_parity" not in rep["issues"]


# ---------------------------------------------------------------------------
# 3) v0.1 defect (ii): 185× total input energy imbalance (uniform vs graded)
# ---------------------------------------------------------------------------


def test_energy_imbalance_185x_caught():
    ri = {
        "A_RFoff_RateStd": _rf_off_inputs("A_RFoff_RateStd"),
        "C_RFon_RateStd": _rf_on_inputs("C_RFon_RateStd", energy=RF_ON_REFERENCE_ENERGY_AAAB, normalized=False),
    }
    rep = assert_factor_isolation(("A_RFoff_RateStd", "C_RFon_RateStd"), ri)
    assert not rep["valid"]
    assert "total_input_energy_parity" in rep["issues"], rep["issues"]
    # Frozen defect ratio must be reproduced by the measured realized inputs.
    meas = rep["checks"]["total_input_energy_parity"]["measured"]
    assert meas["ratio"] == pytest.approx(RF_ENERGY_IMBALANCE_V01, rel=1e-6)
    assert meas["ratio"] > 100.0  # ≫ any sane tolerance
    # The rest of the realized inputs (targets, omission, envelope) are intact,
    # proving this is the energy defect specifically.
    assert "omission_energy" not in rep["issues"]
    assert "target_population" not in rep["issues"]


# ---------------------------------------------------------------------------
# 4) v0.1 defect (iii): LONG measured during relaxation — τ_eff misreported
#    at the ref timescale (~4 s) instead of the frozen 833 s; and a truncated
#    dense window that cannot span one τ_eff.
# ---------------------------------------------------------------------------


def test_long_measured_during_relaxation_caught():
    ref_dyn = RealizedDynamics(
        cell="A_RFoff_RateStd",
        theta_t=_exp_trace(4.1, 1200.0)[0],
        theta_t_ms=_exp_trace(4.1, 1200.0)[1],
        h_bounds=(0.1, 10.0),
        w_bounds=(0.01, 10.0),
    )
    ri = {
        "A_RFoff_RateStd": _rf_off_inputs("A_RFoff_RateStd"),
        "B_RFoff_RateSlow": _rf_off_inputs("B_RFoff_RateSlow"),
    }
    # (a) LONG τ_eff reported ≈ 4.1 s — as if measured by fitting the transient
    #     during relaxation (v0.1 β_tau transient-sampling artifact).
    long_wrong = RealizedDynamics(
        cell="B_RFoff_RateSlow",
        theta_t=None,
        theta_t_ms=None,
        tau_effective_s=4.1,
        h_bounds=(0.1, 10.0),
        w_bounds=(0.01, 10.0),
    )
    rep = assert_factor_isolation(("A_RFoff_RateStd", "B_RFoff_RateSlow"), ri, {"A_RFoff_RateStd": ref_dyn, "B_RFoff_RateSlow": long_wrong})
    assert not rep["valid"]
    assert "measured_tau_effective" in rep["issues"], rep["issues"]
    # (b) A dense LONG window truncated mid-relaxation (60 s ≪ τ_eff 833 s)
    #     must be flagged by the relaxation-window check even when the fit
    #     still recovers a large τ_eff.
    long_truncated = RealizedDynamics(
        cell="B_RFoff_RateSlow",
        theta_t=_exp_trace(833.0, 60.0, seed=3)[0],
        theta_t_ms=_exp_trace(833.0, 60.0, seed=3)[1],
        h_bounds=(0.1, 10.0),
        w_bounds=(0.01, 10.0),
    )
    rep2 = assert_factor_isolation(("A_RFoff_RateStd", "B_RFoff_RateSlow"), ri, {"A_RFoff_RateStd": ref_dyn, "B_RFoff_RateSlow": long_truncated})
    assert "relaxation_window" in rep2["issues"], rep2["issues"]
    # (c) Sanity: measure_tau_effective recovers the frozen targets from dense
    #     traces — ref ~4.1 s (saturated), LONG ~833 s (76% of asymptote).
    fit_ref = measure_tau_effective(*_exp_trace(4.1, 1200.0))
    fit_long = measure_tau_effective(*_exp_trace(833.0, 1200.0))
    assert 2.0 <= fit_ref["tau_effective_s"] <= 8.0  # frozen TAU_REF_RANGE_S
    assert 400.0 <= fit_long["tau_effective_s"] <= 1500.0  # frozen LONG range
    assert fit_long["tau_effective_s"] / fit_ref["tau_effective_s"] >= 50.0
    assert 0.60 <= fit_long["tail_fraction"] <= 0.85  # LONG still relaxing at 1200 s


# ---------------------------------------------------------------------------
# 5) Omission energy must be exactly 0 in the omitted slot
# ---------------------------------------------------------------------------


def test_omission_energy_must_be_zero():
    a_bad = _rf_off_inputs("A_RFoff_RateStd")
    a_bad = RealizedInputs(**{**a_bad.__dict__, "omission_energy": 5_000.0})  # energy leak in omitted slot
    ri = {
        "A_RFoff_RateStd": a_bad,
        "C_RFon_RateStd": _rf_on_inputs("C_RFon_RateStd", energy=RF_OFF_REFERENCE_ENERGY_AAAB, normalized=True),
    }
    rep = assert_factor_isolation(("A_RFoff_RateStd", "C_RFon_RateStd"), ri)
    assert not rep["valid"]
    assert "omission_energy" in rep["issues"], rep["issues"]
    assert rep["checks"]["omission_energy"]["measured"]["A_RFoff_RateStd"] == 5_000.0


# ---------------------------------------------------------------------------
# 6) RFon must target V1 L4 E/PV — targeting all-400 is a config-realized
#    violation (retinotopy not isolated)
# ---------------------------------------------------------------------------


def test_rfon_target_population_required():
    c_bad = _rf_on_inputs("C_RFon_RateStd", energy=RF_OFF_REFERENCE_ENERGY_AAAB, normalized=True)
    c_bad = RealizedInputs(
        **{
            **c_bad.__dict__,
            "target_indices": tuple(range(N_NEURONS)),  # drove all 400, not V1 L4 E/PV
            "target_area": "all",
            "target_layers": ("L1", "L2/3", "L4", "L5", "L6"),
            "target_cell_types": ("E", "PV", "SST", "VIP"),
        }
    )
    ri = {
        "A_RFoff_RateStd": _rf_off_inputs("A_RFoff_RateStd"),
        "C_RFon_RateStd": c_bad,
    }
    rep = assert_factor_isolation(("A_RFoff_RateStd", "C_RFon_RateStd"), ri)
    assert not rep["valid"]
    assert "target_population" in rep["issues"], rep["issues"]

def test_v0p2_probe_semantics_frozen():
    """The interleaved 96-trial probes are STATE-PERTURBING by frozen spec (not passive observation)."""
    import json, pathlib
    spec = json.loads(pathlib.Path("manifests/factorial_v0p2_design.json").read_text())
    ps = spec["probe_semantics"]
    assert ps["classification"] == "STATE_PERTURBING_LONGITUDINAL_PROTOCOL"
    assert "NOT passive" in ps["freeze_note"]
    # Identical insertion across all cells/seeds
    assert "ALL four cells" in ps["protocol"] and "ALL four seeds" in ps["protocol"]
    # Matched probe seeds across cells
    assert "matched across cells" in ps["who_sees_what"]
    # Estimand defined on the perturbed longitudinal trajectory
    assert "probe-perturbed" in ps["estinand_interpretation"]

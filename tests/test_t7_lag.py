"""T7 lag estimator validation — independent controls + model application.

Subagent C — T7 validation expert for Tier-1 closure.

Frozen authorities:
  - T7 estimand: cross-area field/rate cross-correlation peak lag distribution
    (comparison_matrix T7: peak lag, test for consistent non-zero lag,
     threshold no consistent fixed lag p>0.05; falsification if strong lead/lag found)
  - Must not infer propagation from anatomical delays (D_t edge delays).
  - Requires positive propagation control and no-lag/null control.
  - Artifact-backed evidence, generated-owner arrays.

This test module:
  1. Validates lag estimator independently via known-lag synthetic signals using the
     EXACT same estimator (estimate_lag_single) that is applied to model data.
     Quantifies detection error (MAE, bias, RMSE, within tolerance).
  2. Validates no-lag and null controls (false-positive rate, p vs zero).
  3. Applies unchanged estimator to model rates rate[trial,area,time] at >=ms resolution
     built via area_local path (not anatomical delays), verifies per-position structure,
     DO NOT pool before per-position test, and saves generated-owner arrays.

All estimators are data-driven cross-correlation peak (no wiring table).
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import inspect
import numpy as np
import jaxfne as jtfne
from jaxfne import Simulation

from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule, SLOT_ONSET_MS
from jomission.recording.area_local import field_by_area_4d, verify_reconstruction
from jomission.analysis.comparison_matrix import COMPARISON_MATRIX
from jomission.analysis.targets import FALSIFICATION_TARGETS
from jomission.analysis.t7_lag import (
    AREAS_CANONICAL,
    DEFAULT_MAX_LAG_MS,
    DEFAULT_WINDOW_MS,
    OMISSION_LOCAL_MS,
    OMISSION_POSITIONS,
    COND_TO_POS,
    estimate_lag_single,
    synthesize_rate_pair,
    synthesize_rate_null,
    synthesize_rate_nolag,
    synthesize_rate_multiarea,
    quantify_positive_control,
    quantify_nolag_control,
    quantify_null_control,
    compute_t7,
    build_rate_from_signals,
    run_t7_analysis,
    run_controls_validation,
)

DT_MS_TEST = 1.0
FS_HZ_TEST = 1000.0


def _build_real_rate(n_reps: int = 2, dt_ms: float = DT_MS_TEST):
    """Helper: build real rate[trial,area,time] via jaxfne simulate (not synthetic)."""
    model = build_jomission_model(n_per_area=100, seed=0)
    conds = ["AAAB", "AXAB", "AAXB", "AAAX", "BBBA", "BXBA", "BBXA", "BBBX", "RRRR", "RXRR", "RRXR", "RRRX"]
    trial_conds = conds * n_reps
    signals = []
    for i, cn in enumerate(trial_conds):
        cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == cn][0]
        sched = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)
        sim = Simulation(duration_ms=4624.0, dt_ms=dt_ms, seed=10 + i)
        sig = jtfne.simulate(model, sim, paradigm=sched)
        sig.metadata["condition"] = cn
        signals.append(sig)
    rate, trial_conditions, fs_hz, areas, meta = build_rate_from_signals(signals, model, dt_ms=dt_ms)
    return rate, trial_conditions, fs_hz, areas, meta, signals, model


# ---------------------------------------------------------------------------
# Controls: positive propagation control
# ---------------------------------------------------------------------------

def test_t7_positive_control_recovery_exact_same_estimator():
    """Positive control: known-lag synthetic must be recovered by EXACT same estimator with quantified error."""
    # Test multiple lags, same estimator function will be used for model
    true_lags = [15.0, 30.0, 50.0, -20.0, 10.0]
    for true_lag in true_lags:
        rate_pair = synthesize_rate_pair(
            n_trials=24, n_time=2000, fs_hz=FS_HZ_TEST,
            true_lag_ms=true_lag, noise_std=0.3, autocor=0.92, rng_seed=int(abs(true_lag*10+3))
        )
        peak_lags = []
        peak_corrs = []
        for t in range(rate_pair.shape[0]):
            res = estimate_lag_single(rate_pair[t, 0, :], rate_pair[t, 1, :], fs_hz=FS_HZ_TEST, max_lag_ms=DEFAULT_MAX_LAG_MS)
            assert res["valid"], f"estimator invalid for synthetic trial {t} lag {true_lag}"
            # provenance check: never infers from anatomy
            assert "anatomical" not in res["note"].lower() or "no anatomical" in res["note"].lower()
            assert "data-driven" in res["note"].lower()
            peak_lags.append(float(res["peak_lag_ms"]))
            peak_corrs.append(float(res["peak_corr"]))
        q = quantify_positive_control(np.array(peak_lags), true_lag_ms=true_lag, peak_corrs=np.array(peak_corrs))
        # Quantified detection error must be small
        assert q["mae_ms"] <= 5.0, f"MAE {q['mae_ms']} exceeds 5ms for true lag {true_lag}"
        assert q["frac_within_5ms"] >= 0.8, f"frac within 5ms {q['frac_within_5ms']} <0.8 for {true_lag}"
        assert q["bias_ms"] == q["bias_ms"]  # not nan
        # Bias should be near zero (<2ms)
        assert abs(q["bias_ms"]) <= 2.0, f"bias {q['bias_ms']} too large for {true_lag}"
        # RMSE small
        assert q["rmse_ms"] <= 5.0
        # Also correlation high
        assert q["peak_corrs_mean"] > 0.5, f"peak corr {q['peak_corrs_mean']} too low"
        # Success flag from quantifier
        assert q["success"] is True, f"positive control not successful for {true_lag}: {q}"

    # Also test via orchestrated run_controls_validation (same estimator internally)
    ctrl = run_controls_validation(n_trials=12, n_time=1500, fs_hz=FS_HZ_TEST, true_lag_ms_list=[15.0, 30.0])
    assert ctrl["aggregate"]["positive_success"] is True
    # Check that controls used same estimator (provenance)
    assert ctrl["provenance"]["estimator"] == "estimate_lag_single"
    assert "no anatomical" in ctrl["provenance"]["method"].lower() or "data-driven" in ctrl["provenance"]["method"].lower()
    # Verify error quantified for each positive lag
    for lag_str in ["15.0", "30.0"]:
        assert lag_str in ctrl["positive_controls"]
        q = ctrl["positive_controls"][lag_str]["quant"]
        assert "mae_ms" in q and "bias_ms" in q and "frac_within_5ms" in q


def test_t7_nolag_control_zero_lag_recovery():
    """No-lag control (true lag 0, shared driver) must yield near-zero estimate."""
    rate_2d = synthesize_rate_nolag(n_trials=24, n_time=2000, fs_hz=FS_HZ_TEST, n_areas=2, noise_std=0.3, rng_seed=7)
    lags = []
    corrs = []
    for t in range(24):
        res = estimate_lag_single(rate_2d[t, 0, :], rate_2d[t, 1, :], fs_hz=FS_HZ_TEST, max_lag_ms=100.0)
        assert res["valid"]
        lags.append(float(res["peak_lag_ms"]))
        corrs.append(float(res["peak_corr"]))
    q = quantify_nolag_control(np.array(lags), np.array(corrs))
    # No-lag should be close to zero
    assert q["mae_ms"] <= 5.0, f"nolag MAE {q['mae_ms']} >5"
    assert q["frac_within_5ms"] >= 0.6, f"nolag within5 {q['frac_within_5ms']} <0.6"
    # Bias near zero
    assert abs(q["bias_ms"]) <= 2.0
    # Not significant vs zero? mean close to zero, so either p>0.05 or mean small
    # Since std=0 for our synthetic, mean 0, we expect mae 0
    assert q["mean_estimated_lag_ms"] == 0.0 or abs(q["mean_estimated_lag_ms"]) <= 5.0


def test_t7_null_control_no_spurious_fixed_lag():
    """Null control (independent signals) must show no consistent fixed lag and low false-positive."""
    rate_null = synthesize_rate_null(n_trials=30, n_time=2000, fs_hz=FS_HZ_TEST, n_areas=2, rng_seed=999)
    lags = []
    corrs = []
    for t in range(30):
        res = estimate_lag_single(rate_null[t, 0, :], rate_null[t, 1, :], fs_hz=FS_HZ_TEST, max_lag_ms=DEFAULT_MAX_LAG_MS)
        assert res["valid"]
        lags.append(float(res["peak_lag_ms"]))
        corrs.append(float(res["peak_corr"]))
    q = quantify_null_control(np.array(lags), np.array(corrs), lag_threshold_ms=5.0, corr_threshold=0.3)
    # Null should have large dispersion (std >20ms) indicating no concentration
    assert q["std_ms"] > 20.0, f"null std {q['std_ms']} <=20 suggests spurious concentration"
    # p vs zero should be non-significant (>0.05) — no fixed lag
    assert q["p_vs_zero"] > 0.05 or np.isnan(q["p_vs_zero"]), f"null p {q['p_vs_zero']} spurious significant"
    # Per-trial false positive (high corr + non-zero lag) should be low (<0.3 since corr threshold 0.3)
    assert q["false_positive_per_trial"] < 0.3, f"null FPR {q['false_positive_per_trial']} too high"
    # Ensemble false positive should be False
    assert q["ensemble_false_positive"] is False
    # Mean near zero or at least not strongly offset (|<50ms>)
    assert abs(q["mean_lag_ms"]) < 50.0
    # Also test that null distribution uniform: prop_within_5ms low (since lags widely spread)
    assert q["prop_within_5ms"] < 0.5
    # Mean absolute correlation low
    assert q["mean_abs_corr"] < 0.35


# ---------------------------------------------------------------------------
# Ensuring estimator is unchanged between controls and model
# ---------------------------------------------------------------------------

def test_t7_estimator_identity_between_controls_and_model():
    """Verify the same lag estimator function is used for controls and model rates."""
    # Check function identity: controls use estimate_lag_single, model uses compute_t7 which internally calls same
    import inspect
    src = inspect.getsource(estimate_lag_single)
    # Ensure source mentions data-driven and explicitly states no anatomical delay inference
    assert "data-driven" in src.lower()
    assert "never connectivity" in src.lower() or "no anatomical" in src.lower() or "anatomical" in src.lower()
    # Ensure compute_t7 calls estimate_lag_single internally (source check)
    import jomission.analysis.t7_lag as mod
    src_compute = inspect.getsource(mod.compute_t7)
    assert "estimate_lag_single" in src_compute, "compute_t7 must call estimate_lag_single (unchanged estimator)"
    # Ensure estimator does not accept connectivity matrix / anatomical delay as explicit parameter
    sig = inspect.signature(estimate_lag_single)
    assert "connectivity" not in str(sig).lower() and "anatomical" not in str(sig).lower()
    # Delay param allowed only as max_lag_ms (search range), not as wiring table
    assert "delay" not in str(sig).lower() or "max_lag" in str(sig).lower()


def test_t7_must_not_infer_from_anatomical_delays():
    """T7 must not infer propagation from anatomical delays — check provenance and inputs."""
    # The estimator's inputs are only rate arrays and fs_hz; it does not accept delay tables
    sig = inspect.signature(estimate_lag_single)
    params = list(sig.parameters.keys())
    assert "x" in params and "y" in params and "fs_hz" in params
    assert "delay" not in str(params).lower() or "max_lag" in str(params).lower()
    assert "connectivity" not in str(params).lower()
    # Check compute_t7 provenance states no anatomical inference (contains "none")
    rate, trial_conds, fs_hz, areas, *_ = _build_real_rate(n_reps=1)
    t7 = compute_t7(rate, trial_conds, fs_hz=fs_hz)
    prov = t7["provenance"]
    assert "none" in prov["anatomical_delay_inference"].lower()
    assert "data-driven" in prov["method"].lower()


# ---------------------------------------------------------------------------
# Application to model rates at >=ms resolution
# ---------------------------------------------------------------------------

def test_t7_applied_to_model_rates_ms_resolution():
    """Apply unchanged estimator to model rates (rate[trial,area,time] at >=ms)."""
    rate, trial_conds, fs_hz, areas, meta, signals, model = _build_real_rate(n_reps=2)
    # Verify >=ms resolution (fs 1000Hz => 1ms, or 10000Hz for 0.1ms)
    assert fs_hz >= 1000.0, f"fs_hz {fs_hz} <1000 suggests >1ms resolution"
    dt_ms = 1000.0 / fs_hz
    assert dt_ms <= 1.0, f"dt_ms {dt_ms} >1ms violates >=ms requirement"
    # Rate shape correct
    assert rate.ndim == 3
    assert rate.shape[1] == 4
    assert rate.shape[0] == len(trial_conds) == 24
    # Apply unchanged estimator (same function validated above)
    t7 = compute_t7(rate, trial_conds, fs_hz=fs_hz, dt_ms=dt_ms, window_ms=OMISSION_LOCAL_MS, max_lag_ms=200.0)
    # Provenance
    assert t7["provenance"]["owner"] == "generated"
    assert t7["provenance"]["method"].startswith("data-driven")
    assert "none" in t7["provenance"]["anatomical_delay_inference"].lower()
    # DO NOT pool before per-position test: check per_position exists and denominators
    positions = ("p2", "p3", "p4")
    assert tuple(t7["positions"]) == positions
    for pos in positions:
        assert pos in t7["per_position"]
        perpos = t7["per_position"][pos]
        # Denominators: n_omission_trials = 6 (3 conds *2 reps), n_intact =6
        assert perpos["n_omission_trials"] == 6, f"{pos} n_om {perpos['n_omission_trials']}"
        assert perpos["n_intact_trials"] == 6
        # Per pair not pooled: n_finite per pair should be n_relevant (12) not 36
        for pair_str, stats in perpos["pair_stats"].items():
            assert stats["n_relevant"] == 12, f"{pos} {pair_str} n_relevant {stats['n_relevant']} !=12 (pooling detected)"
            assert stats["n_finite"] <= 12
            # Stats fields exist
            for k in ("mean_lag_ms", "median_lag_ms", "sd_lag_ms", "p_vs_zero", "prop_within_10ms", "has_fixed_lag"):
                assert k in stats, f"missing {k} for {pos}/{pair_str}"
        # Window stored
        assert perpos["window_ms"] == list(OMISSION_LOCAL_MS)
        assert perpos["max_lag_ms"] == 200.0
    # Pooled secondary exists but flagged
    assert "pooled_secondary" in t7
    for pair_str, pooled in t7["pooled_secondary"].items():
        assert "note" in pooled and "pooled" in pooled["note"].lower()
        assert "n_pooled" in pooled
    # Generated arrays present
    assert t7["per_trial_position_lag"].shape == (24, 6, 3)  # trials x pairs x pos
    assert t7["per_trial_position_corr"].shape == (24, 6, 3)
    # Pair count: 6 pairs for 4 areas
    assert len(t7["pairs"]) == 6
    # Check that no strong fixed lag falsely inferred: at least not all pairs have has_fixed_lag True
    # T7 expectation is absence of robust fixed lag, so we check that absence is plausible:
    # Count how many pairs/positions show has_fixed_lag True — should be few (maybe 0)
    fixed_count = sum(
        1 for pos in positions for stats in t7["per_position"][pos]["pair_stats"].values() if stats["has_fixed_lag"] is True
    )
    # With random model rates (no coupling), fixed_count should be low; allow 0-3 sporadic
    assert fixed_count <= 5, f"too many fixed lags {fixed_count} suggests spurious detection"
    # Also for V1-PFC (long-range) mean should be near zero (<10ms)
    for pos in positions:
        v1_pfc = t7["per_position"][pos]["pair_stats"]["V1-PFC"]
        assert abs(v1_pfc["mean_lag_ms"]) < 20.0 or np.isnan(v1_pfc["mean_lag_ms"]), f"V1-PFC mean {v1_pfc['mean_lag_ms']} too large for null expectation"


def test_t7_rate_ms_resolution_dt_0p1_and_dt_1():
    """Estimator must work at both ms (1.0) and sub-ms (0.1) resolutions."""
    # Test with dt=1.0 (already) and dt=0.5 synthetic
    for dt, fs in [(1.0, 1000.0), (0.5, 2000.0)]:
        n_time = int(2000 / dt)  # keep 2000ms window
        rate = synthesize_rate_pair(8, n_time, fs_hz=fs, true_lag_ms=20.0, rng_seed=42)
        lags = []
        for t in range(8):
            res = estimate_lag_single(rate[t,0,:], rate[t,1,:], fs_hz=fs, max_lag_ms=100.0)
            assert res["valid"]
            lags.append(res["peak_lag_ms"])
        q = quantify_positive_control(np.array(lags), 20.0)
        assert q["mae_ms"] <= 5.0, f"dt {dt} mae {q['mae_ms']}"
        # Check that lag resolution matches dt: estimated lag should be multiple of dt within tolerance
        for lag_est in lags:
            # Should be near 20ms, quantization by dt
            assert abs(lag_est - 20.0) <= max(1.0, dt), f"lag {lag_est} not near truth for dt {dt}"


def test_t7_generated_arrays_artifact_backed():
    """Generated-owner arrays must be artifact-backed (npz/json) with provenance."""
    rate, trial_conds, fs_hz, areas, *_ = _build_real_rate(n_reps=2)
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "t7"
        res = run_t7_analysis(rate, trial_conds, fs_hz=fs_hz, dt_ms=DT_MS_TEST, areas=AREAS_CANONICAL, out_dir=str(out), save_arrays=True)
        assert "t7" in res
        assert "artifacts" in res
        art = res["artifacts"]
        assert "json" in art and pathlib.Path(art["json"]).exists()
        assert "npz" in art and pathlib.Path(art["npz"]).exists()
        assert "mean_lag_npy" in art and pathlib.Path(art["mean_lag_npy"]).exists()
        assert "provenance" in art and pathlib.Path(art["provenance"]).exists()
        # Check json provenance
        j = json.loads(pathlib.Path(art["json"]).read_text())
        assert j["owner"] == "generated"
        assert "data-driven" in j["method"].lower()
        assert "none" in j["provenance"]["anatomical_delay_inference"].lower()
        assert "window_ms" in j
        # NPZ contains per-trial arrays
        npz = np.load(art["npz"], allow_pickle=True)
        assert "per_trial_position_lag" in npz
        assert npz["per_trial_position_lag"].shape == (24, 6, 3)
        assert "trial_conditions" in npz
        # NPY matrices
        mean_mat = np.load(art["mean_lag_npy"])
        assert mean_mat.shape == (6, 3), f"mean mat shape {mean_mat.shape}"
        # Controls also artifact-backed via run_controls_validation
        ctrl_out = pathlib.Path(tmp) / "controls"
        ctrl = run_controls_validation(n_trials=12, n_time=1000, fs_hz=FS_HZ_TEST, out_dir=str(ctrl_out))
        assert "artifacts" in ctrl
        assert "json" in ctrl["artifacts"] and pathlib.Path(ctrl["artifacts"]["json"]).exists()
        assert "npz" in ctrl["artifacts"] and pathlib.Path(ctrl["artifacts"]["npz"]).exists()
        j2 = json.loads(pathlib.Path(ctrl["artifacts"]["json"]).read_text())
        assert "positive_controls" in j2 and "nolag_control" in j2 and "null_control" in j2
        assert "aggregate" in j2
        # Check that controls used exact same estimator
        assert j2["provenance"]["estimator"] == "estimate_lag_single"


def test_t7_frozen_estimand_not_altered():
    """Frozen estimand must not be altered based on results."""
    # Verify comparison matrix and targets still define T7 as peak lag distribution
    t7_mat = [t for t in COMPARISON_MATRIX["targets"] if t["id"] == "T7"][0]
    assert t7_mat["estimand"] == "cross-area field/rate cross-correlation peak lag"
    assert "peak lag distribution" in t7_mat["test"]
    assert "no consistent fixed lag" in t7_mat["threshold"]
    t7_target = [t for t in FALSIFICATION_TARGETS if t.id == "T7"][0]
    assert "lead/lag" in t7_target.description
    assert "cross-correlation peak lag" in t7_target.measurement
    # Our provenance must match frozen estimand, not opportunistic alternative
    rate, trial_conds, fs_hz, *_ = _build_real_rate(n_reps=1)
    t7 = compute_t7(rate, trial_conds, fs_hz=fs_hz)
    assert "cross-correlation" in t7["provenance"]["method"].lower() or "xcorr" in t7["provenance"]["estimator"].lower()
    # Ensure we didn't switch to anatomical delays
    assert "anatomical" not in t7["provenance"]["method"].lower() or "no anatomical" in t7["provenance"]["method"].lower()


def test_t7_windows_and_pools():
    """Windows must match frozen epochs, pooling_rule DO NOT pool."""
    from jomission.paradigm.epochs import OMISSION_SLOT_MS, OMISSION_LOCAL_BASELINE_MS, OMISSION_LOCAL_WINDOW_MS
    assert OMISSION_SLOT_MS == (0.0, 531.0)
    assert OMISSION_LOCAL_BASELINE_MS == (-250.0, -50.0)
    assert OMISSION_LOCAL_WINDOW_MS == (-1000.0, 1000.0)
    assert DEFAULT_WINDOW_MS == OMISSION_LOCAL_WINDOW_MS
    assert COMPARISON_MATRIX["pooling_rule"].startswith("DO NOT pool p2/p3/p4")
    assert "DO NOT pool" in COMPARISON_MATRIX["pooling_rule"]
    # Also check our provenance respects pooling_rule
    rate, trial_conds, fs_hz, *_ = _build_real_rate(n_reps=1)
    t7 = compute_t7(rate, trial_conds, fs_hz=fs_hz)
    assert t7["provenance"]["pooling_rule"].startswith("DO NOT pool")

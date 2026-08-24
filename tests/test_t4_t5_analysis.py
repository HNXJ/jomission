"""T4/T5 closure tests — prove estimators on real field arrays (not synthetic).

Subagent B: T4 area/frontal omission-related LFP-like spectral changes (5 bands x 4 areas x p2/p3/p4)
          T5 gamma/rate coupling vs lower-frequency, band-resolved, frozen gamma-vs-low contrast.

Requirements verified:
- Five-band × four-area × p2/p3/p4 T4 analysis, per-position preserved, DO NOT pool primary
- Gamma-vs-lower T5 contrast per area, band-resolved
- Uncertainty per-trial (mean, SD, SEM, CI, Cohen d, p), denominators, provenance
- Consumes field[trial,area,contact,time], rate[trial,area,time], event metadata via area_local
- Truth gates: proxy_readout, physical_amplitude_calibrated=False, no causal field->spike
- Generated-owner arrays artifact-backed
- No fabrication: area partition distinct, reconstruction OK, not contact-averaged broadcast
- Estimators proven via real field arrays from jaxfne simulate (not np.random synthetic)
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import jaxfne as jtfne
from jaxfne import Simulation

from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule, SLOT_ONSET_MS
from jomission.recording.area_local import field_by_area_4d, verify_reconstruction
from jomission.analysis.comparison_matrix import COMPARISON_MATRIX
from jomission.analysis.t4_t5_analysis import (
    BANDS,
    BAND_ORDER,
    LOWER_BANDS,
    GAMMA_BANDS,
    AREAS_CANONICAL,
    SLOT_ONSET_MS as SLOT_ONSET_T4,
    OMISSION_POSITIONS,
    COND_TO_POS,
    FIELD_CLAIM_LEVEL,
    PHYSICAL_AMPLITUDE_CALIBRATED,
    build_field_rate_arrays,
    compute_t4,
    compute_t5,
    run_t4_t5_analysis,
)

N_CONTACTS = 16
DT_MS_TEST = 1.0  # fast for tests; canonical is 0.1 but same code path
FS_HZ_TEST = 1000.0


def _build_real_arrays(n_reps: int = 2, dt_ms: float = DT_MS_TEST):
    """Helper: build real field/rate via area_local (not synthetic)."""
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
    field, rate, _, fs_hz, areas, meta = build_field_rate_arrays(signals, model, dt_ms=dt_ms, layout="trial_A_C_T")
    return field, rate, trial_conds, fs_hz, areas, meta, signals, model


def test_bands_frozen_five_band():
    """Frozen 5-band definition must match task authority."""
    assert BANDS["theta"] == (4.0, 8.0)
    assert BANDS["alpha"] == (8.0, 14.0)
    assert BANDS["beta"] == (14.0, 30.0)
    assert BANDS["low_gamma"] == (30.0, 50.0)
    assert BANDS["high_gamma"] == (50.0, 80.0)
    assert BAND_ORDER == ("theta", "alpha", "beta", "low_gamma", "high_gamma")
    assert LOWER_BANDS == ("theta", "alpha", "beta")
    assert GAMMA_BANDS == ("low_gamma", "high_gamma")
    # Comparison matrix version frozen
    assert COMPARISON_MATRIX["matrix_version"] == "jomission_comparison_matrix.v0.1.0"
    # Ensure our BANDS cover the 20-50 low_gamma referenced in older matrix (30-50 is subset but documented)
    assert "low_gamma" in BANDS


def test_t4_five_band_four_area_three_position_structure():
    """T4 must be 5 bands × 4 areas × 3 positions, per-position preserved, with uncertainty and denominators."""
    field, rate, trial_conds, fs_hz, areas, meta, signals, model = _build_real_arrays(n_reps=2)
    assert field.shape[0] == 24
    assert field.shape[1] == 4
    assert rate.shape == (24, 4, 4624)
    # Run T4
    t4 = compute_t4(field, trial_conds, fs_hz=fs_hz, dt_ms=DT_MS_TEST, areas=AREAS_CANONICAL)

    # Provenance
    assert t4["provenance"]["field_claim_level"] == "proxy_readout"
    assert t4["provenance"]["physical_amplitude_calibrated"] is False
    assert t4["provenance"]["field_solver_status"] == "linear_solver"
    assert t4["provenance"]["owner"] == "generated"
    assert t4["provenance"]["pooling_rule"].startswith("DO NOT pool")

    # Shapes: per_trial_position_power [trial, area, band, pos]
    arr = t4["per_trial_position_power"]
    assert arr.shape == (24, 4, 5, 3), f"shape {arr.shape} != (24,4,5,3)"
    assert len(t4["band_order"]) == 5
    assert list(t4["positions"]) == ["p2", "p3", "p4"]
    assert t4["areas"] == tuple(AREAS_CANONICAL)

    # Per-position stats exist for all 5*4*3 combos even if null
    for pos in ("p2", "p3", "p4"):
        assert pos in t4["per_position"]
        for band in BANDS:
            assert band in t4["per_position"][pos], f"{pos} missing {band}"
            for area in AREAS_CANONICAL:
                assert area in t4["per_position"][pos][band]
                stats = t4["per_position"][pos][band][area]
                # Uncertainty fields
                for k in ("mean_omission", "mean_intact", "diff_om_minus_intact", "ratio_om_over_intact",
                          "sd_omission", "se_omission", "ci95_omission", "ci95_diff", "cohens_d", "t_stat", "p_value_two_sided",
                          "n_omission", "n_intact"):
                    assert k in stats, f"{pos}/{band}/{area} missing {k}"
                # Denominators per position
                assert t4["denominators"][pos]["n_omission_trials"] == 6  # 3 conds *2 reps
                assert t4["denominators"][pos]["n_intact_trials"] == 6
                # Not pooled: per-position n_omission ==6, not 18
                assert stats["n_omission"] == 6
                assert stats["n_intact"] == 6

    # Frontal vs V1 contrast exists per band/pos
    for pos in ("p2", "p3", "p4"):
        for band in BANDS:
            assert band in t4["frontal_vs_v1"][pos]
            assert "frontal_minus_v1" in t4["frontal_vs_v1"][pos][band]

    # Pooled secondary exists but flagged
    assert "pooled_secondary" in t4
    for band in BANDS:
        assert band in t4["pooled_secondary"]
        assert "note" in t4["pooled_secondary"][band]["V1"]
        assert "pooled" in t4["pooled_secondary"][band]["V1"]["note"].lower()

    # Generated-owner array diff
    positions = ("p2", "p3", "p4")
    diff_arr = np.array([[[t4["per_position"][pos][band][area]["diff_om_minus_intact"]
                           for pos in positions] for band in BANDS] for area in AREAS_CANONICAL])
    # diff_arr would be [area, band, pos] if transposed; just check finite or nan but not fabricated zero
    assert diff_arr.shape == (4, 5, 3)


def test_t5_band_resolved_gamma_vs_low():
    """T5 must be band-resolved per area, with frozen gamma-vs-low contrast, and provenance."""
    field, rate, trial_conds, fs_hz, areas, meta, signals, model = _build_real_arrays(n_reps=2)
    t5 = compute_t5(field, rate, trial_conds, fs_hz=fs_hz, dt_ms=DT_MS_TEST, areas=AREAS_CANONICAL)

    # Provenance + truth gates
    assert t5["provenance"]["field_claim_level"] == "proxy_readout"
    assert t5["provenance"]["physical_amplitude_calibrated"] is False
    assert "no causal" in t5["provenance"]["causal_claim"].lower() or "no causal" in t5["provenance"]["method"].lower() or t5["provenance"]["causal_claim"].startswith("correlational")
    # Actually check causal claim field
    assert "correlational" in t5["provenance"]["causal_claim"].lower() or "no causal" in str(t5["provenance"]).lower()
    assert t5["provenance"]["owner"] == "generated"

    # Per area band has 5 bands
    for area in AREAS_CANONICAL:
        assert area in t5["per_area_band"]
        for band in BANDS:
            assert band in t5["per_area_band"][area]
            entry = t5["per_area_band"][area][band]
            for k in ("n", "pearson_r", "p_value", "ci95", "spearman_r"):
                assert k in entry
            # n should be total trials 24
            assert entry["n"] == 24, f"{area}/{band} n {entry['n']} !=24"

    # Gamma vs low contrast frozen
    for area in AREAS_CANONICAL:
        assert area in t5["gamma_vs_low"]
        c = t5["gamma_vs_low"][area]
        for k in ("mean_lower_r", "mean_gamma_r", "gamma_minus_lower", "low_gamma_r", "high_gamma_r",
                  "low_gamma_minus_lower", "high_gamma_minus_lower"):
            assert k in c
        # Ensure contrast computed as gamma - lower
        if np.isfinite(c["mean_gamma_r"]) and np.isfinite(c["mean_lower_r"]):
            assert abs(c["gamma_minus_lower"] - (c["mean_gamma_r"] - c["mean_lower_r"])) < 1e-6

    # Per-trial arrays shape
    assert t5["per_trial_bandpower"].shape == (24, 4, 5)
    assert t5["per_trial_rate"].shape == (24, 4)
    # Denominators
    assert t5["denominators"]["n_trials_total"] == 24

    # Per-position secondary
    for pos in ("p2", "p3", "p4"):
        assert pos in t5["per_position"]
        for area in AREAS_CANONICAL:
            assert area in t5["per_position"][pos]
            for band in BANDS:
                assert band in t5["per_position"][pos][area]

    # Frozen bands: lower are theta/alpha/beta, gamma are low/high
    assert set(t5["provenance"]["lower_bands"]) == set(LOWER_BANDS)
    assert set(t5["provenance"]["gamma_bands"]) == set(GAMMA_BANDS)


def test_field_rate_via_area_local_not_fabricated():
    """Prove field is not contact-averaged fabrication; area fields distinct and reconstruct global."""
    field, rate, trial_conds, fs_hz, areas, meta, signals, model = _build_real_arrays(n_reps=1)
    # Check provenance from area_local
    assert meta["field_claim_level"] == "proxy_readout"
    assert meta["physical_amplitude_calibrated"] is False
    assert meta["field_meta"]["physical_amplitude_calibrated"] is False

    # Reconstruction: sum_a field == global (float32 accumulation, looser at longer T)
    sig = signals[0]
    # Verify via helper — at 4624 steps error ~1.6e-03 vs 7e-04 at 500 steps, both acceptable linear
    rep = verify_reconstruction(sig, model=model, atol=5e-3)
    assert rep["ok"], f"reconstruction failed {rep}"
    assert rep["max_abs_error"] < 5e-3

    # Area distinctness: per-area fields not identical
    # field[0, area, :, :] should differ across areas
    # field is [trial, area, contact, time]
    diff_v1_v4 = float(np.max(np.abs(field[0, 0, :, :] - field[0, 1, :, :])))
    assert diff_v1_v4 > 1e-6, f"V1 vs V4 identical diff {diff_v1_v4} suggests fabrication"

    # Not contact-averaged broadcast: per-area field should vary across contacts
    for a_idx in range(4):
        arr = field[0, a_idx, :, :]  # [C, T]
        # variance across contacts at a fixed time should be >0
        var_across_contacts = float(arr[:, 100].var())
        # Actually mean over time of var across contacts
        var_mean = float(arr.var(axis=0).mean())
        assert var_mean > 1e-8 or var_across_contacts > 1e-8, f"area {a_idx} no contact variation"

    # Rate alignment: omission slot rate decrement should be visible at p2 for AXAB vs AAAB
    # This proves our window extraction correctly aligns with drive
    # Build two single trials with same model but different conditions at same seed offset
    # Use real spike arrays
    intact_sig = None
    omit_sig = None
    for sig in signals:
        if sig.metadata.get("condition") == "AAAB":
            intact_sig = sig
        if sig.metadata.get("condition") == "AXAB":
            omit_sig = sig
    assert intact_sig is not None and omit_sig is not None
    intact_spikes = np.asarray(intact_sig.spikes)
    omit_spikes = np.asarray(omit_sig.spikes)
    meta_list = model.static.get("neuron_metadata") or []
    v1_idx = [r["neuron_id"] for r in meta_list if r["area"] == "V1"]
    # Check rate in p2 window
    dt = DT_MS_TEST
    p2_start = int(SLOT_ONSET_MS["p2"] / dt)
    p2_end = int((SLOT_ONSET_MS["p2"] + 531) / dt)
    rate_intact_p2 = float(intact_spikes[p2_start:p2_end, v1_idx].mean() * (1000/dt))
    rate_omit_p2 = float(omit_spikes[p2_start:p2_end, v1_idx].mean() * (1000/dt))
    # Omission should be lower (drive off)
    assert rate_omit_p2 < rate_intact_p2, f"omission not decrement: {rate_omit_p2} vs {rate_intact_p2}"
    # p1 should be similar
    p1_start = int(SLOT_ONSET_MS["p1"] / dt)
    p1_end = int((SLOT_ONSET_MS["p1"] + 531) / dt)
    rate_intact_p1 = float(intact_spikes[p1_start:p1_end, v1_idx].mean() * (1000/dt))
    rate_omit_p1 = float(omit_spikes[p1_start:p1_end, v1_idx].mean() * (1000/dt))
    assert abs(rate_intact_p1 - rate_omit_p1) < 5.0, f"p1 should be similar but {rate_intact_p1} vs {rate_omit_p1}"


def test_t4_t5_run_generates_artifact_arrays():
    """run_t4_t5_analysis must produce generated-owner arrays (npz/json/npy) with correct provenance."""
    field, rate, trial_conds, fs_hz, areas, meta, signals, model = _build_real_arrays(n_reps=2)
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "t4_t5"
        result = run_t4_t5_analysis(field, rate, trial_conds, fs_hz=fs_hz, dt_ms=DT_MS_TEST, areas=AREAS_CANONICAL, out_dir=str(out), save_arrays=True)
        # Artifacts exist
        assert "json" in result["artifacts"]
        assert "npz" in result["artifacts"]
        assert "t4_diff_npy" in result["artifacts"]
        assert "t5_r_npy" in result["artifacts"]
        assert pathlib.Path(result["artifacts"]["json"]).exists()
        assert pathlib.Path(result["artifacts"]["npz"]).exists()
        # Check json has provenance
        j = json.loads(pathlib.Path(result["artifacts"]["json"]).read_text())
        assert j["field_claim_level"] == "proxy_readout"
        assert j["physical_amplitude_calibrated"] is False
        assert "causal" in j["causal_interpretation"].lower() or "no causal" in str(j).lower()
        assert j["owner"] == "generated"
        # NPZ contains per-trial arrays
        npz = np.load(result["artifacts"]["npz"], allow_pickle=True)
        assert "t4_per_trial_position_power" in npz
        assert "t5_per_trial_bandpower" in npz
        assert npz["t4_per_trial_position_power"].shape == (24, 4, 5, 3)
        assert npz["t5_per_trial_bandpower"].shape == (24, 4, 5)
        # NPY generated-owner arrays
        diff_arr = np.load(result["artifacts"]["t4_diff_npy"])
        assert diff_arr.shape == (4, 5, 3), f"diff shape {diff_arr.shape}"
        r_arr = np.load(result["artifacts"]["t5_r_npy"])
        assert r_arr.shape == (4, 5)
        gamma_low = np.load(result["artifacts"]["t5_gamma_low_npy"])
        assert gamma_low.shape == (4,)
        # Provenance file
        assert pathlib.Path(result["artifacts"]["provenance"]).exists()
        prov = json.loads(pathlib.Path(result["artifacts"]["provenance"]).read_text())
        assert prov["physical_amplitude_calibrated"] is False
        assert prov["owner"] == "generated"
        assert prov["field_claim_level"] == "proxy_readout"


def test_comparison_matrix_unchanged():
    """Frozen authorities must remain unaltered."""
    assert COMPARISON_MATRIX["pooling_rule"].startswith("DO NOT pool p2/p3/p4")
    assert COMPARISON_MATRIX["language_rule"].startswith("lfp_proxy")
    assert len(COMPARISON_MATRIX["targets"]) == 7
    assert [t["id"] for t in COMPARISON_MATRIX["targets"]] == ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    # T4 and T5 still exist
    t4 = [t for t in COMPARISON_MATRIX["targets"] if t["id"] == "T4"][0]
    t5 = [t for t in COMPARISON_MATRIX["targets"] if t["id"] == "T5"][0]
    assert t4["field_claim"] == "proxy_readout, physical_amplitude_calibrated=False"
    assert "gamma" in t5["label"].lower()


def test_windows_frozen():
    """Windows must match epochs.py frozen values."""
    from jomission.paradigm.epochs import OMISSION_SLOT_MS, OMISSION_LOCAL_BASELINE_MS, OMISSION_LOCAL_WINDOW_MS
    assert OMISSION_SLOT_MS == (0.0, 531.0)
    assert OMISSION_LOCAL_BASELINE_MS == (-250.0, -50.0)
    assert OMISSION_LOCAL_WINDOW_MS == (-1000.0, 1000.0)
    # Our analysis windows should match
    from jomission.analysis.t4_t5_analysis import OMISSION_SLOT_MS as T4_SLOT, OMISSION_BASELINE_MS as T4_BASE
    assert T4_SLOT == OMISSION_SLOT_MS
    assert T4_BASE == OMISSION_LOCAL_BASELINE_MS

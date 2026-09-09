"""C_min single-cortex tests — generic Gen-2 qualification, NOT omission.

Covers: grammar (4 layers x 4 classes = 16 populations, laminar labels kept),
reproducibility (same seed -> same config_hash), execution (zero-drive
baseline finite, s* measured not assumed), impulse plumbing (perturbation
propagates and stays finite). No T1-T7, no jaxfne._* imports.
"""

import numpy as np

import jaxfne as jtfne
from jaxfne.io import config_hash

from jomission.qualification.cmin import (
    CMIN_CELL_TYPES,
    CMIN_DEFAULT_N,
    CMIN_LAYERS,
    build_cmin,
    coverage,
    run_impulse_decay,
    run_zero_drive,
    summarize_baseline,
)


def test_cmin_grammar_sixteen_populations_laminar():
    model = build_cmin(n_total=CMIN_DEFAULT_N, seed=0)
    cov = coverage(model)
    assert set(cov.keys()) == {(la, ct) for la in CMIN_LAYERS for ct in CMIN_CELL_TYPES}
    assert all(v > 0 for v in cov.values())
    layers_seen = {la for (la, _ct) in cov}
    assert "uniform_3d" not in layers_seen  # laminar geometry must keep labels


def test_cmin_reproducible_config_hash():
    m0 = build_cmin(n_total=CMIN_DEFAULT_N, seed=7)
    m1 = build_cmin(n_total=CMIN_DEFAULT_N, seed=7)
    assert config_hash(m0.cfg) == config_hash(m1.cfg)


def test_cmin_zero_drive_baseline_finite_and_measured():
    model = build_cmin(n_total=160, seed=0)
    sig = run_zero_drive(model, duration_ms=100.0, dt_ms=0.1, seed=0)
    s = summarize_baseline(sig)
    assert s["finite"] is True
    assert s["n_neurons"] == 160
    assert s["n_steps"] == 1000
    assert np.isfinite(s["v_mean"]) and np.isfinite(s["spike_rate_hz"])


def test_cmin_impulse_perturbation_propagates_finite():
    model = build_cmin(n_total=160, seed=0)
    out = run_impulse_decay(model, target_ids=[0, 1, 2, 3], dv=15.0, post_ms=50.0)
    assert np.isfinite(out["V_pert"]).all()
    assert out["mean_abs_dV"][0] > 0.0  # perturbation present at t=0+
    assert (out["mean_abs_dV"] >= 0.0).all()


def test_cmin_uses_public_surface_only():
    import pathlib

    src = pathlib.Path(__file__).parent.parent.joinpath("jomission/qualification/cmin.py").read_text()
    assert "jaxfne._" not in src


def test_repair_operating_point_acceptance():
    from jomission.qualification.cmin import per_class_rates, repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    r = per_class_rates(model, duration_ms=1000.0, seed=1, skip_ms=500.0)
    assert r["PV"]["mean"] > 0.0
    assert r["VIP"]["mean"] > 0.0
    assert r["E"]["std"] > 0.0  # jitter broke exact synchrony
    assert r["SST"]["mean"] < 60.0  # not pathologically dominant
    assert all(np.isfinite([v["mean"] for v in r.values()]))

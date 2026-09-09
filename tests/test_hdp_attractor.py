"""HDP attractor tests — objective math (fast) + small-scale pipeline smoke."""

import numpy as np

from jomission.qualification import hdp_attractor as ha
from jomission.qualification.cmin import build_cmin


def test_basin_zero_inside_positive_outside():
    v_in = ha.objective(20.0, 5.0, np.zeros(4))  # ratio 4, inside bands
    assert v_in["V_B"] == 0.0 or abs(v_in["V_B"]) < 1e-6  # EPS-shifted null
    assert v_in["V"] < 0.5  # soft walls: small but nonzero near edges
    assert ha.objective(100.0, 5.0, np.zeros(4))["V_rate"] > 5.0  # steep wall
    assert ha.objective(0.01, 5.0, np.zeros(4))["V_rate"] > 0.0
    assert ha.objective(20.0, 20.0, np.zeros(4))["V_B"] > 0.0


def test_grad_pushes_back_into_basin():
    g = ha.grad_r_objective(100.0, 4.0)
    assert g[0] > 0.0  # descend by lowering rE
    g2 = ha.grad_r_objective(0.01, 4.0)
    assert g2[0] < 0.0  # descend by raising rE


def test_gains_admissible_and_theta0_identity():
    g = ha.theta_to_gains(np.array([10.0, -10.0, 0.0, 0.0]))
    assert (g <= ha.G_HI).all() and (g >= ha.G_LO).all()
    model = build_cmin(n_total=160, seed=0)
    m2 = ha.apply_theta(model, np.zeros(4))
    assert np.allclose(np.asarray(m2.params["edge_list"].weight),
                       np.asarray(model.params["edge_list"].weight))
    m3 = ha.apply_theta(model, np.array([0.5, 0, 0, 0]))
    assert not np.allclose(np.asarray(m3.params["edge_list"].weight),
                           np.asarray(model.params["edge_list"].weight))


def test_family_masks_partition_edges():
    model = build_cmin(n_total=160, seed=0)
    masks, members = ha.family_masks(model)
    n = int(np.asarray(model.params["edge_list"].pre).shape[0])
    total = sum(int(m.sum()) for m in masks.values())
    assert total == n
    flats = [m.astype(int) for m in masks.values()]
    assert (sum(flats) <= 1).all() or True  # overlap impossible by construction
    assert (sum(flats) == 1).all()


def test_smoke_S_estimation_finite():
    model = build_cmin(n_total=160, seed=0)
    est = ha.estimate_S(model, np.zeros(4), 0.0, dtheta=0.2,
                        n_settle=500, n_meas=500, seed=0)
    assert est["S"].shape == (2, 4)
    assert np.isfinite(est["S"]).all()
    assert np.isfinite(est["r0"]).all()
    assert (est["r0"] >= 0).all()


def test_smoke_closed_loop_finishes():
    model = build_cmin(n_total=160, seed=0)
    est = ha.estimate_S(model, np.zeros(4), 0.0, dtheta=0.2,
                        n_settle=500, n_meas=500, seed=0)
    rec = ha.closed_loop(model, np.zeros(4), est["S"], 0.0, eta=0.01,
                         n_seg=2, seg_ms=200.0, seed=0)
    assert rec["reject"] is None
    assert len(rec["V"]) == 2
    assert np.isfinite(rec["V"]).all()


def test_theta6_identity_and_drive_authority():
    model = build_cmin(n_total=160, seed=0)
    m0 = ha.apply_theta6(model, np.zeros(6))
    assert np.allclose(np.asarray(m0.params["edge_list"].weight),
                       np.asarray(model.params["edge_list"].weight))
    assert np.allclose(np.asarray(m0.params["emitter"].drive),
                       np.asarray(model.params["emitter"].drive))
    md = ha.apply_theta6(model, np.array([0, 0, 0, 0, 0.5, 0]))
    e0 = np.asarray(model.params["emitter"].drive)
    ed = np.asarray(md.params["emitter"].drive)
    tbl = model.neuron_table()
    isE = np.array([r["cell_type"] == "E" for r in tbl])
    assert np.allclose(ed[isE] / e0[isE], np.exp(0.5))
    assert np.allclose(ed[~isE], e0[~isE])


def test_offline_currents_match_kernel():
    import jax.numpy as jnp

    import jaxfne as jtfne
    from jomission.qualification.cmin import initial_state

    model = build_cmin(n_total=160, seed=0)
    ws = ha.edge_weight_sums(model)
    el = model.params["edge_list"]
    w = np.asarray(el.weight)
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    tbl = model.neuron_table()
    isE = np.array([r["cell_type"] == "E" for r in tbl])
    step_fn, _ = jtfne.compile_step_fn(model, dt_ms=0.1, kernel="baseline",
                                       record_weight_trace=False,
                                       record_edge_current=True)
    st = initial_state(model, 0)
    drive = jnp.zeros((500, 160), dtype=model.params["emitter"].v0.dtype)
    st, out = jtfne.run_continuation(step_fn, st, drive)
    spikes = np.asarray(out[1])
    ec = np.asarray(out[4])
    k_exc = ec[:, (w > 0) & isE[post]].sum(axis=1)
    k_inh = ec[:, (w < 0) & isE[post]].sum(axis=1)
    o_exc, o_inh = ha.realized_currents_E(spikes, ws)
    assert abs(o_exc.mean() / k_exc.mean() - 1) < 0.05
    assert abs(o_inh.mean() / abs(k_inh.mean()) - 1) < 0.05
    assert np.corrcoef(k_exc, o_exc)[0, 1] > 0.99


def test_pinv_weighted_properties():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((3, 6))
    lam, R = 0.1, np.array([1, 1, 1, 1, 4, 4])
    Sp = ha.pinv_weighted(S, lam, R)
    assert Sp.shape == (6, 3)
    assert np.isfinite(Sp).all()
    # weighted normal equations: (S R^-1 S^T + lam I) x = y solved exactly
    y = rng.standard_normal(3)
    x = Sp @ y
    # residual of regularized system: KKT stationarity
    # x = R^-1 S^T z, (S R^-1 S^T + lam I) z = y
    z = np.linalg.solve(S @ np.diag(1.0 / R) @ S.T + lam * np.eye(3), y)
    assert np.allclose(x, np.diag(1.0 / R) @ S.T @ z)
    # kernel projection kills range directions, preserves ker directions
    Pker = np.eye(6) - Sp @ S
    v_ker = np.linalg.svd(S)[2][-1]
    assert np.allclose(Pker @ v_ker, v_ker, atol=1e-8)
    # R-weighting: expensive directions (large R) move less for same y
    x_plain = ha.pinv_weighted(S, lam, None) @ y
    assert abs(x[4]) + abs(x[5]) <= abs(x_plain[4]) + abs(x_plain[5]) + 1e-9


def test_controller_step_w_descent_and_drift_pullback():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((3, 6))
    th = rng.standard_normal(6)
    g = rng.standard_normal(3)
    th2 = ha.controller_step_w(th, g, S, eta=0.01, lam=0.1,
                               R_diag=np.ones(6), gamma=0.0)
    # first-order V decrease along the step (linearized): g^T S dTh < 0
    assert float(g @ (S @ (th2 - th))) < 0
    # pure drift pullback stays in ker S (exact in the lam -> 0 limit;
    # at lam > 0 the residual is O(lam)-scale by construction)
    th3 = ha.controller_step_w(th, np.zeros(3), S, eta=0.0, lam=1e-9,
                               R_diag=np.ones(6), gamma=0.5, theta0=np.zeros(6))
    assert np.allclose(S @ (th3 - th), 0, atol=1e-6)


def test_smoke_Sy_finite_with_subclass_health():
    model = build_cmin(n_total=160, seed=0)
    est = ha.estimate_Sy(model, np.zeros(6), 0.0, dtheta=0.3, ddrive=0.3,
                         n_settle=400, n_meas=400, seed=0)
    assert est["Sy"].shape == (3, 6)
    assert np.isfinite(est["Sy"]).all()
    assert np.isfinite(est["y0"]).all()
    assert len(est["branch_sub"]) == 6
    rep = ha.svd_report(est["Sy"])
    assert rep["sigma"].shape == (3,)
    assert 1 <= rep["rank"] <= 3


def test_restricted_objective_has_no_ratio_term():
    o = ha.objective_restricted(10.0, 40.0, np.zeros(6))
    assert o["V_rate"] >= 0.0
    assert abs(o["V"] - (o["V_rate"] + ha.LAM_TH * o["V_Theta"])) < 1e-12
    g = ha.grad_r_restricted(100.0, 4.0)
    assert g[0] > 0.0


def test_guards_and_classifier():
    y_ok = {"b_E": -0.5}
    sub_ok = {"E": 10.0, "PV": 14.0, "SST": 18.0, "VIP": 27.0}
    ok, bad = ha.check_guards(y_ok, sub_ok)
    assert ok and bad == []
    ok2, bad2 = ha.check_guards({"b_E": 5.0}, sub_ok)
    assert not ok2
    ok3, bad3 = ha.check_guards(y_ok, {**sub_ok, "VIP": 0.0})
    assert not ok3
    osc, flips = ha.oscillation_flag([1.0, 0.5, 0.8, 0.4, 0.9, 0.3])
    assert osc and flips >= ha.OSC_FLIP_MIN
    osc2, _ = ha.oscillation_flag([1.0, 0.8, 0.6, 0.5, 0.45, 0.44])
    assert not osc2
    rec = {"V_rate": np.array([1.0, 0.8, 0.6, 0.5, 0.45, 0.44]),
           "b_E": np.array([-0.5] * 6), "theta": np.zeros((6, 6)),
           "reject": None, "subE": np.full(6, 10.0), "subPV": np.full(6, 14.0),
           "subSST": np.full(6, 18.0), "subVIP": np.full(6, 27.0)}
    m = ha.classify_run(rec)
    assert m["class"] == "PASS"
    assert m["V_rate_final"] < m["V_rate_max"]
    rec2 = dict(rec, reject="NUMERICAL")
    assert ha.classify_run(rec2)["class"] == "NUMERICAL FAILURE"


def test_smoke_closed_loop6_and_off():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    S2 = np.array([[0.25, 0.006, -0.4, 0.011, 18.06, -1.24],
                   [-0.017, 1.0, -0.05, -1.38, 1.82, 48.07]])
    rec = ha.closed_loop6(model, S2, np.zeros(6), eta=0.5, drive_step=0.0,
                          n_seg=2, seg_ms=200.0, seed=0)
    assert rec["reject"] is None
    assert len(rec["V_rate"]) == 2
    off = ha.closed_loop6(model, S2, np.zeros(6), eta=0.5, drive_step=0.0,
                          n_seg=2, seg_ms=200.0, seed=0, freeze_theta=True)
    assert off["reject"] is None
    assert np.allclose(off["theta_final"], np.zeros(6))


def test_tau_theta_bounds_and_limits():
    assert ha.TAU_S_DFLT <= ha.tau_theta(0.0) <= ha.TAU_L_DFLT
    assert abs(ha.tau_theta(1e-6) - ha.TAU_S_DFLT) < 1e-3
    assert abs(ha.tau_theta(100.0) - ha.TAU_L_DFLT) < 1e-3
    assert abs(ha.tau_theta(ha.THETA_C_DFLT)
               - (ha.TAU_S_DFLT + ha.TAU_L_DFLT) / 2) < 1e-9
    assert (ha.tau_theta(np.array([0.01, 1.0])).min() >= ha.TAU_S_DFLT
            and ha.tau_theta(np.array([0.01, 1.0])).max() <= ha.TAU_L_DFLT)


def test_zero_error_contraction():
    d = 0.5
    for mode in ("const", "state"):
        d0, prev = d, abs(d)
        for _ in range(200):
            d0 = d0 - float(ha.pullback_step(np.array([d0]), np.zeros(1), mode)[0])
            assert abs(d0) < prev  # strict Lyapunov decrease every step
            prev = abs(d0)
        assert abs(d0) < 1e-6
    assert (ha.pullback_step(np.array([0.3]), np.zeros(1), "none") == 0).all()


def test_synthetic_stm_to_ltm_tau_eff_rises():
    # Synthetic error histories through the discrete law (no simulator):
    # brief pulse vs sustained pulse, then free recovery.
    def run(pulse):
        th, t0 = 0.0, 0.0
        hist = []
        for c in pulse + [0.0] * 60:
            th = th - 4.0 * 0.02 * c - float(
                ha.pullback_step(np.array([th]), np.array([t0]), "state")[0])
            hist.append(th)
        taus, mags = ha.tau_eff_series(np.array(hist)[:, None],
                                       np.zeros(1), comp=0)
        return taus, mags

    tb, mb = run([1.0])          # brief: one error segment
    ts, ms = run([1.0] * 6)      # sustained: six error segments
    assert len(tb) > 0 and len(ts) > 0
    assert ms.max() > mb.max()  # sustained accumulates more
    # tau_eff rises with |d|: sustained early-recovery tau exceeds brief tau
    assert np.median(ts[:5]) > np.median(tb[:5])
    # const-tau control shows flat tau_eff (no consolidation slope)
    def run_const(pulse):
        th = 0.0
        hist = []
        for c in pulse + [0.0] * 60:
            th = th - 4.0 * 0.02 * c - float(
                ha.pullback_step(np.array([th]), np.zeros(1), "const")[0])
            hist.append(th)
        return ha.tau_eff_series(np.array(hist)[:, None], np.zeros(1), comp=0)
    tc, mc = run_const([1.0] * 6)
    assert abs(np.median(tc[:5]) - ha.TAU_S_DFLT) < 0.5


def test_smoke_recovery_assay_plumbing():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    S2 = np.array([[0.25, 0.006, -0.4, 0.011, 18.06, -1.24],
                   [-0.017, 1.0, -0.05, -1.38, 1.82, 48.07]])
    rec = ha.recovery_assay(model, S2, np.zeros(6), pattern=[1], n_recovery=1,
                            drive_step=1.0, seed=0, seg_ms=200.0,
                            pullback="state")
    assert rec["reject"] is None
    assert len(rec["V_rate"]) == 2
    assert rec["pullback"] == "state"


def test_eligibility_mean_preservation_and_envelope():
    rng = np.random.default_rng(0)
    members = {"E": np.arange(6), "PV": np.arange(6, 8),
               "SST": np.arange(8, 9), "VIP": np.arange(9, 10)}
    H = rng.uniform(1, 30, size=10)
    m = ha.susceptibility(H, members, {"E": 10.0, "PV": 14.0, "SST": 18.0, "VIP": 27.0})
    assert (m >= ha.M_MIN_DFLT).all() and (m <= ha.M_MAX_DFLT).all()
    assert ha.check_envelope(m, eta=4.0)
    assert not ha.check_envelope(np.full(10, 3.0), eta=4.0)  # 12 > 8 trips
    dth = ha.allocate_drive_delta(m, members, u_E=0.05, u_I=-0.03, eta=4.0)
    assert abs(dth[:6].mean() - (-4.0 * 0.05)) < 1e-12  # class mean exact
    for c in ("PV", "SST", "VIP"):
        idx = members[c]
        assert abs(dth[idx].mean() - (-4.0 * -0.03)) < 1e-12
    # H dynamics: steps toward activity, bounded
    H2 = ha.update_H(np.zeros(5), np.full(5, 20.0), tau_H=3.0)
    assert np.allclose(H2, np.full(5, 20.0 / 3.0))
    H3 = ha.update_H(H2, np.full(5, 20.0), tau_H=3.0)
    assert (H3 > H2).all() and (H3 < 20.0).all()


def test_synthetic_allocation_corr():
    # Higher-H neurons get larger |delta| under same consensus correction.
    members = {"E": np.arange(20), "PV": np.arange(20, 22),
               "SST": np.arange(22, 23), "VIP": np.arange(23, 24)}
    H = np.zeros(24)
    H[:10] = 25.0  # S1-like history
    H[10:20] = 5.0
    m = ha.susceptibility(H, members, {"E": 10.0, "PV": 14.0, "SST": 18.0, "VIP": 27.0})
    dth = ha.allocate_drive_delta(m, members, u_E=0.05, u_I=0.0, eta=4.0)
    dE, HE = dth[:20], H[:20]
    assert float(np.corrcoef(HE, np.abs(dE))[0, 1]) > 0.9
    assert abs(dE.mean() - (-4.0 * 0.05)) < 1e-12  # mean preserved regardless


def test_smoke_eligibility_assay():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    S2 = np.array([[0.25, 0.006, -0.4, 0.011, 18.06, -1.24],
                   [-0.017, 1.0, -0.05, -1.38, 1.82, 48.07]])
    out = ha.eligibility_assay(model, S2, n_E_sub=10, hist_boost=2.0, n_hist=2,
                               probe_step=1.0, n_probe=1, seed=0, seg_ms=200.0)
    assert out["reject"] is None
    assert np.isfinite(out["corr"])
    assert out["corr"] > 0  # history -> allocation
    assert set(out["h_c"].keys()) == {"E", "PV", "SST", "VIP"}


def test_smoke_integrated_assay_triple_corr():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    S2 = np.array([[0.25, 0.006, -0.4, 0.011, 18.06, -1.24],
                   [-0.017, 1.0, -0.05, -1.38, 1.82, 48.07]])
    out = ha.consolidation_allocation_assay(
        model, S2, n_E_sub=10, hist_boost=2.0, n_hist=1, probe_step=1.0,
        n_probe=1, n_hdecay=1, n_retain=3, seed=0, seg_ms=200.0)
    assert out["reject"] is None
    for k in ("corr_H_dth", "corr_dth_tau", "corr_H_tau"):
        assert k in out
    assert len(out["tau_eff"]) == 120  # all E at N=160


def test_ramp_scaling_and_authority():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    w0 = np.asarray(model.params["edge_list"].weight)
    m2 = ha.scale_recurrence(model, 2.0)
    assert np.allclose(np.asarray(m2.params["edge_list"].weight), 2.0 * w0)
    assert (np.sign(np.asarray(m2.params["edge_list"].weight)) == np.sign(w0)).all()
    assert (np.asarray(m2.params["edge_list"].pre) == np.asarray(model.params["edge_list"].pre)).all()
    m0 = ha.zero_recurrence(model)
    assert np.allclose(np.asarray(m0.params["edge_list"].weight), 0.0)
    mEE = ha.scale_family(model, ("E", "E"), 1.5)
    masks, _ = ha.family_masks(model)
    wE = np.asarray(model.params["edge_list"].weight)
    wM = np.asarray(mEE.params["edge_list"].weight)
    assert np.allclose(wM[np.asarray(masks[("E", "E")])], 1.5 * wE[np.asarray(masks[("E", "E")])])
    assert np.allclose(wM[~np.asarray(masks[("E", "E")])], wE[~np.asarray(masks[("E", "E")])])
    ar = ha.recurrence_authority(model)
    assert set(ar.keys()) == {"r_full", "r_off", "dE", "dI"}
    assert np.isfinite(ar["dE"]) and np.isfinite(ar["dI"])
    pr = ha.probe_response(model)
    assert set(pr.keys()) == {"G_R", "T_R", "r2", "flips", "r_base", "r_peak",
                              "t_ms", "envelope", "spikes_ds", "dt_ms"}
    assert pr["G_R"] >= 0 and np.isfinite(pr["r_base"])
    assert np.asarray(pr["spikes_ds"]).shape[1] == 160


def test_loop_delay_helper():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    m0 = ha.set_loop_delay(model, 0.0)
    assert int(np.asarray(m0.params["edge_list"].delay_steps).max()) == 0
    m5 = ha.set_loop_delay(model, 10.0)
    ds = np.asarray(m5.params["edge_list"].delay_steps)
    masks, _ = ha.family_masks(model)
    assert set(np.unique(ds).tolist()) == {0, 50}  # 5ms/0.1 per direction
    assert (ds[np.asarray(masks[("E", "E")])] == 0).all()
    assert (ds[np.asarray(masks[("E", "I")])] == 50).all()
    assert (ds[np.asarray(masks[("I", "E")])] == 50).all()
    assert np.allclose(np.asarray(m5.params["edge_list"].weight),
                       np.asarray(model.params["edge_list"].weight))


def test_scale_tau_selective():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    el = model.params["edge_list"]
    ri = np.asarray(el.receptor_index)
    tau0 = np.asarray(el.tau_ms, dtype=float)
    m4 = ha.scale_tau(model, 1, 4.0)
    tau4 = np.asarray(m4.params["edge_list"].tau_ms, dtype=float)
    assert np.allclose(tau4[ri == 1], 4.0 * tau0[ri == 1])
    assert np.allclose(tau4[ri == 0], tau0[ri == 0])
    assert np.allclose(np.asarray(m4.params["edge_list"].weight),
                       np.asarray(el.weight))  # gain untouched


def test_tail_selector_synthetic():
    t = np.arange(0, 800, 10.0)
    rng = np.random.default_rng(0)
    y1 = 3.0 * np.exp(-t / 150.0) + 10.0 + rng.normal(0, 0.05, t.shape)
    m1 = ha.select_tail_model(t, y1)
    assert m1["model"] in ("M1", "M2")
    assert abs(m1["T_R"] - 150.0) / 150.0 < 0.3
    assert m1["bic"] < m1["bic_null"]  # beats intercept-only
    assert m1["A"] > 0.02 * (y1.max() - y1.min())  # nonzero residue
    y2 = 2.0 * np.exp(-t / 200.0) * np.cos(2 * np.pi * t / 100.0) + 10.0 \
        + rng.normal(0, 0.05, t.shape)
    m2 = ha.select_tail_model(t, y2)
    assert m2["model"] == "M2"
    assert abs(m2["T_R"] - 200.0) / 200.0 < 0.4
    assert m2["omega"] > 0
    y0 = 10.0 + rng.normal(0, 0.3, t.shape)
    m0 = ha.select_tail_model(t, y0)
    assert m0["model"] == "M0"
    # identifiability: T beyond half the window auto-rejects even if BIC fits
    yslow = 3.0 * np.exp(-t / 5000.0) + 10.0 + rng.normal(0, 0.02, t.shape)
    ms = ha.select_tail_model(t, yslow)
    assert ms["model"] == "M0"  # T=5000 >> window/2=400: extrapolation


def test_current_decomposition_gamma():
    import jax.numpy as jnp

    import jaxfne as jtfne
    from jomission.qualification.cmin import (
        initial_state,
        repair_jitter,
        repair_tonic,
    )

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    step_fn, _ = jtfne.compile_step_fn(model, dt_ms=0.1, kernel="baseline",
                                       record_weight_trace=False)
    st = initial_state(model, 0)
    drive = jnp.zeros((3000, 160), dtype=model.params["emitter"].v0.dtype)
    st, out = jtfne.run_continuation(step_fn, st, drive)
    spikes = np.asarray(out[1])
    ton = np.asarray(model.params["emitter"].drive, dtype=float)
    dec = ha.current_decomposition(spikes[1000:], model, ton)
    assert 0.0 <= dec["Gamma_R"] <= 1.0
    for k in ("Gamma_E", "Gamma_PV", "Gamma_SST", "Gamma_VIP",
              "I_EE", "I_EI", "I_IE", "I_II"):
        assert np.isfinite(dec[k]) and dec[k] >= 0.0
    assert dec["Gamma_R"] < 0.5  # baseline is tonic-supported (diagnostic)


def test_split_charge_preserved():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    q0 = ha.excitatory_charge(model)
    m0 = ha.split_excitatory_kernel(model, 0.0, 50.0, seed=0)
    assert ha.excitatory_charge(m0) == q0
    assert np.allclose(np.asarray(m0.params["edge_list"].weight),
                       np.asarray(model.params["edge_list"].weight))
    for f in (0.05, 0.15, 0.35):
        mf = ha.split_excitatory_kernel(model, f, 50.0, seed=0)
        assert abs(ha.excitatory_charge(mf) / q0 - 1) < 1e-9  # exact
        el, elf = model.params["edge_list"], mf.params["edge_list"]
        ri = np.asarray(el.receptor_index)
        slow = (np.asarray(elf.tau_ms, dtype=float) == 50.0) & (ri == 0)
        assert abs(slow.sum() / (ri == 0).sum() - f) < 0.01  # fraction of exc
        assert (np.asarray(elf.tau_ms)[ri == 1] == np.asarray(el.tau_ms)[ri == 1]).all()
        w, wf = np.asarray(el.weight, float), np.asarray(elf.weight, float)
        assert np.allclose(wf[slow], w[slow] * (2.0 / 50.0))
        assert np.allclose(wf[~slow], w[~slow])
        mg = ha.split_excitatory_kernel(model, f, 50.0, seed=1)
        assert not np.allclose(np.asarray(mg.params["edge_list"].tau_ms, dtype=float),
                               np.asarray(elf.tau_ms, dtype=float))  # seed matters


def test_modular_rebuild_and_ablation():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=400, seed=0)), seed=0)
    m2, info = ha.build_modular(model, M=4, chi=0.9, seed=0)
    assert info["B"] == 40000  # N * 100 degree-ceiling scale
    assert info["within_frac"] > 0.8
    assert info["max_indeg"] < 200
    mod = info["mod"]
    assert set(np.unique(mod).tolist()) == {0, 1, 2, 3}
    # stratified grammar: every module mirrors all 16 layer x class cells
    tbl = model.neuron_table()
    for mm in range(4):
        keys = {(tbl[i]["layer"], tbl[i]["cell_type"]) for i in np.flatnonzero(mod == mm)}
        assert len(keys) == 16
    # determinism + chi control
    m2b, _ = ha.build_modular(model, M=4, chi=0.9, seed=0)
    assert np.allclose(np.asarray(m2b.params["edge_list"].weight),
                       np.asarray(m2.params["edge_list"].weight))
    m0, info0 = ha.build_modular(model, M=4, chi=0.0, seed=0)
    assert abs(info0["within_frac"] - 0.25) < 0.05
    # ablations: loop cut > 0 edges, cross control matched-count-ish
    ml, nl = ha.ablate_loop(m2, 0, mod=mod)
    assert nl > 0
    mx, nx = ha.ablate_cross_matched(m2, nl, 0, seed=0, mod=mod)
    assert nx > 0
    # module tail selector runs on probe output
    pr = ha.probe_response(m2)
    sel = ha.module_tail_select(pr, mod, 0)
    assert sel["model"] in ("M0", "M1", "M2")


def test_continuation_and_kickmap_plumbing():
    from jomission.qualification.cmin import repair_jitter, repair_tonic

    model = repair_jitter(repair_tonic(build_cmin(n_total=160, seed=0)), seed=0)
    mq = ha.scale_tonic(model, 0.5)
    e0 = np.asarray(model.params["emitter"].drive, dtype=float)
    eq = np.asarray(mq.params["emitter"].drive, dtype=float)
    assert np.allclose(eq, 0.5 * e0)
    p = ha.continuation_point(model, 1.0, n_settle=500, n_meas=500,
                              with_S=False)
    assert p["rE"] > 1.0 and 0.0 <= p["Gamma_R"] <= 1.0
    assert set(p.keys()) >= {"rE", "rI", "Gamma_R", "I_EE", "I_IE", "I_EI", "I_II"}
    m0 = ha.scale_tonic(model, 0.0)
    rows = ha.kick_release_map(m0, amps=(2.0, 6.0), pre_ms=200.0,
                               rel_ms=300.0, seed=0)
    assert len(rows) == 2
    for r in rows:
        assert set(r.keys()) >= {"rE_pre", "rI_pre", "rE_post", "rI_post", "R_E", "R_I"}
        assert np.isfinite(r["R_E"]) and np.isfinite(r["R_I"])

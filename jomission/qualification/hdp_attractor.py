"""HDP attractor qualification — outer-loop E/I controller on a single cortex.

Architecture: JaxFNE is the ENGINE (baseline kernel via `compile_step_fn` +
`run_continuation`, HDP-off). The plastic controller lives HERE (outer loop),
exactly implementing the reviewed rule::

    dTheta/dt = -K S^T grad_r V - lambda_Th K R (Theta - Theta_0),  K = eta D

with Theta = (thEE, thEI, thIE, thII) mapped to admissible multiplicative
family gains. S = dr*/dTheta is estimated by central finite perturbations
around the ACTIVE stationary regime (never zero) — 8 branches + reference.

Objective (basin, not point target)::

    V = V_rate + lambda_B V_B + lambda_Th V_Theta

All numeric constants are MODEL_ASSUMPTION (provenance-tagged), not biology.

Scalar-H runtime note: this controller owns Theta explicitly per family, so
it does not depend on JaxFNE's internal H representation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne

# --------------------------------------------------------------------------
# MODEL_ASSUMPTION constants (initial values; provenance-tagged, not biology)
# --------------------------------------------------------------------------
R_E_LO, R_E_HI = 3.0, 25.0  # acceptable E band Hz
R_I_LO, R_I_HI = 3.0, 40.0  # acceptable I band Hz (pooled PV/SST/VIP)
SOFT_S = 2.0  # softplus wall softness Hz
RHO_STAR = 4.0  # target E/I rate ratio (MODEL_ASSUMPTION)
EPS = 1e-3
LAM_B = 1.0
LAM_TH = 0.1
R_DIAG = (1.0, 1.0, 1.0, 1.0)  # Theta drift penalty weights
G_LO, G_HI = 0.2, 5.0  # admissible multiplicative gain box
D_NORM = (1.0, 1.0, 1.0, 1.0)  # actuator scale normalizer (refined from |S|)
RUNAWAY_HZ = 120.0  # pathological firing cap
SUBCLASS_LO, SUBCLASS_HI = 1.0, 60.0  # per-subclass PV/SST/VIP health band

FAMILIES = (("E", "E"), ("E", "I"), ("I", "E"), ("I", "I"))
I_CLASSES = ("PV", "SST", "VIP")


def _softplus(x):
    return np.logaddexp(0.0, x)


def rate_penalty(r, lo, hi, s=SOFT_S):
    """Basin wall: ~0 inside [lo, hi], steep outside."""
    return float(_softplus((lo - r) / s) ** 2 + _softplus((r - hi) / s) ** 2)


def d_rate_penalty(r, lo, hi, s=SOFT_S):
    sig_lo = 1.0 / (1.0 + np.exp(-(lo - r) / s))
    sig_hi = 1.0 / (1.0 + np.exp(-(r - hi) / s))
    return float(2 * _softplus((lo - r) / s) * sig_lo * (-1.0 / s)
                 + 2 * _softplus((r - hi) / s) * sig_hi * (1.0 / s))


def objective(rE, rI, theta, theta0=None):
    """V = V_rate + lamB V_B + lamTh V_Theta + components dict."""
    theta = np.asarray(theta, dtype=float)
    t0 = np.zeros(4) if theta0 is None else np.asarray(theta0, dtype=float)
    v_rate = rate_penalty(rE, R_E_LO, R_E_HI) + rate_penalty(rI, R_I_LO, R_I_HI)
    e_b = float(np.log((rE + EPS) / (rI + EPS)) - np.log(RHO_STAR))
    v_b = 0.5 * e_b ** 2
    d = theta - t0
    v_th = 0.5 * float(np.sum(np.asarray(R_DIAG) * d * d))
    v = v_rate + LAM_B * v_b + LAM_TH * v_th
    return {"V": float(v), "V_rate": float(v_rate), "e_B": e_b, "V_B": float(v_b),
            "V_Theta": float(v_th)}


def grad_r_objective(rE, rI):
    """Analytic grad of V w.r.t. (rE, rI) at fixed Theta."""
    e_b = float(np.log((rE + EPS) / (rI + EPS)) - np.log(RHO_STAR))
    return np.array([
        d_rate_penalty(rE, R_E_LO, R_E_HI) + LAM_B * e_b / (rE + EPS),
        d_rate_penalty(rI, R_I_LO, R_I_HI) - LAM_B * e_b / (rI + EPS),
    ])


# --------------------------------------------------------------------------
# Engine plumbing (public surface only)
# --------------------------------------------------------------------------
def family_masks(model):
    """Boolean edge masks per (src_EI, tgt_EI) family + class member indices."""
    tbl = model.neuron_table()
    cls = [str(r["cell_type"]) for r in tbl]
    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    masks = {}
    for (s, t) in FAMILIES:
        sm = np.array([c == s or (s == "I" and c in I_CLASSES) for c in cls])
        tm = np.array([c == t or (t == "I" and c in I_CLASSES) for c in cls])
        masks[(s, t)] = sm[pre] & tm[post]
    members = {c: np.flatnonzero(np.array(cls) == c) for c in ("E",) + I_CLASSES}
    return masks, members


def theta_to_gains(theta):
    """Admissible map: raw Theta -> multiplicative gains in [G_LO, G_HI]."""
    return np.clip(np.exp(np.asarray(theta, dtype=float)), G_LO, G_HI)


def apply_theta(model, theta):
    """Return model with family gains applied (post-construct EdgeList scaling).

    Sign-preserving multiplicative; Theta_0=0 is identity. No kernel change.
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    gains = theta_to_gains(theta)
    masks, _ = family_masks(model)
    el = model.params["edge_list"]
    w = np.asarray(el.weight, dtype=np.float64)
    scale = np.ones_like(w)
    for j, fam in enumerate(FAMILIES):
        scale[masks[fam]] = gains[j]
    new_el = EdgeList(pre=el.pre, post=el.post,
                      weight=jnp.asarray(w * scale, dtype=el.weight.dtype),
                      receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                      source_calibration_status=el.source_calibration_status,
                      **({"delay_steps": el.delay_steps}
                         if getattr(el, "delay_steps", None) is not None else {}))
    return replace(model, params={**model.params, "edge_list": new_el})


def class_rates(spikes, dt_ms, members):
    """Mean Hz per class + pooled I. spikes: bool [T, N]."""
    sp = np.asarray(spikes, dtype=float)
    hz = sp.mean(axis=0) * (1000.0 / float(dt_ms))
    out = {c: float(hz[idx].mean()) if len(idx) else 0.0 for c, idx in members.items()}
    i_idx = np.concatenate([members[c] for c in I_CLASSES])
    out["I"] = float(hz[i_idx].mean()) if len(i_idx) else 0.0
    return out


def run_segment(step_fn, state, drive):
    """One continuation segment; returns (state, spikes, v)."""
    state, out = jtfne.run_continuation(step_fn, state, drive)
    jax.block_until_ready(out[0])
    return state, np.asarray(out[1]), np.asarray(out[0])


def reference_rates(model, theta, drive_amp, n_settle, n_meas, dt_ms, seed,
                    members, masks=None):
    """r* at fixed Theta under constant drive. Fresh cold start (comparable S).

    Returns (rE, rI, per-class dict, end_state). Cold start per branch keeps
    branches matched (same seed) so S measures gain sensitivity, not history.
    """
    gm = apply_theta(model, theta)
    n_neurons = int(gm.params["emitter"].n_neurons)
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    state = initial_state(gm, seed)
    drive = jnp.full((n_settle + n_meas, n_neurons),
                     float(drive_amp), dtype=gm.params["emitter"].v0.dtype)
    state, spikes, _ = run_segment(step_fn, state, drive)
    tail = spikes[n_settle:]
    rates = class_rates(tail, dt_ms, members)
    return rates["E"], rates["I"], rates, state


def estimate_S(model, theta, drive_amp, dtheta=0.1, n_settle=2000, n_meas=3000,
               dt_ms=0.1, seed=0):
    """Central-difference S in R^{2x4} around active regime. 8 branches + ref."""
    masks, members = family_masks(model)
    r0 = reference_rates(model, theta, drive_amp, n_settle, n_meas, dt_ms, seed,
                         members)[:2]
    S = np.zeros((2, 4))
    for j in range(4):
        dp = np.asarray(theta, float).copy(); dp[j] += dtheta
        dm = np.asarray(theta, float).copy(); dm[j] -= dtheta
        rp = reference_rates(model, dp, drive_amp, n_settle, n_meas, dt_ms, seed,
                             members)[:2]
        rm = reference_rates(model, dm, drive_amp, n_settle, n_meas, dt_ms, seed,
                             members)[:2]
        S[:, j] = (np.array(rp) - np.array(rm)) / (2 * dtheta)
    return {"S": S, "r0": np.array(r0), "members": members, "masks": masks}


def controller_step(theta, grad_r, S, eta, theta0=None):
    """Discrete reviewed rule: dTheta = -eta D (S^T g + lamTh R (Th-Th0))."""
    theta = np.asarray(theta, float)
    t0 = np.zeros(4) if theta0 is None else np.asarray(theta0, float)
    d = np.asarray(D_NORM) * (S.T @ np.asarray(grad_r, float)
                              + LAM_TH * np.asarray(R_DIAG) * (theta - t0))
    return theta - float(eta) * d


def closed_loop(model, theta0, S, drive_amp, eta, n_seg=10, seg_ms=1000.0,
                dt_ms=0.1, seed=0):
    """Run closed loop with CONTINUED state (history matters here).

    Per segment: measure rates -> V -> Theta update -> re-apply gains.
    Returns trajectory record + reject flags.
    """
    from jomission.qualification.cmin import initial_state

    masks, members = family_masks(model)
    n_steps = int(round(seg_ms / dt_ms))
    theta = np.asarray(theta0, float)
    gm = apply_theta(model, theta)
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    n_neurons = int(gm.params["emitter"].n_neurons)
    dtype = gm.params["emitter"].v0.dtype
    state = initial_state(gm, seed)
    drive = jnp.full((n_steps, n_neurons), float(drive_amp), dtype=dtype)
    hist = {"rE": [], "rI": [], "V": [], "V_rate": [], "e_B": [],
            "theta": [], "sub": []}
    reject = None
    for _ in range(n_seg):
        state, spikes, v = run_segment(step_fn, state, drive)
        if not (np.isfinite(spikes).all() and np.isfinite(v).all()):
            reject = "nonfinite_state"; break
        rates = class_rates(spikes, dt_ms, members)
        if rates["E"] > RUNAWAY_HZ or rates["I"] > RUNAWAY_HZ:
            reject = "runaway_firing"; break
        obj = objective(rates["E"], rates["I"], theta, theta0)
        hist["rE"].append(rates["E"]); hist["rI"].append(rates["I"])
        hist["V"].append(obj["V"]); hist["V_rate"].append(obj["V_rate"])
        hist["e_B"].append(obj["e_B"]); hist["theta"].append(theta.copy())
        hist["sub"].append({c: rates[c] for c in I_CLASSES})
        g = grad_r_objective(rates["E"], rates["I"])
        theta = controller_step(theta, g, S, eta, theta0)
        if not np.all(np.isfinite(theta)):
            reject = "nonfinite_theta"; break
        gm = apply_theta(model, theta)  # admissibility by construction (clip)
        step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms),
                                           kernel="baseline",
                                           record_weight_trace=False)
    rec = {k: (np.array(v) if k != "sub" else v) for k, v in hist.items()}
    rec["reject"] = reject
    rec["theta_final"] = theta
    return rec


def acceptance(rec, sub_lo=SUBCLASS_LO, sub_hi=SUBCLASS_HI):
    """First-10s acceptance checks (review box, minus HDP-OFF contrast)."""
    out = {}
    n = len(rec["V"])
    out["finished"] = rec["reject"] is None and n > 0
    out["finite"] = out["finished"]
    out["admissible"] = out["finished"]  # gains clipped by construction
    out["v_rate_down"] = bool(n >= 2 and rec["V_rate"][-1] < rec["V_rate"][0])
    out["eb_down"] = bool(n >= 2 and abs(rec["e_B"][-1]) < abs(rec["e_B"][0]))
    th = np.asarray(rec["theta"]) if n else np.zeros((0, 4))
    out["theta_settles"] = bool(n >= 3 and np.linalg.norm(th[-1] - th[-2])
                                < np.linalg.norm(th[1] - th[0]))
    sub_ok = True
    for snap in rec["sub"]:
        for c in I_CLASSES:
            if not (sub_lo <= snap[c] <= sub_hi):
                sub_ok = False
    out["subclass_healthy"] = bool(sub_ok) and n > 0
    out["all"] = all([out["finite"], out["admissible"], out["v_rate_down"],
                      out["eb_down"], out["theta_settles"], out["subclass_healthy"]])
    return out


# --------------------------------------------------------------------------
# Six-actuator extension: synaptic families + intrinsic-drive gains,
# current-balance observable b_E, S_y plant, weighted pseudoinverse control
# --------------------------------------------------------------------------
N_THETA6 = 6  # (EE, EI, IE, II, dE, dI)
DT_MS_DEFAULT = 0.1


def apply_theta6(model, theta):
    """Theta[0:4] synaptic family gains + Theta[4:6] E/I drive gains.

    Drive gains multiply CURRENT emitter drive (post-repair values), same
    admissible clip box. Theta_0 = 0 is identity on both paths.
    """
    theta = np.asarray(theta, dtype=float)
    gm = apply_theta(model, theta[:4])
    e = gm.params["emitter"]
    tbl = gm.neuron_table()
    base = np.asarray(e.drive, dtype=float)
    gdE, gdI = theta_to_gains(theta[4:6])
    out = base.copy()
    for i, row in enumerate(tbl):
        out[i] = base[i] * (gdE if str(row["cell_type"]) == "E" else gdI)
    return gm.with_emitter_parameters(
        drive_per_neuron=jnp.asarray(out, dtype=e.drive.dtype))


def edge_weight_sums(model):
    """Summed weights per (pre, receptor-into-E-sign) for offline currents.

    Returns dict with Wexc[pre] = sum of w>0 edges pre->E targets,
    Winh[pre] = sum of |w| over w<0 edges pre->E targets, tau_exc/inτ.
    Receptor mapping verified live: receptor 0 = exc (tau 2ms, w>0),
    receptor 1 = inh (tau 5ms, w<0). Raises if mapping ever differs.
    """
    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    ri = np.asarray(el.receptor_index, dtype=np.int64)
    tau = np.asarray(el.tau_ms, dtype=float)
    tbl = model.neuron_table()
    is_E = np.array([str(r["cell_type"]) == "E" for r in tbl])
    for r, sgn, t in ((0, 1.0, 2.0), (1, -1.0, 5.0)):
        sel = ri == r
        assert sel.any(), f"receptor {r} absent"
        assert np.unique(tau[sel]).tolist() == [t], "tau map changed"
        assert ((w[sel] * sgn) > 0).all(), "sign map changed"
    n = len(tbl)
    Wexc = np.zeros(n)
    Winh = np.zeros(n)
    np.add.at(Wexc, pre[(ri == 0) & is_E[post]], w[(ri == 0) & is_E[post]])
    np.add.at(Winh, pre[(ri == 1) & is_E[post]], -w[(ri == 1) & is_E[post]])
    return {"Wexc": Wexc, "Winh": Winh, "tau_exc": 2.0, "tau_inh": 5.0}


def _exp_filter(x, tau_ms, dt_ms):
    """Causal exponential filter (receptor synaptic state, offline)."""
    from scipy.signal import lfilter

    a = float(np.exp(-float(dt_ms) / float(tau_ms)))
    b = [1.0 - a]
    return lfilter(b, [1.0, -a], np.asarray(x, dtype=float), axis=0)


def realized_currents_E(spikes, wsums, dt_ms=DT_MS_DEFAULT, stride=1):
    """Offline realized |I_exc->E|, |I_inh->E| from realized spike trains.

    spikes [T, N] bool; returns per-step (Iexc_E, Iinh_E) summed over E
    targets (population totals), then caller windows/means them.
    """
    sp = np.asarray(spikes, dtype=float)[::int(stride)]
    dt = float(dt_ms) * int(stride)
    f_exc = _exp_filter(sp, wsums["tau_exc"], dt) * (wsums["tau_exc"] / dt)
    f_inh = _exp_filter(sp, wsums["tau_inh"], dt) * (wsums["tau_inh"] / dt)
    Iexc = (f_exc * wsums["Wexc"][None, :]).sum(axis=1)
    Iinh = (f_inh * wsums["Winh"][None, :]).sum(axis=1)
    return Iexc, Iinh


def y_observables(spikes, members, wsums, dt_ms=DT_MS_DEFAULT, eps=EPS):
    """Controlled observable y = (rE, rI, b_E). No target asserted on b_E."""
    rates = class_rates(spikes, dt_ms, members)
    Iexc, Iinh = realized_currents_E(spikes, wsums, dt_ms)
    half = len(Iexc) // 2  # settled half only
    b_E = float(np.log(Iexc[half:].mean() + eps) - np.log(Iinh[half:].mean() + eps))
    return {"rE": rates["E"], "rI": rates["I"], "b_E": b_E,
            "Iexc_mean": float(Iexc[half:].mean()),
            "Iinh_mean": float(Iinh[half:].mean()), "sub": rates}


def reference_y(model, theta, drive_amp, n_settle, n_meas, dt_ms, seed,
                members, wsums):
    """y* at fixed 6-Theta under constant drive (cold-start matched)."""
    gm = apply_theta6(model, theta)
    n_neurons = int(gm.params["emitter"].n_neurons)
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    state = initial_state(gm, seed)
    drive = jnp.full((n_settle + n_meas, n_neurons),
                     float(drive_amp), dtype=gm.params["emitter"].v0.dtype)
    state, spikes, _ = run_segment(step_fn, state, drive)
    y = y_observables(spikes[n_settle:], members, wsums, dt_ms)
    return y, state


def estimate_Sy(model, theta, drive_amp, dtheta=0.2, ddrive=0.2,
                n_settle=15000, n_meas=15000, dt_ms=DT_MS_DEFAULT, seed=0):
    """Central-difference S_y in R^{3x6}. Synaptic step dtheta, drive step
    ddrive (separate scales: different physical meaning). 12 branches + ref,
    same protocol as S_syn for direct comparability.
    """
    masks, members = family_masks(model)
    wsums = edge_weight_sums(model)
    y0 = reference_y(model, theta, drive_amp, n_settle, n_meas, dt_ms, seed,
                     members, wsums)[0]
    y0v = np.array([y0["rE"], y0["rI"], y0["b_E"]])
    Sy = np.zeros((3, N_THETA6))
    branch_sub = []
    for j in range(N_THETA6):
        h = ddrive if j >= 4 else dtheta
        dp = np.asarray(theta, float).copy(); dp[j] += h
        dm = np.asarray(theta, float).copy(); dm[j] -= h
        yp, _ = reference_y(model, dp, drive_amp, n_settle, n_meas, dt_ms,
                            seed, members, wsums)
        ym, _ = reference_y(model, dm, drive_amp, n_settle, n_meas, dt_ms,
                            seed, members, wsums)
        Sy[:, j] = (np.array([yp["rE"], yp["rI"], yp["b_E"]])
                    - np.array([ym["rE"], ym["rI"], ym["b_E"]])) / (2 * h)
        branch_sub.append((yp["sub"], ym["sub"]))
    return {"Sy": Sy, "y0": y0v, "y0full": y0, "members": members,
            "branch_sub": branch_sub}


def svd_report(S):
    """rank (tol 1e-6 relative), singular values, condition number."""
    s = np.linalg.svd(S, compute_uv=False)
    tol = 1e-6 * (s[0] if len(s) and s[0] > 0 else 1.0)
    return {"sigma": s, "rank": int((s > tol).sum()),
            "kappa": float(s[0] / s[-1]) if s[-1] > 0 else float("inf")}


def actuator_authority(S, R_diag=None):
    """Normalized per-actuator authority: column 2-norms / max (scale-aware).

    R_diag weights parameter distance; authority reported both raw and
    R-normalized (column j scaled by 1/sqrt(R_jj): cheap directions are
    down-weighted so leverage-hunting is visible, not hidden).
    """
    R = np.ones(S.shape[1]) if R_diag is None else np.asarray(R_diag, float)
    col = np.linalg.norm(S, axis=0)
    col_n = col / np.sqrt(R)
    m = col_n.max() if col_n.max() > 0 else 1.0
    return {"raw": col, "normalized": col_n / m}


def pinv_weighted(S, lam=1e-2, R_diag=None):
    """Weighted regularized pseudoinverse: R^-1 S^T (S R^-1 S^T + lam I)^-1.

    Minimizes physiological error subject to minimal R-normalized parameter
    displacement; lam regularizes the 3x3 (not the 6x6) solve.
    """
    S = np.asarray(S, float)
    R = np.ones(S.shape[1]) if R_diag is None else np.asarray(R_diag, float)
    Ri = np.diag(1.0 / R)
    A = S @ Ri @ S.T + float(lam) * np.eye(S.shape[0])
    return Ri @ S.T @ np.linalg.inv(A)


def controller_step_w(theta, grad_y, Sy, eta, lam=1e-2, R_diag=None,
                      gamma=0.0, theta0=None):
    """Minimum-change law: dTh = -eta S^dagger grad - gamma P_ker (Th-Th0).

    P_ker = I - S^dagger S projects drift pullback onto ker S so
    dynamically irrelevant directions relax to baseline instead of wandering.
    """
    theta = np.asarray(theta, float)
    t0 = np.zeros_like(theta) if theta0 is None else np.asarray(theta0, float)
    Sp = pinv_weighted(Sy, lam, R_diag)
    Pker = np.eye(len(theta)) - Sp @ np.asarray(Sy, float)
    return theta - float(eta) * (Sp @ np.asarray(grad_y, float)) \
        - float(gamma) * (Pker @ (theta - t0))


# --------------------------------------------------------------------------
# Restricted rate controller + guards + eta boundary search (conditional PASS)
# --------------------------------------------------------------------------
# Guard bounds are MODEL_ASSUMPTION (non-biological). b_E band is wide on
# purpose: the first eta experiment ESTABLISHES the neighborhood; only
# excursion aborts. Subclass band reuses the qualification gate.
B_E_LO, B_E_HI = -2.5, 1.0
GUARD_SUB_LO, GUARD_SUB_HI = SUBCLASS_LO, SUBCLASS_HI
LAM_DEFAULT = 1.0  # MODEL_ASSUMPTION, mid log-interval (1e-3, 1e2); see text
R_DEFAULT = (1.0, 1.0, 1.0, 1.0, 4.0, 4.0)  # drive moves cost 4x (scale-aware)
OSC_FLIP_MIN = 3  # sign flips in last-5 segs to flag oscillation
OSC_AMP_TOL = 0.05  # min |dV| amplitude for a flip to count


def objective_restricted(rE, rI, theta, theta0=None):
    """V = V_E + V_I + lamTh V_Theta. No rate-ratio term (rho* retired)."""
    theta = np.asarray(theta, dtype=float)
    t0 = np.zeros(6) if theta0 is None else np.asarray(theta0, dtype=float)
    v_rate = rate_penalty(rE, R_E_LO, R_E_HI) + rate_penalty(rI, R_I_LO, R_I_HI)
    d = theta - t0
    v_th = 0.5 * float(np.sum(np.asarray(R_DEFAULT) * d * d))
    return {"V": float(v_rate + LAM_TH * v_th), "V_rate": float(v_rate),
            "V_Theta": float(v_th)}


def grad_r_restricted(rE, rI):
    """grad of restricted V w.r.t. (rE, rI) at fixed Theta."""
    return np.array([d_rate_penalty(rE, R_E_LO, R_E_HI),
                     d_rate_penalty(rI, R_I_LO, R_I_HI)])


def check_guards(y, sub):
    """b_E band + subclass band. Returns (ok, failed_list)."""
    bad = []
    if not (B_E_LO < y["b_E"] < B_E_HI):
        bad.append(f"b_E={y['b_E']:.2f}")
    for c in I_CLASSES:
        if not (GUARD_SUB_LO <= sub[c] <= GUARD_SUB_HI):
            bad.append(f"{c}={sub[c]:.1f}")
    if not (GUARD_SUB_LO <= sub["E"] <= 100.0):
        bad.append(f"E={sub['E']:.1f}")
    return (not bad), bad


def oscillation_flag(V_hist):
    """Late-run sign-flip count on segment V_rate deltas (MODEL_ASSUMPTION)."""
    d = np.diff(np.asarray(V_hist[-6:], dtype=float))
    flips = int((((d[:-1] * d[1:]) < 0) & (np.abs(d[1:]) > OSC_AMP_TOL)).sum())
    return flips >= OSC_FLIP_MIN, flips


def closed_loop6(model, S2, theta0, eta, drive_base=0.0, drive_step=0.0,
                 step_segs=(3, 4, 5, 6), n_seg=10, seg_ms=1000.0,
                 dt_ms=DT_MS_DEFAULT, seed=0, lam=LAM_DEFAULT,
                 R_diag=R_DEFAULT, gamma=0.0, freeze_theta=False):
    """Restricted 6-theta closed loop with built-in drive-step assay.

    Segments in step_segs run at drive_base + drive_step (perturbation),
    others at drive_base (settle / reversal). State CONTINUES across segments.
    freeze_theta=True gives the matched HDP_OFF causal control.
    Returns record with per-segment observables + classification inputs.
    """
    from jomission.qualification.cmin import initial_state

    masks, members = family_masks(model)
    wsums = edge_weight_sums(model)
    n_steps = int(round(float(seg_ms) / float(dt_ms)))
    theta = np.asarray(theta0, float)
    n_neurons = int(model.params["emitter"].n_neurons)
    dtype = model.params["emitter"].v0.dtype
    Sp = pinv_weighted(np.asarray(S2, float), lam, R_diag)
    Pker = np.eye(6) - Sp @ np.asarray(S2, float)
    gm = apply_theta6(model, theta)
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    state = initial_state(gm, seed)
    hist = {"rE": [], "rI": [], "b_E": [], "V": [], "V_rate": [],
            "theta": [], "subE": [], "subPV": [], "subSST": [], "subVIP": [],
            "drive": []}
    reject = None
    for s in range(n_seg):
        amp = float(drive_base + (drive_step if s in step_segs else 0.0))
        drive = jnp.full((n_steps, n_neurons), amp, dtype=dtype)
        state, spikes, v = run_segment(step_fn, state, drive)
        if not (np.isfinite(spikes).all() and np.isfinite(v).all()):
            reject = "NUMERICAL"; break
        rates = class_rates(spikes, dt_ms, members)
        if rates["E"] > RUNAWAY_HZ or rates["I"] > RUNAWAY_HZ:
            reject = "RATE"; break
        y = y_observables(spikes, members, wsums, dt_ms)
        ok, bad = check_guards(y, rates)
        if not ok:
            reject = f"GUARD:{'+'.join(bad)}"; break
        obj = objective_restricted(rates["E"], rates["I"], theta, theta0)
        for k, val in (("rE", rates["E"]), ("rI", rates["I"]), ("b_E", y["b_E"]),
                       ("V", obj["V"]), ("V_rate", obj["V_rate"]),
                       ("subE", rates["E"]), ("subPV", rates["PV"]),
                       ("subSST", rates["SST"]), ("subVIP", rates["VIP"]),
                       ("drive", amp)):
            hist[k].append(val)
        hist["theta"].append(theta.copy())
        if not freeze_theta:
            g = grad_r_restricted(rates["E"], rates["I"])
            theta = theta - float(eta) * (Sp @ g) - float(gamma) * (Pker @ (theta - np.asarray(theta0, float)))
            if not np.all(np.isfinite(theta)):
                reject = "NUMERICAL"; break
            gm = apply_theta6(model, theta)
            step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms),
                                               kernel="baseline",
                                               record_weight_trace=False)
    rec = {k: (np.array(v) if k != "theta" else np.array(v)) for k, v in hist.items()}
    rec["reject"] = reject
    rec["theta_final"] = theta
    return rec


def classify_run(rec):
    """PASS / OSCILLATORY / RATE / GUARD / NUMERICAL (+ metrics)."""
    m = {}
    n = len(rec["V_rate"])
    m["n_seg"] = n
    if rec["reject"] is not None:
        kind = rec["reject"].split(":")[0]
        m["class"] = {"NUMERICAL": "NUMERICAL FAILURE", "RATE": "RATE FAILURE"}.get(kind, "GUARD FAILURE")
        return m
    if n < 3:
        m["class"] = "NUMERICAL FAILURE"; return m
    osc, flips = oscillation_flag(rec["V_rate"])
    m["osc_flips"] = flips
    if osc:
        m["class"] = "OSCILLATORY"; return m
    m["class"] = "PASS"
    vr = np.asarray(rec["V_rate"])
    m["V_rate_max"] = float(vr.max()); m["V_rate_final"] = float(vr[-1])
    settle = int(np.argmax(vr <= max(0.1, 1.2 * vr.min())))
    m["T_settle_seg"] = settle
    th = np.asarray(rec["theta"])
    m["max_dtheta"] = float(np.abs(th - th[0]).max())
    m["max_dbE"] = float(np.abs(np.asarray(rec["b_E"]) - rec["b_E"][0]).max())
    for c in ("subE", "subPV", "subSST", "subVIP"):
        a = np.asarray(rec[c])
        m[f"{c}_min"] = float(a.min()); m[f"{c}_max"] = float(a.max())
    return m


def eta_boundary_search(model, S2, theta0, eta0=0.5, drive_step=2.0, **kw):
    """Geometric eta search until first non-PASS. Returns list of (eta, class)."""
    out = []
    eta = float(eta0)
    while True:
        rec = closed_loop6(model, S2, theta0, eta, drive_step=drive_step, **kw)
        cls = classify_run(rec)["class"]
        out.append((eta, cls, rec))
        if cls != "PASS" or eta > 64:
            break
        eta *= 2.0
    return out


# --------------------------------------------------------------------------
# Lineage edge: state-dependent consolidation (m(H) = 1 first).
# Reference eta=4 controller preserved; pullback modes select the delta.
# --------------------------------------------------------------------------
# MODEL_ASSUMPTION initial consolidation params (segment units; 1 seg = 1 s).
# Calibrated so single-segment displacements (~0.02-0.05) sit below theta_c
# while sustained 4-seg exposure (~0.1+) crosses it. Coarse scan to follow.
TAU_S_DFLT = 3.0
TAU_L_DFLT = 60.0
THETA_C_DFLT = 0.06
Q_DFLT = 2.0


def tau_theta(dtheta, tau_S=TAU_S_DFLT, tau_L=TAU_L_DFLT,
              theta_c=THETA_C_DFLT, q=Q_DFLT):
    """Bounded consolidation law: tau_S <= tau <= tau_L guaranteed."""
    a = np.abs(np.asarray(dtheta, dtype=float))
    f = (a ** q) / (theta_c ** q + a ** q)
    return tau_S + (tau_L - tau_S) * f


def pullback_step(theta, theta0, mode="none", tau_S=TAU_S_DFLT,
                  tau_L=TAU_L_DFLT, theta_c=THETA_C_DFLT, q=Q_DFLT):
    """Per-component pullback increment (Euler, segment units).

    modes: 'none' (reference controller), 'const' (matched constant-tau
    control), 'state' (new consolidation law). C=0 contraction holds for
    const/state since tau > 0 always.
    """
    d = np.asarray(theta, float) - np.asarray(theta0, float)
    if mode == "none":
        return np.zeros_like(d)
    if mode == "const":
        return d / float(tau_S)
    if mode == "state":
        return d / tau_theta(d, tau_S, tau_L, theta_c, q)
    raise ValueError(f"unknown pullback mode {mode!r}")


def closed_loop6_consol(model, S2, theta0, eta=4.0, drive_base=0.0,
                        drive_schedule=None, n_seg=10, seg_ms=1000.0,
                        dt_ms=DT_MS_DEFAULT, seed=0, lam=LAM_DEFAULT,
                        R_diag=R_DEFAULT, gamma=0.0, freeze_theta=False,
                        pullback="none", tau_S=TAU_S_DFLT, tau_L=TAU_L_DFLT,
                        theta_c=THETA_C_DFLT, q=Q_DFLT, ref=None):
    """closed_loop6 with selectable pullback. drive_schedule: list of per-seg
    additive drives (None = zeros). Default args reproduce closed_loop6.

    ref: pullback/objective reference (default theta0 = current behavior).
    Displacement assays pass initial theta as theta0 and nominal zeros as ref.
    """
    from jomission.qualification.cmin import initial_state

    masks, members = family_masks(model)
    wsums = edge_weight_sums(model)
    n_steps = int(round(float(seg_ms) / float(dt_ms)))
    theta = np.asarray(theta0, float)
    t0 = np.asarray(theta0, float)
    ref = t0 if ref is None else np.asarray(ref, float)
    n_neurons = int(model.params["emitter"].n_neurons)
    dtype = model.params["emitter"].v0.dtype
    Sp = pinv_weighted(np.asarray(S2, float), lam, R_diag)
    Pker = np.eye(6) - Sp @ np.asarray(S2, float)
    gm = apply_theta6(model, theta)
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    state = initial_state(gm, seed)
    sched = [0.0] * n_seg if drive_schedule is None else list(drive_schedule)
    hist = {"rE": [], "rI": [], "b_E": [], "V": [], "V_rate": [],
            "theta": [], "subE": [], "subPV": [], "subSST": [], "subVIP": [],
            "drive": []}
    reject = None
    for s in range(n_seg):
        amp = float(drive_base + sched[s])
        drive = jnp.full((n_steps, n_neurons), amp, dtype=dtype)
        state, spikes, v = run_segment(step_fn, state, drive)
        if not (np.isfinite(spikes).all() and np.isfinite(v).all()):
            reject = "NUMERICAL"; break
        rates = class_rates(spikes, dt_ms, members)
        if rates["E"] > RUNAWAY_HZ or rates["I"] > RUNAWAY_HZ:
            reject = "RATE"; break
        y = y_observables(spikes, members, wsums, dt_ms)
        ok, bad = check_guards(y, rates)
        if not ok:
            reject = f"GUARD:{'+'.join(bad)}"; break
        obj = objective_restricted(rates["E"], rates["I"], theta, ref)
        for k, val in (("rE", rates["E"]), ("rI", rates["I"]), ("b_E", y["b_E"]),
                       ("V", obj["V"]), ("V_rate", obj["V_rate"]),
                       ("subE", rates["E"]), ("subPV", rates["PV"]),
                       ("subSST", rates["SST"]), ("subVIP", rates["VIP"]),
                       ("drive", amp)):
            hist[k].append(val)
        hist["theta"].append(theta.copy())
        if not freeze_theta:
            g = grad_r_restricted(rates["E"], rates["I"])
            step = Sp @ g
            theta = (theta - float(eta) * step
                     - float(gamma) * (Pker @ (theta - ref))
                     - pullback_step(theta, ref, pullback, tau_S, tau_L, theta_c, q))
            if not np.all(np.isfinite(theta)):
                reject = "NUMERICAL"; break
            gm = apply_theta6(model, theta)
            step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms),
                                               kernel="baseline",
                                               record_weight_trace=False)
    rec = {k: np.array(v) for k, v in hist.items()}
    rec["reject"] = reject
    rec["theta_final"] = theta
    rec["pullback"] = pullback
    return rec


def tau_eff_series(theta_hist, theta0, comp=5):
    """Per-segment effective time constant for one component during recovery.

    tau_eff,k = -d_k / (d_{k+1} - d_k), d = theta - theta0. Returns arrays
    (tau_eff, |d|) over valid steps (denominator nonzero, same sign).
    """
    th = np.asarray(theta_hist, dtype=float)
    d = th[:, comp] - float(np.asarray(theta0, float)[comp])
    taus, mags = [], []
    for k in range(len(d) - 1):
        dd = d[k + 1] - d[k]
        if dd != 0 and d[k] != 0 and np.sign(dd) != np.sign(d[k]):
            taus.append(-d[k] / dd)
            mags.append(abs(d[k]))
    return np.array(taus), np.array(mags)


def recovery_assay(model, S2, theta0, pattern, n_recovery=8, drive_step=2.0,
                   seed=0, seg_ms=1000.0, **kw):
    """Run exposure pattern then clean recovery. pattern: list of 0/1 per
    exposure seg (1 = drive_step on). Returns rec incl. exposure+recovery."""
    sched = [float(drive_step) if p else 0.0 for p in pattern] + [0.0] * n_recovery
    return closed_loop6_consol(model, S2, theta0, drive_base=0.0,
                               drive_schedule=sched, n_seg=len(sched),
                               seg_ms=seg_ms, seed=seed, **kw)


def recovery_assay(model, S2, theta0, pattern, n_recovery=8, drive_step=2.0,
                   seed=0, seg_ms=1000.0, **kw):
    """Run exposure pattern then clean recovery. pattern: list of 0/1 per
    exposure seg (1 = drive_step on). Returns rec incl. exposure+recovery."""
    sched = [float(drive_step) if p else 0.0 for p in pattern] + [0.0] * n_recovery
    return closed_loop6_consol(model, S2, theta0, drive_base=0.0,
                               drive_schedule=sched, n_seg=len(sched),
                               seg_ms=seg_ms, seed=seed, **kw)


# --------------------------------------------------------------------------
# Lineage edge: H-gated allocation (m(H) eligibility, NOT global gain).
# C(t) = what changes (class consensus); H = where (per-neuron history);
# Theta = retained. Envelope: eta * m(H) <= ETA_SAFE from measured boundary.
# --------------------------------------------------------------------------
ETA_SAFE = 8.0  # last measured PASS; boundary in (8, 12)
M_MIN_DFLT = 0.2
M_MAX_DFLT = 2.0  # eta * m_max = 8 = ETA_SAFE exactly
P_ELIG_DFLT = 2.0
TAU_H_SEGS_DFLT = 3.0  # MODEL_ASSUMPTION eligibility timescale
# Per-neuron drive-gain consolidation scale. Family-theta theta_c=0.27 is
# NOT transferable: single per-neuron writes are ~0.01 (30x smaller), the
# same scale-relativity lesson as R-weighting. Calibrated from measured
# single-write size (~2-3x): MODEL_ASSUMPTION, stated, not biology.
THETA_C_NEURON_DFLT = 0.025


def update_H(H, rates_hz, tau_H=TAU_H_SEGS_DFLT):
    """Discrete eligibility: H += (A - H) / tau_H. A = per-neuron Hz."""
    H = np.asarray(H, dtype=float)
    A = np.asarray(rates_hz, dtype=float)
    return H + (A - H) / float(tau_H)


def susceptibility(H, members, h_c, p=P_ELIG_DFLT, m_min=M_MIN_DFLT,
                   m_max=M_MAX_DFLT):
    """Bounded m(H) in [m_min, m_max]; h_c per class (dict or scalar)."""
    H = np.asarray(H, dtype=float)
    m = np.empty_like(H)
    cls_of = {}
    for c, idx in members.items():
        cls_of.update({int(i): c for c in
                       (["E"] if c == "E" else [c]) for i in
                       (idx if c in ("E",) + I_CLASSES else [])})
    for i in range(len(H)):
        c = cls_of.get(i, "E")
        hc = h_c[c] if isinstance(h_c, dict) else float(h_c)
        f = (H[i] ** p) / (hc ** p + H[i] ** p) if H[i] > 0 else 0.0
        m[i] = m_min + (m_max - m_min) * f
    return m


def allocate_drive_delta(m, members, u_E, u_I, eta=4.0):
    """Per-neuron drive-theta increments, class-mean normalized.

    delta_i = -eta * (m_i / <m>_class) * u_class. Class mean of delta is
    exactly -eta * u_class (reference authority preserved). Envelope
    eta * m_i <= ETA_SAFE must hold (asserted by caller/test).
    """
    deltas = np.zeros_like(np.asarray(m, dtype=float))
    for c, u in (("E", u_E),) + tuple((k, u_I) for k in I_CLASSES):
        idx = np.asarray(members[c])
        mu = float(np.asarray(m)[idx].mean()) if len(idx) else 1.0
        deltas[idx] = -float(eta) * (np.asarray(m)[idx] / mu) * float(u)
    return deltas


def check_envelope(m, eta=4.0):
    """True iff eta * m(H) <= ETA_SAFE everywhere."""
    return bool((float(eta) * np.asarray(m)).max() <= ETA_SAFE)


def eligibility_assay(model, S2, n_E_sub=30, hist_boost=3.0, n_hist=4,
                      probe_step=2.0, n_probe=2, seed=0, seg_ms=1000.0,
                      dt_ms=DT_MS_DEFAULT, lam=LAM_DEFAULT, R_diag=R_DEFAULT,
                      tau_H=TAU_H_SEGS_DFLT, h_c=None, p=P_ELIG_DFLT,
                      pullback="none", **consol_kw):
    """Differential-history allocation assay.

    Phase 1: E subset S1 gets +hist_boost drive for n_hist segs (S2 rest).
    Phase 2: uniform probe_step for n_probe segs (same consensus error).
    Correlates H_E with per-neuron drive-theta increments in phase 2.
    Synaptic thetas follow the uniform reference law throughout.
    Returns record + corr(H_E, dtheta_E) + guard status.
    """
    from jomission.qualification.cmin import initial_state

    masks, members = family_masks(model)
    wsums = edge_weight_sums(model)
    n_neurons = int(model.params["emitter"].n_neurons)
    dtype = model.params["emitter"].v0.dtype
    n_steps = int(round(float(seg_ms) / float(dt_ms)))
    Sp = pinv_weighted(np.asarray(S2, float), lam, R_diag)
    E_idx = np.asarray(members["E"])
    S1 = E_idx[:int(n_E_sub)]
    base_drive = np.asarray(model.params["emitter"].drive, dtype=float)
    # per-neuron drive-gain state (log space) + eligibility
    logG = np.zeros(n_neurons)
    H = np.zeros(n_neurons)
    theta_syn = np.zeros(4)
    gm = apply_theta6(model, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    state = initial_state(gm, seed)
    hist = {"phase": [], "rE": [], "rI": [], "V_rate": [], "corr": [],
            "mean_dth_E": [], "b_E": [], "subPV": [], "subSST": [],
            "subVIP": [], "reject": None}
    hc = h_c
    probe_deltas, probe_H = None, None
    step = 0

    def run_seg(extra):
        nonlocal state, step_fn, step
        drive = jnp.asarray(extra, dtype=dtype)
        st, spikes, v = run_segment(step_fn, state,
                                    drive.reshape(1, -1).repeat(n_steps, axis=0)
                                    if extra.ndim == 1 else drive)
        step += 1
        return st, spikes, v

    nH = n_neurons
    for seg in range(n_hist):
        extra = np.zeros(nH)
        extra[S1] = hist_boost
        state, spikes, v = run_seg(extra)
        if not (np.isfinite(spikes).all() and np.isfinite(v).all()):
            hist["reject"] = "NUMERICAL"; break
        hz = np.asarray(spikes, dtype=float).mean(axis=0) * (1000.0 / dt_ms)
        H = update_H(H, hz, tau_H)
        rates = class_rates(spikes, dt_ms, members)
        y = y_observables(spikes, members, wsums, dt_ms)
        hist["phase"].append("hist"); hist["rE"].append(rates["E"])
        hist["rI"].append(rates["I"]); hist["corr"].append(float("nan"))
        hist["mean_dth_E"].append(0.0)
        hist["b_E"].append(y["b_E"]); hist["subPV"].append(rates["PV"])
        hist["subSST"].append(rates["SST"]); hist["subVIP"].append(rates["VIP"])
        obj = objective_restricted(rates["E"], rates["I"],
                                   np.concatenate([theta_syn, [0.0, 0.0]]), np.zeros(6))
        hist["V_rate"].append(obj["V_rate"])
    if hc is None:
        # data-driven half-max: class mean rates at end of history phase
        hz_last = np.asarray(spikes, dtype=float).mean(axis=0) * (1000.0 / dt_ms)
        hc = {c: max(float(hz_last[np.asarray(members[c])].mean()), 1.0)
              for c in ("E",) + I_CLASSES}
    for seg in range(n_probe):
        extra = np.full(nH, probe_step)
        state, spikes, v = run_seg(extra)
        if not (np.isfinite(spikes).all() and np.isfinite(v).all()):
            hist["reject"] = "NUMERICAL"; break
        hz = np.asarray(spikes, dtype=float).mean(axis=0) * (1000.0 / dt_ms)
        H = update_H(H, hz, tau_H)
        rates = class_rates(spikes, dt_ms, members)
        y = y_observables(spikes, members, wsums, dt_ms)
        ok, bad = check_guards(y, rates)
        if not ok:
            hist["reject"] = f"GUARD:{'+'.join(bad)}"; break
        m = susceptibility(H, members, hc, p)
        assert check_envelope(m), "eta*m envelope violated"
        g = grad_r_restricted(rates["E"], rates["I"])
        u = Sp @ g  # 6-vector; u[4], u[5] are class consensus corrections
        dth = allocate_drive_delta(m, members, u[4], u[5])
        # synaptic thetas: uniform reference law
        theta6 = np.concatenate([theta_syn, [0.0, 0.0]])
        theta6 = theta6 - 4.0 * (Sp @ g) - pullback_step(
            theta6, np.zeros(6), pullback, **{k: consol_kw[k] for k in
                                              ("tau_S", "tau_L", "theta_c", "q")
                                              if k in consol_kw})
        theta_syn = theta6[:4]
        logG = logG + dth  # per-neuron drive allocation (log space)
        gm_cur = apply_theta(model, theta_syn)
        e = gm_cur.params["emitter"]
        gm_cur = gm_cur.with_emitter_parameters(
            drive_per_neuron=jnp.asarray(base_drive * np.exp(logG),
                                         dtype=e.drive.dtype))
        step_fn, _ = jtfne.compile_step_fn(gm_cur, dt_ms=float(dt_ms),
                                           kernel="baseline",
                                           record_weight_trace=False)
        if seg == 0:
            probe_deltas, probe_H = dth[E_idx].copy(), H[E_idx].copy()
        obj = objective_restricted(rates["E"], rates["I"],
                                   np.concatenate([theta_syn, [0.0, 0.0]]), np.zeros(6))
        hist["phase"].append("probe"); hist["rE"].append(rates["E"])
        hist["rI"].append(rates["I"]); hist["V_rate"].append(obj["V_rate"])
        hist["mean_dth_E"].append(float(dth[E_idx].mean()))
        hist["corr"].append(float("nan"))
        hist["b_E"].append(y["b_E"]); hist["subPV"].append(rates["PV"])
        hist["subSST"].append(rates["SST"]); hist["subVIP"].append(rates["VIP"])
    corr = float(np.corrcoef(probe_H, np.abs(probe_deltas))[0, 1]) \
        if probe_deltas is not None else float("nan")
    hist["corr"][-1] = corr
    return {"hist": hist, "corr": corr, "H_E": H[E_idx] if probe_H is not None else None,
            "dth_E": probe_deltas, "h_c": hc, "reject": hist["reject"],
            "logG": logG, "theta_syn": theta_syn}


    corr = float(np.corrcoef(probe_H, np.abs(probe_deltas))[0, 1]) \
        if probe_deltas is not None else float("nan")
    hist["corr"][-1] = corr
    return {"hist": hist, "corr": corr, "H_E": H[E_idx] if probe_H is not None else None,
            "dth_E": probe_deltas, "h_c": hc, "reject": hist["reject"],
            "logG": logG, "theta_syn": theta_syn}


def consolidation_allocation_assay(
        model, S2, n_E_sub=30, hist_boost=5.0, n_hist=5, probe_step=2.0,
        n_probe=6, n_hdecay=6, n_retain=10, seed=0, seg_ms=1000.0,
        dt_ms=DT_MS_DEFAULT, lam=LAM_DEFAULT, R_diag=R_DEFAULT,
        tau_H=TAU_H_SEGS_DFLT, h_c=None, p=P_ELIG_DFLT, use_m=True,
        theta_c_n=THETA_C_NEURON_DFLT, q=Q_DFLT, tau_S=TAU_S_DFLT,
        tau_L=TAU_L_DFLT, theta_c=THETA_C_DFLT, hold_control=False):
    """Integrated edge: m(H) write + per-neuron consolidation pullback.

    Phases (continuous state throughout):
      history  S1 boosted (H contrast builds; no plasticity updates)
      write    uniform probe_step; drive-theta updates allocated by m(H)
               (or uniform if use_m=False = counterfactual); per-neuron
               logG pullback with scale theta_c_n; synaptic uniform law
      hdecay   base drive; H contrast decays; Theta retained
      retain   base drive; per-neuron logG decay tracked -> tau_eff,i
    H_write frozen at end of write. Triple corr evaluated post-H-decay.
    Guards checked every segment; b_E/subclass recorded throughout.
    hold_control: freeze consensus corrections during hdecay+retain
      (pullback continues) so retention reads the law rather than
      ongoing H-asymmetric controller writes. Default False = closed loop.
    """
    from jomission.qualification.cmin import initial_state

    masks, members = family_masks(model)
    wsums = edge_weight_sums(model)
    n_neurons = int(model.params["emitter"].n_neurons)
    dtype = model.params["emitter"].v0.dtype
    n_steps = int(round(float(seg_ms) / float(dt_ms)))
    Sp = pinv_weighted(np.asarray(S2, float), lam, R_diag)
    E_idx = np.asarray(members["E"])
    S1 = E_idx[:int(n_E_sub)]
    base_drive = np.asarray(model.params["emitter"].drive, dtype=float)
    logG = np.zeros(n_neurons)
    H = np.zeros(n_neurons)
    theta_syn = np.zeros(4)
    gm = apply_theta6(model, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    state = initial_state(gm, seed)
    hist = {"phase": [], "rE": [], "rI": [], "V_rate": [], "b_E": [],
            "subPV": [], "subSST": [], "subVIP": [], "reject": None,
            "H_contrast": [], "theta": [], "logG_E": []}
    hc = h_c
    H_write, logG_write = None, None

    def seg_run(extra):
        nonlocal state, step_fn
        drive = jnp.asarray(extra, dtype=dtype).reshape(1, -1).repeat(n_steps, axis=0)
        return run_segment(step_fn, state, drive)

    def rebuild():
        nonlocal step_fn
        e = model.params["emitter"]
        gm2 = apply_theta(model, theta_syn).with_emitter_parameters(
            drive_per_neuron=jnp.asarray(base_drive * np.exp(logG),
                                         dtype=e.drive.dtype))
        step_fn, _ = jtfne.compile_step_fn(gm2, dt_ms=float(dt_ms),
                                           kernel="baseline",
                                           record_weight_trace=False)

    def observe(spikes, v, phase):
        rates = class_rates(spikes, dt_ms, members)
        y = y_observables(spikes, members, wsums, dt_ms)
        ok, bad = check_guards(y, rates)
        obj = objective_restricted(rates["E"], rates["I"],
                                   np.concatenate([theta_syn, [0.0, 0.0]]), np.zeros(6))
        hist["phase"].append(phase); hist["rE"].append(rates["E"])
        hist["rI"].append(rates["I"]); hist["V_rate"].append(obj["V_rate"])
        hist["b_E"].append(y["b_E"]); hist["subPV"].append(rates["PV"])
        hist["subSST"].append(rates["SST"]); hist["subVIP"].append(rates["VIP"])
        s1h = H[S1].mean() if len(S1) else float("nan")
        s2h = H[np.setdiff1d(E_idx, S1)][:30].mean()
        hist["H_contrast"].append(float(s1h - s2h))
        hist["theta"].append(np.concatenate([theta_syn, [logG[E_idx].mean(), 0.0]]).copy())
        hist["logG_E"].append(logG[E_idx].copy())
        return rates, y, ok, bad

    for _ in range(n_hist):
        extra = np.zeros(n_neurons); extra[S1] = hist_boost
        state, spikes, v = seg_run(extra)
        if not (np.isfinite(spikes).all() and np.isfinite(v).all()):
            hist["reject"] = "NUMERICAL"; break
        hz = np.asarray(spikes, dtype=float).mean(axis=0) * (1000.0 / dt_ms)
        H = update_H(H, hz, tau_H)
        observe(spikes, v, "hist")
    if hc is None:
        hz_last = np.asarray(spikes, dtype=float).mean(axis=0) * (1000.0 / dt_ms)
        hc = {c: max(float(hz_last[np.asarray(members[c])].mean()), 1.0)
              for c in ("E",) + I_CLASSES}
    for _ in range(n_probe):
        if hist["reject"] is not None:
            break
        extra = np.full(n_neurons, probe_step)
        state, spikes, v = seg_run(extra)
        if not (np.isfinite(spikes).all() and np.isfinite(v).all()):
            hist["reject"] = "NUMERICAL"; break
        hz = np.asarray(spikes, dtype=float).mean(axis=0) * (1000.0 / dt_ms)
        H = update_H(H, hz, tau_H)
        rates, y, ok, bad = observe(spikes, v, "write")
        if not ok:
            hist["reject"] = f"GUARD:{'+'.join(bad)}"; break
        m = susceptibility(H, members, hc, p) if use_m else np.ones(n_neurons)
        assert check_envelope(m), "eta*m envelope violated"
        g = grad_r_restricted(rates["E"], rates["I"])
        u = Sp @ g
        dth = allocate_drive_delta(m, members, u[4], u[5])
        theta6 = np.concatenate([theta_syn, [0.0, 0.0]])
        theta6 = theta6 - 4.0 * (Sp @ g) - pullback_step(
            theta6, np.zeros(6), "state", tau_S, tau_L, theta_c, q)
        theta_syn = theta6[:4]
        logG = logG + dth - pullback_step(logG, np.zeros(n_neurons), "state",
                                          tau_S, tau_L, theta_c_n, q)
        rebuild()
    H_write, logG_write = H[E_idx].copy(), logG[E_idx].copy()
    for ph, nseg in (("hdecay", n_hdecay), ("retain", n_retain)):
        for _ in range(nseg):
            if hist["reject"] is not None:
                break
            state, spikes, v = seg_run(np.zeros(n_neurons))
            if not (np.isfinite(spikes).all() and np.isfinite(v).all()):
                hist["reject"] = "NUMERICAL"; break
            hz = np.asarray(spikes, dtype=float).mean(axis=0) * (1000.0 / dt_ms)
            H = update_H(H, hz, tau_H)
            rates, y, ok, bad = observe(spikes, v, ph)
            if not ok:
                hist["reject"] = f"GUARD:{'+'.join(bad)}"; break
            # retention: pullback always; consensus corrections only when
            # not held (hold isolates the law from H-asymmetric writes)
            g = grad_r_restricted(rates["E"], rates["I"])
            u = Sp @ g
            m = susceptibility(H, members, hc, p) if use_m else np.ones(n_neurons)
            if hold_control:
                dth = np.zeros(n_neurons)
            else:
                dth = allocate_drive_delta(m, members, u[4], u[5])
            theta6 = np.concatenate([theta_syn, [0.0, 0.0]])
            if hold_control:
                theta6 = theta6 - pullback_step(
                    theta6, np.zeros(6), "state", tau_S, tau_L, theta_c, q)
            else:
                theta6 = theta6 - 4.0 * (Sp @ g) - pullback_step(
                    theta6, np.zeros(6), "state", tau_S, tau_L, theta_c, q)
            theta_syn = theta6[:4]
            logG = logG + dth - pullback_step(logG, np.zeros(n_neurons), "state",
                                              tau_S, tau_L, theta_c_n, q)
            rebuild()
    out = {"hist": hist, "reject": hist["reject"], "h_c": hc,
           "H_write": H_write, "logG_write": logG_write,
           "logG_final": logG[E_idx].copy(), "H_final": H[E_idx].copy()}
    # triple corr on E neurons, post-H-decay retention; per-neuron tau from
    # retain-phase logG series (median over valid steps)
    n_tot = n_hist + n_probe + n_hdecay + n_retain
    ret_start = n_hist + n_probe + n_hdecay
    series = np.array(hist["logG_E"][ret_start:])
    taus = np.full(len(E_idx), np.nan)
    for i in range(len(E_idx)):
        te, _ = tau_eff_series(series[:, i][:, None], np.zeros(1), comp=0)
        if len(te):
            taus[i] = float(np.median(te))
    out["tau_eff"] = taus
    if H_write is not None:
        d_write = np.abs(logG_write)
        out["corr_H_dth"] = float(np.corrcoef(H_write, d_write)[0, 1]) \
            if d_write.max() > 0 else float("nan")
        valid = np.isfinite(taus)
        out["corr_dth_tau"] = float(np.corrcoef(d_write[valid], taus[valid])[0, 1]) \
            if valid.sum() > 5 and d_write[valid].max() > 0 else float("nan")
        out["corr_H_tau"] = float(np.corrcoef(H_write[valid], taus[valid])[0, 1]) \
            if valid.sum() > 5 else float("nan")
        out["n_tau_valid"] = int(valid.sum())
    return out


# --------------------------------------------------------------------------
# Recurrence ramp: W(g) = g * W0, same topology/params/HDP law/input.
# Four gates per g: stability, A_R (FULL vs REC_OFF), S(g)+margin, T_R/G_R.
# --------------------------------------------------------------------------
def scale_recurrence(model, g):
    """Uniform recurrent multiplier, sign/topology preserving."""
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    el = model.params["edge_list"]
    kwargs = dict(pre=el.pre, post=el.post,
                  weight=el.weight * float(g),
                  receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model, params={**model.params,
                                  "edge_list": EdgeList(**kwargs)})


def zero_recurrence(model):
    """REC_OFF twin (matched state carrier for A_R)."""
    return scale_recurrence(model, 0.0)


def scale_tonic(model, q):
    """Uniform tonic fraction q in [0,1] (continuation parameter).

    Multiplies CURRENT emitter drive. HDP stays OFF (caller uses baseline
    kernel); no other change. q=1 reproduces the reference operating point.
    """
    e = model.params["emitter"]
    base = np.asarray(e.drive, dtype=float)
    return model.with_emitter_parameters(
        drive_per_neuron=jnp.asarray(base * float(q), dtype=e.drive.dtype))


def current_decomposition(spikes, model, tonic, dt_ms=DT_MS_DEFAULT):
    """Realized current fractions: Gamma_R global + per-class, EE/EI/IE/II.

    Recurrent currents reconstructed offline (validated vs kernel traces):
    per-receptor exponential filtering of realized spikes, kernel gain
    tau/dt. Tonic = emitter drive + external schedule (passed in, per
    neuron, time-averaged). Returns means over the (settled) window.
    Gamma_R = <|I_rec|> / (<|I_rec|> + <|I_tonic|>).
    """
    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    ri = np.asarray(el.receptor_index, dtype=np.int64)
    tau = np.asarray(el.tau_ms, dtype=float)
    tbl = model.neuron_table()
    n = len(tbl)
    cls = np.array([r["cell_type"] for r in tbl])
    sp = np.asarray(spikes, dtype=float)
    out = {}
    Irec = np.zeros((sp.shape[0], n))
    comps = {}
    for r, t in ((0, 2.0), (1, 5.0)):
        sel = ri == r
        tau_r = float(np.unique(tau[sel])[0]) if sel.any() else t
        f = _exp_filter(sp, tau_r, dt_ms) * (tau_r / dt_ms)
        for (sname, smask) in (("E", cls[pre] == "E"),
                               ("I", cls[pre] != "E")):
            m = sel & smask
            if not m.any():
                continue
            acc = np.zeros_like(f)
            np.add.at(acc, (slice(None), post[m]),
                      (f[:, pre[m]] * w[m][None, :]))
            Irec += acc
            comps[f"{sname}->{r}"] = acc
    ton = np.asarray(tonic, dtype=float)
    if ton.ndim == 1:
        ton = np.broadcast_to(ton[None, :], sp.shape)
    aR, aT = np.abs(Irec).mean(), np.abs(ton).mean()
    out["Gamma_R"] = float(aR / (aR + aT))
    out["Irec_mean"] = float(aR)
    out["Itonic_mean"] = float(aT)
    for c in ("E", "PV", "SST", "VIP"):
        idx = cls == c
        aRc, aTc = np.abs(Irec[:, idx]).mean(), np.abs(ton[:, idx]).mean()
        out[f"Gamma_{c}"] = float(aRc / (aRc + aTc))
    # family components (magnitudes; signed sums live in Irec). acc entries
    # are already per-neuron values, so .mean() over the block IS the
    # per-neuron mean -- no further division (a /n_targets here once hid
    # a 300x scale error; fixed on direct kernel-trace comparison).
    for tgt in ("E", "I"):
        tm = (cls == "E") if tgt == "E" else (cls != "E")
        for src, r in (("E", 0), ("I", 1)):
            acc = comps.get(f"{src}->{r}", np.zeros((sp.shape[0], n)))
            out[f"I_{src}{tgt}"] = float(np.abs(acc[:, tm]).mean())
    return out


def assign_modules(model, M):
    """Stratified module ids: round-robin within (layer, cell_type) groups.

    Each module mirrors the full layer x class grammar. Deterministic.
    """
    tbl = model.neuron_table()
    key_of = [(r["layer"], r["cell_type"]) for r in tbl]
    groups: dict = {}
    for i, k in enumerate(key_of):
        groups.setdefault(k, []).append(i)
    mod = np.full(len(tbl), -1, dtype=np.int64)
    for members in groups.values():
        for j, i in enumerate(members):
            mod[i] = j % int(M)
    return mod


def build_modular(model, M=4, budget=None, chi=0.0, seed=0):
    """Fixed-budget modular rebuild. Total edges B (default N*100 degree
    ceiling scale); chi fraction lands within-module, rest distributed.
    Weights/receptors/taus sampled from the parent edge pool (distribution
    preserved); posts redrawn (same layer+class as original post) either
    within the pre's module (chi) or anywhere (1-chi). No self-loops.
    Returns (model2, info). Sparsity effect isolated via chi=0 control.
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    rng = np.random.default_rng(int(seed))
    tbl = model.neuron_table()
    n = len(tbl)
    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    ri = np.asarray(el.receptor_index, dtype=np.int64)
    tau = np.asarray(el.tau_ms, dtype=float)
    mod = assign_modules(model, M)
    B = int(budget) if budget else n * 100
    B = min(B, len(pre))
    take = rng.choice(len(pre), size=B, replace=False)
    layers = np.array([r["layer"] for r in tbl])
    clss = np.array([r["cell_type"] for r in tbl])
    by_lc: dict = {}
    for i in range(n):
        by_lc.setdefault((layers[i], clss[i]), []).append(i)
    for v in by_lc.values():
        rng.shuffle(v)
    new_pre, new_post, new_w, new_ri, new_tau = [], [], [], [], []
    indeg = np.zeros(n, dtype=np.int64)
    for e in take:
        p = int(pre[e])
        key = (layers[int(post[e])], clss[int(post[e])])
        cand = by_lc[key]
        if rng.random() < float(chi):
            pool = [i for i in cand if mod[i] == mod[p] and i != p]
            if not pool:
                pool = [i for i in cand if i != p]
        else:
            pool = [i for i in cand if i != p]
        # degree ceiling: prefer low-indegree targets
        pool = sorted(pool, key=lambda i: indeg[i])[:32]
        q = pool[int(rng.integers(len(pool)))]
        indeg[q] += 1
        new_pre.append(p); new_post.append(q)
        new_w.append(w[e]); new_ri.append(int(ri[e])); new_tau.append(float(tau[e]))
    new_el = EdgeList(pre=jnp.asarray(np.array(new_pre, dtype=np.int64)),
                      post=jnp.asarray(np.array(new_post, dtype=np.int64)),
                      weight=jnp.asarray(np.array(new_w, dtype=np.float32)),
                      receptor_index=jnp.asarray(np.array(new_ri, dtype=np.int32)),
                      tau_ms=jnp.asarray(np.array(new_tau, dtype=np.float32)),
                      source_calibration_status=el.source_calibration_status)
    m2 = replace(model, params={**model.params, "edge_list": new_el})
    within = float(np.mean(mod[np.array(new_pre)] == mod[np.array(new_post)]))
    info = {"M": int(M), "B": B, "chi": float(chi), "within_frac": within,
            "max_indeg": int(indeg.max()), "seed": int(seed), "mod": mod}
    return m2, info


def module_impulse_response(model_g, target_mod=0, n_steps_pulse=2000,
                            pulse_amp=3.0, n_settle=10000, n_tail=5000,
                            dt_ms=DT_MS_DEFAULT, seed=0, mod=None, M=4):
    """Stimulate E neurons of one module; A_module = dR_target/dR_others.

    Early-window (first 100ms post-onset) mean-rate contrast vs matched
    no-pulse baseline (same seed/state). Returns dict with A_module.
    """
    masks, members = family_masks(model_g)
    tbl = model_g.neuron_table()
    n = len(tbl)
    if mod is None:
        mod = assign_modules(model_g, M)
    cls = np.array([r["cell_type"] for r in tbl])
    gm = apply_theta6(model_g, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    E_t = np.flatnonzero((mod == int(target_mod)) & (cls == "E"))
    outs = {}
    for tag, boost in (("pulse", pulse_amp), ("base", 0.0)):
        state = initial_state(gm, seed)
        nN = n
        extra = np.zeros(nN); extra[E_t] = boost
        drive = jnp.asarray(np.tile(extra, (n_settle + n_steps_pulse + n_tail, 1)),
                            dtype=gm.params["emitter"].v0.dtype)
        state, spikes, _ = run_segment(step_fn, state, drive)
        outs[tag] = np.asarray(spikes, dtype=float)
    win = slice(n_settle, n_settle + 1000)  # first 100ms of pulse
    dR = (outs["pulse"][win].mean(axis=0) - outs["base"][win].mean(axis=0)) * (1000.0 / dt_ms)
    tgt = float(dR[(mod == int(target_mod))].mean())
    oth = float(dR[(mod != int(target_mod))].mean())
    return {"A_module": float(tgt / max(oth, 1e-9)), "dR_target": tgt,
            "dR_others": oth, "mod": mod}


def ablate_loop(model_g, target_mod=0, mod=None, M=4):
    """Zero E1->I1 and I1->E1 (local loop OFF). Returns (model, n_cut)."""
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    masks, members = family_masks(model_g)
    if mod is None:
        mod = assign_modules(model_g, M)
    el = model_g.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    loop = ((masks[("E", "I")] | masks[("I", "E")])
            & (mod[pre] == int(target_mod)) & (mod[post] == int(target_mod)))
    w2 = w.copy()
    w2[loop] = 0.0
    kwargs = dict(pre=el.pre, post=el.post,
                  weight=jnp.asarray(w2, dtype=el.weight.dtype),
                  receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model_g, params={**model_g.params, "edge_list": EdgeList(**kwargs)}), int(loop.sum())


def module_tail_select(pr, mod, target_mod, bin_ms=10.0):
    """M0/M1/M2 on one module's post-release rate tail (causal memory test).

    Returns selector dict. Memory attribution requires FULL-selects while
    LOOPoff does not (matched seeds).
    """
    import numpy as np

    sp = np.asarray(pr["spikes_ds"], dtype=float)  # stride-10
    dt = float(pr["dt_ms"]) * 10.0
    n_set = 10000 // 10
    n_pulse = 2000 // 10
    post = sp[n_set + n_pulse:]
    sel = np.asarray(mod) == int(target_mod)
    b = int(round(bin_ms / dt))
    nb = post.shape[0] // b
    r = post[:nb * b, :][:, sel].reshape(nb, b, -1).mean(axis=(1, 2)) * (1000.0 / dt)
    t = np.arange(nb) * float(bin_ms)
    return select_tail_model(t, r)


def ablate_cross_matched(model_g, n_cut, target_mod=0, seed=0, mod=None, M=4):
    """Zero n_cut random cross-module edges (matched-count control)."""
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    rng = np.random.default_rng(int(seed))
    if mod is None:
        mod = assign_modules(model_g, 4)
    el = model_g.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    cross = np.flatnonzero(mod[pre] != mod[post])
    cut = rng.choice(cross, size=min(int(n_cut), len(cross)), replace=False)
    w2 = w.copy()
    w2[cut] = 0.0
    kwargs = dict(pre=el.pre, post=el.post,
                  weight=jnp.asarray(w2, dtype=el.weight.dtype),
                  receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model_g, params={**model_g.params, "edge_list": EdgeList(**kwargs)}), int(len(cut))


def set_loop_delay(model, D_ms, dt_ms=DT_MS_DEFAULT, families=(("E", "I"), ("I", "E"))):
    """E-I-E loop delay: D split symmetrically across both directions.

    Only the listed families get nonzero delay_steps; all else stays 0.
    Weights, taus, topology untouched. The edge kernel carries delay_state
    (verified live); no backend switch required.
    """
    from jaxfne.emitters import edge_list_with_delay_ms

    masks, _ = family_masks(model)
    n = int(np.asarray(model.params["edge_list"].pre).shape[0])
    dms = np.zeros(n)
    for fam in families:
        dms[np.asarray(masks[fam])] = float(D_ms) / 2.0
    from dataclasses import replace

    el2 = edge_list_with_delay_ms(model.params["edge_list"], dms, float(dt_ms))
    return replace(model, params={**model.params, "edge_list": el2})


def split_excitatory_kernel(model, f_S, tau_S, seed=0, receptor_exc=0):
    """Slow-excitatory fraction at constant integrated charge.

    Random subset (fraction f_S, seed-fixed) of excitatory edges becomes
    the slow channel: tau -> tau_S with weight *= tau_F/tau_S, so each
    converted edge delivers identical steady-state charge (kernel gain
    per edge is w*r*tau/dt, validated against edge_current_trace).
    Total excitatory charge Q = sum(w*tau) is exactly preserved; only
    the temporal distribution changes. Topology, signs, taus of all
    other edges untouched.
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    el = model.params["edge_list"]
    ri = np.asarray(el.receptor_index, dtype=np.int64)
    tau = np.asarray(el.tau_ms, dtype=float)
    w = np.asarray(el.weight, dtype=float)
    exc = np.where(ri == int(receptor_exc))[0]
    assert (exc.size > 0) and np.unique(tau[exc]).tolist() == [float(tau[exc][0])], \
        "exc channel must have a single fast tau"
    tau_F = float(tau[exc][0])
    rng = np.random.default_rng(int(seed))
    n_slow = int(round(float(f_S) * exc.size))
    slow = np.sort(rng.choice(exc, size=n_slow, replace=False)) if n_slow else np.array([], dtype=np.int64)
    tau2 = tau.copy()
    w2 = w.copy()
    tau2[slow] = float(tau_S)
    w2[slow] = w[slow] * (tau_F / float(tau_S))
    new_el = EdgeList(pre=el.pre, post=el.post,
                      weight=jnp.asarray(w2, dtype=el.weight.dtype),
                      receptor_index=el.receptor_index,
                      tau_ms=jnp.asarray(tau2, dtype=el.tau_ms.dtype),
                      source_calibration_status=el.source_calibration_status,
                      **({"delay_steps": el.delay_steps}
                         if getattr(el, "delay_steps", None) is not None else {}))
    return replace(model, params={**model.params, "edge_list": new_el})


def excitatory_charge(model, receptor_exc=0):
    """Integrated excitatory action Q = sum(w*tau) over exc edges."""
    el = model.params["edge_list"]
    ri = np.asarray(el.receptor_index, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    tau = np.asarray(el.tau_ms, dtype=float)
    return float((w[ri == int(receptor_exc)] * tau[ri == int(receptor_exc)]).sum())


def scale_tau(model, receptor, mult):
    """Selective synaptic-timescale multiplier (temporal-kernel edge).

    receptor: 0 (exc, tau 2ms) or 1 (inh, tau 5ms). Scales tau_ms on
    matching edges only; weights, topology, delays untouched. The kernel
    implements per-edge receptor-filtered exponential states, so the
    kinetic distinction is genuinely represented (verified by edge
    tau_ms values, not assumed).
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    el = model.params["edge_list"]
    ri = np.asarray(el.receptor_index, dtype=np.int64)
    tau = np.asarray(el.tau_ms, dtype=float)
    t2 = tau.copy()
    t2[ri == int(receptor)] = tau[ri == int(receptor)] * float(mult)
    kwargs = dict(pre=el.pre, post=el.post, weight=el.weight,
                  receptor_index=el.receptor_index,
                  tau_ms=jnp.asarray(t2, dtype=el.tau_ms.dtype),
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model, params={**model.params,
                                  "edge_list": EdgeList(**kwargs)})


def select_tail_model(t, y, amp_floor=0.02):
    """M0/M1/M2 tail selection for perturbation envelopes.

    M0: no identifiable tail. M1: A exp(-t/T)+c. M2: A exp(-t/T)cos(wt+p)+c.
    Selection: BIC + amplitude floor + out-of-sample RMSE gate (fit first
    2/3, predict last 1/3), plus identifiability: T_R < window/2 required
    (longer fits are extrapolation -> M0 regardless of BIC).
    Returns dict(model, T_R, A, omega, r2_oos, bic).
    Interpretation guard (permanent): A_R>0 and G_R up NEVER imply T_R>0;
    only M1/M2 selection with residue + OOS superiority counts.
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    window = float(t[-1] - t[0]) if n > 1 else 0.0
    k = max(8, 2 * n // 3)
    tf, yf, tp, yp = t[:k], y[:k], t[k:], y[k:]
    span = float(yf.max() - yf.min())
    if span <= 0:
        return {"model": "M0", "T_R": float("nan"), "A": 0.0,
                "omega": 0.0, "r2_oos": float("nan"), "bic": float("inf")}
    c0 = float(yf[-len(yf) // 4:].mean())
    a0 = float(yf.max() - c0)
    out = {"model": "M0", "T_R": float("nan"), "A": 0.0, "omega": 0.0,
           "r2_oos": float("nan"), "bic": float("inf")}

    def passes_null(fit_pred, oos_pred):
        resid = yf - fit_pred
        sig = float(np.sqrt((resid ** 2).mean()))
        oos_rmse = float(np.sqrt(((yp - oos_pred) ** 2).mean()))
        return oos_rmse < 5.0 * max(sig, 1e-9)

    def oos_r2(ypred):
        ss = float(((yp - ypred) ** 2).sum())
        vv = float(((yp - yp.mean()) ** 2).sum())
        return float(1 - ss / vv) if vv > 0 else float("nan")

    def bic_of(rss, npar):
        return float(n * np.log(max(rss, 1e-12) / n) + npar * np.log(n))

    out["bic"] = bic_of(float(((yf - c0) ** 2).sum()), 1)  # null (M0) BIC
    out["bic_null"] = out["bic"]
    # M1 via log-linear on positive part
    try:
        yc = yf - c0
        pos = yc > 0.05 * abs(a0)
        if pos.sum() >= 4:
            co = np.polyfit(tf[pos], np.log(yc[pos]), 1)
            T1 = float(-1.0 / co[0]) if co[0] < 0 else float("nan")
            A1 = float(np.exp(co[1]))
            if np.isfinite(T1) and T1 > 0 and abs(A1) >= amp_floor * span:
                fit1 = A1 * np.exp(-tf / T1) + c0
                pred = A1 * np.exp(-tp / T1) + c0
                rss = float(((yf - fit1) ** 2).sum())
                bic1 = bic_of(rss, 3)
                if bic1 < out["bic"] and passes_null(fit1, pred):
                    out.update({"model": "M1", "T_R": T1, "A": A1, "omega": 0.0,
                                "r2_oos": oos_r2(pred), "bic": bic1})
    except Exception:
        pass
    # M2: FFT init for omega, then damped-cosine fit
    try:
        yc = yf - yf.mean()
        fr = np.fft.rfftfreq(len(tf), d=float(tf[1] - tf[0]) if len(tf) > 1 else 1.0)
        pw = np.abs(np.fft.rfft(yc - yc.mean()))
        w0 = 2 * np.pi * float(fr[1 + int(np.argmax(pw[1:]))]) if len(pw) > 2 else 0.0
        Tguess = out["T_R"] if out["model"] == "M1" and np.isfinite(out["T_R"]) else float(t[-1] - t[0])
        def m2(tt, A, T, w, p, c):
            return A * np.exp(-tt / T) * np.cos(w * tt + p) + c
        po, _ = curve_fit(m2, tf, yf, p0=[a0, Tguess, w0, 0.0, c0],
                          maxfev=20000)
        A2, T2, w2, _, c2 = (float(v) for v in po)
        if T2 > 0 and abs(A2) >= amp_floor * span and abs(w2) > 0:
            fit2 = m2(tf, *po)
            pred = m2(tp, *po)
            rss = float(((yf - fit2) ** 2).sum())
            bic2 = bic_of(rss, 5)
            if bic2 < out["bic"] and passes_null(fit2, pred):
                out.update({"model": "M2", "T_R": T2, "A": A2, "omega": abs(w2),
                            "r2_oos": oos_r2(pred), "bic": bic2})
    except Exception:
        pass
    if out["model"] != "M0":
        out["A"] = float(out["A"])  # residue kept explicit for acceptance
        if not (np.isfinite(out["T_R"]) and out["T_R"] < window / 2.0):
            out.update({"model": "M0", "T_R": float("nan"), "A": 0.0,
                        "omega": 0.0, "r2_oos": float("nan"),
                        "bic": out["bic_null"]})
    return out


def scale_family(model, family, mult):
    """Selective single-family multiplier (loop-structure edge).

    family in FAMILIES, e.g. ('E','E'). Sign/topology preserving; other
    families untouched. Synaptic taus and delays unchanged.
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    masks, _ = family_masks(model)
    el = model.params["edge_list"]
    w = np.asarray(el.weight, dtype=float)
    sel = np.asarray(masks[family])
    w2 = w.copy()
    w2[sel] = w[sel] * float(mult)
    kwargs = dict(pre=el.pre, post=el.post,
                  weight=jnp.asarray(w2, dtype=el.weight.dtype),
                  receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model, params={**model.params,
                                  "edge_list": EdgeList(**kwargs)})


def settle_rates(model, theta, drive_amp=0.0, n_settle=10000, n_meas=5000,
                 dt_ms=DT_MS_DEFAULT, seed=0):
    """Open-loop settled rates + spikes tail (matched-state A_R reader)."""
    masks, members = family_masks(model)
    gm = apply_theta6(model, theta)
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    state = initial_state(gm, seed)
    nN = int(gm.params["emitter"].n_neurons)
    drive = jnp.full((n_settle + n_meas, nN), float(drive_amp),
                     dtype=gm.params["emitter"].v0.dtype)
    state, spikes, _ = run_segment(step_fn, state, drive)
    tail = np.asarray(spikes)[n_settle:]
    rates = class_rates(tail, dt_ms, members)
    return rates, tail, state


def recurrence_authority(model_g, theta=None, **kw):
    """A_R(g) = r_FULL(g) - r_REC_OFF, matched seeds/states. Returns dict."""
    theta = np.zeros(6) if theta is None else np.asarray(theta, float)
    r_full, _, _ = settle_rates(model_g, theta, **kw)
    r_off, _, _ = settle_rates(zero_recurrence(model_g), theta, **kw)
    return {"r_full": r_full, "r_off": r_off,
            "dE": float(r_full["E"] - r_off["E"]),
            "dI": float(r_full["I"] - r_off["I"])}


def probe_response(model_g, theta=None, pulse_amp=2.0, pulse_ms=200.0,
                   tail_ms=800.0, dt_ms=DT_MS_DEFAULT, seed=0, bin_ms=10.0):
    """Ensemble rate perturbation: settle, pulse, release; fit decay.

    Returns G_R (peak |dR|/dI, population rate) and T_R (exp envelope fit
    on post-release |r - r*|, 10ms bins). Non-monotone/oscillatory tails
    flagged via flip count. Rate-based by construction (no micro-dV).
    """
    theta = np.zeros(6) if theta is None else np.asarray(theta, float)
    masks, members = family_masks(model_g)
    gm = apply_theta6(model_g, theta)
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    state = initial_state(gm, seed)
    nN = int(gm.params["emitter"].n_neurons)
    dt = gm.params["emitter"].v0.dtype
    n_set = 10000
    n_pulse = int(round(pulse_ms / dt_ms))
    n_tail = int(round(tail_ms / dt_ms))
    drive = jnp.concatenate([
        jnp.zeros((n_set, nN), dtype=dt),
        jnp.full((n_pulse, nN), float(pulse_amp), dtype=dt),
        jnp.zeros((n_tail, nN), dtype=dt)])
    state, spikes, _ = run_segment(step_fn, state, drive)
    sp = np.asarray(spikes, dtype=float)
    b = int(round(bin_ms / dt_ms))
    base = sp[n_set - 2000:n_set].mean() * (1000.0 / dt_ms)
    post = sp[n_set + n_pulse:]
    nb = len(post) // b
    r = post[:nb * b].reshape(nb, b, nN).mean(axis=(1, 2)) * (1000.0 / dt_ms)
    e = np.abs(r - base)
    peak = int(np.argmax(e[:nb // 2]))
    G_R = float(e[peak] / pulse_amp)
    tail = e[peak:]
    flips = int((((np.diff(tail)[:-1]) * (np.diff(tail)[1:])) < 0).sum())
    T_R, slope_r2 = float("nan"), float("nan")
    use = tail[tail > 0.05 * e[peak]]
    if len(use) >= 4:
        t = np.arange(len(use)) * bin_ms
        A = np.vstack([t, np.ones_like(t)]).T
        coef, res, _, _ = np.linalg.lstsq(A, np.log(use), rcond=None)
        T_R = float(-1.0 / coef[0]) if coef[0] < 0 else float("inf")
        ss = float(((np.log(use) - A @ coef) ** 2).sum())
        slope_r2 = float(1 - ss / (np.log(use).var() * len(use))) \
            if np.log(use).var() > 0 else float("nan")
    return {"G_R": G_R, "T_R": T_R, "r2": slope_r2, "flips": flips,
            "r_base": float(base), "r_peak": float(r[peak]),
            "t_ms": (np.arange(nb) * float(bin_ms)).tolist(),
            "envelope": r.tolist(),
            "spikes_ds": sp[::10].tolist(),
            "dt_ms": float(dt_ms)}


# --------------------------------------------------------------------------
# Tonic continuation + kick/release empirical vector field (no bridging).
# HDP OFF throughout. Tonic fraction q scales CURRENT emitter drive.
# --------------------------------------------------------------------------
def continuation_point(model, q, n_settle=15000, n_meas=10000,
                       dt_ms=DT_MS_DEFAULT, seed=0, with_S=True, dtheta=0.2):
    """One tonic-continuation point: settled r(q), currents, S(q).

    S(q) via central differences (8 branches + ref, same protocol as S_syn
    for comparability). Tonic = base drive * q (HDP off, baseline kernel).
    """
    mq = scale_tonic(model, q)
    masks, members = family_masks(mq)
    wsums = edge_weight_sums(mq)
    gm = apply_theta6(mq, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    state = initial_state(gm, seed)
    nN = int(gm.params["emitter"].n_neurons)
    drive = jnp.zeros((n_settle + n_meas, nN), dtype=gm.params["emitter"].v0.dtype)
    state, spikes, _ = run_segment(step_fn, state, drive)
    sp = np.asarray(spikes)
    tail = sp[n_settle:]
    rates = class_rates(tail, dt_ms, members)
    ton = np.asarray(gm.params["emitter"].drive, dtype=float)
    dec = current_decomposition(tail, gm, ton, dt_ms)
    out = {"q": float(q), "rE": rates["E"], "rI": rates["I"], "sub": rates,
           "Gamma_R": dec["Gamma_R"], "Gamma_E": dec["Gamma_E"],
           "I_EE": dec["I_EE"], "I_IE": dec["I_IE"],
           "I_EI": dec["I_EI"], "I_II": dec["I_II"]}
    if with_S:
        est = estimate_S(mq, np.zeros(4), 0.0, dtheta=dtheta,
                         n_settle=n_settle, n_meas=n_meas, dt_ms=dt_ms, seed=seed)
        out["S"] = est["S"]
    return out


def linearity_probe(model, q, dths=(0.1, 0.25, 0.5, -0.25), n_settle=15000,
                    n_meas=10000, dt_ms=DT_MS_DEFAULT, seed=0):
    """Forward EE-theta steps at fixed q: Delta rE vs S_EE(q) * dtheta.

    Returns per-step (dtheta, drE, linear_pred, rel_err). Tests where the
    local derivative ceases to be predictive. No extrapolation beyond.
    """
    mq = scale_tonic(model, q)
    masks, members = family_masks(mq)
    base = reference_rates(mq, np.zeros(4), 0.0, n_settle, n_meas, dt_ms,
                           seed, members)[:2]
    ref = np.array(base, float)
    S = estimate_S(mq, np.zeros(4), 0.0, dtheta=0.1, n_settle=n_settle,
                   n_meas=n_meas, dt_ms=dt_ms, seed=seed)["S"]
    rows = []
    for h in dths:
        th = np.zeros(4)
        th[0] = h
        r = np.array(reference_rates(mq, th, 0.0, n_settle, n_meas, dt_ms,
                                     seed, members)[:2], float)
        pred = S[0, 0] * h
        rows.append({"dtheta": float(h), "drE": float(r[0] - ref[0]),
                     "pred": float(pred),
                     "rel_err": float(abs(r[0] - ref[0] - pred) / max(abs(pred), 1e-9))})
    return {"q": float(q), "S_EE": float(S[0, 0]), "rows": rows}


def kick_release_map(model, amps=(2.0, 4.0, 6.0, 9.0), pre_ms=500.0,
                     rel_ms=1000.0, dt_ms=DT_MS_DEFAULT, seed=0):
    """Zero-tonic kick/release: pre-drive at amp, release, measure flow.

    Model must already carry zero tonic (caller scales q=0). Cold start per
    amp (matched). Returns list of (r_pre, r_post) + flow R = post - pre
    in (rE, rI). Short runs; no HDP; no bridging assumptions.
    """
    masks, members = family_masks(model)
    gm = apply_theta6(model, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    nN = int(gm.params["emitter"].n_neurons)
    dtype = gm.params["emitter"].v0.dtype
    n_pre = int(round(float(pre_ms) / float(dt_ms)))
    n_rel = int(round(float(rel_ms) / float(dt_ms)))
    n_win = int(round(200.0 / float(dt_ms)))
    rows = []
    for amp in amps:
        state = initial_state(gm, seed)
        drive = jnp.concatenate([jnp.full((n_pre, nN), float(amp), dtype=dtype),
                                 jnp.zeros((n_rel, nN), dtype=dtype)])
        state, spikes, _ = run_segment(step_fn, state, drive)
        sp = np.asarray(spikes, dtype=float)
        r_pre = sp[n_pre - n_win:n_pre].mean(axis=0) * (1000.0 / dt_ms)
        r_post = sp[-n_win:].mean(axis=0) * (1000.0 / dt_ms)
        E_idx = np.asarray(members["E"])
        I_idx = np.concatenate([np.asarray(members[c]) for c in I_CLASSES])
        rows.append({"amp": float(amp),
                     "rE_pre": float(r_pre[E_idx].mean()),
                     "rI_pre": float(r_pre[I_idx].mean()),
                     "rE_post": float(r_post[E_idx].mean()),
                     "rI_post": float(r_post[I_idx].mean())})
    for r in rows:
        r["R_E"] = r["rE_post"] - r["rE_pre"]
        r["R_I"] = r["rI_post"] - r["rI_pre"]
    return rows


# --------------------------------------------------------------------------
# Fork-flow machinery: same microstate, different params (D_p inverse
# problem) + Markov/closure test of the (rE, rI) reduction.
# --------------------------------------------------------------------------
def prepare_release(model, amp, pre_ms, dt_ms=DT_MS_DEFAULT, seed=0,
                    win_ms=200.0):
    """Run pre-drive history; return (end_state, r_rel, step_fn_base).

    r_rel measured over last win_ms of drive. State carries full microstate
    (X, syn, H, w, RNG) for forking.
    """
    gm = apply_theta6(model, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    state = initial_state(gm, seed)
    nN = int(gm.params["emitter"].n_neurons)
    dtype = gm.params["emitter"].v0.dtype
    n_pre = int(round(float(pre_ms) / float(dt_ms)))
    n_win = int(round(float(win_ms) / float(dt_ms)))
    drive = jnp.full((n_pre, nN), float(amp), dtype=dtype)
    state, spikes, _ = run_segment(step_fn, state, drive)
    sp = np.asarray(spikes, dtype=float)
    masks, members = family_masks(gm)
    E_idx = np.asarray(members["E"])
    I_idx = np.concatenate([np.asarray(members[c]) for c in I_CLASSES])
    r_rel = (sp[-n_win:].mean(axis=0) * (1000.0 / dt_ms))
    return {"state": state, "step_fn": step_fn, "model": gm,
            "rE": float(r_rel[E_idx].mean()), "rI": float(r_rel[I_idx].mean())}


def fork_flow(prep, model_variant=None, rel_ms=500.0, dt_ms=DT_MS_DEFAULT,
              win_ms=200.0):
    """Continue prep['state'] under variant (or base) params, zero drive.

    Returns (r_end, R) with R = r_end - r_rel over win_ms windows.
    Variant shares microstate: pure parameter effect on flow.
    """
    from jomission.qualification.cmin import initial_state  # noqa (namespace)

    masks, members = family_masks(prep["model"])
    if model_variant is None:
        step_fn = prep["step_fn"]
    else:
        step_fn, _ = jtfne.compile_step_fn(model_variant, dt_ms=float(dt_ms),
                                           kernel="baseline",
                                           record_weight_trace=False)
    nN = int(prep["model"].params["emitter"].n_neurons)
    dtype = prep["model"].params["emitter"].v0.dtype
    n_rel = int(round(float(rel_ms) / float(dt_ms)))
    n_win = int(round(float(win_ms) / float(dt_ms)))
    drive = jnp.zeros((n_rel, nN), dtype=dtype)
    state, spikes, _ = run_segment(step_fn, prep["state"], drive)
    sp = np.asarray(spikes, dtype=float)
    E_idx = np.asarray(members["E"])
    I_idx = np.concatenate([np.asarray(members[c]) for c in I_CLASSES])
    r_end = sp[-n_win:].mean(axis=0) * (1000.0 / dt_ms)
    return {"rE": float(r_end[E_idx].mean()), "rI": float(r_end[I_idx].mean()),
            "R_E": float(r_end[E_idx].mean()) - prep["rE"],
            "R_I": float(r_end[I_idx].mean()) - prep["rI"]}


# --------------------------------------------------------------------------
# Funnel ownership: release-state audit (v/u/currents, matched dv units)
# + u-replacement fork. No parameter optimization.
# --------------------------------------------------------------------------
def release_audit(model_q0, amp, pre_ms=500.0, rel_ms=500.0,
                  dt_ms=DT_MS_DEFAULT, seed=0, bin_ms=50.0):
    """Run pre-drive then zero release with u_trace (no edge recording;
    currents reconstructed offline via validated path).

    Returns release distributions (v_E, u_E), binned collapse series of
    u_E mean, intrinsic proxy (-u_E), realized I_rec->E (exc/inh), and
    per-bin E rate. All currents in dv units (directly comparable).
    """
    gm = apply_theta6(model_q0, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False,
                                       record_u_trace=True)
    from jomission.qualification.cmin import initial_state

    state = initial_state(gm, seed)
    nN = int(gm.params["emitter"].n_neurons)
    dtype = gm.params["emitter"].v0.dtype
    n_pre = int(round(float(pre_ms) / float(dt_ms)))
    n_rel = int(round(float(rel_ms) / float(dt_ms)))
    drive = jnp.concatenate([jnp.full((n_pre, nN), float(amp), dtype=dtype),
                             jnp.zeros((n_rel, nN), dtype=dtype)])
    state, out = jtfne.run_continuation(step_fn, state, drive)
    import jax

    jax.block_until_ready(out[0])
    v, sp, u = (np.asarray(out[0]), np.asarray(out[1], dtype=float),
                np.asarray(out[4]))
    masks, members = family_masks(gm)
    E_idx = np.asarray(members["E"])
    wsums = edge_weight_sums(gm)
    rel = n_pre
    out_d = {
        "vE_rel": {"mean": float(v[rel, E_idx].mean()),
                   "std": float(v[rel, E_idx].std()),
                   "min": float(v[rel, E_idx].min()),
                   "max": float(v[rel, E_idx].max())},
        "uE_rel": {"mean": float(u[rel, E_idx].mean()),
                   "std": float(u[rel, E_idx].std()),
                   "min": float(u[rel, E_idx].min()),
                   "max": float(u[rel, E_idx].max())},
    }
    Iexc_full, Iinh_full = realized_currents_E(sp, wsums, dt_ms)
    b = int(round(float(bin_ms) / float(dt_ms)))
    post = slice(rel, rel + n_rel)
    nb = n_rel // b
    series = []
    for k in range(nb):
        sl = slice(rel + k * b, rel + (k + 1) * b)
        series.append({
            "t_ms": float((rel + k * b) * dt_ms),
            "uE": float(u[sl][:, E_idx].mean()),
            "Iexc_E": float(Iexc_full[sl].mean()),
            "Iinh_E": float(Iinh_full[sl].mean()),
            "rE": float(sp[sl][:, E_idx].mean() * (1000.0 / dt_ms)),
        })
    out_d["series"] = series
    out_d["end_state"] = state
    return out_d


def u_replace_fork(model_q0, u_mode="rest", u_ref=None, rel_ms=500.0,
                   dt_ms=DT_MS_DEFAULT, seed=0, amp=6.0, pre_ms=500.0):
    """Pre-drive, replace u_E at release with reference, continue at zero.

    u_mode 'rest': u = b*v per neuron (resting recovery).
    u_mode 'tonic': u_E = u_ref scalar (tonic-supported mean, measured live).
    Same microstate otherwise. Returns orig/replaced E rates + dR_E.
    Causal u-authority test over the funnel.
    """
    gm = apply_theta6(model_q0, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    masks, members = family_masks(gm)
    E_idx = np.asarray(members["E"])
    nN = int(gm.params["emitter"].n_neurons)
    dtype = gm.params["emitter"].v0.dtype
    n_pre = int(round(float(pre_ms) / float(dt_ms)))
    n_rel = int(round(float(rel_ms) / float(dt_ms)))
    n_win = int(round(200.0 / float(dt_ms)))
    state = initial_state(gm, seed)
    pre_drive = jnp.full((n_pre, nN), float(amp), dtype=dtype)
    state, _, _ = run_segment(step_fn, state, pre_drive)
    # release microstate captured; fork two continuations
    results = {}
    for tag in ("orig", "replaced"):
        st = state
        if tag == "replaced":
            e = gm.params["emitter"]
            b = np.asarray(e.b, dtype=float)
            v = np.asarray(st.dynamic.v, dtype=float)
            u = np.asarray(st.dynamic.u, dtype=float).copy()
            if u_mode == "rest":
                u[E_idx] = b[E_idx] * v[E_idx]
            else:
                u[E_idx] = float(u_ref)
            st = st._replace(dynamic=st.dynamic._replace(
                u=jnp.asarray(u, dtype=st.dynamic.u.dtype)))
        st, sp, _ = run_segment(step_fn, st, jnp.zeros((n_rel, nN), dtype=dtype))
        sp = np.asarray(sp, dtype=float)
        results[tag] = float(sp[-n_win:][:, E_idx].mean() * (1000.0 / dt_ms))
    results["dR_E"] = results["replaced"] - results["orig"]
    return results


# --------------------------------------------------------------------------
# Clamp vs recurrent fork: current-clamp response map + waveform replay.
# Same release microstate; HDP OFF; no substrate change.
# --------------------------------------------------------------------------
def release_prep(model_q0, amp=6.0, pre_ms=500.0, dt_ms=DT_MS_DEFAULT, seed=0):
    """Shared release microstate + base step_fn. Returns dict."""
    gm = apply_theta6(model_q0, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    state = initial_state(gm, seed)
    nN = int(gm.params["emitter"].n_neurons)
    dtype = gm.params["emitter"].v0.dtype
    n_pre = int(round(float(pre_ms) / float(dt_ms)))
    state, sp_pre, _ = run_segment(step_fn, state,
                               jnp.full((n_pre, nN), float(amp), dtype=dtype))
    masks, members = family_masks(gm)
    E_idx = np.asarray(members["E"])
    I_idx = np.concatenate([np.asarray(members[c]) for c in I_CLASSES])
    r_rel = np.asarray(sp_pre, dtype=float)[-2000:].mean(axis=0) * (1000.0 / dt_ms)
    return {"state": state, "step_fn": step_fn, "model": gm,
            "members": members, "masks": masks,
            "rE": float(r_rel[E_idx].mean()), "rI": float(r_rel[I_idx].mean())}


def clamp_curve(prep, currents=(0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0),
                rel_ms=1000.0, dt_ms=DT_MS_DEFAULT, win_ms=200.0,
                target="E"):
    """Constant additive current to target class post-release.

    Returns list of (I, r_end, R) with R vs the recorded release rate.
    Sustained = r_end holds above floor without saturation; finds I_sustain.
    """
    members = prep["members"]
    nN = int(prep["model"].params["emitter"].n_neurons)
    dtype = prep["model"].params["emitter"].v0.dtype
    tbl = prep["model"].neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    tgt = np.flatnonzero(cls == target) if target != "I" else np.flatnonzero(cls != "E")
    E_idx = np.asarray(members["E"])
    n_rel = int(round(float(rel_ms) / float(dt_ms)))
    n_win = int(round(float(win_ms) / float(dt_ms)))
    rows = []
    for I in currents:
        extra = np.zeros(nN)
        extra[tgt] = float(I)
        drive = jnp.asarray(np.tile(extra, (n_rel, 1)), dtype=dtype)
        _, sp, _ = run_segment(prep["step_fn"], prep["state"], drive)
        sp = np.asarray(sp, dtype=float)
        r_end = float(sp[-n_win:][:, E_idx].mean() * (1000.0 / dt_ms))
        rows.append({"I": float(I), "rE_end": r_end})
    return rows


def per_neuron_Irec(spikes, model, dt_ms=DT_MS_DEFAULT):
    """Per-neuron realized recurrent current waveforms [T, N].

    Offline reconstruction (validated): per-receptor exponential filtering
    of realized spikes through signed weights. Includes kernel tau/dt gain.
    """
    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    ri = np.asarray(el.receptor_index, dtype=np.int64)
    tau = np.asarray(el.tau_ms, dtype=float)
    n = int(w.shape[0] and max(int(post.max()), int(pre.max())) + 1)
    sp = np.asarray(spikes, dtype=float)
    Irec = np.zeros_like(sp)
    for r in np.unique(ri):
        sel = ri == r
        tau_r = float(np.unique(tau[sel])[0])
        f = _exp_filter(sp, tau_r, dt_ms) * (tau_r / dt_ms)
        np.add.at(Irec, (slice(None), post[sel]), f[:, pre[sel]] * w[sel][None, :])
    return Irec


def waveform_replay(prep, rel_ms=1000.0, dt_ms=DT_MS_DEFAULT, win_ms=200.0):
    """A vs C: reference release vs REC_OFF + recorded per-neuron I_rec
    waveforms injected as drive. Same microstate. Compares trajectories.

    Returns dict with both E-rate tails + max abs divergence + correlation.
    Agreement => architectural (magnitude) problem; disagreement =>
    representation/coupling problem.
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    members = prep["members"]
    E_idx = np.asarray(members["E"])
    gm = prep["model"]
    nN = int(gm.params["emitter"].n_neurons)
    dtype = gm.params["emitter"].v0.dtype
    n_rel = int(round(float(rel_ms) / float(dt_ms)))
    n_win = int(round(float(win_ms) / float(dt_ms)))
    # A: reference release, record spikes for waveform extraction
    _, spA, _ = run_segment(prep["step_fn"], prep["state"],
                            jnp.zeros((n_rel, nN), dtype=dtype))
    spA = np.asarray(spA, dtype=float)
    Irec = per_neuron_Irec(spA, gm, dt_ms)
    # C: REC_OFF model, inject recorded waveforms as drive
    el = gm.params["edge_list"]
    gm_off = replace(gm, params={**gm.params, "edge_list": EdgeList(
        pre=el.pre, post=el.post, weight=jnp.zeros_like(el.weight),
        receptor_index=el.receptor_index, tau_ms=el.tau_ms,
        source_calibration_status=el.source_calibration_status)})
    step_off, _ = jtfne.compile_step_fn(gm_off, dt_ms=float(dt_ms),
                                        kernel="baseline",
                                        record_weight_trace=False)
    _, spC, _ = run_segment(step_off, prep["state"],
                            jnp.asarray(Irec[:n_rel], dtype=dtype))
    spC = np.asarray(spC, dtype=float)
    b = max(1, n_rel // 20)
    rA = spA[:, E_idx].mean(axis=1).reshape(-1) * (1000.0 / dt_ms)
    rC = spC[:, E_idx].mean(axis=1).reshape(-1) * (1000.0 / dt_ms)
    return {"rA_end": float(rA[-n_win:].mean()), "rC_end": float(rC[-n_win:].mean()),
            "max_div": float(np.abs(rA - rC).max()),
            "corr": float(np.corrcoef(rA, rC)[0, 1]) if rA.std() > 0 and rC.std() > 0 else float("nan"),
            "Irec_mean": float(np.abs(Irec).mean())}


# --------------------------------------------------------------------------
# Kernel calibration: star-graph assay mapping (w_EE, r_pre) -> I_EE(t).
# Public surface only: build_laminar_column + construct + post-construct
# EdgeList surgery (established pattern) + compile/run with edge currents.
# No closed substrate, no HDP. Posts passive (zero drive); edge currents
# are presynaptic-driven so post spiking is irrelevant to K.
# --------------------------------------------------------------------------
def calibration_star(n_post=50, seed=0, w_base=0.014, tau_ms=2.0,
                     layers=("L2/3",), fractions=None):
    """Single E driver + passive E targets, star edges driver->targets.

    Returns (model, w_base). All-E column; caller sets drives/weights.
    """
    import jaxfne as jtfne

    fr = {"E": 1.0} if fractions is None else dict(fractions)
    cfg = jtfne.build_laminar_column(
        "CAL", n=int(n_post) + 1, layers=list(layers),
        cell_type_fractions=fr, ei_profile="flat", geometry="laminar",
        edge_seed=int(seed))
    cfg = (cfg.runtime(seed=int(seed), duration_ms=1000.0, dt_ms=0.1, dtype="float32")
           .set_emitter("izhikevich", "cortical_eig")
           .probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=4)
           .field(domain="laminar_column", conductivity="proxy",
                  boundary="mean_zero_neumann"))
    model = jtfne.construct(cfg)
    el = model.params["edge_list"]
    n = int(n_post) + 1
    pre = np.zeros(n_post, dtype=np.int64)
    post = np.arange(1, n, dtype=np.int64)
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    new_el = EdgeList(pre=jnp.asarray(pre), post=jnp.asarray(post),
                      weight=jnp.asarray(np.full(n_post, float(w_base),
                                                 dtype=np.float32)),
                      receptor_index=jnp.asarray(np.zeros(n_post, dtype=np.int32)),
                      tau_ms=jnp.asarray(np.full(n_post, float(tau_ms),
                                                 dtype=np.float32)),
                      source_calibration_status=el.source_calibration_status)
    return (replace(model, params={**model.params, "edge_list": new_el}),
            float(w_base))


def kernel_map(model, w_mults=(1.0, 4.0, 16.0, 64.0, 256.0),
               drives=(3.0, 4.5, 6.0, 8.0, 12.0), dur_ms=2000.0,
               dt_ms=DT_MS_DEFAULT, seed=0, skip_ms=1000.0):
    """K_EE(w, r_pre): per-edge mean |current| + RMS + quantiles, r measured.

    Drive applied to neuron 0 only; targets passive. Returns rows with
    (w_mult, drive, r_pre, Imean, Irms, q10, q50, q90) over post-drive window.
    """
    from dataclasses import replace

    rows = []
    el0 = model.params["edge_list"]
    w0 = np.asarray(el0.weight, dtype=float)
    nN = int(model.params["emitter"].n_neurons)
    dtype = model.params["emitter"].v0.dtype
    from jomission.qualification.cmin import initial_state

    n_steps = int(round(float(dur_ms) / float(dt_ms)))
    n_skip = int(round(float(skip_ms) / float(dt_ms)))
    for wm in w_mults:
        el = el0.__class__(pre=el0.pre, post=el0.post,
                           weight=jnp.asarray(w0 * float(wm), dtype=el0.weight.dtype),
                           receptor_index=el0.receptor_index, tau_ms=el0.tau_ms,
                           source_calibration_status=el0.source_calibration_status)
        mg = replace(model, params={**model.params, "edge_list": el})
        step_fn, _ = jtfne.compile_step_fn(mg, dt_ms=float(dt_ms), kernel="baseline",
                                           record_weight_trace=False,
                                           record_edge_current=True)
        for dv in drives:
            state = initial_state(mg, seed)
            d = np.zeros(nN)
            d[0] = float(dv)
            drive = jnp.asarray(np.tile(d, (n_steps, 1)), dtype=dtype)
            state, out = jtfne.run_continuation(step_fn, state, drive)
            import jax

            jax.block_until_ready(out[0])
            sp = np.asarray(out[1], dtype=float)[n_skip:]
            ec = np.asarray(out[4], dtype=float)[n_skip:]
            r_pre = float(sp[:, 0].mean() * (1000.0 / dt_ms))
            per_edge = np.abs(ec).mean(axis=0)
            rows.append({"w_mult": float(wm), "drive": float(dv),
                         "r_pre": round(r_pre, 2),
                         "Imean": float(per_edge.mean()),
                         "Irms": float(np.sqrt((per_edge ** 2).mean())),
                         "q10": float(np.quantile(per_edge, 0.1)),
                         "q50": float(np.quantile(per_edge, 0.5)),
                         "q90": float(np.quantile(per_edge, 0.9))})
    return rows


def ablate_src_to_E(model_g, src_class):
    """Zero src_class->E edges only (class-resolved veto fork).

    Returns (model, n_cut). Same-microstate forks test whether one
    inhibitory class's transient recruitment vetoes basin capture.
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    tbl = model_g.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    el = model_g.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    cut = (cls[pre] == str(src_class)) & (cls[post] == "E")
    w2 = w.copy()
    w2[cut] = 0.0
    kwargs = dict(pre=el.pre, post=el.post,
                  weight=jnp.asarray(w2, dtype=el.weight.dtype),
                  receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model_g, params={**model_g.params, "edge_list": EdgeList(**kwargs)}), int(cut.sum())


# --------------------------------------------------------------------------
# Intrinsic-compartment decomposition: subthreshold vs spike/reset history.
# Same release microstate; DynamicState fields replaced one at a time.
# No parameter changes. HDP OFF.
# --------------------------------------------------------------------------
def intrinsic_forks(model_q0, variants=("full", "syn0", "vdep", "urest", "prev0"),
                    amp=12.0, pre_ms=500.0, rel_ms=2000.0, dt_ms=DT_MS_DEFAULT,
                    seed=0, vdep_mv=-50.0):
    """Fork release microstate with single-field replacements (plus rebirth
    conjunction: v+u+syn+prev reset jointly).
    full: unmodified control. syn0: syn_state -> 0 (kill network history).
    vdep: v_E -> vdep_mv (subthreshold position). urest: u_E -> b*v
    (reset-history accumulation removed). prev0: prev_spikes -> 0.
    Returns {variant: E-tail rate + full per-class tails}.
    """
    gm = apply_theta6(model_q0, np.zeros(6))
    step_fn, _ = jtfne.compile_step_fn(gm, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    from jomission.qualification.cmin import initial_state

    masks, members = family_masks(gm)
    E_idx = np.asarray(members["E"])
    nN = int(gm.params["emitter"].n_neurons)
    dtype = gm.params["emitter"].v0.dtype
    n_pre = int(round(float(pre_ms) / float(dt_ms)))
    n_rel = int(round(float(rel_ms) / float(dt_ms)))
    n_win = int(round(200.0 / float(dt_ms)))
    state = initial_state(gm, seed)
    state, _, _ = run_segment(step_fn, state,
                              jnp.full((n_pre, nN), float(amp), dtype=dtype))
    e = gm.params["emitter"]
    out = {}
    for tag in variants:
        st = state
        dyn = st.dynamic
        v = np.asarray(dyn.v, dtype=float)
        u = np.asarray(dyn.u, dtype=float)
        b = np.asarray(e.b, dtype=float)
        if tag == "syn0":
            st = st._replace(dynamic=dyn._replace(
                syn_state=jnp.zeros_like(dyn.syn_state)))
        elif tag == "vdep":
            v = v.copy()
            v[E_idx] = float(vdep_mv)
            st = st._replace(dynamic=dyn._replace(
                v=jnp.asarray(v, dtype=dyn.v.dtype)))
        elif tag == "urest":
            u = u.copy()
            u[E_idx] = b[E_idx] * v[E_idx]
            st = st._replace(dynamic=dyn._replace(
                u=jnp.asarray(u, dtype=dyn.u.dtype)))
        elif tag == "prev0":
            st = st._replace(dynamic=dyn._replace(
                prev_spikes=jnp.zeros_like(dyn.prev_spikes)))
        elif tag == "rebirth":
            v = v.copy()
            v[E_idx] = float(vdep_mv)
            u = u.copy()
            u[E_idx] = b[E_idx] * v[E_idx]
            st = st._replace(dynamic=dyn._replace(
                v=jnp.asarray(v, dtype=dyn.v.dtype),
                u=jnp.asarray(u, dtype=dyn.u.dtype),
                syn_state=jnp.zeros_like(dyn.syn_state),
                prev_spikes=jnp.zeros_like(dyn.prev_spikes)))
        elif tag != "full":
            raise ValueError(tag)
        st, sp, _ = run_segment(step_fn, st, jnp.zeros((n_rel, nN), dtype=dtype))
        sp = np.asarray(sp, dtype=float)
        tail = sp[-n_win:]
        out[tag] = {c: float(tail[:, np.asarray(members[c])].mean() * (1000.0 / dt_ms))
                    for c in ("E", "PV", "SST", "VIP")}
        # synaptic magnitude at release (first bin) for accounting
        if tag == "full":
            out["syn_rel_mean"] = float(np.abs(np.asarray(dyn.syn_state, dtype=float)).mean())
    return out


def scale_pair(model_g, src_class, tgt_class, mult):
    """Multiply src->tgt edge weights (class-pair efficacy dial).

    Connectivity fixed (same edges, only weights scale). Workhorse for
    transfer qualification (e.g. E->PV) and loop-geometry construction.
    Returns (model, n_affected).
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    tbl = model_g.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    el = model_g.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    sel = (cls[pre] == str(src_class)) & (cls[post] == str(tgt_class))
    w2 = w.copy()
    w2[sel] = w[sel] * float(mult)
    kwargs = dict(pre=el.pre, post=el.post,
                  weight=jnp.asarray(w2, dtype=el.weight.dtype),
                  receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model_g, params={**model_g.params, "edge_list": EdgeList(**kwargs)}), int(sel.sum())

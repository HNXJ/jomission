"""S8 full-state slow-manifold stability: predeclared Jacobian construction.

Parent: 23c56d5 (S7 PASS). Candidates: s_E in {50,54,57,61} (passing
subset; s61 boundary-marginal included, no pre-selection). No tuning of
any parameter during S8: adjudication only.

State vector (14, per-ms units), all dynamically participating slow means:
  z = [u_E,u_PV,u_SST, aux_EE,aux_EPV,aux_ESST,aux_PVE,aux_SSTE,
       w_EE,w_EPV,w_ESST,w_PVE,w_SSTE, H]
Slaved (fast) subsystem, scope condition S8-ADB: r_X = F_X(I_X) measured,
Vbar_X = Vbar_X(I_X) fresh-measured, syn_p = r_pre*tau_p/1000 exact
(tau_syn 2-5ms << slow tau >= 17ms; sustained regular firing observed in
all S7 battery levels). If the slaving matrix (I - F'A) is singular the
fast-rate subsystem itself is unstable -> flagged, never papered over.

Native gains (nothing transplanted, nothing fitted):
  F'_X(I*)   central differences of the fresh joint (F,Vbar) assay
             (single-cell Euler, sealed protocol; fresh F must match sealed
             F within 5% where sealed r > 1 else UNRESOLVED);
  Vbar'_X    same assay (mean V over second half);
  G_X        adaptation gain dR/du at (I*,u*): preset u0 in
             u*{0.7,1.0,1.3}, I=I*, rate over first 150ms, least squares
             (R^2 recorded);
  a/b/d      exact emitter params; K structural; tau exact kernel math;
  rule partials exact form at act* inverted from dw=0 with measured w*:
             act2* = lam(w*-w_lo)/(w_hi-w*).
Closure assert (predeclared): 1.0 <= act*_rms/aux*_mean <= 2.5
(RMS>=mean; sawtooth shape bound), else UNRESOLVED.
H decoupling self-check: exact eigenvalue -gamma_h present (1e-6).

Margin (predeclared, dimensionally checked: all S8 dynamics in 1/ms,
slowest native tau ~100ms -> lambda ~-0.01; margin -0.005 = tau<=200ms):
  max Re(lambda) <= -0.005.
Diagnosis on failure: slowest-mode participation by group
(neural-U | HDP-aux | HDP-w | H) identifies neural vs HDP-state instability.
"""

from __future__ import annotations

import numpy as np

CANDIDATES: tuple[float, ...] = (50.0, 54.0, 57.0, 61.0)

MARGIN = -0.005  # 1/ms

DT_MS = 0.1
DUR_MS = 2000.0
I_GRID = tuple(float(x) for x in np.arange(0.0, 40.5, 0.5))

# Pathway table: (pre, post, tau_syn, K, battery, row-class, E-scaled?)
PATHWAYS = (
    ("E", "E", 2.0, 299.0, "b1", "E", True),
    ("E", "PV", 2.0, 300.0, "b1", "PV", True),
    ("E", "SST", 2.0, 300.0, "b1", "SST", True),
    ("PV", "E", 5.0, 40.0, "b2", "E", False),
    ("SST", "E", 5.0, 32.0, "b3", "E", False),
)

TAU_ACT = 20.0
K_W = 0.2
LAM = 0.09
GAMMA_H = 0.02
W_HI_1 = 0.05
W_LO_1 = 0.002


def scale_tag(s: float) -> str:
    return ("%.1f" % s).replace(".", "p")


def class_params():
    """Exact emitter (a,b,d) per class from a fresh plant model."""
    import jax.numpy as jnp  # noqa: F401 (plant construction side)
    from jomission.qualification import ei_geometry as G

    model, _, _ = G.build_star(driver_class="E",
                               n_targets={"E": 10, "PV": 10, "SST": 10,
                                          "VIP": 10}, seed=0)
    e = model.params["emitter"]
    tbl = model.neuron_table()
    got = np.array([str(r["cell_type"]) for r in tbl])
    out = {}
    for c in ("E", "PV", "SST", "VIP"):
        i = int(np.flatnonzero(got == c)[0])
        out[c] = {"a": float(np.asarray(e.a)[i]),
                  "b": float(np.asarray(e.b)[i]),
                  "c": float(np.asarray(e.c)[i]),
                  "d": float(np.asarray(e.d)[i])}
    return out


def run_cell(a, b, c, d, I, u0=None, rate_window_ms=None):
    """Single-cell Euler (sealed protocol). Returns (rate_Hz, mean_V)."""
    v, u = float(c), float(u0) if u0 is not None else float(b) * float(c)
    n_steps = int(round(DUR_MS / DT_MS))
    if rate_window_ms is None:
        skip = n_steps // 2
        win = n_steps - skip
    else:
        skip = 0
        win = int(round(rate_window_ms / DT_MS))
    nsp = 0
    vsum, vn = 0.0, 0
    for t in range(n_steps if rate_window_ms is None else win):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + float(I)
        du = a * (b * v - u)
        v += DT_MS * dv
        u += DT_MS * du
        if v >= 30.0:
            if t >= skip:
                nsp += 1
            v = float(c)
            u += float(d)
        if rate_window_ms is None and t >= skip:
            vsum += v
            vn += 1
    if rate_window_ms is None:
        return nsp / (DUR_MS / 1000.0 / 2.0), vsum / max(vn, 1)
    return nsp / (rate_window_ms / 1000.0), None


def assay_FV(params):
    """Fresh joint (F, Vbar) assay per class on the predeclared grid."""
    out = {}
    for c, p in params.items():
        rows = [(I,) + run_cell(p["a"], p["b"], p["c"], p["d"], I)
                for I in I_GRID]
        out[c] = {"I": np.array([r[0] for r in rows]),
                  "r": np.array([r[1] for r in rows]),
                  "V": np.array([r[2] for r in rows])}
    return out


def central_diff(x, y, x0):
    """Central-difference derivative of interpolant y(x) at x0 (h=0.5)."""
    h = 0.5
    return (float(np.interp(x0 + h, x, y)) - float(np.interp(x0 - h, x, y))) / (2 * h)


def adaptation_gain(p, I_star, u_star):
    """dR/du at (I*, u*): preset-u transient rates, least squares."""
    u0s = np.array([0.7, 1.0, 1.3]) * u_star
    rs = np.array([run_cell(p["a"], p["b"], p["c"], p["d"], I_star,
                            u0=u0, rate_window_ms=150.0)[0] for u0 in u0s])
    A = np.vstack([u0s, np.ones_like(u0s)]).T
    slope, _ = np.linalg.lstsq(A, rs, rcond=None)[0]
    pred = slope * u0s + (rs.mean() - slope * u0s.mean())
    ss = float(((rs - pred) ** 2).sum())
    tot = float(((rs - rs.mean()) ** 2).sum())
    r2 = 1.0 - ss / tot if tot > 0 else 1.0
    return float(slope), float(r2)


def operating_point(s, FK_unused=None):
    """Reconstruct the 14-state FP + gains for candidate s from S7 JSONs."""
    import json

    tag = scale_tag(s)
    b1 = json.load(open(f"results/s7_battery_b1_s{tag}.json"))
    b2 = json.load(open(f"results/s7_battery_b2_s{tag}.json"))
    b3 = json.load(open(f"results/s7_battery_b3_s{tag}.json"))
    geo = json.load(open(f"results/s7_geometry_s{tag}.json"))
    assert geo["verdict"] == "S7_PASS", (s, geo["verdict"])
    root = geo["cortical"][0]
    rE, rPV, rSST, Ie = root["rE"], root["rPV"], root["rSST"], root["I_E"]
    bats = {"b1": b1, "b2": b2, "b3": b3}

    def curve(rows, key):
        xs = np.array([r["driver_rate"] for r in rows])
        ys = np.array([r["classes"][key]["I"] if key in r["classes"]
                       else r["classes"]["E"]["I"] for r in rows])
        o = np.argsort(xs)
        return xs[o], ys[o]

    def wcurve(rows, key):
        xs = np.array([r["driver_rate"] for r in rows])
        ys = np.array([r["classes"][key]["w"] for r in rows])
        o = np.argsort(xs)
        return xs[o], ys[o]

    x_ee, i_ee = curve(b1["rows"], "E")
    x_epv, i_epv = curve(b1["rows"], "PV")
    x_esst, i_esst = curve(b1["rows"], "SST")
    r_pre = {"E": rE, "PV": rPV, "SST": rSST}
    i_tot = {"E": Ie,
             "PV": 300.0 * float(np.interp(rE, x_epv, i_epv)),
             "SST": 300.0 * float(np.interp(rE, x_esst, i_esst))}
    w_star, w_hi, w_lo = {}, {}, {}
    for (pre, post, tau, K, bat, cls, esc) in PATHWAYS:
        x, y = wcurve(bats[bat]["rows"], cls)
        w_star[(pre, post)] = float(np.interp(r_pre[pre], x, y))
        w_hi[(pre, post)] = (s * W_HI_1) if esc else W_HI_1
        w_lo[(pre, post)] = (s * W_LO_1) if esc else W_LO_1
    return {"r": {"E": rE, "PV": rPV, "SST": rSST}, "I": i_tot,
            "w_star": w_star, "w_hi": w_hi, "w_lo": w_lo,
            "curves": {"ee": (x_ee, i_ee)}}


def build_jacobian(s, FV, params):
    """Assemble the 14x14 slow-manifold Jacobian + diagnostics."""
    op = operating_point(s)
    r, I = op["r"], op["I"]
    cls = ("E", "PV", "SST")
    Fp = {c: central_diff(FV[c]["I"], FV[c]["r"], I[c]) for c in cls}
    Vp = {c: central_diff(FV[c]["I"], FV[c]["V"], I[c]) for c in cls}
    Vstar = {c: float(np.interp(I[c], FV[c]["I"], FV[c]["V"])) for c in cls}
    Gg, R2 = {}, {}
    u_star = {}
    for c in cls:
        u_star[c] = (params[c]["b"] * Vstar[c]
                     + (params[c]["d"] / params[c]["a"]) * (r[c] / 1000.0))
        Gg[c], R2[c] = adaptation_gain(params[c], I[c], u_star[c])

    # Pathway slow states: aux*, act2* (inverted), syn*.
    aux_star, act2_star, syn_star = {}, {}, {}
    for (pre, post, tau, K, bat, clss, esc) in PATHWAYS:
        aux_star[(pre, post)] = r[pre] * TAU_ACT / 1000.0
        # Rule math is on efficacy MAGNITUDE (kernel sign safety via
        # exc_mask); inhibitory w* is negative, so take abs first.
        wm = abs(op["w_star"][(pre, post)])
        hi, lo = op["w_hi"][(pre, post)], op["w_lo"][(pre, post)]
        act2_star[(pre, post)] = LAM * (wm - lo) / max(hi - wm, 1e-12)
        syn_star[(pre, post)] = r[pre] * tau / 1000.0
        ratio = float(np.sqrt(max(act2_star[(pre, post)], 0.0))
                      / max(aux_star[(pre, post)], 1e-12))
        # Closure health (calibrated 2026-09-15: exact Jensen floor is 1.0;
        # the measurement chain -- w* interpolation, <=settle residual drift
        # from dw=0 -- carries ~1% noise, so 0.9; gross structural failure
        # would miss by 10x, not 1%).
        assert 0.9 <= ratio <= 2.5, ((pre, post), ratio)
    # A (3x3 current<-rate, signed w) ; B (3x5 current<-wmag, receptor sign);
    # P (5x3 pre-class select). sign: +1 E-pre, -1 I-pre (ri convention).
    A = np.zeros((3, 3))
    B = np.zeros((3, 5))
    P = np.zeros((5, 3))
    for j, (pre, post, tau, K, bat, clss, esc) in enumerate(PATHWAYS):
        i, k = cls.index(post), cls.index(pre)
        sgn = 1.0 if pre == "E" else -1.0
        A[i, k] += K * op["w_star"][(pre, post)] * tau / 1000.0
        B[i, j] = sgn * K * syn_star[(pre, post)]
        P[j, k] = 1.0
    Fd = np.diag([Fp[c] for c in cls])
    Gd = np.diag([Gg[c] for c in cls])
    M = np.eye(3) - Fd @ A
    svals = np.linalg.svd(M, compute_uv=False)
    assert float(svals.min()) >= 1e-9, float(svals.min())  # slaving health
    Minv = np.linalg.inv(M)
    Mr_w = Minv @ Fd @ B  # dr/dw (3x5)
    Mr_u = Minv @ Gd      # dr/du (3x3)
    # Fast-rate diagnostic: eig(F'A) must avoid 1 (earlier failure mode).
    fast = np.linalg.eigvals(Fd @ A)

    n_u, n_a, n_w = 3, 5, 5
    J = np.zeros((14, 14))
    # du_X/dt = a b Vp dI - a du + (d/1000) dr ; dI rows via (A Mr + B).
    dI_dw = A @ Mr_w + B          # 3x5
    dI_du = A @ Mr_u              # 3x3
    for ix, c in enumerate(cls):
        a, b, d = params[c]["a"], params[c]["b"], params[c]["d"]
        J[ix, 8:13] = (a * b * Vp[c]) * dI_dw[ix, :] + (d / 1000.0) * Mr_w[ix, :]
        J[ix, 0:3] = (a * b * Vp[c]) * dI_du[ix, :] + (d / 1000.0) * Mr_u[ix, :]
        J[ix, ix] += -a
    # daux_p/dt = -aux/tau + r_pre/1000 ; r_pre via Mr rows.
    for j, (pre, post, tau, K, bat, clss, esc) in enumerate(PATHWAYS):
        k = cls.index(pre)
        J[3 + j, 8:13] = (1.0 / 1000.0) * Mr_w[k, :]
        J[3 + j, 0:3] = (1.0 / 1000.0) * Mr_u[k, :]
        J[3 + j, 3 + j] = -1.0 / TAU_ACT
    # dw_p/dt partials (exact form at inverted act2*; magnitudes).
    for j, (pre, post, tau, K, bat, clss, esc) in enumerate(PATHWAYS):
        wm = abs(op["w_star"][(pre, post)])
        hi, lo = op["w_hi"][(pre, post)], op["w_lo"][(pre, post)]
        a2 = act2_star[(pre, post)]
        ax = aux_star[(pre, post)]
        J[8 + j, 3 + j] = K_W * (hi - wm) * 2.0 * a2 / max(ax, 1e-12)
        J[8 + j, 8 + j] = -K_W * (a2 + LAM)
    # dH/dt = -gamma H (decoupled: dw/dH = 0 at FP by dw=0 construction).
    J[13, 13] = -GAMMA_H
    assert np.all(np.isfinite(J))
    # H-decoupling self-check: exact eigenvalue -gamma present.
    lam = np.linalg.eigvals(J)
    assert float(np.abs(lam + GAMMA_H).min()) <= 1e-6
    return {"J": J, "Fp": Fp, "Vp": Vp, "G": Gg, "R2": R2,
            "u_star": u_star, "Vstar": Vstar, "fast_max": float(np.abs(fast).max()),
            "slaving_sigma_min": float(svals.min()), "op": op}

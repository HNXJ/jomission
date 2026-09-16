"""V2.1c intrinsic-heterogeneity design: reduced E population, single-cell numerics.

No plant execution. Each of M independent E cells integrates the JaxFNE
baseline Izhikevich update (engine order: simultaneous Euler on v, u with
I = tonic + schedule + recurrent + 0.5*N(0,1); spike at v_next >= 30 ->
v = c, u = u_next + d) under:

* the low_sm2p0 shot schedule, bit-identical E columns of the V2.1b generator
  (tests/test_v21b_operation.py shot_chunks, seed 21100);
* native noise 0.5*N(0,1) per step (numpy stream; the engine's JAX stream is
  not reproducible outside the plant);
* surrogate recurrent input: for each presynaptic class X, Poisson events at
  K_X * r_X (K_X = plant in-degree to E, r_X = sealed network class rate) feed
  a per-cell exponential trace with tau_X; each step's increment is
  n*mean(w_X) + sqrt(n)*sd(w_X)*N(0,1) with w_X the plant's X->E weights.

Shapes: currents and states are (M,) per step; spike rasters are (steps, M)
bool; rates in Hz; currents in native JaxFNE units; time in ms. Float64.
"""

from __future__ import annotations

import numpy as np

DT_MS = 0.1
V_PEAK = 30.0
NATIVE_NOISE = 0.5
TAU_MS = {"E": 2.0, "PV": 5.0, "SST": 5.0, "VIP": 5.0}
CLASSES = ("E", "PV", "SST", "VIP")


def lognormal_dispersion(p0, width, m, seed):
    """Zero-mean dispersion p_i = p0 * g_i / mean(g), sd(p)/|p0| == width exactly.

    g_i = exp(s * z_i), z_i = standard-normal quantiles (k - 0.5)/m, permuted by
    PCG64(seed). s solved by bisection so the realized relative sd equals width.
    Sign of p0 preserved for every cell. Returns (m,) float64.
    """
    from statistics import NormalDist
    if width == 0.0:
        return np.full(m, float(p0))
    nd = NormalDist()
    z = np.array([nd.inv_cdf((k + 0.5) / m) for k in range(m)])
    z = np.random.Generator(np.random.PCG64(int(seed))).permutation(z)

    def rel_sd(s):
        g = np.exp(s * z)
        g = g / g.mean()
        return float(g.std())

    lo, hi = 0.0, 5.0
    assert rel_sd(hi) > width, width
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if rel_sd(mid) < width:
            lo = mid
        else:
            hi = mid
    g = np.exp(0.5 * (lo + hi) * z)
    g = g / g.mean()
    return float(p0) * g


def recurrent_spec(model, cls, class_rates_hz):
    """Per presynaptic class onto E: in-degree K, weight mean/sd, tau, rate."""
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    w = np.asarray(el.weight, dtype=np.float64)
    spec = {}
    for x in CLASSES:
        sel = (cls[pre] == x) & (cls[post] == "E")
        if not sel.any():
            continue
        indeg = np.bincount(post[sel], minlength=len(cls))[cls == "E"]
        assert indeg.min() == indeg.max(), "non-uniform in-degree onto E"
        spec[x] = {"K": int(indeg[0]), "w_mean": float(w[sel].mean()),
                   "w_sd": float(w[sel].std()), "tau_ms": TAU_MS[x],
                   "rate_hz": float(class_rates_hz[x])}
    return spec


def simulate_population(params, shot_chunks_e, rec, n_chunk, steps, seed,
                        v0=None, windows=(1, 2, 3)):
    """Reduced E population. params: dict a,b,c,d -> (M,) arrays; tonic in params['I0'].

    shot_chunks_e: list of (steps, M) float arrays (schedule, zero-mean shot).
    Returns list of (steps, M) bool spike rasters for the analysis windows.
    """
    a, b, c, d = (np.asarray(params[k], dtype=np.float64) for k in "abcd")
    I0 = float(params["I0"])
    m = a.shape[0]
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    v = np.full(m, -65.0) if v0 is None else np.array(v0, dtype=np.float64)
    u = b * v
    traces = {x: np.zeros(m) for x in rec}
    decay = {x: np.exp(-DT_MS / rec[x]["tau_ms"]) for x in rec}
    out = []
    for ch in range(n_chunk):
        noise = rng.standard_normal((steps, m))
        ev = {x: rng.poisson(rec[x]["K"] * rec[x]["rate_hz"] * DT_MS / 1000.0, size=(steps, m))
              for x in rec}
        wn = {x: rng.standard_normal((steps, m)) for x in rec}
        sched = shot_chunks_e[ch]
        keep = ch in windows
        raster = np.zeros((steps, m), dtype=bool) if keep else None
        for t in range(steps):
            syn = 0.0
            for x, r in rec.items():
                syn = syn + traces[x]
            I = I0 + sched[t] + syn + NATIVE_NOISE * noise[t]
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            du = a * (b * v - u)
            v = v + DT_MS * dv
            u = u + DT_MS * du
            spk = v >= V_PEAK
            v = np.where(spk, c, v)
            u = np.where(spk, u + d, u)
            for x, r in rec.items():
                n = ev[x][t]
                traces[x] = traces[x] * decay[x] + (n * r["w_mean"] + np.sqrt(n) * r["w_sd"] * wn[x][t])
            if keep:
                raster[t] = spk
        if not np.isfinite(v).all():
            return None  # non-finite state: caller records pathology
        if keep:
            out.append(raster)
    return out


def window_metrics(raster, window_s=5.0):
    """V2.1 estimators restricted to E, plus pathology counts. raster (steps, M) bool."""
    m = raster.shape[1]
    nsp = raster.sum(axis=0)
    rates = nsp / window_s
    cvs = np.full(m, np.nan)
    for i in range(m):
        idx = np.flatnonzero(raster[:, i])
        if len(idx) >= 4:
            isi = np.diff(idx).astype(float) * DT_MS
            cvs[i] = float(isi.std() / max(isi.mean(), 1e-9))
    act = ~np.isnan(cvs)
    return {
        "rate_mean": float(rates.mean()),
        "rate_cv": float(rates.std() / max(rates.mean(), 1e-9)),
        "rate_cv_active": (float(rates[act].std() / max(rates[act].mean(), 1e-9)) if act.any() else None),
        "frac_in": float(((cvs[act] >= 0.5) & (cvs[act] <= 1.5)).mean()) if act.any() else 0.0,
        "cv_median": float(np.median(cvs[act])) if act.any() else None,
        "silent_frac": float((~act).mean()),
        "above_60Hz_frac": float((rates > 60.0).mean()),
    }


def deterministic_rate_cv(a, b, c, d, I, dur_s=10.0):
    """Vectorized H1 protocol (constant I, last half analysed). Returns rates, cvs (M,)."""
    a, b, c, d = (np.asarray(x, dtype=np.float64) for x in (a, b, c, d))
    m = a.shape[0]
    v = c.copy()
    u = b * c
    n = int(round(dur_s * 1000.0 / DT_MS))
    skip = n // 2
    last = np.full(m, -1)
    cnt = np.zeros(m)
    s1 = np.zeros(m)
    s2 = np.zeros(m)
    for t in range(n):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = a * (b * v - u)
        v = v + DT_MS * dv
        u = u + DT_MS * du
        spk = v >= V_PEAK
        v = np.where(spk, c, v)
        u = np.where(spk, u + d, u)
        if t >= skip and spk.any():
            has = spk & (last >= 0)
            isi = (t - last[has]) * DT_MS
            s1[has] += isi
            s2[has] += isi * isi
            last = np.where(spk, t, last)
            cnt += spk
    k = np.maximum(cnt - 1, 1)
    mean = s1 / k
    sd = np.sqrt(np.maximum(s2 / k - mean * mean, 0.0))
    cv = np.where(cnt >= 4, sd / np.maximum(mean, 1e-9), 0.0)
    return cnt / (dur_s / 2.0), cv


def rheobase(a, b, c, d, lo=0.0, hi=40.0, tol=0.005):
    """Vectorized bisection: smallest constant I giving >=1 spike in the last 1 s of 2 s.

    Same criterion as symmetry.rheobase (rate > 0.5 Hz over the analysed second),
    resolved to tol instead of a 1-unit grid. Returns (M,); nan if silent at hi.
    """
    a, b, c, d = (np.asarray(x, dtype=np.float64) for x in (a, b, c, d))
    lo = np.full(a.shape, lo)
    hi = np.full(a.shape, hi)
    r_hi, _ = deterministic_rate_cv(a, b, c, d, hi, dur_s=2.0)
    reachable = r_hi > 0.5
    while (hi - lo).max() > tol:
        mid = 0.5 * (lo + hi)
        r, _ = deterministic_rate_cv(a, b, c, d, mid, dur_s=2.0)
        fire = r > 0.5
        hi = np.where(fire, mid, hi)
        lo = np.where(fire, lo, mid)
    return np.where(reachable, hi, np.nan)  # nan: no spike at I=hi

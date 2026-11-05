"""Neuron-resolved mean-field of a calibrated candidate (spec: docs/designs/SCI-EI-MEANFIELD_spec.md).

Usage: python scripts/ei_meanfield.py run
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402
import hysteresis_probe as HP  # noqa: E402
import vip_drive_probe as P  # noqa: E402

SPEC = REPO / "docs/designs/SCI-EI-MEANFIELD_spec.md"
OUT = REPO / "docs/receipts/ei_meanfield"
CAND, SEED = "AGS_S0_06", 0
DI = (-0.10, -0.05, 0.0, 0.05, 0.10)
T0_MS = 500.0
CLASSES = ("E", "PV", "SST", "VIP")
CONDS = {"C_a": (1.0, 1.0), "C_b": (2.25, 1.0), "C_c": (2.25, 0.5)}


def sim_rates(model, cand) -> np.ndarray:
    spikes, _ = P.simulate(model, cand, SEED)
    S = np.asarray(spikes, dtype=np.float64)
    k = round(T0_MS / A.DT_MS)
    return S[k:].sum(axis=0) / ((A.DURATION_MS - T0_MS) / 1000.0)


def with_drive_offset(model, di: float):
    import jax.numpy as jnp
    em = model.params["emitter"]
    d = np.asarray(em.drive, dtype=np.float64) + di
    p = dict(model.params)
    p["emitter"] = replace(em, drive=jnp.asarray(d, dtype=jnp.float32))
    return replace(model, params=p)


def uncoupled(model):
    import jax.numpy as jnp
    el = model.params["edge_list"]
    p = dict(model.params)
    p["edge_list"] = replace(el, weight=jnp.zeros_like(el.weight))
    return replace(model, params=p)


def coupling_matrix(model) -> np.ndarray:
    el = model.params["edge_list"]
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    w, tau = np.asarray(el.weight, dtype=np.float64), np.asarray(el.tau_ms, dtype=np.float64)
    n = len(model.neuron_table())
    W = np.zeros((n, n))
    np.add.at(W, (post, pre), w * tau / 1000.0)
    return W


def phi_factory(table: np.ndarray):
    x = np.asarray(DI)

    def phi(mu):
        out = np.empty_like(mu)
        for i, m in enumerate(mu):
            y = table[:, i]
            if m <= x[0]:
                out[i] = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (m - x[0])
            elif m >= x[-1]:
                out[i] = y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (m - x[-1])
            else:
                out[i] = np.interp(m, x, y)
        return np.maximum(out, 0.0)

    def slope(mu):
        out = np.empty_like(mu)
        for i, m in enumerate(mu):
            j = int(np.clip(np.searchsorted(x, m) - 1, 0, len(x) - 2))
            out[i] = (table[j + 1, i] - table[j, i]) / (x[j + 1] - x[j])
        return out
    return phi, slope


def fixed_point(phi, W, r0):
    r = r0.copy()
    for it in range(2000):
        rn = 0.5 * r + 0.5 * phi(W @ r)
        if np.max(np.abs(rn - r)) < 1e-3:
            return rn, it + 1, True
        r = rn
    return r, 2000, False


def simulated(cond: str) -> dict:
    if cond == "C_a":
        return json.loads((REPO / "docs/receipts/vip_drive_probe/AGS_S0_06_seed0.json").read_text())["rows"][0]["class_mean_Hz"]
    if cond == "C_b":
        return json.loads((REPO / "docs/receipts/agsdr_stage0_a4/AGS_S0_06.json").read_text())["class_mean_Hz"]
    rows = json.loads((REPO / "docs/receipts/hysteresis_probe/AGS_S0_06_seed0.json").read_text())["axes"]["G"]["up"]
    return next(r["class_rate_Hz"] for r in rows if r["value"] == 0.5)


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = P.load_candidate(CAND)
    base = A.build_candidate_model(cand, SEED)
    cells = np.array([r["cell_type"] for r in base.neuron_table()])
    W = coupling_matrix(base)
    res = {"spec": "docs/designs/SCI-EI-MEANFIELD_spec.md", "spec_sha256": A.sha256_of(SPEC),
           "script_sha256": A.sha256_of(Path(__file__).resolve()), "candidate": CAND, "seed": SEED, "conditions": {}}
    cls = lambda r: {c: float(r[cells == c].mean()) for c in CLASSES}  # noqa: E731
    for name, (f, g) in CONDS.items():
        m, *_ = HP.scaled_model(base, f, g)
        m0 = uncoupled(m)
        table = np.stack([sim_rates(with_drive_offset(m0, di), cand) for di in DI])
        phi, slope = phi_factory(table)
        starts = {"zero": np.zeros(len(cells)), "phi0": phi(np.zeros(len(cells))), "high": 3 * phi(np.zeros(len(cells)))}
        fps = {k: fixed_point(phi, W, v) for k, v in starts.items()}
        r = fps["phi0"][0]
        mu = W @ r
        J = slope(mu)[:, None] * W
        rho = float(np.max(np.abs(np.linalg.eigvals(J))))
        sim = simulated(name)
        mf = cls(r)
        res["conditions"][name] = {
            "f": f, "g": g, "mf_class_Hz": mf, "sim_class_Hz": sim, "open_loop_phi0_class_Hz": cls(starts["phi0"]),
            "fixed_points": {k: {"class_Hz": cls(v[0]), "iterations": v[1], "converged": v[2]} for k, v in fps.items()},
            "mu_range": [float(mu.min()), float(mu.max())], "mu_within_grid": bool(mu.min() >= DI[0] and mu.max() <= DI[-1]),
            "spectral_radius_loop_gain": rho,
            "K1_within_tol": {c: bool(abs(mf[c] - sim[c]) <= max(2.0, 0.25 * sim[c])) for c in CLASSES},
            "K4_unique": bool(max(abs(cls(v[0])[c] - mf[c]) for v in fps.values() for c in CLASSES) < 0.01
                              and all(v[2] for v in fps.values()))}
        print(name, {c: (round(mf[c], 2), round(sim[c], 2)) for c in CLASSES}, "rho", round(rho, 4), flush=True)
    c = res["conditions"]
    k1 = all(all(c[n]["K1_within_tol"].values()) for n in ("C_a", "C_b"))
    k2 = c["C_c"]["mf_class_Hz"]["PV"] < 1.0
    k3 = c["C_a"]["mf_class_Hz"]["VIP"] < 3.0 and 8.0 <= c["C_b"]["mf_class_Hz"]["VIP"] <= 15.0
    k4 = all(c[n]["K4_unique"] for n in CONDS)
    res.update({"K1": k1, "K2": k2, "K3": k3, "K4": k4})
    failed = [k for k, v in (("K1", k1), ("K2", k2), ("K3", k3), ("K4", k4)) if not v]
    res["verdict"] = "MEANFIELD_REPRODUCES" if not failed else "MEANFIELD_FAILS_" + "_".join(failed)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("K1", "K2", "K3", "K4", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

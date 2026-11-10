"""E2 stack run (docs/laminar_motif/PLAN_LANSA_E2.md, owner rulings 2026-11-09).

Usage:
  python scripts/run_e2_stack.py SEED T_S TAG [LAYOUT]

LAYOUT is a caminos_stack.layout name: e2_3unit (default), unitNNNN, e2_21bin, with modifiers
(caminos_stack.variant), e.g. e2_3unit_n4000_dens_k3_dsz2_sst30.

Writes results/laminar_e2/<TAG>.npz (untracked: block counts, LFP proxy, raster sample) and
results/laminar_e2/<TAG>.json (sizes, edges, wall time, versions). JaxFNE 0.5.9 must be importable;
set JAXFNE_SITE to a side-install directory to put it first on sys.path. Exit 0 when both files are written.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("JAXFNE_SITE"):
    sys.path.insert(0, os.environ["JAXFNE_SITE"])
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from jomission.laminar import caminos_e0 as c0  # noqa: E402
from jomission.laminar import caminos_stack as cs  # noqa: E402

OUT = ROOT / "results" / "laminar_e2"
V_REF_MV = -65.0  # fixed driving force for the current proxy (first pass, declared in the plan)
RASTER_N = {"E": 200, "PV": 50, "SOM": 50}  # cells per block kept for the raster
CHUNK = 10000


def main(seed: int, T_s: float, tag: str, lay: str) -> None:
    import jax
    import jaxfne as jtfne

    OUT.mkdir(parents=True, exist_ok=True)
    sizes, rho, p_inter, depth = cs.layout(lay)
    mods = cs.variant(lay)[1]
    tau_sst_e = cs.sst_taus(lay, sizes)
    names = list(sizes)
    t0 = time.time()
    p_local = cs.local_p(lay, sizes)
    model, rows = cs.construct(seed, sizes, p_inter, p_local, tau_sst_e_ms=tau_sst_e, k_scale=mods.get("k", 1.0),
                               deep_size=mods.get("dsz", 1.0))
    t_construct = time.time() - t0
    # Drive per E and PV cell as E0: 560 private 4 Hz Poisson sources (round(0.07 * 8000)), any unit size
    sim = jtfne.Simulation(duration_ms=T_s * 1000.0, dt_ms=c0.DT_MS, poisson_drive=c0.drive_spec(0.07, seed))
    if str(sim.resolved_runtime.actual_dtype) != "float32":
        raise RuntimeError(f"runtime resolved {sim.resolved_runtime.actual_dtype}, float32 required")
    t1 = time.time()
    sig = model.simulate(sim)
    jax.block_until_ready(sig.spikes)
    t_sim = time.time() - t1
    spikes = np.asarray(sig.spikes, dtype=bool)
    n_steps = spikes.shape[0]

    sl = cs.block_slices(sizes)
    keys = [(m, t) for m in names for t in cs.POPS]
    counts = np.stack([spikes[:, sl[k]].sum(axis=1) for k in keys], axis=1).astype(np.int32)

    W, gcols, gtau, grev = cs.lfp_weights(model, rows, names)
    W = W.astype(np.float32)
    X = np.empty((n_steps, W.shape[1]), dtype=np.float32)
    for a in range(0, n_steps, CHUNK):
        X[a:a + CHUNK] = spikes[a:a + CHUNK].astype(np.float32) @ W
    d = int(round(c0.DELAY_MS / c0.DT_MS))
    decay = np.exp(-c0.DT_MS / gtau)
    g = np.zeros_like(X, dtype=np.float64)
    acc = np.zeros(W.shape[1])
    for t in range(n_steps):
        acc = acc * decay + (X[t - d] if t >= d else 0.0)
        g[t] = acc
    I = np.abs(g * (V_REF_MV - grev))  # pA per unit and receptor
    lfp = np.stack([I[:, [c for c, (m, _) in enumerate(gcols) if m == u]].sum(axis=1) for u in names],
                   axis=1).astype(np.float32)

    rng = np.random.default_rng(0)
    raster = {}
    for k in keys:
        idx = np.arange(sl[k].start, sl[k].stop)
        pick = np.sort(rng.choice(idx, size=min(RASTER_N[k[1]], idx.size), replace=False))
        st, cell = np.nonzero(spikes[:, pick])
        raster[f"{k[0]}_{k[1]}_t"] = st.astype(np.int32)
        raster[f"{k[0]}_{k[1]}_c"] = cell.astype(np.int16)

    np.savez_compressed(OUT / f"{tag}.npz", counts=counts, block_keys=np.array([f"{m}_{t}" for m, t in keys]),
                        lfp_pA=lfp, g_nS=g.astype(np.float32), **raster)
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = {
        "experiment": "E2 stack, first pass (unsealed; owner rulings 2026-11-09)",
        "layout": lay, "units": names, "depth_rel_L4": depth,
        "seed": seed, "T_s": T_s, "dt_ms": c0.DT_MS, "p_inter": p_inter, "k_inter": cs.K_INTER if p_inter is None else None,
        "rho": rho, "sizes": sizes, "edges": cs.edge_table(model), "tau_sst_e_ms": tau_sst_e, "modifiers": mods, "p_local": p_local,
        "g_columns": [list(c) for c in gcols], "g_reversal_mV": grev.tolist(), "g_tau_ms": gtau.tolist(),
        "lfp_proxy": f"|g_ampa (V-0)| + |g_gaba (V+80)| onto E cells per module, V fixed {V_REF_MV} mV, recurrent only",
        "wall_s": {"construct": t_construct, "simulate": t_sim, "total": time.time() - t0},
        "jaxfne_version": jtfne.__version__, "jaxfne_file": str(jtfne.__file__),
        "jax_devices": [str(x) for x in jax.devices()], "git_sha": sha,
    }
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("seed", "T_s", "wall_s", "jaxfne_version")}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (3, 4):
        raise SystemExit(__doc__)
    main(int(a[0]), float(a[1]), a[2], a[3] if len(a) == 4 else "e2_3unit")

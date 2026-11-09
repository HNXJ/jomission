"""Run the Allen V1 replica (docs/laminar_motif/ALLEN_V1_REPLICA_SPEC.md).

Usage:
  python scripts/run_allen_v1_replica.py SEED T_S TAG [WIRING] [SCALE] [SILENCE] [SWAP]

WIRING H (default) or A0. SCALE: fraction of the core counts, in-degree kept (default 1). SILENCE: '-' (default),
PV, SST, VIP, layer:<L>, isolate:<L>. SWAP: 'swap' exchanges L4 and L6 counts and within-layer wiring.
Writes results/allen_v1/<TAG>.npz (untracked: population counts, LFP proxy per E layer, raster sample) and
<TAG>.json (plan, edge check, rates, silencing check, wall time, versions). Set JAXFNE_SITE to put JaxFNE 0.5.9
first on sys.path. Exit 0 when both files are written and the edge and silencing checks pass; 1 otherwise.
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

from jomission.laminar import allen_v1_replica as av  # noqa: E402
from jomission.laminar import caminos_e0 as c0  # noqa: E402

OUT = ROOT / "results" / "allen_v1"
V_REF_MV = -65.0  # fixed driving force of the LFP proxy (as run_e2_stack.py)
RASTER_N = 50
CHUNK = 10000


def main(seed: int, T_s: float, tag: str, wiring: str, scale: float, silence: str | None, swap: bool) -> int:
    import jax
    import jaxfne as jtfne

    OUT.mkdir(parents=True, exist_ok=True)
    pl = av.plan(wiring, scale, swap)
    silenced = av.silence_targets(silence, pl["pops"])
    t0 = time.time()
    model, rows = av.construct(seed, pl, silenced)
    t_construct = time.time() - t0
    check = av.edge_check(model, pl)
    sim = jtfne.Simulation(duration_ms=T_s * 1000.0, dt_ms=c0.DT_MS, poisson_drive=c0.drive_spec(0.07, seed))
    if str(sim.resolved_runtime.actual_dtype) != "float32":
        raise RuntimeError(f"runtime resolved {sim.resolved_runtime.actual_dtype}, float32 required")
    t1 = time.time()
    sig = model.simulate(sim)
    jax.block_until_ready(sig.spikes)
    t_sim = time.time() - t1
    spikes = np.asarray(sig.spikes, dtype=bool)
    n_steps = spikes.shape[0]

    sl = av.pop_slices(pl)
    counts = np.stack([spikes[:, sl[p]].sum(axis=1) for p in pl["pops"]], axis=1).astype(np.int32)
    rates = {p: float(counts[:, i].sum() / (pl["sizes"][p] * n_steps * c0.DT_MS / 1000.0)) for i, p in enumerate(pl["pops"])}
    silent_ok = all(counts[:, pl["pops"].index(p)].sum() == 0 for p in silenced)

    W, layers = av.lfp_weights(model, rows, pl)
    W = W.astype(np.float32)
    X = np.empty((n_steps, W.shape[1]), dtype=np.float32)
    for a in range(0, n_steps, CHUNK):
        X[a:a + CHUNK] = spikes[a:a + CHUNK].astype(np.float32) @ W
    d = int(round(c0.DELAY_MS / c0.DT_MS))
    decay = np.array([np.exp(-c0.DT_MS / (c0.TAU_E_MS if j % 2 == 0 else c0.TAU_I_MS)) for j in range(W.shape[1])])
    g = np.zeros_like(X, dtype=np.float64)
    acc = np.zeros(W.shape[1])
    for t in range(n_steps):
        acc = acc * decay + (X[t - d] if t >= d else 0.0)
        g[t] = acc
    drive = np.array([V_REF_MV - c0.REV_E_MV if j % 2 == 0 else V_REF_MV - c0.REV_I_MV for j in range(W.shape[1])])
    I = np.abs(g * drive)
    lfp = np.stack([I[:, 2 * k] + I[:, 2 * k + 1] for k in range(len(layers))], axis=1).astype(np.float32)

    rng = np.random.default_rng(0)
    raster = {}
    for p in pl["pops"]:
        idx = np.arange(sl[p].start, sl[p].stop)
        pick = np.sort(rng.choice(idx, size=min(RASTER_N, idx.size), replace=False))
        st, cell = np.nonzero(spikes[:, pick])
        raster[f"{av.tag(p)}_t"] = st.astype(np.int32)
        raster[f"{av.tag(p)}_c"] = cell.astype(np.int16)
    np.savez_compressed(OUT / f"{tag}.npz", counts=counts, pops=np.array(pl["pops"]), lfp_pA=lfp,
                        lfp_layers=np.array(layers), g_nS=g.astype(np.float32), **raster)
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ok = not check["failed"] and silent_ok
    rec = {
        "experiment": "Allen V1 replica (ALLEN_V1_REPLICA_SPEC.md; unsealed first pass)",
        "seed": seed, "T_s": T_s, "dt_ms": c0.DT_MS, "wiring": wiring, "scale": scale, "swap": swap,
        "silence": silence, "silenced": silenced, "silenced_zero_spikes": silent_ok,
        "sizes": pl["sizes"], "n_total": int(sum(pl["sizes"].values())), "capped_blocks": pl["capped"],
        "edge_check": {k: check[k] for k in ("examined", "failed", "edges_total", "edges_expected")},
        "edge_blocks": check["blocks"], "rates_hz": rates,
        "lfp_proxy": f"|g_ampa (V-0)| + |g_gaba (V+80)| onto each layer's E cells, V fixed {V_REF_MV} mV, recurrent only",
        "lfp_layers": layers,
        "wall_s": {"construct": t_construct, "simulate": t_sim, "total": time.time() - t0},
        "jaxfne_version": jtfne.__version__, "jaxfne_file": str(jtfne.__file__),
        "jax_devices": [str(x) for x in jax.devices()], "git_sha": sha, "checks_pass": ok,
    }
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("n_total", "edge_check", "silenced_zero_spikes", "wall_s", "checks_pass")}))
    print(json.dumps({p: round(v, 2) for p, v in rates.items()}))
    return 0 if ok else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if not 3 <= len(a) <= 7:
        raise SystemExit(__doc__)
    sil = a[5] if len(a) > 5 and a[5] != "-" else None
    raise SystemExit(main(int(a[0]), float(a[1]), a[2], a[3] if len(a) > 3 else "H",
                          float(a[4]) if len(a) > 4 else 1.0, sil, len(a) > 6 and a[6] == "swap"))

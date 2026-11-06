"""CAMINOS E-PV-SST reference module through the JaxFNE public API (E0, docs/laminar_motif/E0_SEAL.md).

Ported from the JaxFNE etude runner (artifacts/etudes/papers/caminos/caminos_api.py, PREREG_P6 route):
only the p = 0.07 path with the port's 2 ms delay and the private p*NE Poisson drive. Every value
comes from the port specification (Tahvili, Vinck and di Volo; JaxFNE PORT_SPEC Tables 1 and 2).
Units are JaxFNE's (mV, ms, pF, nS, pA). Nothing here tunes a parameter.
"""

from __future__ import annotations

import numpy as np

NE, NPV, NSST = 8000, 750, 750
N = NE + NPV + NSST
DT_MS = 0.1
DELAY_MS = 2.0
TAU_E_MS, TAU_I_MS = 2.0, 8.0
REV_E_MV, REV_I_MV = 0.0, -80.0
Q_E_E, Q_E_INH, Q_INH, Q_EXT = 2.0, 3.0, 5.0, 2.5
EXT_RATE_HZ = 4.0

# Seven realized blocks. PV -> SST and SST -> SST are absent in the reference model.
RULES = (
    ("e_e", "E", "E", Q_E_E, "ampa"),
    ("e_pv", "E", "PV", Q_E_INH, "ampa"),
    ("e_sst", "E", "SOM", Q_E_INH, "ampa"),
    ("pv_e", "PV", "E", Q_INH, "gaba"),
    ("pv_pv", "PV", "PV", Q_INH, "gaba"),
    ("sst_e", "SOM", "E", Q_INH, "gaba"),
    ("sst_pv", "SOM", "PV", Q_INH, "gaba"),
)
# JaxFNE labels the SST class "SOM" (the ADEX_CELL_TYPE_DEFAULTS key); the order below is E, PV, SOM.
POPS = ("E", "PV", "SOM")


def ext_source_count(p: float) -> int:
    """Private Poisson sources per E and PV cell: round(p * NE)."""
    return int(round(float(p) * NE))


def drive_spec(p: float, seed: int) -> dict:
    return {
        "n_sources": ext_source_count(p),
        "rate_hz": EXT_RATE_HZ,
        "weight_nS": Q_EXT,
        "mechanism": "ampa",
        "target_cell_types": ["E", "PV"],
        "delay_ms": DELAY_MS,
        "seed": int(seed),
    }


def build_spec(p: float, seed: int):
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .network(name="caminos_e0", n=N, cell_types={"E": NE / N, "PV": NPV / N, "SOM": NSST / N})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": TAU_E_MS, "reversal_mV": REV_E_MV})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": TAU_I_MS, "reversal_mV": REV_I_MV})
    )
    for name, pre, post, weight, mech in RULES:
        cfg = cfg.connections(
            name=name,
            source={"cell_type": pre},
            target={"cell_type": post},
            probability=float(p),
            weight=weight,
            mechanism=mech,
            delay_ms=DELAY_MS,
            sampling="bernoulli",
        )
    return cfg.connectivity(edge_seed=int(seed))


def build_sim(p: float, seed: int, T_s: float):
    import jaxfne as jtfne

    return jtfne.Simulation(duration_ms=float(T_s) * 1000.0, dt_ms=DT_MS, poisson_drive=drive_spec(p, seed))


def spikes_to_counts(spikes, sizes: dict) -> np.ndarray:
    """Per-step population counts [steps, 3] (int32) in E, PV, SOM order from a boolean [steps, N] raster."""
    arr = np.asarray(spikes, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"spikes must be [steps, N], got shape {arr.shape!r}")
    n_e, n_pv, n_s = int(sizes["E"]), int(sizes["PV"]), int(sizes["SOM"])
    if arr.shape[1] != n_e + n_pv + n_s:
        raise ValueError(f"spikes has {arr.shape[1]} cells but sizes sum to {n_e + n_pv + n_s}")
    out = np.empty((arr.shape[0], 3), dtype=np.int32)
    out[:, 0] = arr[:, :n_e].sum(axis=1)
    out[:, 1] = arr[:, n_e : n_e + n_pv].sum(axis=1)
    out[:, 2] = arr[:, n_e + n_pv :].sum(axis=1)
    return out


def run(p: float, seed: int, T_s: float) -> dict:
    """Construct and simulate through the public API; return counts, sizes and wall time. float32 required."""
    import time

    import jax
    import jaxfne as jtfne

    t0 = time.time()
    model = jtfne.construct(build_spec(p, seed))
    sim = build_sim(p, seed, T_s)
    if str(sim.resolved_runtime.actual_dtype) != "float32":
        raise RuntimeError(f"E0 runs are float32, runtime resolved {sim.resolved_runtime.actual_dtype}")
    signals = model.simulate(sim)
    jax.block_until_ready(signals.spikes)
    labels = tuple(model.params["emitter"].labels)
    sizes = {t: labels.count(t) for t in POPS}
    counts = spikes_to_counts(np.asarray(signals.spikes), sizes)
    return {"counts": counts, "sizes": sizes, "steps": int(sim.n_steps), "wall_s": time.time() - t0,
            "jax_devices": [str(d) for d in jax.devices()], "jaxfne_version": jtfne.__version__}

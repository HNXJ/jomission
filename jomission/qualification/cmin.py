"""C_min single-cortex timescale unit — Gen-2 generic qualification assay (Q_A).

One minimal cortical unit::

    C = {L2/3, L4, L5, L6} x {E, PV, SST, VIP}

built ONLY from the public jaxfne surface
(`build_laminar_column` + `construct` + `simulate` / `run_continuation`).
No new simulator, no omission T1-T7 metrics, no hierarchy-dependent
parameters (identical columns only — composition question Q_B comes later).

Scientific order per review: Q_A (intrinsic decay modes of one C) precedes
Q_B (composition C_0..C_{N-1} -> T_k). s* is MEASURED, never assumed zero:
tonic drive, H reference dynamics, synapses, and Izhikevich resets may make
zero not-a-fixed-point. Persistence claims additionally require residue,
not merely a slow eigenvalue.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig

CMIN_LAYERS = ("L2/3", "L4", "L5", "L6")
CMIN_CELL_TYPES = ("E", "PV", "SST", "VIP")
CMIN_FRACTIONS = {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07}
CMIN_DEFAULT_N = 400
CMIN_DT_MS = 0.1


def build_cmin(n_total: int = CMIN_DEFAULT_N, seed: int = 0, name: str = "C0",
               layers: tuple[str, ...] | None = None):
    """Instantiate one C_min column. Public jaxfne API only.

    Laminar geometry preserves per-neuron layer labels (unlike uniform3d
    scatter). Flat E:I profile repeats CMIN_FRACTIONS per layer.
    Default layers are the tested 4-layer C; pass an explicit 6-layer
    sequence only when six-layer representation is genuinely required —
    never silently changed.
    """
    lay = list(layers) if layers is not None else list(CMIN_LAYERS)
    cfg = jtfne.build_laminar_column(
        name,
        n=int(n_total),
        layers=lay,
        cell_type_fractions=dict(CMIN_FRACTIONS),
        ei_profile="flat",
        geometry="laminar",
        edge_seed=int(seed),
    )
    # Full staging chain (mirrors default_cortical_column_config): runtime +
    # Izhikevich emitter + probe suite + laminar proxy field. Without these
    # construct() fails loudly (no_emitters_declared / no_field_declared).
    cfg = (
        cfg.runtime(seed=int(seed), duration_ms=1000.0, dt_ms=CMIN_DT_MS, dtype="float32")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    return jtfne.construct(cfg)


def coverage(model) -> dict[tuple[str, str], int]:
    """Count neurons per (layer, cell_type). Default C yields 16 populations."""
    cov: dict[tuple[str, str], int] = {}
    for row in model.neuron_table():
        cov[(row["layer"], row["cell_type"])] = cov.get((row["layer"], row["cell_type"]), 0) + 1
    return cov


def run_zero_drive(model, duration_ms: float = 200.0, dt_ms: float = CMIN_DT_MS, seed: int = 0):
    """Baseline run with no paradigm (candidate s* observation window)."""
    runtime = RuntimeConfig(enable_hdp=False)
    sim = Simulation(duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=int(seed), runtime=runtime)
    return jtfne.simulate(model, sim)


def summarize_baseline(sig) -> dict:
    """s* candidacy summary: finiteness, mean/std V, population rate."""
    import numpy as np

    v = np.asarray(sig.V_m)
    sp = np.asarray(sig.spikes)
    dt = float(getattr(sig, "dt_ms", CMIN_DT_MS))
    return {
        "finite": bool(np.isfinite(v).all() and np.isfinite(sp).all()),
        "v_mean": float(v.mean()),
        "v_std": float(v.std()),
        "spike_rate_hz": float(sp.mean() * (1000.0 / dt)),
        "n_steps": int(v.shape[0]),
        "n_neurons": int(v.shape[1]),
    }


def initial_state(model, seed: int = 0):
    """Cold-start ContinuationState on the public surface."""
    dynamic = jtfne.dynamic_state_from_model(model)
    n_neurons = int(model.params["emitter"].n_neurons)
    delay_state = None
    try:
        ds = getattr(model.params["edge_list"], "delay_steps", None)
        if ds is not None:
            import numpy as _np

            arr = _np.asarray(ds)
            if arr.size and int(arr.max()) > 0:
                delay_state = jnp.zeros((int(arr.max()) + 1, n_neurons), dtype=model.params["emitter"].v0.dtype)
    except Exception:
        delay_state = None
    return jtfne.ContinuationState(
        dynamic=dynamic,
        prng_key=jax.random.PRNGKey(int(seed)),
        step_index=0,
        delay_state=delay_state,
    )


def run_impulse_decay(model, target_ids, dv: float = 15.0, post_ms: float = 500.0,
                      dt_ms: float = CMIN_DT_MS, seed: int = 0):
    """delta-s(0) -> delta-s(t): perturb V of targets by dv, zero drive after.

    Returns dict with V trace [T, N] and per-step population-mean |dV| vs the
    unperturbed continuation (residue quantification comes next).
    """
    import numpy as np

    n_neurons = int(model.params["emitter"].n_neurons)
    n_steps = int(round(float(post_ms) / float(dt_ms)))
    step_fn, _ = jtfne.compile_step_fn(model, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    s_pert = initial_state(model, seed)
    s_ref = initial_state(model, seed)
    idx = np.asarray(sorted(set(int(i) for i in target_ids)), dtype=np.int64)
    v0 = s_pert.dynamic.v
    dv_arr = jnp.zeros_like(v0).at[idx].set(float(dv))
    s_pert = s_pert._replace(dynamic=s_pert.dynamic._replace(v=v0 + dv_arr))
    drive = jnp.zeros((n_steps, n_neurons), dtype=v0.dtype)
    s_pert, out_p = jtfne.run_continuation(step_fn, s_pert, drive)
    _, out_r = jtfne.run_continuation(step_fn, s_ref, drive)
    vp = np.asarray(out_p[0])
    vr = np.asarray(out_r[0])
    return {
        "V_pert": vp,
        "V_ref": vr,
        "mean_abs_dV": np.abs(vp - vr).mean(axis=1),
        "dt_ms": float(dt_ms),
        "dv": float(dv),
    }


# --------------------------------------------------------------------------
# Operating-regime repairs (Gen-2 W2 lineage; each isolated + provenanced)
# --------------------------------------------------------------------------
REPAIR1_TONIC = {
    "provenance": "REPAIR1 tonic rebalance: PV x1.3 lineage GEN2_C016 (derated "
                  "from x1.7: 46Hz overshoot); SST x0.35 MODEL_ASSUMPTION "
                  "de-escalation; VIP x10 MODEL_ASSUMPTION (resonator b=-0.1 "
                  "rheobase ~25, single-cell verified; Poisson alternative "
                  "rejected: continuation incompatible with poisson_drive, "
                  "_model_simulate.py:748); E unchanged",
    "scale": {"E": 1.0, "PV": 1.3, "SST": 0.35, "VIP": 10.0},
}
REPAIR2_JITTER = {
    "provenance": "REPAIR2 b-jitter: sigma=0.10 clip [-0.20, 0.35], "
                  "lineage jomission builder.py:39 heterogeneity",
    "b_sigma": 0.10,
    "b_lo": -0.20,
    "b_hi": 0.35,
}


def repair_tonic(model, scale_map=None):
    """REPAIR1: per-class tonic rescale (single isolated mutation)."""
    import numpy as _np

    sm = REPAIR1_TONIC["scale"] if scale_map is None else scale_map
    tbl = model.neuron_table()
    e = model.params["emitter"]
    base = _np.asarray(e.drive, dtype=float)
    out = base.copy()
    for i, row in enumerate(tbl):
        out[i] = base[i] * float(sm[str(row["cell_type"])])
    return model.with_emitter_parameters(drive_per_neuron=jnp.asarray(out, dtype=e.drive.dtype))


def repair_jitter(model, seed=0, b_sigma=None):
    """REPAIR2: per-neuron b jitter (breaks exact synchrony)."""
    import numpy as _np

    sg = REPAIR2_JITTER["b_sigma"] if b_sigma is None else b_sigma
    e = model.params["emitter"]
    rng = _np.random.default_rng(int(seed))
    b = _np.asarray(e.b, dtype=float)
    jb = _np.clip(b * (1.0 + sg * rng.standard_normal(b.shape)),
                  REPAIR2_JITTER["b_lo"], REPAIR2_JITTER["b_hi"])
    return model.with_emitter_parameters(b_per_neuron=jnp.asarray(jb, dtype=e.b.dtype))


def per_class_rates(model, duration_ms=2000.0, dt_ms=CMIN_DT_MS, seed=1,
                    skip_ms=1000.0):
    """Post-transient per-class rate summary (mean/std/min/max Hz)."""
    import numpy as _np

    sig = run_zero_drive(model, duration_ms, dt_ms, seed)
    sp = _np.asarray(sig.spikes)
    n_skip = int(round(skip_ms / dt_ms))
    hz = sp[n_skip:].mean(axis=0) * (1000.0 / float(dt_ms))
    tbl = model.neuron_table()
    cls = _np.array([r["cell_type"] for r in tbl])
    return {c: {"mean": float(hz[cls == c].mean()), "std": float(hz[cls == c].std()),
                "min": float(hz[cls == c].min()), "max": float(hz[cls == c].max()),
                "n": int((cls == c).sum())}
            for c in ("E", "PV", "SST", "VIP")}

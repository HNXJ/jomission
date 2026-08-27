"""Compiled experiment observables union — must be owned by recorder before expensive compute.

Union of all observables needed for RF validation, Δ_exposure, T1–T7, plasticity-timescale analysis and mechanistic interpretation.
"""

REQUIRED_OBSERVABLES = {
    "visual_field": "visual_field[trial,time,32,32] — 32×32 pixel lattice input per trial slot (A/B/R patterns)",
    "external_drive": "external_drive[trial,time,unit] — post-RF drive per unit (target_indices × amplitude)",
    "RF_operator": "RF_operator[unit,1024] — per V1 unit Gaussian weights, L1-normalized",
    "spikes": "spikes[trial,unit,time/event] — per-unit spike raster",
    "rate": "rate[trial,area,time] — ms area population rates",
    "metadata": "area/layer/class/unit metadata — neuron_metadata per unit",
    "field_proxy": "field_proxy[trial,area_attribution,contact,time] where scientifically valid — area-attributed contributions at declared probe geometry, proxy_readout",
    "H_t": "H(t) — per-neuron hidden state trajectory, dense early-time sampling to resolve tau_Theta ~2.5s effective",
    "Theta_t": "Theta(t) — HDP w trajectory, dense early-time sampling, bounds [w_floor,w_ceiling]",
    "event_ledger": "event ledger — trial/condition/onset/duration per slot, omission zero-drive preserved",
    "continuation_state": "continuation/checkpoint state — (X,H,Θ,D,RNG,cursor) per trial boundary",
    # GEN2_C004 (B3 E/I currents): opt-in for B3 UNRESOLVED policy. When
    # record_edge_current=True, edge_current_trace[t,n_edges] is available via
    # jaxfne.compile_step_fn(edge_current_trace) / Model.last_hdp_diagnostics()
    # seam (emitters.py:2846, _pipeline.py:395). Marked optional so existing
    # callers without currents remain PASS; B3 qualification requires it.
    "I_edge_current": "I_edge_current[t,n_edges] — per-edge synaptic current w*syn_state, partitioned I_e/I_i by presynaptic sign (optional, B3 UNRESOLVED; requires record_edge_current=True)",
}

# Optional observables that do not block recorder ownership but are needed for B3/B4 when requested
OPTIONAL_OBSERVABLES = {"I_edge_current"}

# Recorder ownership check: must be able to produce all before long runs
def assert_recorder_owns(available: set[str]) -> list[str]:
    missing = [k for k in REQUIRED_OBSERVABLES if k not in available and k not in OPTIONAL_OBSERVABLES]
    return missing


def partition_currents_by_motif(edge_currents, edge_list, neuron_table, *, reduce: str = "mean"):
    """Partition per-edge currents into E vs I and aggregate per motif.

    Splits I_e (presynaptic excitatory, sign +1) vs I_i (presynaptic inhibitory,
    sign -1) per edge per step and aggregates per motif
    (area×layer×class → area×layer×class).

    Args:
        edge_currents: array [n_steps, n_edges] — w*syn_state per edge per step
            (jaxfne/emitters.py:2846 edge_current_trace).
        edge_list: EdgeList with .pre/.post (int) and optionally .weight (sign).
        neuron_table: list[dict] from model.static["neuron_metadata"] or
            model.neuron_table(), each with area/layer/cell_type.
        reduce: "mean" | "sum" — aggregation over time and edges within motif.

    Returns:
        dict with:
          - I_e[t] and I_i[t] time series (if reduce=="mean" also per-step)
          - per_motif: {(pre_area,pre_layer,pre_class)→(post_area,post_layer,post_class): stats}
          - Efrac[t] = |I_e|/(|I_e|+|I_i|) per step
          - summary: global mean Efrac, per-area Efrac
    Uses sign from emitter per presynaptic neuron (labels or area/layer/class
    E vs PV/SST/VIP). If edge_list.weight sign is available, it is used as
    fallback. No second simulator.
    """
    import numpy as np

    ec = np.asarray(edge_currents)
    if ec.ndim != 2:
        raise ValueError(f"edge_currents must be [n_steps,n_edges], got {ec.shape}")
    n_steps, n_edges = ec.shape
    pre = np.asarray(edge_list.pre, dtype=int)
    post = np.asarray(edge_list.post, dtype=int)
    if pre.shape[0] != n_edges or post.shape[0] != n_edges:
        raise ValueError(f"edge_list length {pre.shape[0]} != n_edges {n_edges}")

    # Determine excitatory presynaptic mask via neuron_table cell_type
    is_exc_pre = np.zeros(n_edges, dtype=bool)
    for ei, pid in enumerate(pre):
        try:
            ct = str(neuron_table[int(pid)].get("cell_type", "")).upper()
            is_exc_pre[ei] = ct.startswith("E")
        except Exception:
            # Fallback to weight sign if table unavailable
            try:
                w = float(np.asarray(edge_list.weight)[ei])
                is_exc_pre[ei] = w > 0
            except Exception:
                is_exc_pre[ei] = True

    I_e = ec[:, is_exc_pre]  # [n_steps, n_e]
    I_i = ec[:, ~is_exc_pre]  # [n_steps, n_i] (negative weights typically)

    # Per-step aggregates
    # Use absolute currents for Efrac balance (magnitudes), but preserve sign for means
    abs_e = np.abs(I_e).sum(axis=1) if I_e.size else np.zeros(n_steps)
    abs_i = np.abs(I_i).sum(axis=1) if I_i.size else np.zeros(n_steps)
    denom = abs_e + abs_i
    Efrac = np.where(denom > 0, abs_e / denom, 0.5)

    out: dict = {
        "I_e_shape": tuple(I_e.shape),
        "I_i_shape": tuple(I_i.shape),
        "I_e_mean_per_step": I_e.mean(axis=1).tolist() if I_e.size else [0.0] * n_steps,
        "I_i_mean_per_step": I_i.mean(axis=1).tolist() if I_i.size else [0.0] * n_steps,
        "Efrac_per_step": Efrac.tolist(),
        "Efrac_mean": float(Efrac.mean()),
        "Efrac_min": float(Efrac.min()) if n_steps else 0.5,
        "Efrac_max": float(Efrac.max()) if n_steps else 0.5,
        "n_e_edges": int(is_exc_pre.sum()),
        "n_i_edges": int((~is_exc_pre).sum()),
        "n_steps": int(n_steps),
        "n_edges": int(n_edges),
    }

    # Per-motif aggregation: key = (pre_motif -> post_motif)
    try:
        per_motif: dict[str, dict] = {}
        for ei in range(n_edges):
            pre_id = int(pre[ei])
            post_id = int(post[ei])
            pre_meta = neuron_table[pre_id] if pre_id < len(neuron_table) else {}
            post_meta = neuron_table[post_id] if post_id < len(neuron_table) else {}
            pre_key = f"{pre_meta.get('area','?')}×{pre_meta.get('layer','?')}×{pre_meta.get('cell_type','?')}"
            post_key = f"{post_meta.get('area','?')}×{post_meta.get('layer','?')}×{post_meta.get('cell_type','?')}"
            k = f"{pre_key}->{post_key}"
            if k not in per_motif:
                per_motif[k] = {"count": 0, "sum_mean": 0.0, "is_exc": bool(is_exc_pre[ei])}
            # mean current for this edge over time
            per_motif[k]["count"] += 1
            per_motif[k]["sum_mean"] += float(ec[:, ei].mean())
        for k, v in per_motif.items():
            v["mean_current"] = v["sum_mean"] / max(v["count"], 1)
            del v["sum_mean"]
        out["per_motif"] = per_motif
        # Per-area Efrac (post area)
        area_Efrac: dict[str, float] = {}
        for area in ("V1", "V4", "FEF", "PFC"):
            # edges targeting this area
            mask_post_area = np.array([
                str(neuron_table[int(post[ei])].get("area","")) == area
                for ei in range(n_edges)
                if int(post[ei]) < len(neuron_table)
            ])
            # Need to map to edge indices targeting area
            idx_area = [ei for ei in range(n_edges) if int(post[ei]) < len(neuron_table) and str(neuron_table[int(post[ei])].get("area","")) == area]
            if not idx_area:
                continue
            is_exc_area = np.array([is_exc_pre[ei] for ei in idx_area])
            ec_area = ec[:, idx_area]
            ae = np.abs(ec_area[:, is_exc_area].sum(axis=1)) if is_exc_area.any() else np.zeros(n_steps)
            ai = np.abs(ec_area[:, ~is_exc_area].sum(axis=1)) if (~is_exc_area).any() else np.zeros(n_steps)
            d = ae + ai
            ef = np.where(d > 0, ae / d, 0.5)
            area_Efrac[area] = float(ef.mean())
        out["Efrac_by_post_area"] = area_Efrac
    except Exception:
        pass

    out["claim_level"] = "realized_currents_by_presynaptic_sign"
    out["seam"] = "jaxfne/emitters.py:2846, jaxfne/_pipeline.py:395"
    return out

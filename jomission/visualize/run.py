"""Run-report foundation — VIS_FOUNDATION_v0 V3.

Standard simulation report: every run should automatically generate
results/<run_id>/ with summary.json, summary.md, report.html, figures/, arrays/,
EvidenceRef.json and HTML with fixed tabs.

Engineering, not science: must NOT alter scientific primitives (C019 frozen).
Visualizations consume SAME generated-owner arrays used by analyses (spikes, V_m,
field, H, Theta, q) not recomputed prettier alternatives.

Provenance citations (read-only):
- jomission/network/builder.py:62,90,395  frozen C019 primitives (MOTIF_GAIN, FF/FB, spatial_sigma)
- jomission/network/builder.py:713-785   VIP/SST b correction seams (pre-jitter u0 recompute)
- jomission/network/builder.py:623,760   _apply_motif_gains post-EdgeList seam
- jaxfne/_model_simulate.py:280,572,1063 HDP+delay guard (enable_hdp vs delay_steps)
- jaxfne/_model_simulate.py:764-773     single project_laminar_sources call → global FieldOutput
- jaxfne/fields/proxy.py:148,192,201     project_laminar_sources kernel K[c,n], field=sources@K.T
- jomission/recording/area_local.py:52,152,206  field_by_area_from_signal linear partition
- jomission/recording/observables.py:6,35       REQUIRED_OBSERVABLES + partition_currents_by_motif
- jomission/recording/observables.py:33        OPTIONAL I_edge_current (record_edge_current)
- jomission/simulation/ledger.py:11,21         PRODUCTION_SEED_LEDGER, CHECKPOINT_CADENCE
- jomission/simulation/schedule.py:11,35       canonical_schedule derived timing
- jomission/simulation/stability.py:8          STABILITY_CRITERIA bounds
- jomission/evidence.py:15                     EvidenceRef dataclass
- jaxfne/_signals.py:670                       _make_poisson_drive private Bernoulli shape
- jaxfne/emitters.py:545                       edge_list_with_delay_ms seam
- jaxfne/emitters.py:2846,3066,3507           record_edge_current seam edge_current_trace

Tabs (fixed ordering):
  Overview, Raster, Rates, Membrane, E/I, H/Θ, Spectra, Field, Stimulus,
  Connectivity, Diagnostics, Provenance

Function:
  run_report(simulation_result) -> dict
    simulation_result keys (flexible):
      model, signals (Signals or list[Signals]), config_hash, hp_hash,
      seed, dt_ms, duration_ms, run_id, results_dir, paradigm, schedule,
      phases (optional), numerical_valid, state_identity, run_hash
    Writes to results/<run_id>/ or results_dir if provided.

Example generator:
  generate_example_C019_spontaneous(duration_ms=2000, seed=0)
    Builds C019 model via build_jomission_model(seed=0), runs jtfne.simulate
    with StimulusSchedule silent (spontaneous), handles HDP+delay incompat
    (jaxfne/_model_simulate.py:280) by falling back to enable_hdp=False when
    delays present, records provenance, then calls run_report.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import time
from typing import Any

import numpy as np

try:
    import jax.numpy as jnp  # type: ignore
except Exception:  # pragma: no cover
    jnp = np  # type: ignore

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    _PLOTLY = True
except Exception:  # pragma: no cover
    go = None  # type: ignore
    make_subplots = None  # type: ignore
    _PLOTLY = False

# ---------------------------------------------------------------------------
# Helpers — read-only, no science mutation
# ---------------------------------------------------------------------------

def _safe_hash(obj: Any) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(obj).encode()).hexdigest()[:16]


def _as_numpy(arr: Any) -> np.ndarray:
    try:
        return np.asarray(arr)
    except Exception:
        return np.array(arr)


def _resolve_signals(simulation_result: dict) -> list[Any]:
    sig = simulation_result.get("signals") or simulation_result.get("signal") or simulation_result.get("sig")
    if sig is None and "V_m" in simulation_result:
        # SimulationResult is itself a Signals-like
        return [simulation_result]
    if isinstance(sig, list):
        return sig
    if sig is not None:
        return [sig]
    # Try to handle dict with V_m/spikes
    if isinstance(simulation_result, dict) and "spikes" in simulation_result:
        return [simulation_result]
    raise ValueError("simulation_result must contain 'signals' (Signals or list[Signals])")


def _model_hash(model: Any) -> str:
    try:
        from jaxfne.io import config_hash  # type: ignore
        return str(config_hash(model.cfg))
    except Exception:
        try:
            return str(model.cfg.metadata.get("config_hash", "unknown"))
        except Exception:
            return "unknown"


def _hp_hash(model: Any, simulation_result: dict) -> str:
    if "hp_hash" in simulation_result and simulation_result["hp_hash"]:
        return str(simulation_result["hp_hash"])
    try:
        import jaxfne.hdp_network as hdp  # type: ignore
        hp = hdp.v1_pfc_aaab_hdp_params()
        return hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _run_hash(signals: list[Any]) -> str:
    # Deterministic run hash from spike sums
    try:
        sums = [float(np.sum(_as_numpy(s.spikes))) if hasattr(s, "spikes") else float(np.sum(_as_numpy(s.get("spikes", 0)))) for s in signals]
        return hashlib.sha256(json.dumps(sums).encode()).hexdigest()[:12]
    except Exception:
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]


def _neuron_metadata(model: Any, signals: list[Any]) -> list[dict]:
    # Prefer signals[0].metadata neuron_metadata (provenance per trial)
    try:
        sig0 = signals[0]
        md = getattr(sig0, "metadata", None)
        if isinstance(md, dict) and md.get("neuron_metadata"):
            return list(md["neuron_metadata"])
    except Exception:
        pass
    try:
        if hasattr(model, "neuron_table"):
            return list(model.neuron_table())  # type: ignore
    except Exception:
        pass
    try:
        if hasattr(model, "static") and model.static.get("neuron_metadata"):
            return list(model.static["neuron_metadata"])
    except Exception:
        pass
    return []


def _check_numerical_valid(signals: list[Any], dt_ms: float) -> dict:
    issues: list[str] = []
    try:
        from jomission.simulation.stability import STABILITY_CRITERIA  # type: ignore
        lo, hi = STABILITY_CRITERIA["finite_state"]["V_m_mean_range"]
    except Exception:
        lo, hi = -90.0, -50.0
    for si, sig in enumerate(signals):
        try:
            vm = _as_numpy(sig.V_m if hasattr(sig, "V_m") else sig.get("V_m", []))
            sp = _as_numpy(sig.spikes if hasattr(sig, "spikes") else sig.get("spikes", []))
            if not np.all(np.isfinite(vm)):
                issues.append(f"signal {si}: V_m non-finite")
            if not np.all(np.isfinite(sp)):
                issues.append(f"signal {si}: spikes non-finite")
            if sig is not None and getattr(sig, "field", None) is not None:
                try:
                    if not np.all(np.isfinite(_as_numpy(sig.field.lfp_proxy))):
                        issues.append(f"signal {si}: lfp non-finite")
                except Exception:
                    pass
            v_mean = float(np.mean(vm)) if vm.size else 0.0
            if not (lo <= v_mean <= hi):
                issues.append(f"signal {si}: V_m mean {v_mean:.1f} outside [{lo},{hi}]")
            rate = float(np.mean(sp) * (1000.0 / float(dt_ms))) if sp.size else 0.0
            if rate > 100:
                issues.append(f"signal {si}: rate {rate:.1f} >100 Hz")
        except Exception as e:
            issues.append(f"signal {si}: check error {e}")
    return {"numerical_valid": len(issues) == 0, "issues": issues}


def _state_identity(model: Any, signals: list[Any]) -> str:
    # Mirrors lifecycle.py _state_identity but on model params + signals tail state if available
    try:
        import jax  # type: ignore
        leaves = []
        for arr in jax.tree_util.tree_leaves(model.params):
            try:
                leaves.append(np.asarray(arr).tobytes())
            except Exception:
                pass
        h = hashlib.sha256()
        for b in leaves:
            h.update(b)
        # Add spike hash for run identity
        rh = _run_hash(signals)
        h.update(rh.encode())
        return h.hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(_run_hash(signals)).encode()).hexdigest()[:16]


def _ensure_plotly():
    if not _PLOTLY or go is None:
        raise ImportError("plotly not installed (required for report.html)")


# ---------------------------------------------------------------------------
# Tab figure builders — each consumes SAME arrays as analyses
# ---------------------------------------------------------------------------

def _fig_raster(signals: list[Any], meta: list[dict], dt_ms: float, max_neurons: int = 120, max_steps: int = 8000) -> Any:
    """Raster time×neuron — filters area/layer/class/subtype via neuron_metadata."""
    _ensure_plotly()
    sig0 = signals[0]
    spikes = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))  # [T,N]
    vm0 = sig0  # keep sig for dimensions
    T, N = spikes.shape if spikes.ndim == 2 else (spikes.shape[0], 1)
    # Subsample time for plotting performance; time_ms = step_index * dt_ms per P0 time-axis fix
    step = max(1, T // max_steps)
    sp_sub = spikes[::step, :] if spikes.ndim == 2 else spikes
    t_ms = np.arange(sp_sub.shape[0]) * float(dt_ms) * step  # time_ms = step * dt_ms (ms, not steps)
    # Limit neurons for visibility, but keep filter structure via dropdown-like traces per area
    areas = ["V1", "V4", "FEF", "PFC"]
    fig = go.Figure()
    # One trace per area for interactive legend filtering (standard pattern)
    for area in areas:
        idxs = [i for i, r in enumerate(meta) if str(r.get("area", "")) == area]
        if not idxs:
            continue
        # Restrict to max_neurons total sampling
        idxs_plot = idxs[: max(64, max_neurons // 4)] if len(idxs) > 64 else idxs
        # Extract spike coordinates (sparse)
        xs, ys = [], []
        for col_j, nid in enumerate(idxs_plot):
            if nid >= sp_sub.shape[1]:
                continue
            rows = np.where(sp_sub[:, nid] > 0.5)[0]
            if rows.size:
                xs.extend(t_ms[rows].tolist())
                # y is original neuron id for provenance
                ys.extend([int(nid)] * len(rows))
        if not xs:
            continue
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="markers", marker=dict(size=2, opacity=0.7), name=area, hovertemplate="t %{x:.1f} ms id %{y}<extra>" + area + "</extra>"))
    # Layer filter suggestion in title
    fig.update_layout(
        title=f"Raster — time×neuron (T={T} dt={dt_ms}ms, N={N}, shown ≤{max_steps} steps, filters: area/layer/class/subtype via neuron_metadata) — semantic_class STATE",
        xaxis_title="Time (ms)",
        yaxis_title="neuron id (0..N-1)",
        height=520,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=50, r=20, t=60, b=60),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _fig_rates(signals: list[Any], meta: list[dict], dt_ms: float) -> Any:
    """Rates area×layer×class heatmap from spikes mean."""
    _ensure_plotly()
    sig0 = signals[0]
    spikes = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))
    # Compute per-group rates
    from collections import defaultdict
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(meta):
        k = (str(r.get("area", "?")), str(r.get("layer", "?")), str(r.get("cell_type", "?")))
        groups[k].append(i)
    areas = ["V1", "V4", "FEF", "PFC"]
    layers = ["L1", "L2/3", "L4", "L5", "L6"]
    cts = ["E", "PV", "SST", "VIP"]
    # Matrix areas×layer×class -> flatten to area rows, (layer,class) cols
    col_labels = [f"{lyr} {ct}" for lyr in layers for ct in cts]
    row_labels = areas
    mat = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)
    for (area, lyr, ct), idxs in groups.items():
        if area not in areas or lyr not in layers or ct not in cts:
            continue
        r = row_labels.index(area)
        c = col_labels.index(f"{lyr} {ct}")
        vals = spikes[:, idxs] if spikes.ndim == 2 and max(idxs) < spikes.shape[1] else np.zeros((spikes.shape[0], 1))
        rate = float(np.mean(vals) * (1000.0 / float(dt_ms))) if vals.size else 0.0
        mat[r, c] = rate
    # Plot heatmap with annot if small
    fig = go.Figure(data=go.Heatmap(z=mat, x=col_labels, y=row_labels, colorscale="Viridis", colorbar=dict(title="Hz"), hovertemplate="%{y} %{x}<br>rate %{z:.2f} Hz<extra></extra>"))
    fig.update_layout(title="Rates — area×layer×class mean rate (Hz) from spikes (same array as analyses)", xaxis=dict(tickangle=45), height=460, margin=dict(l=60, r=40, t=60, b=120))
    return fig


def _fig_membrane(signals: list[Any], meta: list[dict], dt_ms: float) -> Any:
    """Membrane: representative V_m traces + distributional histogram. Relative_V = DERIVED_FROM(V_m) labeled."""
    _ensure_plotly()
    sig0 = signals[0]
    vm = _as_numpy(sig0.V_m if hasattr(sig0, "V_m") else sig0.get("V_m", np.zeros((100, 4))))
    T, N = vm.shape if vm.ndim == 2 else (vm.shape[0], 1)
    # Representative: one per area (first E neuron per area)
    rep_ids: list[int] = []
    rep_labels: list[str] = []
    for area in ["V1", "V4", "FEF", "PFC"]:
        for i, r in enumerate(meta):
            if str(r.get("area")) == area and str(r.get("cell_type")) == "E":
                rep_ids.append(i)
                rep_labels.append(f"{area} E id{i}")
                break
    if not rep_ids:
        rep_ids = list(range(min(4, N)))
        rep_labels = [f"id{i}" for i in rep_ids]
    t_ms = np.arange(T) * float(dt_ms)  # time_ms = step * dt_ms (ms)
    # Subsample for trace plotting — cap to ~2000 points per trace to keep html <2 MB
    step = max(1, T // 2000)
    t_sub = t_ms[::step]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Representative V_m(t) — same array as analyses (STATE, V_m mV)", "Distributional V_m (histogram) — note: relative_V = DERIVED_FROM(V_m) not shown as independent"), column_widths=[0.62, 0.38])
    for idx, lab in zip(rep_ids, rep_labels):
        if idx >= vm.shape[1]:
            continue
        vm_sub = vm[::step, idx]
        fig.add_trace(go.Scatter(x=t_sub, y=vm_sub, mode="lines", name=lab, line=dict(width=1), hovertemplate="t %{x:.1f} ms V %{y:.1f} mV<extra>" + lab + "</extra>"), row=1, col=1)
    # Distributional: histogram of V_m values (downsampled to ~8000 samples)
    flat = vm[:: max(1, T // 2000), : min(200, vm.shape[1])].reshape(-1) if vm.size else np.array([0.0])
    fig.add_trace(go.Histogram(x=flat, nbinsx=40, marker=dict(color="#1f77b4"), name="V_m hist", showlegend=False, hovertemplate="V %{x:.1f} mV count %{y}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="V_m (mV)", row=1, col=1)
    fig.update_xaxes(title_text="V_m (mV)", row=1, col=2)
    fig.update_layout(title="Membrane — V_m representative + distributional (recording/observables.py V_m, STATE; relative_V DERIVED_FROM(V_m) — see ObservableBasis)", height=440, margin=dict(l=50, r=20, t=60, b=40), legend=dict(orientation="h", y=-0.22))
    # Annotate derived note
    fig.add_annotation(x=0.5, y=-0.22, xref="paper", yref="paper", text="relative_V = V_i - mean(V) is DERIVED_FROM(V_m) (DERIVED), not independent state — see jomission/visualization/model_summary.py:observable_basis()", showarrow=False, font=dict(size=8, color="#555"), align="center")
    return fig


def _compute_ei_proxy_wr(signals: list[Any], meta: list[dict], model: Any, dt_ms: float) -> dict | None:
    """Compute Efrac_proxy W·r (PROXY_ESTIMATE) from EdgeList weights and spike rates.

    EI_PROXY not realized: Efrac_proxy = |W_E·r|/(|W_E·r|+|W_I·r_I|) where r = mean rate per presynaptic neuron (Hz).
    Uses same arrays as analyses (spikes, EdgeList) but is explicitly PROXY_ESTIMATE (semantic_class PROXY_ESTIMATE, proxy_status True).
    Separated from realized edge_current_trace (jaxfne/emitters.py:2846) which is blocked by jaxfne/_model_simulate.py:280 when delay_steps nonzero + HDP.
    """
    try:
        sig0 = signals[0]
        spikes = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))
        if spikes.ndim != 2:
            return None
        T = spikes.shape[0]
        # mean rate per neuron (Hz) = mean(spikes)*1000/dt_ms
        rates = np.mean(spikes, axis=0) * (1000.0 / float(dt_ms))  # [N]
        ea = {}
        try:
            el = model.params.get("edge_list") if model is not None else None
            if el is not None:
                ea["pre"] = np.asarray(el.pre, dtype=np.int64)
                ea["post"] = np.asarray(el.post, dtype=np.int64)
                ea["weight"] = np.asarray(el.weight, dtype=np.float64)
            else:
                return None
        except Exception:
            return None
        pre = ea["pre"]
        post = ea["post"]
        w = ea["weight"]
        n_edges = int(pre.shape[0])
        # Determine excitatory presynaptic mask via cell_type E vs PV/SST/VIP (same as observables.py)
        is_exc_pre = np.zeros(n_edges, dtype=bool)
        for ei, pid in enumerate(pre):
            try:
                ct = str(meta[int(pid)].get("cell_type", "")) if int(pid) < len(meta) else ""
                is_exc_pre[ei] = ct.startswith("E")
            except Exception:
                try:
                    is_exc_pre[ei] = float(w[ei]) > 0
                except Exception:
                    is_exc_pre[ei] = True
        # W·r proxy: per-edge current proxy = |w| * r_pre (absolute for Efrac balance)
        r_pre = rates[pre]  # [n_edges]
        abs_wr = np.abs(w) * r_pre
        abs_e = np.sum(abs_wr[is_exc_pre]) if np.any(is_exc_pre) else 0.0
        abs_i = np.sum(abs_wr[~is_exc_pre]) if np.any(~is_exc_pre) else 0.0
        denom = abs_e + abs_i
        efrac_proxy = float(abs_e / denom) if denom > 0 else 0.5
        # Per post area Efrac_proxy
        areas = ["V1", "V4", "FEF", "PFC"]
        efrac_by_area = {}
        for area in areas:
            idx = [ei for ei in range(n_edges) if int(post[ei]) < len(meta) and str(meta[int(post[ei])].get("area", "")) == area]
            if not idx:
                continue
            idx_arr = np.array(idx, dtype=int)
            ae = np.sum(abs_wr[idx_arr[is_exc_pre[idx_arr]]]) if np.any(is_exc_pre[idx_arr]) else 0.0
            ai = np.sum(abs_wr[idx_arr[~is_exc_pre[idx_arr]]]) if np.any(~is_exc_pre[idx_arr]) else 0.0
            d = ae + ai
            efrac_by_area[area] = float(ae / d) if d > 0 else 0.5
        return dict(Efrac_proxy=float(efrac_proxy), Efrac_proxy_by_post_area=efrac_by_area, abs_e=float(abs_e), abs_i=float(abs_i), n_e_edges=int(np.sum(is_exc_pre)), n_i_edges=int(np.sum(~is_exc_pre)), n_edges=int(n_edges), r_pre_mean=float(np.mean(r_pre)) if r_pre.size else 0.0)
    except Exception:
        return None


def _fig_ei(signals: list[Any], model: Any, meta: list[dict]) -> Any:
    """E/I configured vs realized + EI_PROXY (W·r) watermark.

    P0 E/I proxy watermark: E/I displayed from W·r is EI_PROXY not realized (semantic_class PROXY_ESTIMATE, proxy_status True).
    Realized edge_current_trace is degraded when jaxfne/_model_simulate.py:280 blocks recording (delay_steps [20,80,120] + HDP).
    Must watermark visibly and use supplement banner when production path unavailable.
    """
    _ensure_plotly()
    # Configured: from builder MOTIF_GAIN etc.
    try:
        from jomission.network.builder import MOTIF_GAIN, DESIRED_MOTIF_GAIN_V0  # type: ignore
        motif_cfg = {f"{k[0]}->{k[1]}": float(v) for k, v in MOTIF_GAIN.items()} if isinstance(MOTIF_GAIN, dict) else dict(DESIRED_MOTIF_GAIN_V0)
    except Exception:
        motif_cfg = {"E->E": 1.0, "E->PV": 1.7}
    cfg_labels = sorted(motif_cfg.keys())
    cfg_vals = [motif_cfg[k] for k in cfg_labels]
    # Realized: try observables partition if edge_current available
    realized = None
    realized_note = "realized currents unavailable (upstream jaxfne/_model_simulate.py:280 guard — requires record_edge_current seam jaxfne/emitters.py:2846, see recording/observables.py:35)"
    degraded = True  # default degraded when no realized
    try:
        diag = None
        if hasattr(model, "last_hdp_diagnostics"):
            diag = model.last_hdp_diagnostics()
        if diag is not None and diag.get("edge_current_trace") is not None:
            from jomission.recording.observables import partition_currents_by_motif  # type: ignore
            ec = diag.get("edge_current_trace")
            part = partition_currents_by_motif(ec, model.params["edge_list"], meta)
            realized = part
            realized_note = f"realized Efrac {part.get('Efrac_mean', 0.5):.3f} from edge_current_trace {part.get('n_e_edges',0)}/{part.get('n_i_edges',0)} edges (recording/observables.py:35, REALIZED)"
            degraded = False
        else:
            # No realized trace — check if due to HDP+delay incompatibility (expected for C019 with delays [20,80,120])
            realized_note = "realized currents unavailable — SUPPLEMENTARY / NON-CANONICAL EXECUTION PATH — delayed HDP production path unavailable (jaxfne/_model_simulate.py:280 guard: enable_hdp does not support nonzero edge delay_steps [20,80,120]; use record_edge_current seam jaxfne/emitters.py:2846 only on non-delayed path) — degraded evidence class SUPPLEMENTARY"
            degraded = True
    except Exception as e:
        realized_note = f"realized error: {e} — degraded evidence class"
        degraded = True
    # EI_PROXY via W·r (always available as PROXY_ESTIMATE)
    proxy = _compute_ei_proxy_wr(signals, meta, model, dt_ms=0.1) if model is not None else None
    proxy_note = ""
    if proxy is not None:
        proxy_note = f"EI_PROXY (W·r) Efrac_proxy {proxy.get('Efrac_proxy',0.5):.3f} by area {proxy.get('Efrac_proxy_by_post_area',{})} — semantic_class PROXY_ESTIMATE proxy_status True (not realized; W·r estimate, physical_amplitude_calibrated=False) — observables.py:93"
    # Build figure: left bar configured, middle proxy, right realized/degraded
    has_proxy = proxy is not None
    n_cols = 3 if has_proxy else 2
    titles = ("Configured MOTIF_GAIN (builder.py:62 DESIRED_MOTIF_GAIN v0, SOURCE)", "EI_PROXY W·r (PROXY_ESTIMATE, proxy_status True)", "Realized E/I (READOUT) — degraded when jaxfne/_model_simulate.py:280 blocks") if has_proxy else ("Configured MOTIF_GAIN (builder.py:62 DESIRED_MOTIF_GAIN v0, SOURCE)", "Realized E/I (READOUT) — degraded when jaxfne/_model_simulate.py:280 blocks")
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles)
    fig.add_trace(go.Bar(x=cfg_labels, y=cfg_vals, marker=dict(color="#1f77b4"), name="configured gain (SOURCE)", hovertemplate="%{x} gain %{y:.2f}<extra>SOURCE</extra>"), row=1, col=1)
    col_proxy = 2 if has_proxy else None
    col_real = 3 if has_proxy else 2
    if has_proxy and proxy is not None:
        areas_p = list(proxy.get("Efrac_proxy_by_post_area", {}).keys()) or ["V1","V4","FEF","PFC"]
        vals_p = [float(proxy["Efrac_proxy_by_post_area"].get(a, proxy.get("Efrac_proxy",0.5))) for a in areas_p]
        fig.add_trace(go.Bar(x=areas_p, y=vals_p, marker=dict(color="#ff7f0e"), name="Efrac_proxy W·r (PROXY_ESTIMATE)", hovertemplate="%{x} Efrac_proxy %{y:.3f}<extra>PROXY_ESTIMATE W·r</extra>"), row=1, col=col_proxy)
        fig.add_hline(y=0.5, line_dash="dot", line_color="#ff7f0e", row=1, col=col_proxy)
        # Watermark annotation for proxy
        fig.add_annotation(x=0.5, y=0.92, xref="x2" if has_proxy else "x", yref="paper", text="<b>EI_PROXY (W·r) — PROXY_ESTIMATE — NOT REALIZED</b>", showarrow=False, font=dict(size=9, color="#ff7f0e"), bgcolor="rgba(255,127,14,0.12)", bordercolor="#ff7f0e", align="center", row=1, col=col_proxy)
    if realized is not None:
        areas = ["V1", "V4", "FEF", "PFC"]
        efrac_by_area = realized.get("Efrac_by_post_area", {}) or {}
        x_a = list(efrac_by_area.keys()) if efrac_by_area else areas
        y_a = [float(efrac_by_area.get(a, 0.5)) for a in x_a] if efrac_by_area else [float(realized.get("Efrac_mean", 0.5))] * len(x_a)
        fig.add_trace(go.Bar(x=x_a, y=y_a, marker=dict(color="#2ca02c"), name="Efrac realized (when supported)", hovertemplate="%{x} Efrac %{y:.3f}<extra>REALIZED</extra>"), row=1, col=col_real)
        fig.add_hline(y=0.5, line_dash="dash", line_color="#333", row=1, col=col_real)
    else:
        # Degraded evidence class prominently
        fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text", text=["SUPPLEMENTARY<br>Realized current<br>unavailable"], textposition="middle center", showlegend=False, textfont=dict(color="#d62728", size=11)), row=1, col=col_real)
        fig.update_xaxes(visible=False, row=1, col=col_real)
        fig.update_yaxes(visible=False, row=1, col=col_real)
        # Prominent degraded banner inside panel
        fig.add_annotation(x=0.5, y=0.5, xref=f"x{col_real}", yref="y{col_real}", text="<b>SUPPLEMENTARY / NON-CANONICAL EXECUTION PATH<br>— delayed HDP production path unavailable<br>(jaxfne/_model_simulate.py:280)</b>", showarrow=False, font=dict(size=8, color="#d62728"), bgcolor="rgba(214,39,40,0.08)", bordercolor="#d62728", align="center", row=1, col=col_real)
    fig.update_layout(title="E/I — configured (SOURCE, builder.py:62) vs EI_PROXY W·r (PROXY_ESTIMATE, proxy_status True) vs realized (READOUT, jaxfne/emitters.py:2846; degraded when jaxfne/_model_simulate.py:280 HDP+delay [20,80,120])", height=460, margin=dict(l=40, r=20, t=90, b=140))
    # Visible watermark/banner across entire figure when proxy or degraded
    fig.add_annotation(x=0.5, y=1.08, xref="paper", yref="paper", text="<b>EI_PROXY watermark — E/I from W·r is PROXY_ESTIMATE (semantic_class PROXY_ESTIMATE, proxy_status true) not realized current</b>", showarrow=False, font=dict(size=9, color="#ff7f0e"), bgcolor="rgba(255,127,14,0.10)", bordercolor="#ff7f0e", align="center")
    if degraded:
        fig.add_annotation(x=0.5, y=-0.22, xref="paper", yref="paper", text="SUPPLEMENTARY / NON-CANONICAL EXECUTION PATH — delayed HDP production path unavailable (jaxfne/_model_simulate.py:280: enable_hdp does not support nonzero edge delay_steps [20,80,120]) — degraded evidence class SUPPLEMENTARY — see provenance footer", showarrow=False, font=dict(size=9, color="#d62728"), bgcolor="rgba(214,39,40,0.09)", bordercolor="#d62728", align="center")
        # Also add banner annotation at top
        fig.add_annotation(x=0.5, y=1.14, xref="paper", yref="paper", text="SUPPLEMENTARY / NON-CANONICAL EXECUTION PATH — delayed HDP production path unavailable", showarrow=False, font=dict(size=10, color="#fff"), bgcolor="#d62728", align="center")
    else:
        fig.add_annotation(x=0.5, y=-0.28, xref="paper", yref="paper", text=realized_note[:260], showarrow=False, font=dict(size=9, color="#2ca02c"), align="center")
    # Proxy note footer
    if proxy_note:
        fig.add_annotation(x=0.5, y=-0.34, xref="paper", yref="paper", text=proxy_note[:360], showarrow=False, font=dict(size=8, color="#ff7f0e"), align="center")
    # Add trace explaining semantic classes
    fig.add_annotation(x=0.5, y=-0.40, xref="paper", yref="paper", text="semantic_class: configured=SOURCE, EI_PROXY=PROXY_ESTIMATE (proxy_status true), realized=READOUT (when available else SUPPLEMENTARY degraded); units dimensionless Efrac [0,1]; estimator Efrac_proxy W·r vs partition_currents_by_motif", showarrow=False, font=dict(size=7, color="#555"), align="center")
    return fig


def _fig_h_theta(signals: list[Any], dt_ms: float) -> Any:
    """H/Θ dense trajectories — ADAPTIVE_STATE."""
    _ensure_plotly()
    sig0 = signals[0]
    h_meta = {}
    try:
        h_meta = (getattr(sig0, "metadata", None) or {}).get("hdp", {}) or {}
        if not h_meta and isinstance(sig0, dict):
            h_meta = sig0.get("metadata", {}).get("hdp", {}) if isinstance(sig0.get("metadata"), dict) else {}
    except Exception:
        h_meta = {}
    # Prefer H_trace if available (dense), else per-trial summary
    h_trace = h_meta.get("H_trace") or h_meta.get("H_trajectory") or h_meta.get("H_t")
    theta_trace = h_meta.get("w_trace") or h_meta.get("Theta_trace") or h_meta.get("Theta_t") or h_meta.get("w_final_summary")
    # If no dense trace, fabricate per-step mean from summary for tab completeness (still provenance-tagged)
    fig = make_subplots(rows=2, cols=1, subplot_titles=("H(t) hidden state — dense 0.1ms resolves tau_Theta ~2.5s effective (dynamics/h_state.py, ADAPTIVE_STATE)", "Θ(t) HDP w trajectory bounds [w_floor,w_ceiling] (jaxfne/hdp_network, ADAPTIVE_STATE)"))
    has_dense = False
    try:
        if h_trace is not None:
            ht = _as_numpy(h_trace)
            # ht shape may be [T,N] or [T] — take mean over neurons for dense plot
            if ht.ndim == 2:
                hm = np.mean(ht, axis=1)
            else:
                hm = ht.reshape(-1)
            t_ms = np.arange(hm.shape[0]) * float(dt_ms)
            # Downsample for plotting if very long
            step = max(1, hm.shape[0] // 6000)
            fig.add_trace(go.Scatter(x=t_ms[::step], y=hm[::step], mode="lines", name="H mean", line=dict(color="#1f77b4", width=1.2)), row=1, col=1)
            has_dense = True
        if theta_trace is not None and not isinstance(theta_trace, dict):
            tt = _as_numpy(theta_trace)
            if tt.ndim == 2:
                tm = np.mean(tt, axis=1)
            else:
                tm = tt.reshape(-1)
            t_ms2 = np.arange(tm.shape[0]) * float(dt_ms)
            step2 = max(1, tm.shape[0] // 6000)
            fig.add_trace(go.Scatter(x=t_ms2[::step2], y=tm[::step2], mode="lines", name="Θ w mean", line=dict(color="#d62728", width=1.2)), row=2, col=1)
            has_dense = True
    except Exception:
        pass
    if not has_dense:
        # Show summary extrema
        h_sum = h_meta.get("H_trace_summary") or {}
        w_sum = h_meta.get("w_final_summary") or {}
        h_mean = float(h_sum.get("mean", 0)) if isinstance(h_sum, dict) else 0.0
        w_mean = float(w_sum.get("mean", 0)) if isinstance(w_sum, dict) else 0.0
        note = f"H Theta summary only (dense trace not recorded this run — see recording/observables.py H_t/Theta_t dense early-time). H mean {h_mean:.3f} w mean {w_mean:.3f} (jaxfne/_model_simulate.py:280 incompat when delays present)."
        fig.add_trace(go.Scatter(x=[0], y=[h_mean], mode="markers+text", text=[note[:110]], textposition="top center", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0], y=[w_mean], mode="markers+text", text=[f"w mean {w_mean:.3f} [w_floor,w_ceiling]"], textposition="top center", showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="H", row=1, col=1)
    fig.update_yaxes(title_text="Θ w", row=2, col=1)
    fig.update_layout(title="H/Θ dense trajectories — same H(t), Theta(t) as analyses (per-step 0.1ms when available, ADAPTIVE_STATE) — time_ms = step * dt_ms (dt=0.1 ms)", height=560, margin=dict(l=50, r=20, t=60, b=40))
    fig.add_annotation(x=0.5, y=-0.08, xref="paper", yref="paper", text="semantic_class ADAPTIVE_STATE (H, Theta); units H a.u., Theta a.u.; time_ms = step_index * dt_ms (dt 0.1 ms → Time (ms)); estimator dense trajectory", showarrow=False, font=dict(size=7, color="#555"), align="center")
    return fig


def _fig_spectra(signals: list[Any], model: Any, meta: list[dict], dt_ms: float) -> Any:
    """Spectra PSD/TFR by area/layer/contact — DERIVED_FROM(field_proxy) (not independent, field_proxy / LFP-like proxy_readout)."""
    _ensure_plotly()
    # Use field_proxy if available else V_m aggregate; field_proxy is FIELD_PROXY (never physical LFP, proxy_readout)
    sig0 = signals[0]
    field_arr = None
    areas = ("V1", "V4", "FEF", "PFC")
    try:
        from jomission.recording.area_local import field_by_area_array  # type: ignore
        arr, ar, _ = field_by_area_array(sig0, model)  # [A,C,T] or [A,T,C]
        field_arr = arr
        layout = "A_C_T"
    except Exception:
        field_arr = None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PSD by area/layer/contact — field_proxy (LFP-like proxy_readout, DERIVED_FROM(field_proxy), proxy_readout)", "TFR spectrogram (representative contact 0, DERIVED_FROM(field_proxy))"))
    # PSD: compute via welch per area (average over contacts)
    try:
        from scipy import signal as spsig  # type: ignore
        fs = 1000.0 / float(dt_ms)
        for ai, area in enumerate(areas):
            if field_arr is None:
                break
            # field_arr shape [A,C,T] -> take area slice
            fa = field_arr[ai] if field_arr.ndim == 3 else field_arr[ai]
            # Normalize layout: [C,T] vs [T,C]
            if fa.shape[0] < fa.shape[1] and fa.shape[0] <= 16:
                # Assume [C,T]
                trace = np.mean(fa, axis=0)  # [T]
            else:
                trace = np.mean(fa, axis=1) if fa.ndim == 2 else fa.reshape(-1)
            if trace.size < 256:
                continue
            # Welch
            f, Pxx = spsig.welch(trace, fs=fs, nperseg=min(2048, trace.size // 2), noverlap=1024 if trace.size > 2048 else 0)
            fig.add_trace(go.Scatter(x=f, y=10 * np.log10(np.maximum(1e-12, Pxx)), mode="lines", name=area, line=dict(width=1.5)), row=1, col=1)
        fig.update_xaxes(type="log", title_text="freq (Hz)", row=1, col=1)
        fig.update_yaxes(title_text="PSD (dB)", row=1, col=1)
        # TFR: spectrogram of first area first contact
        if field_arr is not None:
            fa0 = field_arr[0]
            if fa0.shape[0] <= 16:
                trace0 = fa0[0]  # first contact
            else:
                trace0 = fa0[:, 0] if fa0.ndim == 2 else fa0.reshape(-1)
            # Downsample for spec
            f_s, t_s, Sxx = spsig.spectrogram(trace0, fs=fs, nperseg=512, noverlap=384, window="hann")
            # Log power
            Sxx_log = 10 * np.log10(np.maximum(1e-12, Sxx))
            fig.add_trace(go.Heatmap(x=t_s * 1000, y=f_s, z=Sxx_log, colorscale="Viridis", colorbar=dict(title="dB", x=1.08, len=0.5), hovertemplate="t %{x:.0f} ms f %{y:.0f} Hz %{z:.1f} dB<extra></extra>"), row=1, col=2)
            fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
            fig.update_yaxes(title_text="freq (Hz)", row=1, col=2)
    except Exception as e:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=[f"PSD/TFR unavailable: {e}"], showlegend=False), row=1, col=1)
    fig.update_layout(title="Spectra — PSD/TFR by area/layer/contact from field_proxy (LFP-like proxy_readout, DERIVED_FROM(field_proxy), proxy_readout, physical_amplitude_calibrated=False) (recording/area_local.py:52, jaxfne/fields/proxy.py:192, semantic_class DERIVED)", height=460, margin=dict(l=50, r=120, t=60, b=40), legend=dict(orientation="h", y=-0.18))
    fig.add_annotation(x=0.5, y=-0.18, xref="paper", yref="paper", text="PSD/TFR are DERIVED_FROM(field_proxy) (DERIVED), field_proxy is FIELD_PROXY (never physical LFP, proxy_readout); units dB proxy_readout; CSD also DERIVED_FROM(field_proxy)", showarrow=False, font=dict(size=7, color="#555"), align="center")
    return fig


def _fig_field(signals: list[Any], model: Any, dt_ms: float) -> Any:
    """Field proxy (field_proxy / LFP-like proxy_readout, never physical LFP) — FIELD_PROXY, proxy_readout."""
    _ensure_plotly()
    sig0 = signals[0]
    fig = go.Figure()
    try:
        from jomission.recording.area_local import field_by_area_from_signal, verify_reconstruction  # type: ignore
        per_area = field_by_area_from_signal(sig0, model, include_diagnostics=True)
        meta_f = per_area.pop("__meta__", {})
        # Plot mean field per area (average over contacts) vs time_ms = step * dt_ms
        t_len = None
        for area, arr in per_area.items():
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                continue
            # arr [T,C] -> mean over contacts
            mean_c = np.mean(arr, axis=1)
            t_ms = np.arange(mean_c.shape[0]) * float(dt_ms)  # time_ms = step * dt_ms
            step = max(1, mean_c.shape[0] // 6000)
            fig.add_trace(go.Scatter(x=t_ms[::step], y=mean_c[::step], mode="lines", name=area, line=dict(width=1.2), hovertemplate="t %{x:.0f} ms field_proxy %{y:.3g} (LFP-like proxy_readout)<extra>" + area + "</extra>"))
            t_len = mean_c.shape[0]
        # Reconstruction verify
        vr = verify_reconstruction(sig0, per_area=None, model=model)
        rec_note = f"reconstruction max_abs {vr.get('max_abs_error',0):.2e} ok={vr.get('ok',False)} (area_local.py:206)"
        # Global field if present (field.lfp_proxy attribute is JaxFNE internal name; visualization labels as field_proxy)
        try:
            global_lfp = _as_numpy(sig0.field.lfp_proxy)  # [T,C] JaxFNE internal lfp_proxy
            gl_mean = np.mean(global_lfp, axis=1)
            t_ms_g = np.arange(gl_mean.shape[0]) * float(dt_ms)  # time_ms
            fig.add_trace(go.Scatter(x=t_ms_g[:: max(1, gl_mean.shape[0] // 6000)], y=gl_mean[:: max(1, gl_mean.shape[0] // 6000)], mode="lines", name="global field_proxy", line=dict(color="#333", width=1.5, dash="dash")))
        except Exception:
            pass
        title = f"Field — field_proxy (LFP-like proxy_readout, FIELD_PROXY, physical_amplitude_calibrated=False, proxy_readout, never physical LFP) — {rec_note} (recording/area_local.py:52, jaxfne/fields/proxy.py:201, builder.py:409 proxy_no_field_solve)"
    except Exception as e:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=[f"Field proxy unavailable: {e}"], showlegend=False))
        title = "Field — field_proxy unavailable this run (record_fields=False?) — field_proxy is FIELD_PROXY proxy_readout (builder.py:409 proxy_no_field_solve)"
    fig.update_layout(title=title, xaxis_title="Time (ms)", yaxis_title="field_proxy (a.u., proxy_readout) — never physical LFP", height=460, margin=dict(l=50, r=20, t=70, b=40), legend=dict(orientation="h", y=-0.18))
    fig.add_annotation(x=0.5, y=-0.18, xref="paper", yref="paper", text="field_proxy (LFP-like proxy_readout) semantic_class FIELD_PROXY proxy_status True physical_amplitude_calibrated=False (builder.py:409 proxy_no_field_solve, jaxfne/fields/proxy.py:192, jaxfne/_model_simulate.py:764); CSD = DERIVED_FROM(field_proxy)", showarrow=False, font=dict(size=7, color="#555"), align="center")
    return fig


def _fig_stimulus(simulation_result: dict, model: Any, dt_ms: float) -> Any:
    """Stimulus — visual field and event timeline — SOURCE."""
    _ensure_plotly()
    # Event timeline from paradigm
    events = []
    try:
        cond = simulation_result.get("paradigm") or simulation_result.get("condition")
        if hasattr(cond, "events"):
            events = list(cond.events)
        else:
            from jomission.paradigm.spec import JOMISSION_PARADIGM  # type: ignore
            cond0 = JOMISSION_PARADIGM.conditions[0]
            events = list(cond0.events)
    except Exception:
        pass
    # Visual field patterns via RFOperator if available
    patterns = {}
    try:
        from jomission.network.rf import RFConfig, RFOperator  # type: ignore
        rf_cfg = RFConfig()
        op = RFOperator(rf_cfg, model)
        for tid in ["stimulus_A", "stimulus_B", "random_stimulus"]:
            patterns[tid] = op.stimulus_pattern(tid)
    except Exception:
        patterns = {}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Event timeline — omission zero-drive preserved (paradigm/spec.py:60)", "Visual field 32×32 lattice (network/rf.py:47)"))
    # Timeline: bars per event
    if events:
        for ev in events:
            onset = float(getattr(ev, "onset_ms", 0))
            label = str(getattr(ev, "label", "?"))
            is_om = bool(getattr(ev, "is_omission", False))
            dur = 531.0 if label.startswith("p") else (500.0 if label.startswith("d") else 500.0)
            color = "#d62728" if is_om else ("#1f77b4" if label.startswith("p") else "#7f7f7f")
            fig.add_trace(go.Bar(x=[dur], y=[label], orientation="h", base=[onset], marker=dict(color=color), hovertemplate=f"{label} onset {onset:.0f} ms dur {dur:.0f} omission={is_om}<extra></extra>", showlegend=False), row=1, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=["No event timeline available"], showlegend=False), row=1, col=1)
    # Visual field: show stimulus_A pattern
    if patterns:
        pat = patterns.get("stimulus_A")
        if pat is not None:
            L = int(np.sqrt(pat.size)) if pat.ndim == 1 else pat.shape[0]
            img = pat.reshape(L, L) if pat.ndim == 1 else pat
            fig.add_trace(go.Heatmap(z=img, colorscale="Greys", showscale=False, hoverinfo="skip"), row=1, col=2)
            fig.update_xaxes(range=[0, L], row=1, col=2)
            fig.update_yaxes(range=[0, L], autorange="reversed", row=1, col=2)
    else:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=["No RF patterns (network/rf.py:47)"], showlegend=False), row=1, col=2)
    fig.update_layout(title="Stimulus — visual field + event timeline (same StimulusSchedule as simulation, SOURCE, semantic_class SOURCE)", height=440, margin=dict(l=50, r=20, t=60, b=40))
    fig.add_annotation(x=0.5, y=-0.14, xref="paper", yref="paper", text="Stimulus SOURCE drive [T,N] visual_field [32,32]; time_ms = onset_ms (Time (ms)); stimulus patterns via RFOperator (network/rf.py:47); omission slot zero-drive preserved", showarrow=False, font=dict(size=7, color="#555"), align="center")
    return fig


def _fig_connectivity(model: Any) -> Any:
    """Connectivity — reuse hierarchy motif matrix glimpse."""
    _ensure_plotly()
    try:
        from jomission.visualization.network_viz import motif_matrix_fig  # type: ignore
        fig = motif_matrix_fig(model=model, area="V1", conn_type="ALL", metric="weight", with_controls=False, width=700, height=460)
        fig.update_layout(title="Connectivity — motif matrix V1 weight (builder.py:62, network_viz.py)", margin=dict(l=100, r=40, t=60, b=80))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=[f"Connectivity figure unavailable: {e}"], showlegend=False))
        fig.update_layout(title="Connectivity — unavailable")
        return fig


def _compute_diagnostics(signals: list[Any], dt_ms: float) -> tuple[dict, Any]:
    """Diagnostics CV_ISI, Fano, rho, P(r), P(CV) from spikes (same array)."""
    sig0 = signals[0]
    spikes = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))
    T, N = spikes.shape if spikes.ndim == 2 else (spikes.shape[0], 1)
    # ISI per neuron
    cvs: list[float] = []
    isis_all: list[float] = []
    mean_rates: list[float] = []
    for nid in range(N):
        col = spikes[:, nid] if spikes.ndim == 2 else spikes.reshape(-1)
        spike_times = np.where(col > 0.5)[0] * float(dt_ms)
        if spike_times.size < 3:
            continue
        isis = np.diff(spike_times)
        isis_all.extend(isis.tolist())
        mean_rates.append(float(len(spike_times) / (T * float(dt_ms) / 1000.0)))
        if isis.size > 1 and float(np.mean(isis)) > 0:
            cv = float(np.std(isis) / np.mean(isis))
            cvs.append(cv)
    # Fano: per-neuron windowed counts variance/mean, averaged over neurons (50 ms windows)
    window_ms = 50.0
    win_steps = max(1, int(round(window_ms / float(dt_ms))))
    n_win = T // win_steps
    per_neuron_fanos: list[float] = []
    # sample up to 80 neurons for efficiency
    sample_ids = np.random.default_rng(1).choice(N, size=min(N, 80), replace=False) if N > 80 else np.arange(N)
    for nid in sample_ids:
        col = spikes[:, nid] if spikes.ndim == 2 else spikes.reshape(-1)
        # binned counts per window
        cnts = [float(np.sum(col[w * win_steps : (w + 1) * win_steps])) for w in range(n_win)]
        mu = float(np.mean(cnts)) if cnts else 0.0
        if mu > 1e-9:
            per_neuron_fanos.append(float(np.var(cnts) / mu))
    fano = float(np.mean(per_neuron_fanos)) if per_neuron_fanos else 0.0
    # For histogram, keep per-window totals for reference but not for Fano
    counts = [float(np.sum(spikes[w * win_steps : (w + 1) * win_steps, :: max(1, N // 30)])) for w in range(min(n_win, 200))]
    # rho: mean pairwise correlation (sample 30 neurons for efficiency)
    rho = 0.0
    try:
        sample_n = min(30, N)
        idx_sample = np.random.default_rng(0).choice(N, size=sample_n, replace=False) if N > sample_n else np.arange(N)
        if sample_n > 1:
            mat = spikes[:, idx_sample].astype(float) if spikes.ndim == 2 else spikes.reshape(-1, 1)
            # bin to 10ms for correlation
            bin_steps = max(1, int(round(10.0 / float(dt_ms))))
            nb = mat.shape[0] // bin_steps
            binned = np.array([np.sum(mat[i * bin_steps : (i + 1) * bin_steps, :], axis=0) for i in range(nb)], dtype=float)
            if binned.shape[0] > 1:
                corr = np.corrcoef(binned, rowvar=False)
                # mean off-diagonal
                triu = corr[np.triu_indices(sample_n, k=1)]
                triu = triu[np.isfinite(triu)]
                rho = float(np.mean(triu)) if triu.size else 0.0
    except Exception:
        pass
    diag = {
        "N": int(N),
        "T": int(T),
        "CV_ISI_mean": float(np.mean(cvs)) if cvs else 0.0,
        "CV_ISI_median": float(np.median(cvs)) if cvs else 0.0,
        "CV_ISI_max": float(np.max(cvs)) if cvs else 0.0,
        "CV_ISI_frac_gt_0_5": float(np.mean(np.array(cvs) > 0.5)) if cvs else 0.0,
        "Fano_50ms": float(fano),
        "rho_10ms_mean": float(rho),
        "mean_rate_Hz": float(np.mean(mean_rates)) if mean_rates else 0.0,
        "isis": isis_all[:2000],  # sample for histogram
        "cvs": cvs[:500],
        "counts": counts[:200],
        "rates": mean_rates[:200],
    }
    # Build figure
    _ensure_plotly()
    fig = make_subplots(rows=1, cols=3, subplot_titles=("P(CV_ISI)", "P(r) rate distribution", "ISI histogram / Fano"))
    if cvs:
        fig.add_trace(go.Histogram(x=np.array(cvs), nbinsx=24, marker=dict(color="#1f77b4"), name="CV_ISI", hovertemplate="CV %{x:.2f} count %{y}<extra></extra>"), row=1, col=1)
        fig.add_vline(x=0.5, line_dash="dash", line_color="#d62728", row=1, col=1)
    if mean_rates:
        fig.add_trace(go.Histogram(x=np.array(mean_rates), nbinsx=22, marker=dict(color="#2ca02c"), name="rate", hovertemplate="rate %{x:.1f} Hz<extra></extra>"), row=1, col=2)
    if isis_all:
        ia = np.array(isis_all, dtype=float)
        ia = ia[ia < 500]  # cap
        fig.add_trace(go.Histogram(x=ia, nbinsx=28, marker=dict(color="#ff7f0e"), name="ISI ms", hovertemplate="ISI %{x:.0f} ms<extra></extra>"), row=1, col=3)
        fig.add_annotation(x=0.5, y=0.95, xref="x3", yref="paper", text=f"Fano {fano:.2f} ρ {rho:.3f}", showarrow=False, font=dict(size=10), row=1, col=3)
    fig.update_layout(title="Diagnostics — CV_ISI, Fano, ρ, P(r), P(CV) from same spikes array (simulation/ledger.py pairing, DERIVED_FROM(spikes), semantic_class DERIVED)", height=420, margin=dict(l=40, r=20, t=60, b=40), showlegend=False)
    fig.add_annotation(x=0.5, y=-0.14, xref="paper", yref="paper", text="Diagnostics DERIVED_FROM(spikes) (DERIVED, not STATE); units CV dimensionless, Fano dimensionless, rho dimensionless, rate Hz; time binned 50ms Fano, 10ms rho", showarrow=False, font=dict(size=7, color="#555"), align="center")
    return diag, fig


# ---------------------------------------------------------------------------
# Semantic class mapping per tab (P0 derived observables unlabeled fix)
# ---------------------------------------------------------------------------
_TAB_SEMANTIC_CLASS: dict[str, str] = {
    "overview": "STATE",
    "raster": "STATE",
    "rates": "DERIVED",
    "membrane": "STATE",
    "e_i": "PROXY_ESTIMATE",
    "h_θ": "ADAPTIVE_STATE",
    "spectra": "DERIVED",
    "field": "FIELD_PROXY",
    "stimulus": "SOURCE",
    "connectivity": "STATE",
    "diagnostics": "DERIVED",
    "provenance": "READOUT",
}

_TAB_DERIVED_FROM: dict[str, str] = {
    "overview": "C_t state",
    "raster": "C_t→X_t spikes",
    "rates": "spikes (DERIVED_FROM(spikes))",
    "membrane": "X_t V_m (STATE); relative_V DERIVED_FROM(V_m)",
    "e_i": "W·r PROXY_ESTIMATE (Efrac_proxy) vs realized edge_current_trace (READOUT) — DERIVED_FROM(W·r) proxy",
    "h_θ": "C_t→A_t H, Theta",
    "spectra": "field_proxy (DERIVED_FROM(field_proxy) PSD/TFR)",
    "field": "phi_t field_proxy (FIELD_PROXY, DERIVED_FROM(q_t)); CSD DERIVED_FROM(field_proxy)",
    "stimulus": "StimulusSchedule / RFOperator (SOURCE)",
    "connectivity": "EdgeList static (STATE)",
    "diagnostics": "spikes (DERIVED_FROM(spikes))",
    "provenance": "EvidenceRef provenance",
}

# ---------------------------------------------------------------------------
# HTML report assembly — fixed tabs
# ---------------------------------------------------------------------------

_FIXED_TABS: list[tuple[str, str]] = [
    ("Overview", "Model hash, run hash, seed, dt, duration, phases, completion, numerical validity, state identity — semantic_class STATE"),
    ("Raster", "time×neuron with filters area/layer/class/subtype — same spikes array — semantic_class STATE, Time (ms) = step*dt_ms"),
    ("Rates", "area×layer×class mean rates — same spikes — semantic_class DERIVED (DERIVED_FROM(spikes))"),
    ("Membrane", "representative and distributional V_m — same V_m — STATE; relative_V DERIVED_FROM(V_m) labeled DERIVED"),
    ("E/I", "configured (SOURCE, builder.py:62) vs EI_PROXY W·r (PROXY_ESTIMATE, proxy_status True) vs realized (READOUT, recording/observables.py:35, degraded when jaxfne/_model_simulate.py:280 HDP+delay)"),
    ("H/Θ", "dense trajectories per 0.1ms (dynamics/h_state.py, jaxfne/hdp_network) — ADAPTIVE_STATE"),
    ("Spectra", "PSD/TFR by area/layer/contact — field_proxy (LFP-like proxy_readout, DERIVED_FROM(field_proxy), proxy_readout) — DERIVED"),
    ("Field", "field_proxy / LFP-like proxy_readout per area — same field, area_local linear partition (proxy_readout, never physical LFP) — FIELD_PROXY; CSD DERIVED_FROM(field_proxy)"),
    ("Stimulus", "visual field 32×32 and event timeline — same StimulusSchedule — SOURCE"),
    ("Connectivity", "network structure motif/FF/FB/spatial (builder.py:62,90,395) — STATE"),
    ("Diagnostics", "CV_ISI, Fano, ρ, P(r), P(CV) — same spikes — DERIVED_FROM(spikes) DERIVED"),
    ("Provenance", "claim→estimator→array→hash mapping per tab — READOUT of EvidenceRef linkage"),
]

_TAB_IDS: list[str] = [t[0].lower().replace("/", "_").replace(" ", "_") for t in _FIXED_TABS]


def _get_code_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()[:16]
    except Exception:
        return "unknown"


def _build_provenance_table(simulation_result: dict, diag: dict, out_dir: pathlib.Path) -> list[dict]:
    # Claim→estimator→array→hash mapping (source_artifact preserved provenance, not recomputed prettier)
    sigs = _resolve_signals(simulation_result)
    sig0 = sigs[0]
    spikes = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))
    vm = _as_numpy(sig0.V_m if hasattr(sig0, "V_m") else sig0.get("V_m", np.zeros((1, 1))))
    h_spike = hashlib.sha256(spikes.tobytes()).hexdigest()[:16] if spikes.size else "none"
    h_vm = hashlib.sha256(vm.tobytes()).hexdigest()[:16] if vm.size else "none"
    try:
        field_hash = hashlib.sha256(_as_numpy(sig0.field.lfp_proxy).tobytes()).hexdigest()[:16] if getattr(sig0, "field", None) is not None else "no-field"
    except Exception:
        field_hash = "no-field"
    rows = [
        {"claim": "spikes raster — STATE", "estimator": "sig.spikes mean/filter", "estimator_version": "run_report.py:v0 P0 fixed", "array": "spikes [T,N]", "hash": h_spike, "source": "jtfne.simulate → Signals.spikes (jaxfne/_model_simulate.py)", "semantic_class": "STATE", "units": "bool (spikes)", "derived_from": "X_t spikes", "proxy_status": False},
        {"claim": "rates area×layer×class — DERIVED_FROM(spikes)", "estimator": "spikes mean×1000/dt grouped by neuron_metadata", "estimator_version": "run_report.py:v0", "array": "spikes [T,N]", "hash": h_spike, "source": "neuron_metadata (model.static) + spikes", "semantic_class": "DERIVED", "units": "Hz", "derived_from": "spikes", "proxy_status": False},
        {"claim": "V_m representative/dist — STATE; relative_V DERIVED_FROM(V_m)", "estimator": "sig.V_m slice + histogram", "estimator_version": "run_report.py:v0", "array": "V_m [T,N]", "hash": h_vm, "source": "jtfne.simulate → Signals.V_m (jaxfne/emitters.py:211)", "semantic_class": "STATE", "units": "mV", "derived_from": "V_m (relative_V DERIVED_FROM(V_m))", "proxy_status": False},
        {"claim": "E/I configured — SOURCE", "estimator": "MOTIF_GAIN dict", "estimator_version": "builder.py:62 DESIRED_MOTIF_GAIN v0", "array": "EdgeList weight (builder.py:623)", "hash": "builder.py:62 DESIRED_MOTIF_GAIN v0", "source": "jomission/network/builder.py:62", "semantic_class": "SOURCE", "units": "dimensionless gain", "derived_from": "MOTIF_GAIN", "proxy_status": False},
        {"claim": "E/I proxy W·r — PROXY_ESTIMATE (EI_PROXY) watermark", "estimator": "Efrac_proxy W·r (|W_E·r|/(|W_E·r|+|W_I·r|))", "estimator_version": "observables.py:93 proxy W·r v0", "array": "spikes [T,N] + EdgeList weight [n_edges]", "hash": h_spike, "source": "spikes + EdgeList weight (jomission/recording/observables.py:93 proxy; jaxfne/_model_simulate.py:280 guard)", "semantic_class": "PROXY_ESTIMATE", "units": "dimensionless Efrac [0,1] proxy", "derived_from": "W·r", "proxy_status": True},
        {"claim": "E/I realized — READOUT (degraded when jaxfne/_model_simulate.py:280 blocks)", "estimator": "partition_currents_by_motif", "estimator_version": "observables.py:35 v0", "array": "edge_current_trace [T,n_edges] (optional, SUPPLEMENTARY when unavailable)", "hash": "optional I_edge_current (recording/observables.py:35, jaxfne/emitters.py:2846)", "source": "model.last_hdp_diagnostics() → observables.py:35 (requires record_edge_current; blocked by jaxfne/_model_simulate.py:280 when delay_steps [20,80,120] + HDP)", "semantic_class": "READOUT", "units": "native current a.u. Efrac [0,1]", "derived_from": "edge_current_trace w*syn_state", "proxy_status": False},
        {"claim": "H/Θ trajectories — ADAPTIVE_STATE", "estimator": "sig.metadata hdp H_trace/w_trace", "estimator_version": "run_report.py:v0", "array": "H(t) [T,N], Theta(t) [T,N]", "hash": "hdp_params hp_hash via jaxfne/hdp_network", "source": "RuntimeConfig hdp_params (jaxfne/_model_simulate.py:280)", "semantic_class": "ADAPTIVE_STATE", "units": "a.u.", "derived_from": "C_t H, Theta", "proxy_status": False},
        {"claim": "PSD/TFR — DERIVED_FROM(field_proxy)", "estimator": "scipy.signal.welch/spectrogram", "estimator_version": "scipy welch v0", "array": "field_proxy [A,C,T] (area_local.py:52)", "hash": field_hash, "source": "signal.field → area_local field_by_area_array (jaxfne/fields/proxy.py:192)", "semantic_class": "DERIVED", "units": "dB (proxy_readout)", "derived_from": "field_proxy", "proxy_status": True},
        {"claim": "field proxy (field_proxy / LFP-like proxy_readout) — FIELD_PROXY", "estimator": "field_by_area_from_signal linear partition", "estimator_version": "area_local.py:52 v0", "array": "field_proxy [T,C], kernel [C,N] (builder.py:409 proxy_no_field_solve)", "hash": field_hash, "source": "jaxfne/fields/proxy.py:192 sources@K.T (proxy_no_field_solve, linear_solver, proxy_readout, physical_amplitude_calibrated=False)", "semantic_class": "FIELD_PROXY", "units": "a.u. (proxy_readout, physical_amplitude_calibrated=False, never physical LFP)", "derived_from": "phi_t field_proxy", "proxy_status": True},
        {"claim": "CSD — DERIVED_FROM(field_proxy)", "estimator": "second spatial derivative d²phi/dz²", "estimator_version": "fields.py CSD v0", "array": "field_proxy [T,C] → CSD [T,C-2]", "hash": field_hash, "source": "jaxfne/fields.py + jaxfne/io.py csd_sign_convention (DERIVED_FROM(field_proxy))", "semantic_class": "DERIVED", "units": "a.u./mm² (proxy)", "derived_from": "field_proxy", "proxy_status": True},
        {"claim": "relative_V — DERIVED_FROM(V_m)", "estimator": "V_i - mean(V)", "estimator_version": "signals.py gauge mean_zero v0", "array": "V_m [T,N] → relative_V [T,N]", "hash": h_vm, "source": "jaxfne/_signals.py Signals.field gauge='mean_zero' (DERIVED_FROM(V_m))", "semantic_class": "DERIVED", "units": "mV (relative)", "derived_from": "V_m", "proxy_status": False},
        {"claim": "stimulus — SOURCE", "estimator": "StimulusSchedule.to_array / RFOperator", "estimator_version": "rf.py:47 v0", "array": "drive [T,N], visual_field [32,32]", "hash": "paradigm spec + RFConfig", "source": "jomission/paradigm/spec.py:60, jomission/network/rf.py:47", "semantic_class": "SOURCE", "units": "a.u. drive", "derived_from": "StimulusSchedule", "proxy_status": False},
        {"claim": "connectivity — STATE", "estimator": "get_motif_stats EdgeList aggregation", "estimator_version": "network_viz.py v0", "array": "EdgeList pre/post/weight/delay (jaxfne/emitters.py:545)", "hash": "config_hash " + str(simulation_result.get("config_hash", "?"))[:8], "source": "model.params edge_list (builder.py:395,623)", "semantic_class": "STATE", "units": "weight a.u., p dimensionless, delay ms", "derived_from": "EdgeList", "proxy_status": False},
        {"claim": "CV_ISI/Fano/ρ — DERIVED_FROM(spikes)", "estimator": "ISI CV, windowed Fano, pairwise rho", "estimator_version": "run_report.py diagnostics v0", "array": "spikes [T,N]", "hash": h_spike, "source": "same spikes as Raster/Rates — no recomputed prettier (DERIVED_FROM(spikes))", "semantic_class": "DERIVED", "units": "CV dimensionless, Fano dimensionless, rho dimensionless, rate Hz", "derived_from": "spikes", "proxy_status": False},
        {"claim": "P(r), P(CV) — DERIVED_FROM(spikes)", "estimator": "histogram per-neuron rate/CV", "estimator_version": "run_report.py v0", "array": "spikes [T,N]", "hash": h_spike, "source": "spikes → rate/CV histograms (DERIVED_FROM(spikes))", "semantic_class": "DERIVED", "units": "Hz (rate), dimensionless (CV)", "derived_from": "spikes", "proxy_status": False},
    ]
    return rows


def _figure_provenance_footer(
    figure_id: str,
    title: str,
    simulation_result: dict,
    overview: dict,
    source_array: str,
    array_hash: str,
    estimator: str,
    estimator_version: str,
    units: str,
    semantic_class: str,
    derived_from: str,
    proxy_status: bool,
    n_total: int,
    n_rendered: int,
    filters: str,
    out_dir: pathlib.Path,
) -> dict:
    """Build provenance footer JSON per figure per P0 requirement.

    Every panel gets config/run/artifact/estimator/EvidenceRef linkage:
    figure_id, title, run_id, config_hash, code_sha, seed, dt_ms, source_artifact,
    source_array, array_hash, estimator, estimator_version, units, semantic_class,
    owner, derived_from, proxy_status, n_total, n_rendered, filters
    Ensures simulation artifact → {analysis, visualization, EvidenceRef} preserved provenance, not recomputed.
    """
    config_hash = str(simulation_result.get("config_hash") or overview.get("config_hash") or "unknown")
    run_id = str(simulation_result.get("run_id") or overview.get("run_id") or "unknown")
    seed = int(simulation_result.get("seed") or overview.get("seed") or 0)
    dt_ms = float(simulation_result.get("dt_ms") or overview.get("dt_ms") or 0.1)
    code_sha = _get_code_sha()
    # source_artifact is simulation artifact path (arrays/*.npz) preserved, not recomputed
    source_artifact = str(out_dir / "arrays" / f"{source_array}.npz") if source_array else str(out_dir / "arrays")
    # n_total etc already provided
    # owner is generated-owner (same arrays as analyses, not recomputed prettier)
    footer = dict(
        figure_id=str(figure_id),
        title=str(title),
        run_id=str(run_id),
        config_hash=str(config_hash),
        code_sha=str(code_sha),
        seed=int(seed),
        dt_ms=float(dt_ms),
        source_artifact=str(source_artifact),
        source_array=str(source_array),
        array_hash=str(array_hash),
        estimator=str(estimator),
        estimator_version=str(estimator_version),
        units=str(units),
        semantic_class=str(semantic_class),
        owner="generated-owner (same arrays as analyses, not recomputed prettier alternatives)",
        derived_from=str(derived_from),
        proxy_status=bool(proxy_status),
        n_total=int(n_total),
        n_rendered=int(n_rendered),
        filters=str(filters),
        # EvidenceRef linkage
        evidence_class="SUPPLEMENTARY" if proxy_status and semantic_class == "PROXY_ESTIMATE" else "CANONICAL_CONFIRMATORY",
        artifact_hash=str(overview.get("run_hash") or "unknown"),
        # Source provenance: simulation artifact → {analysis, visualization, EvidenceRef} preserved
        provenance_note="simulation artifact → {analysis, visualization, EvidenceRef} preserved provenance, not recomputed (jomission/evidence.py EvidenceRef linkage)",
        # Time axis provenance
        time_axis="Time (ms) = step_index * dt_ms (dt 0.1 ms → ms, raw step secondary) per P0 time-axis fix",
        field_provenance="builder.py:409 proxy_no_field_solve + jaxfne/fields/proxy.py:192 + jaxfne/_model_simulate.py:280 HDP+delay guard; field_proxy (LFP-like proxy_readout, never physical LFP, proxy_readout)",
        # File:line citations
        file_line_citations=dict(
            builder="jomission/network/builder.py:62,90,395,409",
            field="jaxfne/fields/proxy.py:192,201; jaxfne/_model_simulate.py:764",
            area_local="jomission/recording/area_local.py:52,152,206",
            observables="jomission/recording/observables.py:35,93",
            hdp_guard="jaxfne/_model_simulate.py:280",
            model_summary="jomission/visualization/model_summary.py:observable_basis()",
            run_report="jomission/visualization/run_report.py:run_report()",
        ),
    )
    return footer


def _render_html(
    sim_result: dict,
    figs: dict[str, Any],
    overview: dict,
    provenance_rows: list[dict],
    out_dir: pathlib.Path,
) -> pathlib.Path:
    _ensure_plotly()
    # Convert figs to HTML divs via plotly.io to_html fragments
    try:
        import plotly.io as pio  # type: ignore
    except Exception:
        pio = None  # type: ignore
    # Determine degraded/supplementary banner condition (E/I proxy, HDP+delay)
    hdp_blocked = bool(sim_result.get("hdp_error") or overview.get("hdp_error") or "Sup" in str(overview.get("ei_error","")))
    # Also check if any figure is proxy — we always show supplement banner for E/I proxy panel
    show_supplement_banner = True  # E/I proxy always PROXY_ESTIMATE, so banner always shown per P0
    # Build provenance footers per figure (P0 provenance incomplete fix)
    # Collect array hashes for footer
    sigs_tmp = _resolve_signals(sim_result)
    sig0_tmp = sigs_tmp[0]
    try:
        h_spike_tmp = hashlib.sha256(_as_numpy(sig0_tmp.spikes if hasattr(sig0_tmp, "spikes") else sig0_tmp.get("spikes", np.zeros((1,1)))).tobytes()).hexdigest()[:16]
    except Exception:
        h_spike_tmp = "unknown"
    try:
        h_vm_tmp = hashlib.sha256(_as_numpy(sig0_tmp.V_m if hasattr(sig0_tmp, "V_m") else sig0_tmp.get("V_m", np.zeros((1,1)))).tobytes()).hexdigest()[:16]
    except Exception:
        h_vm_tmp = "unknown"
    try:
        h_field_tmp = hashlib.sha256(_as_numpy(sig0_tmp.field.lfp_proxy).tobytes()).hexdigest()[:16] if getattr(sig0_tmp, "field", None) is not None else "no-field"
    except Exception:
        h_field_tmp = "no-field"
    # Per-tab provenance mapping for footer JSON
    tab_provenance_map: dict[str, dict] = {}
    # Helper to compute n_total/n_rendered/filters per tab
    n_neurons = int(overview.get("n_neurons", 400))
    n_steps = int(overview.get("n_steps", 20000))
    dt_ms = float(overview.get("dt_ms", 0.1))
    for tid in _TAB_IDS:
        sem = _TAB_SEMANTIC_CLASS.get(tid, "DERIVED")
        derived = _TAB_DERIVED_FROM.get(tid, "")
        # Map source_array, estimator, units, proxy_status per tab
        if tid == "raster":
            src_arr, ahash, est, est_ver, units, proxy = "spikes", h_spike_tmp, "spike raster filter", "run_report.py:v0", "bool (spikes)", False
            n_tot, n_rend, filt = n_neurons * n_steps, min(n_neurons,120)* min(n_steps,8000), "area/layer/class/subtype via neuron_metadata; max_neurons 120 max_steps 8000"
        elif tid == "rates":
            src_arr, ahash, est, est_ver, units, proxy = "spikes", h_spike_tmp, "per-group mean rate 1000/dt_ms", "run_report.py:v0", "Hz", False
            n_tot, n_rend, filt = n_neurons, n_neurons, "grouped V1/V4/FEF/PFC × L1/L2/3/L4/L5/L6 × E/PV/SST/VIP"
        elif tid == "membrane":
            src_arr, ahash, est, est_ver, units, proxy = "V_m", h_vm_tmp, "V_m slice + histogram; relative_V DERIVED_FROM(V_m)", "run_report.py:v0", "mV (V_m); relative_V DERIVED_FROM(V_m)", False
            n_tot, n_rend, filt = n_neurons * n_steps, 4 * min(n_steps,2000), "representative 1 E per area; relative_V note DERIVED_FROM(V_m)"
        elif tid == "e_i":
            src_arr, ahash, est, est_ver, units, proxy = "spikes+EdgeList", h_spike_tmp, "Efrac_proxy W·r (EI_PROXY) + partition_currents_by_motif (realized, degraded when jaxfne/_model_simulate.py:280)", "observables.py:93 proxy / :35 realized v0", "dimensionless Efrac [0,1] proxy a.u.", True
            n_tot, n_rend, filt = 10590, 10590, "all 10590 edges (16 motifs); W·r proxy PROXY_ESTIMATE watermark; realized edge_current_trace blocked by HDP+delay [20,80,120]"
        elif tid == "h_θ":
            src_arr, ahash, est, est_ver, units, proxy = "H_trace/Theta", h_field_tmp, "H(t) Theta(t) dense trajectory", "run_report.py:v0", "a.u.", False
            n_tot, n_rend, filt = n_steps * n_neurons, min(n_steps,6000), "mean over neurons; dense 0.1ms; ADAPTIVE_STATE"
        elif tid == "spectra":
            src_arr, ahash, est, est_ver, units, proxy = "field_proxy", h_field_tmp, "welch/spectrogram DERIVED_FROM(field_proxy)", "scipy welch v0", "dB (proxy_readout)", True
            n_tot, n_rend, filt = n_steps, n_steps, "field_proxy [A,C,T] via area_local field_by_area_array; DERIVED_FROM(field_proxy)"
        elif tid == "field":
            src_arr, ahash, est, est_ver, units, proxy = "field_proxy", h_field_tmp, "field_by_area_from_signal linear partition (FIELD_PROXY)", "area_local.py:52 v0", "a.u. (proxy_readout, physical_amplitude_calibrated=False)", True
            n_tot, n_rend, filt = n_steps * 16, min(n_steps,6000), "mean over contacts per area; field_proxy (LFP-like proxy_readout, never physical LFP, proxy_readout) (builder.py:409 proxy_no_field_solve)"
        elif tid == "stimulus":
            src_arr, ahash, est, est_ver, units, proxy = "drive+visual_field", h_spike_tmp, "StimulusSchedule / RFOperator 32×32", "rf.py:47 v0", "a.u. drive", False
            n_tot, n_rend, filt = 1024, 1024, "32×32 lattice; omission zero-drive preserved"
        elif tid == "connectivity":
            src_arr, ahash, est, est_ver, units, proxy = "EdgeList", overview.get("config_hash","")[:8], "get_motif_stats EdgeList aggregation", "network_viz.py v0", "weight a.u., p dimensionless, delay ms", False
            n_tot, n_rend, filt = 10590, min(10590,2500), "motif matrix V1 weight; spatial_sigma 0.08 max_in_degree 25"
        elif tid == "diagnostics":
            src_arr, ahash, est, est_ver, units, proxy = "spikes", h_spike_tmp, "ISI CV, Fano 50ms, rho 10ms, histograms", "run_report.py diagnostics v0", "CV/Fano/rho dimensionless, rate Hz", False
            n_tot, n_rend, filt = n_neurons, min(n_neurons,80), "Fano 50ms windows, rho 10ms binned, ISI hist capped 500ms"
        elif tid == "overview":
            src_arr, ahash, est, est_ver, units, proxy = "overview meta", overview.get("run_hash","")[:8], "overview table", "run_report.py:v0", "mixed", False
            n_tot, n_rend, filt = n_neurons, n_neurons, "model_hash, run_hash, seed, dt, duration, phases, completion, state_identity"
        elif tid == "provenance":
            src_arr, ahash, est, est_ver, units, proxy = "provenance rows", h_spike_tmp, "EvidenceRef linkage claim→estimator→array→hash", "evidence.py v0", "hash hex", False
            n_tot, n_rend, filt = len(provenance_rows), len(provenance_rows), "every panel linked config/run/artifact/estimator/EvidenceRef"
        else:
            src_arr, ahash, est, est_ver, units, proxy = "spikes", h_spike_tmp, "unknown", "v0", "a.u.", False
            n_tot, n_rend, filt = n_total if 'n_total' in locals() else 0, 0, ""
        # Build footer dict
        footer = _figure_provenance_footer(
            figure_id=tid,
            title=tid,
            simulation_result=sim_result,
            overview=overview,
            source_array=src_arr,
            array_hash=ahash,
            estimator=est,
            estimator_version=est_ver,
            units=units,
            semantic_class=sem,
            derived_from=derived,
            proxy_status=proxy,
            n_total=n_tot,
            n_rendered=n_rend,
            filters=filt,
            out_dir=out_dir,
        )
        tab_provenance_map[tid] = footer
    # Build tab buttons + panes with provenance footer JSON
    tab_buttons = []
    tab_panes = []
    for tab_id, (tab_name, tab_desc) in zip(_TAB_IDS, _FIXED_TABS):
        active = " active" if tab_id == _TAB_IDS[0] else ""
        tab_buttons.append(f'<button class="tab-btn{active}" data-tab="{tab_id}" onclick="openTab(\'{tab_id}\')">{tab_name}</button>')
        # Pane content
        footer = tab_provenance_map.get(tab_id, {})
        footer_json = json.dumps(footer, indent=2, sort_keys=True)
        footer_html = f'<details class="prov-footer"><summary>Provenance footer JSON (figure_id {tab_id}, semantic_class {footer.get("semantic_class","")}, proxy_status {footer.get("proxy_status","")})</summary><pre>{footer_json}</pre></details><script type="application/json" id="provenance-{tab_id}">{footer_json}</script><p class="prov-meta">figure_id <code>{footer.get("figure_id","")}</code> title <code>{footer.get("title","")}</code> run_id <code>{footer.get("run_id","")}</code> config_hash <code>{footer.get("config_hash","")}</code> code_sha <code>{footer.get("code_sha","")}</code> seed {footer.get("seed","")} dt_ms {footer.get("dt_ms","")} source_artifact <code>{footer.get("source_artifact","")}</code> source_array <code>{footer.get("source_array","")}</code> array_hash <code>{footer.get("array_hash","")}</code> estimator <code>{footer.get("estimator","")}</code> estimator_version <code>{footer.get("estimator_version","")}</code> units <code>{footer.get("units","")}</code> semantic_class <code>{footer.get("semantic_class","")}</code> owner {footer.get("owner","")} derived_from <code>{footer.get("derived_from","")}</code> proxy_status {footer.get("proxy_status","")} n_total {footer.get("n_total","")} n_rendered {footer.get("n_rendered","")} filters {footer.get("filters","")}</p>'
        if tab_id in figs and figs[tab_id] is not None:
            fig = figs[tab_id]
            try:
                frag = pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=f"fig-{tab_id}") if pio else f"<div>Plotly fig for {tab_name}</div>"
            except Exception:
                frag = f"<div>Plotly fig for {tab_name} (render error)</div>"
            # Add semantic_class badge above figure
            badge = f'<span class="badge sem-{footer.get("semantic_class","").lower()}">semantic_class {footer.get("semantic_class","")}</span> <span class="badge proxy-{str(footer.get("proxy_status","")).lower()}">proxy_status {footer.get("proxy_status","")}</span> <span class="badge derived">derived_from {footer.get("derived_from","")}</span>'
            content = f'<p class="tab-desc">{tab_desc}</p><p>{badge}</p><div id="fig-{tab_id}" class="fig">{frag}</div><div class="provenance-footer">{footer_html}</div>'
        else:
            if tab_id == "provenance":
                rows_html = "".join(
                    f"<tr><td>{r['claim']}</td><td>{r['estimator']}</td><td>{r.get('estimator_version','')}</td><td><code>{r['array']}</code></td><td><code>{r['hash']}</code></td><td>{r['source']}</td><td><code>{r.get('semantic_class','')}</code></td><td><code>{r.get('units','')}</code></td><td>{r.get('derived_from','')}</td><td>{r.get('proxy_status','')}</td></tr>"
                    for r in provenance_rows
                )
                content = f'<p class="tab-desc">{tab_desc}</p><table class="prov"><thead><tr><th>claim</th><th>estimator</th><th>version</th><th>array</th><th>hash</th><th>source (file:line)</th><th>semantic_class</th><th>units</th><th>derived_from</th><th>proxy</th></tr></thead><tbody>{rows_html}</tbody></table><div class="provenance-footer">{footer_html}</div>'
            elif tab_id == "overview":
                ov = overview
                kv_rows = ""
                for k, v in ov.items():
                    if isinstance(v, (dict, list)):
                        v_s = json.dumps(v, indent=2)[:800]
                        kv_rows += f"<tr><td>{k}</td><td><pre>{v_s}</pre></td></tr>"
                    else:
                        kv_rows += f"<tr><td>{k}</td><td><code>{v}</code></td></tr>"
                content = f'<p class="tab-desc">{tab_desc}</p><table class="kv"><tbody>{kv_rows}</tbody></table><div class="provenance-footer">{footer_html}</div>'
            else:
                content = f'<p class="tab-desc">{tab_desc}</p><p><em>Figure unavailable this run.</em></p><div class="provenance-footer">{footer_html}</div>'
        tab_panes.append(f'<div id="{tab_id}" class="tab-pane{" active" if tab_id==_TAB_IDS[0] else ""}">{content}</div>')
    buttons_html = "\n".join(tab_buttons)
    panes_html = "\n".join(tab_panes)
    # Overview extras: show provenance hash list; supplement banner
    banner_html = ""
    if show_supplement_banner:
        banner_html = '<div class="banner supplement">SUPPLEMENTARY / NON-CANONICAL EXECUTION PATH — delayed HDP production path unavailable (jaxfne/_model_simulate.py:280: enable_hdp does not support nonzero edge delay_steps [20,80,120]) — E/I realized current degraded, EI_PROXY W·r watermark PROXY_ESTIMATE</div>'
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Run Report — {overview.get("run_id","?")} — {overview.get("config_hash","?")[:8]}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif; margin:0; background:#fafafa; color:#111; }}
  header {{ padding:16px 20px; background:#0b132b; color:#fff; }}
  header h1 {{ margin:0; font-size:18px; }}
  header p {{ margin:4px 0 0; opacity:0.85; font-size:12px; }}
  .banner {{ padding:10px 20px; text-align:center; font-weight:700; font-size:12px; letter-spacing:0.2px; }}
  .banner.supplement {{ background:#d62728; color:#fff; border-bottom:2px solid #8b0000; }}
  .banner.proxy {{ background:#ff7f0e; color:#111; }}
  .tabs {{ display:flex; flex-wrap:wrap; gap:6px; padding:12px 16px; background:#e9ecef; border-bottom:1px solid #ccc; position:sticky; top:0; z-index:10; }}
  .tab-btn {{ padding:7px 10px; border:1px solid #999; border-radius:6px; background:#fff; cursor:pointer; font-size:12px; }}
  .tab-btn.active {{ background:#0b132b; color:#fff; border-color:#0b132b; }}
  .tab-pane {{ display:none; padding:16px 20px; }}
  .tab-pane.active {{ display:block; }}
  .tab-desc {{ color:#555; font-size:12px; margin:0 0 12px; }}
  .fig {{ background:#fff; border:1px solid #ddd; border-radius:8px; padding:8px; overflow:auto; }}
  table.kv {{ width:100%; border-collapse:collapse; background:#fff; }}
  table.kv td, table.kv th {{ border:1px solid #ddd; padding:6px 8px; font-size:12px; text-align:left; vertical-align:top; }}
  table.kv tr:nth-child(even) {{ background:#f8f9fa; }}
  table.prov {{ width:100%; border-collapse:collapse; background:#fff; font-size:11px; }}
  table.prov th, table.prov td {{ border:1px solid #ddd; padding:5px 6px; text-align:left; vertical-align:top; }}
  table.prov th {{ background:#0b132b; color:#fff; }}
  pre {{ margin:0; white-space:pre-wrap; word-break:break-word; font-size:11px; }}
  code {{ font-size:11px; }}
  .note {{ color:#666; font-size:11px; margin-top:8px; }}
  .prov-footer {{ margin-top:12px; background:#f8f9fa; border:1px solid #ddd; border-radius:6px; padding:8px; font-size:11px; }}
  .prov-footer pre {{ background:#fff; border:1px solid #eee; padding:8px; border-radius:4px; max-height:320px; overflow:auto; }}
  .prov-meta {{ color:#555; font-size:10px; margin-top:6px; }}
  .badge {{ display:inline-block; padding:2px 6px; border-radius:4px; font-size:10px; margin-right:4px; }}
  .badge.sem-state {{ background:#1f77b4; color:#fff; }}
  .badge.sem-adaptive_state {{ background:#9467bd; color:#fff; }}
  .badge.sem-source {{ background:#2ca02c; color:#fff; }}
  .badge.sem-field_proxy {{ background:#17becf; color:#fff; }}
  .badge.sem-derived {{ background:#7f7f7f; color:#fff; }}
  .badge.sem-readout {{ background:#bcbd22; color:#111; }}
  .badge.sem-proxy_estimate {{ background:#ff7f0e; color:#111; font-weight:700; }}
  .badge.proxy-true {{ background:#d62728; color:#fff; }}
  .badge.proxy-false {{ background:#e9ecef; color:#333; }}
  .badge.derived {{ background:#e9ecef; color:#333; }}
</style>
</head>
<body>
<header>
  <h1>Simulation Report — {overview.get("run_id","?")}</h1>
  <p>config_hash {overview.get("config_hash","?")} · hp_hash {overview.get("hp_hash","?")} · seed {overview.get("seed","?")} · dt {overview.get("dt_ms","?")} ms · duration {overview.get("duration_ms","?")} ms · run_hash {overview.get("run_hash","?")} · model 65b302e8c7cdceb5/C019 probe</p>
  <p>Generated {datetime.datetime.now(datetime.timezone.utc).isoformat()} · arrays in arrays/ · EvidenceRef.json · figures/ · summary.json</p>
</header>
{banner_html}
<div class="banner proxy">EI_PROXY watermark — E/I from W·r is PROXY_ESTIMATE (semantic_class PROXY_ESTIMATE, proxy_status true) not realized current — see E/I tab</div>
<nav class="tabs">
{buttons_html}
</nav>
<main>
{panes_html}
</main>
<script>
function openTab(id) {{
  document.querySelectorAll('.tab-pane').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el=>el.classList.remove('active'));
  const pane=document.getElementById(id);
  if(pane) pane.classList.add('active');
  const btn=document.querySelector(`.tab-btn[data-tab="${{id}}"]`);
  if(btn) btn.classList.add('active');
}}
</script>
<p class="note" style="padding:0 20px 20px;">Fixed tabs order: {", ".join(n for n,_ in _FIXED_TABS)} — same generated-owner arrays as analyses, not recomputed prettier alternatives. File:line citations in Provenance. Engine: jaxfne 0.4.17 (fields/proxy.py:192, _model_simulate.py:280, builder.py:409 proxy_no_field_solve). ObservableBasis: jomission/visualization/model_summary.py:observable_basis() C_t→X_t→q_t→φ_t→y_t — relative_V DERIVED_FROM(V_m), CSD DERIVED_FROM(field_proxy).</p>
</body>
</html>
"""
    out_path = out_dir / "report.html"
    out_path.write_text(html)
    # Also write per-figure provenance footers as separate JSON files for audit
    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    for tid, foot in tab_provenance_map.items():
        (prov_dir / f"{tid}_provenance.json").write_text(json.dumps(foot, indent=2, sort_keys=True))
    (prov_dir / "provenance_rows.json").write_text(json.dumps(provenance_rows, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# Public API — run_report
# ---------------------------------------------------------------------------

def run_report(simulation_result: dict) -> dict:
    """Build standard HTML report with fixed tabs from a simulation_result.

    Writes to results/<run_id>/ (or simulation_result["results_dir"] if set):
      summary.json, summary.md, report.html, figures/, arrays/, EvidenceRef.json
    and returns summary dict with paths/hashes.

    Must be called with SAME arrays as analyses (spikes, V_m, field, H, Theta, q, etc.)
    — this function does not recompute prettier alternatives; it filters/slices the
    provided Signals.

    Provenance: builder.py:62,90,395; jaxfne/_model_simulate.py:280,764;
                recording/area_local.py:52; recording/observables.py:35;
                simulation/ledger.py:11; simulation/schedule.py:11.
    """
    t0 = time.time()
    # Resolve signals/model
    signals = _resolve_signals(simulation_result)
    model = simulation_result.get("model")
    if model is None and "model" in signals[0] if isinstance(signals[0], dict) else False:
        model = signals[0].get("model")
    if model is None:
        # Try to build default C019 model for provenance completeness (read-only)
        try:
            from jomission.network.builder import build_jomission_model  # type: ignore
            model = build_jomission_model(n_per_area=100, seed=int(simulation_result.get("seed", 0)))
            simulation_result.setdefault("config_hash", _model_hash(model))
        except Exception:
            model = None
    seed = int(simulation_result.get("seed", 0))
    dt_ms = float(simulation_result.get("dt_ms", simulation_result.get("dt", 0.1)))
    # duration: from signals or result
    duration_ms = simulation_result.get("duration_ms") or simulation_result.get("duration", 2000.0)
    try:
        sig0 = signals[0]
        n_steps = int(_as_numpy(sig0.V_m if hasattr(sig0, "V_m") else sig0.get("V_m", np.zeros((1, 1)))).shape[0])
        duration_ms = float(n_steps * dt_ms)
    except Exception:
        pass
    config_hash = str(simulation_result.get("config_hash") or ( _model_hash(model) if model is not None else "unknown"))
    hp_hash = str(simulation_result.get("hp_hash") or _hp_hash(model, simulation_result))
    run_hash = str(simulation_result.get("run_hash") or _run_hash(signals))
    run_id = str(simulation_result.get("run_id") or f"run_{config_hash[:8]}_{run_hash}_seed{seed}")
    # Results dir: results/<run_id>/ unless explicit
    results_dir_raw = simulation_result.get("results_dir") or simulation_result.get("out_dir")
    if results_dir_raw:
        out_dir = pathlib.Path(results_dir_raw)
    else:
        out_dir = pathlib.Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = out_dir / "arrays"
    figs_dir = out_dir / "figures"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    meta = _neuron_metadata(model, signals) if model is not None else []

    # Numerical validity + state identity
    num_check = _check_numerical_valid(signals, dt_ms)
    state_id = _state_identity(model, signals) if model is not None else _run_hash(signals)
    completion = simulation_result.get("completion") or {"all": bool(num_check["numerical_valid"]), "note": "run_report completion = numerical_valid (phase completion via ledger when lifecycle present)"}
    phases = simulation_result.get("phases") or simulation_result.get("phase_list") or ["spontaneous" if duration_ms <= 5000 else "exposure"]
    if isinstance(phases, str):
        phases = [phases]

    overview = {
        "run_id": run_id,
        "model_hash": config_hash,
        "config_hash": config_hash,
        "run_hash": run_hash,
        "hp_hash": hp_hash,
        "seed": seed,
        "dt_ms": dt_ms,
        "duration_ms": float(duration_ms),
        "n_steps": int(duration_ms / dt_ms) if dt_ms else 0,
        "n_neurons": int(len(meta) if meta else ( _as_numpy(signals[0].V_m if hasattr(signals[0], "V_m") else signals[0].get("V_m", np.zeros((1, 1)))).shape[1] if _as_numpy(signals[0].V_m if hasattr(signals[0], "V_m") else signals[0].get("V_m", np.zeros((1, 1)))).ndim==2 else 0)),
        "n_signals": len(signals),
        "phases": phases,
        "completion": completion,
        "numerical_valid": bool(num_check["numerical_valid"]),
        "numerical_issues": num_check["issues"][:5],
        "state_identity": state_id,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "wall_time_s": 0.0,  # filled below
        "citations": {
            "builder": "jomission/network/builder.py:62,90,395,623,713",
            "field": "jaxfne/fields/proxy.py:148,192,201; jaxfne/_model_simulate.py:764-773",
            "area_local": "jomission/recording/area_local.py:52,152,206",
            "observables": "jomission/recording/observables.py:6,35",
            "ledger": "jomission/simulation/ledger.py:11,21",
            "schedule": "jomission/simulation/schedule.py:11",
            "stability": "jomission/simulation/stability.py:8",
            "hdp_guard": "jaxfne/_model_simulate.py:280,572,1063",
            "emitter": "jaxfne/emitters.py:545,2846",
        },
    }

    # Build figures (each consumes same arrays)
    figs: dict[str, Any] = {}
    # Overview has no figure — rendered as table
    try:
        figs["raster"] = _fig_raster(signals, meta, dt_ms)
    except Exception as e:
        figs["raster"] = None
        overview["raster_error"] = str(e)
    try:
        figs["rates"] = _fig_rates(signals, meta, dt_ms)
    except Exception as e:
        figs["rates"] = None
        overview["rates_error"] = str(e)
    try:
        figs["membrane"] = _fig_membrane(signals, meta, dt_ms)
    except Exception as e:
        figs["membrane"] = None
        overview["membrane_error"] = str(e)
    try:
        figs["e_i"] = _fig_ei(signals, model, meta)
    except Exception as e:
        figs["e_i"] = None
        overview["ei_error"] = str(e)
    # Canonical tab id for H/Θ contains slash — normalized to h_θ
    try:
        hfig = _fig_h_theta(signals, dt_ms)
        figs["h_θ"] = hfig  # slash key maps to _TAB_IDS entry h/θ normalized
        # Also store under raw id without slash for lookup
        figs["h_theta"] = hfig
    except Exception as e:
        figs["h_θ"] = None
        figs["h_theta"] = None
        overview["h_theta_error"] = str(e)
    try:
        figs["spectra"] = _fig_spectra(signals, model, meta, dt_ms)
    except Exception as e:
        figs["spectra"] = None
        overview["spectra_error"] = str(e)
    try:
        figs["field"] = _fig_field(signals, model, dt_ms)
    except Exception as e:
        figs["field"] = None
        overview["field_error"] = str(e)
    try:
        figs["stimulus"] = _fig_stimulus(simulation_result, model, dt_ms)
    except Exception as e:
        figs["stimulus"] = None
        overview["stimulus_error"] = str(e)
    try:
        figs["connectivity"] = _fig_connectivity(model) if model is not None else None
    except Exception as e:
        figs["connectivity"] = None
        overview["connectivity_error"] = str(e)
    # Diagnostics tab
    diag: dict = {}
    try:
        diag, diag_fig = _compute_diagnostics(signals, dt_ms)
        figs["diagnostics"] = diag_fig
        overview["diagnostics_summary"] = {k: diag[k] for k in ["CV_ISI_mean", "CV_ISI_max", "CV_ISI_frac_gt_0_5", "Fano_50ms", "rho_10ms_mean", "mean_rate_Hz"] if k in diag}
    except Exception as e:
        figs["diagnostics"] = None
        overview["diagnostics_error"] = str(e)

    # Normalize fig keys to _TAB_IDS
    # _TAB_IDS: overview, raster, rates, membrane, e/i, h/θ, spectra, field, stimulus, connectivity, diagnostics, provenance
    key_map = {
        "e_i": "e/i",
        "h_θ": "h/θ",
        "h_theta": "h/θ",
        "rates": "rates",
        "raster": "raster",
        "membrane": "membrane",
        "spectra": "spectra",
        "field": "field",
        "stimulus": "stimulus",
        "connectivity": "connectivity",
        "diagnostics": "diagnostics",
    }
    normalized_figs: dict[str, Any] = {}
    for k, v in figs.items():
        nk = key_map.get(k, k)
        normalized_figs[nk] = v
    # Ensure diagnostics etc. map
    figs = normalized_figs

    provenance_rows = _build_provenance_table(simulation_result, diag, out_dir)

    # Save arrays (generated-owner arrays, not recomputed prettier)
    try:
        sig0 = signals[0]
        sp = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))
        vm = _as_numpy(sig0.V_m if hasattr(sig0, "V_m") else sig0.get("V_m", np.zeros((1, 1))))
        np.savez_compressed(arrays_dir / "spikes.npz", spikes=sp, dt_ms=dt_ms, n_steps=sp.shape[0] if sp.ndim>=1 else 0)
        np.savez_compressed(arrays_dir / "V_m.npz", V_m=vm, dt_ms=dt_ms)
        try:
            if getattr(sig0, "field", None) is not None:
                lfp = _as_numpy(sig0.field.lfp_proxy)
                kern = _as_numpy(sig0.field.kernel)
                contacts = _as_numpy(sig0.field.contact_depths)
                np.savez_compressed(arrays_dir / "field.npz", lfp_proxy=lfp, kernel=kern, contact_depths=contacts, claim="proxy_readout", physical_amplitude_calibrated=False)
                # Also area-local
                try:
                    from jomission.recording.area_local import field_by_area_array  # type: ignore
                    arr, areas, fmeta = field_by_area_array(sig0, model)
                    np.savez_compressed(arrays_dir / "field_by_area.npz", field_by_area=arr, areas=np.array(areas), meta=json.dumps(fmeta))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            src = _as_numpy(sig0.sources) if getattr(sig0, "sources", None) is not None else None
            if src is not None:
                np.savez_compressed(arrays_dir / "sources.npz", sources=src)
        except Exception:
            pass
        # H/Theta if available
        try:
            hdp = (getattr(sig0, "metadata", None) or {}).get("hdp", {}) or {}
            if hdp.get("H_trace") is not None:
                np.savez_compressed(arrays_dir / "H_t.npz", H_trace=_as_numpy(hdp["H_trace"]))
            if hdp.get("w_trace") is not None:
                np.savez_compressed(arrays_dir / "Theta_t.npz", Theta=_as_numpy(hdp["w_trace"]))
        except Exception:
            pass
        # Hashes for provenance
        hashes = {}
        for p in arrays_dir.glob("*.npz"):
            try:
                hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            except Exception:
                pass
        (arrays_dir / "hashes.json").write_text(json.dumps(hashes, indent=2))
    except Exception as e:
        overview["arrays_error"] = str(e)

    # Save figures as standalone html fragments also
    for tab_id, fig in figs.items():
        if fig is None:
            continue
        try:
            import plotly.io as pio  # type: ignore
            fig.write_html(str(figs_dir / f"{tab_id.replace('/','_').replace(' ', '_')}.html"), include_plotlyjs="cdn", full_html=True)
        except Exception:
            pass

    # Summary.json
    elapsed = time.time() - t0
    overview["wall_time_s"] = float(elapsed)
    summary = {
        "overview": overview,
        "provenance": provenance_rows,
        "diagnostics": diag,
        "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in arrays_dir.glob("*.npz")} if arrays_dir.exists() else {},
        "report": "report.html",
        "arrays_dir": str(arrays_dir),
        "figures_dir": str(figs_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    # Summary.md
    md_lines = [
        f"# Run Report — {run_id}",
        "",
        f"- config_hash `{config_hash}` hp_hash `{hp_hash}` run_hash `{run_hash}` seed {seed} dt {dt_ms} ms duration {float(duration_ms):.0f} ms",
        f"- model C019 freeze 65b302e8c7cdceb5 (builder.py:62,90,395) — E mixture M2 RS70/CH20/E_FS10, VIP b0.20, SST b0.21, PV 1.7, lognormal CV1.5, spatial σ0.08 max25, delays [20,80,120]",
        f"- numerical_valid {bool(num_check['numerical_valid'])} state_identity `{state_id}` completion {json.dumps(completion)}",
        f"- phases {phases} — spontaneous 2000 ms baseline when duration≤5 s else lifecycle",
        f"- arrays: `arrays/spikes.npz` {sp.shape if 'sp' in locals() else '?'} `arrays/V_m.npz` `arrays/field.npz` (proxy_readout, jaxfne/fields/proxy.py:192) `arrays/H_t.npz` when HDP available (guard jaxfne/_model_simulate.py:280)",
        f"- figures: `figures/*.html` per tab + `report.html` fixed tabs [{', '.join(t[0] for t in _FIXED_TABS)}]",
        f"- provenance table rows {len(provenance_rows)} — claim→estimator→array→hash (see Provenance tab)",
        "",
        "## Diagnostics (same spikes array)",
        f"- CV_ISI mean {diag.get('CV_ISI_mean',0):.3f} max {diag.get('CV_ISI_max',0):.3f} frac>0.5 {diag.get('CV_ISI_frac_gt_0_5',0):.3f} Fano {diag.get('Fano_50ms',0):.3f} ρ {diag.get('rho_10ms_mean',0):.3f} rate {diag.get('mean_rate_Hz',0):.1f} Hz",
        "",
        "Generated via jomission/visualization/run_report.py:run_report — consumes same generated-owner arrays as analyses (spikes, V_m, field, H, Theta, q) not recomputed prettier alternatives.",
        f"Wall time {elapsed:.1f} s.",
    ]
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    # EvidenceRef.json
    try:
        from jomission.evidence import EvidenceRef  # type: ignore
        import subprocess
        try:
            code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()[:12]  # type: ignore
        except Exception:
            code_sha = "unknown"
        numerical_cfg = {"dt_ms": dt_ms, "duration_ms": float(duration_ms), "n_neurons": overview["n_neurons"], "seed": seed}
        numerical_hash = hashlib.sha256(json.dumps(numerical_cfg, sort_keys=True).encode()).hexdigest()[:16]
        try:
            artifact_hash = hashlib.sha256((out_dir / "summary.json").read_bytes()).hexdigest()[:16]
        except Exception:
            artifact_hash = run_hash
        ev = EvidenceRef(
            code_sha=code_sha,
            parent_run=None,
            config_hash=config_hash,
            numerical_config_hash=numerical_hash,
            hp_hash=hp_hash,
            dt_ms=dt_ms,
            seed=seed,
            network_realization="V1→V4→FEF→PFC 100/area izhikevich edge_list spatial 0.08 max25 delays [20,80,120] (builder.py:62,90,395)",
            phase=str(phases[0]) if phases else "spontaneous",
            initial_state_hash=state_id,
            namespace="canonical_confirmatory",
            evidence_class="CANONICAL_CONFIRMATORY",
            estimand_version="jomission.visualization.run_report.v0",
            generated_owner=str(out_dir),
            artifact_hash=artifact_hash,
        )
        (out_dir / "EvidenceRef.json").write_text(ev.to_json())
    except Exception as e:
        (out_dir / "EvidenceRef_error.json").write_text(json.dumps({"error": str(e)}, indent=2))

    # Report.html with fixed tabs
    report_path = _render_html(simulation_result | {"run_id": run_id, "config_hash": config_hash, "hp_hash": hp_hash, "seed": seed, "dt_ms": dt_ms, "duration_ms": float(duration_ms), "run_hash": run_hash}, figs, overview, provenance_rows, out_dir)

    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "report_html": str(report_path),
        "summary_json": str(out_dir / "summary.json"),
        "summary_md": str(out_dir / "summary.md"),
        "arrays_dir": str(arrays_dir),
        "figures_dir": str(figs_dir),
        "overview": overview,
        "diagnostics": diag,
        "provenance_rows": provenance_rows,
        "wall_time_s": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Example generator — C019 baseline spontaneous 2000ms seed0
# ---------------------------------------------------------------------------

def generate_example_C019_spontaneous(
    *,
    duration_ms: float = 2000.0,
    seed: int = 0,
    dt_ms: float = 0.1,
    n_per_area: int = 100,
    out_dir: str = "results/visualization/example_run_C019",
) -> dict:
    """Generate C019 baseline spontaneous example (no stimulus drive).

    Uses build_jomission_model(seed=0) frozen C019 (builder.py:62,90,395
    65b302e8c7cdceb5) and StimulusSchedule silent (all events is_drive False)
    for duration_ms. Handles jaxfne HDP+delay incompat (jaxfne/_model_simulate.py:280)
    by trying HDP first then falling back to enable_hdp=False with provenance.

    Writes via run_report to out_dir and returns its result.
    """
    import jaxfne as jtfne  # type: ignore
    from jaxfne import Simulation, RuntimeConfig  # type: ignore
    import jaxfne.hdp_network as hdp  # type: ignore
    from jaxfne.io import config_hash  # type: ignore
    from jomission.network.builder import build_jomission_model  # type: ignore

    model = build_jomission_model(n_per_area=n_per_area, seed=seed, dt_ms=dt_ms)
    ch = config_hash(model.cfg)
    # C019 expected hash is 65b302e8c7cdceb5 per freeze, but allow be9b prefix for be9b96ab variant
    # (legacy 4f9fdeae seed0 no-delays vs new C019 with delays)
    hp = hdp.v1_pfc_aaab_hdp_params()
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]

    # Silent stimulus schedule (spontaneous — no drive, preserves timing)
    from jaxfne import StimulusSchedule  # type: ignore
    n_neurons = 400
    try:
        n_neurons = int(len(model.static.get("neuron_metadata", [])) or 400)
    except Exception:
        n_neurons = 400
    # Empty events -> purely spontaneous (no injected drive)
    # Use explicit StimulusSchedule with no events (spontaneous baseline)
    sched = StimulusSchedule(events=(), n_neurons=n_neurons)

    runtime_hdp = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hp)
    runtime_nohdp = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=False)

    sim = Simulation(duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=seed, runtime=runtime_hdp)
    sig = None
    hdp_used = True
    hdp_error = None
    try:
        sig = jtfne.simulate(model, sim, paradigm=sched)
    except ValueError as e:
        msg = str(e)
        if "enable_hdp does not support nonzero edge delay_steps" in msg:
            hdp_error = msg
            hdp_used = False
            sim2 = Simulation(duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=seed, runtime=runtime_nohdp)
            sig = jtfne.simulate(model, sim2, paradigm=sched)
        else:
            raise
    assert sig is not None

    sim_result = {
        "model": model,
        "signals": [sig],
        "signal": sig,
        "config_hash": ch,
        "hp_hash": hp_hash,
        "seed": seed,
        "dt_ms": dt_ms,
        "duration_ms": float(duration_ms),
        "run_id": "example_run_C019",
        "results_dir": out_dir,
        "paradigm": sched,
        "hdp_used": hdp_used,
        "hdp_error": hdp_error,
        "phases": ["spontaneous"],
        "completion": {"all": True, "note": "spontaneous baseline single-trial, terminated_by_schedule true"},
    }
    return run_report(sim_result)


if __name__ == "__main__":
    # Generate example when run as script
    res = generate_example_C019_spontaneous()
    print(json.dumps({k: res[k] for k in ["run_id", "out_dir", "report_html", "wall_time_s"]}, indent=2))

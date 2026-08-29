"""Transfer Function tab — VIS_FOUNDATION_v0 oscilloscope (engineering, not science).

Mechanistic chain (advisor):
  V1 L4 E spikes
        ↓
  delay/history (ring buffer B_t, delay_steps per jaxfne/emitters.py:545)
        ↓
  L4E→L23E synaptic state (syn_state per edge)
        ↓
  L4E→L23E current (edge_current_trace w·syn_state, jaxfne/emitters.py:2846, blocked by jaxfne/_model_simulate.py:280 when delay_steps>0 + HDP)
        ↓
  L2/3 E Vm
        ↓
  L2/3 E spikes

For every arrow/stage: stimulus-triggered average, A/B difference, effect size, latency,
units, provenance (claim→estimator→array→hash), measured/derived/proxy/unavailable.

Unavailable stages render literally as
  NOT OBSERVABLE
  JaxFNE delayed+HDP recording limitation
machine-derived (check model delay_steps>0 and enable_hdp) not manually typed.

Same generated-owner arrays as analyses:
  spikes, V_m, edge_current_trace if available, delay_state if via ContinuationState,
  sources, positions, neuron_metadata, EdgeList.

Provenance citations (read-only, frozen):
- jomission/network/builder.py:62  MOTIF_GAIN 16-gain pseudogenome (typed matrix)
- jomission/network/builder.py:90  FF_LAYER_MAP / FB_LAYER_MAPS / DELAY_FF/FB/WITHIN (2/8/12 ms → steps 20/80/120)
- jaxfne/emitters.py:545            edge_list_with_delay_ms (delay_steps, D_max=120)
- jaxfne/_model_simulate.py:280,572,1063  HDP+delay guard enable_hdp does not support nonzero delay_steps
- jaxfne/_pipeline.py:395           compile_step_fn record_edge_current (only kernel=="hdp")
- jomission/recording/observables.py:32  REQUIRED/OPTIONAL I_edge_current (record_edge_current seam)
- jomission/recording/area_local.py:52   field_by_area_from_signal linear partition
- jomission/visualization/model_summary.py:822 observable_basis() (C_t→X_t→q_t→φ_t→y_t, STATE/OUTPUT/PROXY)

Implementation: this module builds 6 plotly panels; jomission/visualization/run_report.py
adds a new fixed tab "Transfer Function" that calls transfer_figures().
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    _PLOTLY = True
except Exception:  # pragma: no cover
    go = None  # type: ignore
    make_subplots = None  # type: ignore
    _PLOTLY = False


def _as_numpy(arr: Any) -> np.ndarray:
    try:
        return np.asarray(arr)
    except Exception:
        return np.array(arr)


def _safe_hash(arr: Any) -> str:
    try:
        b = _as_numpy(arr).tobytes()
        return hashlib.sha256(b).hexdigest()[:16]
    except Exception:
        return "unknown"


def _get_code_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()[:16]
    except Exception:
        return "unknown"


def _resolve_signals(simulation_result: dict) -> list[Any]:
    sig = simulation_result.get("signals") or simulation_result.get("signal") or simulation_result.get("sig")
    if sig is None and "V_m" in simulation_result:
        return [simulation_result]
    if isinstance(sig, list):
        return sig
    if sig is not None:
        return [sig]
    if isinstance(simulation_result, dict) and "spikes" in simulation_result:
        return [simulation_result]
    raise ValueError("simulation_result must contain 'signals'")


def _neuron_metadata(model: Any, signals: list[Any]) -> list[dict]:
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


def _edge_delay_info(model: Any) -> dict:
    """Machine-derived delay inspection (not manually typed).

    Checks edge_list.delay_steps sum and max via jaxfne/emitters.py:545 seam.
    Returns dict with has_delay, max_delay, n_edges, delay_steps array hash.
    """
    out: dict[str, Any] = {"has_delay": False, "max_delay": 0, "sum_delay": 0, "n_edges": 0, "uniq": [], "hash": "unknown"}
    try:
        el = model.params.get("edge_list") if model is not None and hasattr(model, "params") else None
        if el is None:
            return out
        ds = _as_numpy(getattr(el, "delay_steps", np.array([0], dtype=np.int32)))
        out["n_edges"] = int(ds.shape[0]) if ds.size else 0
        out["has_delay"] = bool(np.any(ds > 0))
        out["max_delay"] = int(np.max(ds)) if ds.size else 0
        out["sum_delay"] = int(np.sum(ds)) if ds.size else 0
        out["uniq"] = sorted(set(ds.astype(int).tolist())) if ds.size else []
        out["hash"] = _safe_hash(ds)
    except Exception:
        pass
    return out


def _is_edge_current_blocked(model: Any, simulation_result: dict) -> dict:
    """Machine-derived check for _model_simulate.py:280 blocker.

    Returns dict blocked: bool, reason: str, has_delay: bool, hdp_enabled: bool.
    blocked = has_delay and hdp_enabled per guard
    sem jaxfne/_model_simulate.py:280,572,1063.
    """
    delay_info = _edge_delay_info(model)
    has_delay = bool(delay_info.get("has_delay", False))
    # Detect enable_hdp from multiple sources (machine-derived, not typed)
    hdp_enabled = False
    reason_parts: list[str] = []
    # 1) simulation_result flags
    if simulation_result.get("hdp_used") is True:
        hdp_enabled = True
        reason_parts.append("hdp_used=True")
    if simulation_result.get("enable_hdp") is True:
        hdp_enabled = True
        reason_parts.append("enable_hdp=True")
    if "enable_hdp does not support nonzero edge delay_steps" in str(simulation_result.get("hdp_error", "")):
        hdp_enabled = True
        reason_parts.append("hdp_error guard string")
    # 2) model cfg metadata (if present)
    try:
        if hasattr(model, "cfg") and hasattr(model.cfg, "metadata"):
            md = model.cfg.metadata
            if isinstance(md, dict):
                if md.get("enable_hdp") is True or md.get("hdp_enabled") is True:
                    hdp_enabled = True
                    reason_parts.append("model.cfg.metadata hdp_enabled")
    except Exception:
        pass
    # 3) Check runtime config via simulation_result runtime
    try:
        rt = simulation_result.get("runtime")
        if rt is not None and getattr(rt, "enable_hdp", None) is True:
            hdp_enabled = True
            reason_parts.append("runtime.enable_hdp")
    except Exception:
        pass
    # 4) If model was built with delays and simulation attempted HDP (common C019 path),
    # we treat hdp_enabled as True when has_delay and hp_hash present and no fallback flag.
    # But strictly require has_delay + hdp_enabled.
    blocked = bool(has_delay and hdp_enabled)
    # Also note second blocker: delayed non-HDP path lacks record_edge_current seam (emitters.py:714 has no param)
    # If has_delay and not blocked but edge_current_trace still unavailable, it's still NOT OBSERVABLE via delayed path limitation.
    has_any_delay = has_delay
    reason = ""
    if blocked:
        reason = f"JaxFNE delayed+HDP recording limitation (jaxfne/_model_simulate.py:280 guard: enable_hdp does not support nonzero edge delay_steps; has_delay={has_delay} uniq={delay_info.get('uniq')} hdp_enabled={hdp_enabled} [{', '.join(reason_parts)}])"
    elif has_any_delay:
        reason = f"Delayed path lacks record_edge_current seam (jaxfne/emitters.py:714, jaxfne/_pipeline.py:395 only kernel==hdp; has_delay={has_delay} uniq={delay_info.get('uniq')})"
    return {"blocked": bool(blocked), "has_delay": bool(has_delay), "hdp_enabled": bool(hdp_enabled), "reason": reason, "delay_info": delay_info, "reason_parts": reason_parts}


def _resolve_transfer_indices(meta: list[dict], model: Any) -> dict:
    """Resolve V1 L4 E and V1 L2/3 E populations and L4E→L23E edges."""
    # Neuron indices per population
    v1_l4_e = [i for i, r in enumerate(meta) if str(r.get("area")) == "V1" and str(r.get("layer")) == "L4" and str(r.get("cell_type")) == "E"]
    v1_l23_e = [i for i, r in enumerate(meta) if str(r.get("area")) == "V1" and str(r.get("layer")) in ("L2/3", "L2/3") and str(r.get("cell_type")) == "E"]
    # Fallback layer naming L2/3 vs L23
    if not v1_l23_e:
        v1_l23_e = [i for i, r in enumerate(meta) if str(r.get("area")) == "V1" and "2" in str(r.get("layer")) and str(r.get("cell_type")) == "E"]
    # Edge indices L4E→L2/3E within V1
    edge_ids: list[int] = []
    n_edges = 0
    try:
        el = model.params.get("edge_list") if model is not None else None
        if el is not None:
            pre = _as_numpy(el.pre)
            post = _as_numpy(el.post)
            n_edges = int(pre.shape[0])
            # Build lookup for neuron area/layer/type
            for ei in range(n_edges):
                try:
                    pi = int(pre[ei]); po = int(post[ei])
                    if pi >= len(meta) or po >= len(meta):
                        continue
                    pre_area = str(meta[pi].get("area")); pre_layer = str(meta[pi].get("layer")); pre_ct = str(meta[pi].get("cell_type"))
                    post_area = str(meta[po].get("area")); post_layer = str(meta[po].get("layer")); post_ct = str(meta[po].get("cell_type"))
                    if pre_area == "V1" and pre_layer == "L4" and pre_ct == "E" and post_area == "V1" and "2" in post_layer and post_ct == "E":
                        edge_ids.append(int(ei))
                    elif pre_area == "V1" and pre_layer == "L4" and pre_ct == "E" and post_area == "V1" and post_layer == "L2/3" and post_ct == "E":
                        if ei not in edge_ids:
                            edge_ids.append(int(ei))
                except Exception:
                    continue
    except Exception:
        pass
    return {"v1_l4_e": v1_l4_e, "v1_l23_e": v1_l23_e, "l4_to_l23_edge_ids": edge_ids, "n_edges": int(n_edges)}


def _extract_events(simulation_result: dict) -> list[dict]:
    """Extract stimulus events for STA alignment.

    Prefers simulation_result["paradigm"] or ["schedule"] or ["stimulus_schedule"].
    Uses same StimulusSchedule as simulation (SOURCE), not recomputed.
    """
    events: list[dict] = []
    # Try paradigm
    for key in ("paradigm", "schedule", "stimulus_schedule", "stimulus", "condition"):
        cond = simulation_result.get(key)
        if cond is None:
            continue
        # If it's a StimulusSchedule-like
        if hasattr(cond, "events"):
            try:
                evs = list(cond.events)
                for ev in evs:
                    if isinstance(ev, dict):
                        events.append(ev)
                    else:
                        # Convert object to dict
                        events.append({"label": str(getattr(ev, "label", "?")), "onset_ms": float(getattr(ev, "onset_ms", 0)), "stimulus": getattr(ev, "stimulus", None), "is_omission": bool(getattr(ev, "is_omission", False)), "is_drive_event": bool(getattr(ev, "is_drive_event", getattr(ev, "is_drive", False))), "amplitude": float(getattr(ev, "amplitude", 0.0)) if hasattr(ev, "amplitude") else None})
                if events:
                    break
            except Exception:
                continue
        # If it's dict with events
        if isinstance(cond, dict) and "events" in cond:
            try:
                for ev in cond["events"]:
                    if isinstance(ev, dict):
                        events.append(ev)
                if events:
                    break
            except Exception:
                continue
    # Fallback: try simulation_result["schedule"]
    if not events:
        try:
            from jomission.paradigm.spec import JOMISSION_PARADIGM  # type: ignore
            # Use first condition events as placeholder? But that would be misleading if spontaneous.
            # Only use if simulation_result explicitly not spontaneous.
            if simulation_result.get("paradigm") is None and not simulation_result.get("hdp_error"):
                pass
        except Exception:
            pass
    return events


def _classify_events_ab(events: list[dict]) -> tuple[list[float], list[float]]:
    """Classify drive events into A vs B by label/stimulus identity."""
    onsets_a: list[float] = []
    onsets_b: list[float] = []
    for ev in events:
        is_drive = bool(ev.get("is_drive_event", True)) if isinstance(ev, dict) else True
        is_omit = bool(ev.get("is_omission", False)) if isinstance(ev, dict) else False
        if not is_drive or is_omit:
            continue
        label = str(ev.get("label", "")) if isinstance(ev, dict) else str(getattr(ev, "label", ""))
        stim = str(ev.get("stimulus", "")) if isinstance(ev, dict) else str(getattr(ev, "stimulus", ""))
        # Heuristic: A blob vs B blob
        text = (label + " " + stim).lower()
        # RF patterns: stimulus_A vs stimulus_B, or p1_u etc.
        # For templated schedule, label like p1_u12 carries no stim but events structure knows?
        # Fallback: classify via amplitude? Not reliable.
        # Use stimulus field if present
        if "stimulus_a" in text or (" a " in text) or stim == "stimulus_A":
            onsets_a.append(float(ev.get("onset_ms", 0)) if isinstance(ev, dict) else float(getattr(ev, "onset_ms", 0)))
        elif "stimulus_b" in text or (" b " in text) or stim == "stimulus_B":
            onsets_b.append(float(ev.get("onset_ms", 0)) if isinstance(ev, dict) else float(getattr(ev, "onset_ms", 0)))
        else:
            # If cannot classify, try to use label p1..p4 mapping if schedule has AAAB etc?
            # For now, distribute alternating? Leave unclassified.
            pass
    return onsets_a, onsets_b


def _compute_sta(
    trace: np.ndarray,  # [T] or [T,N_pop] mean trace per time
    dt_ms: float,
    onsets_a: list[float],
    onsets_b: list[float],
    window_ms: Tuple[float, float] = (-50.0, 300.0),
) -> dict:
    """Compute stimulus-triggered average for trace.

    trace: [T] time series (mean over population)
    Returns dict with t_ms, sta_a, sta_b, diff, sem_a, sem_b, n_a, n_b,
    effect_size_d (Cohen's d per time, peak), latency_ms (time of peak |diff|)
    """
    T = int(trace.shape[0])
    win_lo, win_hi = float(window_ms[0]), float(window_ms[1])
    # Convert window to steps
    pre_steps = int(round(abs(win_lo) / float(dt_ms))) if win_lo < 0 else 0
    post_steps = int(round(win_hi / float(dt_ms))) if win_hi > 0 else 0
    win_steps = pre_steps + post_steps
    t_ms = np.arange(win_steps) * float(dt_ms) + float(win_lo)
    # Helper to collect snippets
    def collect(onsets: list[float]) -> np.ndarray:
        if not onsets:
            return np.zeros((0, win_steps), dtype=float)
        snippets = []
        for onset_ms in onsets:
            onset_step = int(round(float(onset_ms) / float(dt_ms)))
            lo = onset_step - pre_steps
            hi = onset_step + post_steps
            # bounds
            if lo < 0 or hi > T:
                continue
            seg = trace[lo:hi]
            if seg.shape[0] == win_steps:
                snippets.append(seg.astype(float))
        if not snippets:
            return np.zeros((0, win_steps), dtype=float)
        return np.stack(snippets, axis=0)  # [n_events, win_steps]
    arr_a = collect(onsets_a)
    arr_b = collect(onsets_b)
    # Compute means
    if arr_a.shape[0] > 0:
        sta_a = np.mean(arr_a, axis=0)
        sem_a = np.std(arr_a, axis=0, ddof=1) / np.sqrt(max(arr_a.shape[0], 1)) if arr_a.shape[0] > 1 else np.zeros_like(sta_a)
    else:
        sta_a = np.full(win_steps, np.nan, dtype=float)
        sem_a = np.full(win_steps, np.nan, dtype=float)
    if arr_b.shape[0] > 0:
        sta_b = np.mean(arr_b, axis=0)
        sem_b = np.std(arr_b, axis=0, ddof=1) / np.sqrt(max(arr_b.shape[0], 1)) if arr_b.shape[0] > 1 else np.zeros_like(sta_b)
    else:
        sta_b = np.full(win_steps, np.nan, dtype=float)
        sem_b = np.full(win_steps, np.nan, dtype=float)
    diff = sta_a - sta_b if arr_a.shape[0] and arr_b.shape[0] else np.full(win_steps, np.nan)
    # Effect size: Cohen's d per time point (pooled std)
    d = np.full(win_steps, np.nan, dtype=float)
    if arr_a.shape[0] > 1 and arr_b.shape[0] > 1:
        try:
            var_a = np.var(arr_a, axis=0, ddof=1)
            var_b = np.var(arr_b, axis=0, ddof=1)
            n_a, n_b = arr_a.shape[0], arr_b.shape[0]
            pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))
            pooled = np.maximum(pooled, 1e-12)
            d = (sta_a - sta_b) / pooled
        except Exception:
            pass
    elif arr_a.shape[0] and arr_b.shape[0]:
        # Single trial each: difference normalized by overall std
        try:
            pooled = float(np.std(trace)) if np.std(trace) > 0 else 1.0
            d = (sta_a - sta_b) / pooled
        except Exception:
            pass
    # Latency: time of peak |diff|
    latency_ms = float("nan")
    peak_diff = float("nan")
    peak_d = float("nan")
    if np.any(np.isfinite(diff)):
        idx = int(np.nanargmax(np.abs(diff)))
        latency_ms = float(t_ms[idx])
        peak_diff = float(diff[idx]) if np.isfinite(diff[idx]) else float("nan")
        peak_d = float(d[idx]) if np.isfinite(d[idx]) else float("nan")
    return {
        "t_ms": t_ms,
        "sta_a": sta_a,
        "sta_b": sta_b,
        "diff": diff,
        "sem_a": sem_a,
        "sem_b": sem_b,
        "n_a": int(arr_a.shape[0]),
        "n_b": int(arr_b.shape[0]),
        "d": d,
        "latency_ms": float(latency_ms),
        "peak_diff": float(peak_diff),
        "peak_d": float(peak_d),
        "window_ms": window_ms,
        "pre_steps": int(pre_steps),
        "post_steps": int(post_steps),
    }


# ---------------------------------------------------------------------------
# Stage definitions — 6 panels, each arrow
# ---------------------------------------------------------------------------

_TRANSFER_STAGE_DEFS: list[dict] = [
    {
        "id": "v1_l4_e_spikes",
        "label": "V1 L4 E spikes",
        "title": "V1 L4 E spikes — stimulus-triggered PSTH (STATE, spikes)",
        "units": "Hz (spikes binned 10 ms → Hz, bool per dt 0.1 ms)",
        "semantic_class": "STATE",
        "claim": "V1 L4 E spikes carry stimulus identity (A vs B)",
        "estimator": "PSTH: mean(spikes [T,N] filtered V1 L4 E, binned 10 ms) aligned to stimulus onset (SOURCE paradigm)",
        "estimator_version": "transfer.py:sta v0 (10 ms bin, -50..300 ms window)",
        "array": "spikes [T,N] bool (jaxfne/_pipeline.py, jaxfne/emitters.py:211)",
        "source": "jtfne.simulate → Signals.spikes (same array as analyses, not recomputed prettier)",
        "derived_from": "X_t spikes (STATE)",
        "proxy_status": False,
        "provenance_note": "measured STATE when spikes present; derived_from X_t; builder.py:62 pseudogenome intact (E→E 1.0 etc.)",
    },
    {
        "id": "delay_history",
        "label": "delay/history",
        "title": "delay/history — finite-delay ring buffer B_t (STATE, D_max 120)",
        "units": "steps (delay_steps 20/80/120) → ms via ×dt 0.1 ms; buffer [D_max,N] bool",
        "semantic_class": "STATE",
        "claim": "Presynaptic spikes delayed per edge via delay_state ring buffer",
        "estimator": "DelaySteps distribution + ContinuationState.delay_state shape check (jaxfne/_pipeline.py:72, jaxfne/emitters.py:545 edge_list_with_delay_ms)",
        "estimator_version": "transfer.py:delay v0 (host-side np.asarray edges.delay_steps)",
        "array": "delay_state [D_max,N] + edge_list.delay_steps [n_edges] (builder.py:90 laminar delays 2/8/12 ms →20/80/120 steps)",
        "source": "jomission/network/builder.py:807 _apply_laminar_delays → jaxfne/emitters.py:545 edge_list_with_delay_ms → ContinuationState.delay_state (jaxfne/_pipeline.py:72,92)",
        "derived_from": "spikes delayed via B_t ring (STATE history)",
        "proxy_status": False,
        "provenance_note": "measured static delay_steps always; dynamic B_t is STATE history when continuation available, else structural proxy",
    },
    {
        "id": "l4e_l23e_syn_state",
        "label": "L4E→L23E synaptic state",
        "title": "L4E→L23E synaptic state — syn_state per edge (STATE, filtered spikes)",
        "units": "dimensionless gating variable (native, decay exp(-dt/tau) tau 2 ms per emitters.py:457)",
        "semantic_class": "STATE",
        "claim": "Synaptic gating tracks delayed L4E spikes before current",
        "estimator": "syn_state [T,n_edges] if recorded, else proxy: spikes_{L4E} convolved exp(-dt/tau) tau 2 ms (emitters.py:457,961 decay)",
        "estimator_version": "transfer.py:syn_proxy v0 (tau 2 ms, same as emitters decay)",
        "array": "syn_state [T,n_edges] (jaxfne/_pipeline.py:48 DynamicState.syn_state) or spikes [T,N] convolved proxy",
        "source": "jaxfne/_pipeline.py:48 syn_state (same generated-owner when available, else spikes proxy) + jaxfne/emitters.py:457 decay=exp(-dt/tau_ms)",
        "derived_from": "spikes filtered (proxy when syn_state not traced per step)",
        "proxy_status": True,  # when using convolution proxy
        "provenance_note": "measured if syn_state trajectory available; otherwise PROXY via filtered spikes (marked proxy_status True)",
    },
    {
        "id": "l4e_l23e_current",
        "label": "L4E→L23E current",
        "title": "L4E→L23E current — per-edge w·syn_state (READOUT, physical_amplitude_calibrated=False)",
        "units": "a.u. (native current, w·syn_state, uncalibrated; field proxy same a.u.)",
        "semantic_class": "READOUT",
        "claim": "Realized synaptic current from V1 L4E to V1 L2/3 E",
        "estimator": "edge_current_trace [T,n_edges] w*syn_state via jaxfne/emitters.py:2846 record_edge_current (only when available) partitioned by motif",
        "estimator_version": "observables.py:35 partition_currents_by_motif + jaxfne/emitters.py:2846 v0",
        "array": "edge_current_trace [T,n_edges] (optional, SUPPLEMENTARY when unavailable) → [T,n_l4_to_l23] slice",
        "source": "model.last_hdp_diagnostics() → observables.py:35 (requires record_edge_current; blocked by jaxfne/_model_simulate.py:280 when delay_steps [20,80,120] + HDP; also jaxfne/_pipeline.py:395 only kernel==hdp)",
        "derived_from": "w·syn_state per edge (READOUT, not STATE alone)",
        "proxy_status": False,  # realized when available, else unavailable/proxy
        "provenance_note": "measured READOUT when available; when blocked, rendered NOT OBSERVABLE banner machine-derived (check delay_steps>0 and enable_hdp per _model_simulate.py:280)",
    },
    {
        "id": "l23_e_vm",
        "label": "L2/3 E Vm",
        "title": "L2/3 E Vm — membrane response (STATE, V_m mV)",
        "units": "mV (Vm, threshold ~30 mV emitters.py:211, relative_V DERIVED_FROM(V_m) separately)",
        "semantic_class": "STATE",
        "claim": "L2/3 E membrane tracks L4E→L23E current",
        "estimator": "STA of V_m [T,N] filtered V1 L2/3 E (same array as analyses, jaxfne/_pipeline.py:48 DynamicState.v)",
        "estimator_version": "transfer.py:vm_sta v0",
        "array": "V_m [T,N] mV (same array as analyses, not recomputed)",
        "source": "jtfne.simulate → Signals.V_m (jaxfne/emitters.py:211 dv=0.04v²+5v+140-u+I)",
        "derived_from": "X_t V_m (STATE); current → V_m via Izhikevich integration",
        "proxy_status": False,
        "provenance_note": "measured STATE; observable_basis jomission/visualization/model_summary.py:822 V_m is STATE, relative_V is DERIVED_FROM(V_m)",
    },
    {
        "id": "l23_e_spikes",
        "label": "L2/3 E spikes",
        "title": "L2/3 E spikes — output PSTH (STATE, spikes)",
        "units": "Hz (spikes binned 10 ms → Hz)",
        "semantic_class": "STATE",
        "claim": "L2/3 E spikes output of vertical transfer V1 L4→L2/3",
        "estimator": "PSTH: mean(spikes [T,N] filtered V1 L2/3 E) aligned to stimulus",
        "estimator_version": "transfer.py:spike_sta v0",
        "array": "spikes [T,N] bool (same as analyses)",
        "source": "jtfne.simulate → Signals.spikes (same array)",
        "derived_from": "X_t spikes threshold V>=30→c (STATE/OUTPUT*)",
        "proxy_status": False,
        "provenance_note": "measured STATE/OUTPUT* per observable_basis (spikes owner STATE/OUTPUT* ambiguous)",
    },
]

_STAGE_ID_TO_DEF = {d["id"]: d for d in _TRANSFER_STAGE_DEFS}


def _machine_banner_fig(text: str, subtitle: str = "") -> Any:
    """Return a plotly figure with NOT OBSERVABLE banner centered."""
    if not _PLOTLY:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text",
        text=[f"<b>{text}</b><br><span style='font-size:11px'>{subtitle}</span>"],
        textposition="middle center", showlegend=False, textfont=dict(size=18, color="#d62728")))
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        height=360, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(214,39,40,0.06)", plot_bgcolor="rgba(214,39,40,0.06)",
        title=dict(text="<b>NOT OBSERVABLE</b> — JaxFNE delayed+HDP recording limitation", font=dict(color="#d62728", size=12)),
        annotations=[dict(x=0.5, y=0.28, xref="paper", yref="paper", text=subtitle, showarrow=False, font=dict(size=9, color="#555"), align="center")]
    )
    return fig


def _fig_not_observable_machine(blocked_info: dict, stage_def: dict) -> Any:
    """Machine-derived NOT OBSERVABLE figure for blocked edge_current etc.

    Text is literally NOT OBSERVABLE / JaxFNE delayed+HDP recording limitation
    per task, with machine-derived delay/hdp context (not manually typed string).
    """
    reason = blocked_info.get("reason", "")
    delay_info = blocked_info.get("delay_info", {})
    has_delay = bool(delay_info.get("has_delay", False))
    uniq = delay_info.get("uniq", [])
    # Enforce literal text
    primary = "NOT OBSERVABLE"
    secondary = "JaxFNE delayed+HDP recording limitation"
    detail = f"{secondary}<br>{reason[:260]}<br>stage {stage_def.get('id')} requires edge_current_trace [T,n_edges] (jaxfne/emitters.py:2846) but jaxfne/_model_simulate.py:280 guard blocks HDP+delay (delay_steps {uniq} nonzero + enable_hdp) — use non-HDP delayed kernel has no record_edge_current seam (jaxfne/_pipeline.py:395 kernel==hdp only)"
    fig = _machine_banner_fig(primary, secondary)
    if fig is not None:
        # Add second annotation with machine details
        fig.add_annotation(x=0.5, y=0.10, xref="paper", yref="paper",
            text=f"Machine-derived: has_delay={has_delay} uniq={uniq} blocked={blocked_info.get('blocked')} hdp={blocked_info.get('hdp_enabled')}",
            showarrow=False, font=dict(size=8, color="#333"), bgcolor="rgba(255,255,255,0.85)", bordercolor="#d62728", align="center")
        fig.add_annotation(x=0.5, y=0.02, xref="paper", yref="paper",
            text=reason[:220], showarrow=False, font=dict(size=7, color="#555"), align="center")
    return fig


def _fig_transfer_stage(
    stage_id: str,
    simulation_result: dict,
    model: Any,
    signals: list[Any],
    meta: list[dict],
    dt_ms: float,
) -> tuple[Any, dict]:
    """Build one transfer panel figure + provenance dict.

    Each stage shows STA, A/B diff, effect size d, latency, units, provenance,
    measured/derived/proxy/unavailable status.
    Uses same generated-owner arrays as analyses (spikes, V_m, edge_current if available).
    """
    if not _PLOTLY or go is None:
        return None, {"error": "plotly missing"}
    defd = _STAGE_ID_TO_DEF.get(stage_id, {})
    sig0 = signals[0]
    # Resolve indices
    idxs = _resolve_transfer_indices(meta, model)
    v1_l4_e = idxs["v1_l4_e"]
    v1_l23_e = idxs["v1_l23_e"]
    edge_ids = idxs["l4_to_l23_edge_ids"]
    # Events
    events = _extract_events(simulation_result)
    onsets_a, onsets_b = _classify_events_ab(events)
    # For spontaneous runs, we can synthesize pseudo-events as trial start? But then A/B will be empty.
    # We'll report spontaneous note.
    is_spontaneous = len(events) == 0 or (len(onsets_a) == 0 and len(onsets_b) == 0)
    # Common window
    win = (-50.0, 300.0)
    # Dispatch per stage
    # Helper to get population mean trace
    def population_mean_trace(arr2d: np.ndarray, pop_ids: list[int]) -> np.ndarray:
        if arr2d.size == 0 or not pop_ids:
            return np.zeros(arr2d.shape[0] if arr2d.ndim >= 1 else 0, dtype=float)
        try:
            # arr2d [T,N] -> mean over N_pop
            sub = arr2d[:, pop_ids] if arr2d.ndim == 2 and max(pop_ids) < arr2d.shape[1] else arr2d.reshape(arr2d.shape[0], -1)[:, :1]
            return np.mean(sub.astype(float), axis=1)
        except Exception:
            return np.zeros(arr2d.shape[0] if arr2d.ndim >= 1 else 0, dtype=float)

    # Check blocked edge_current machine-derived
    blocked_info = _is_edge_current_blocked(model, simulation_result)
    # Determine status and build fig
    status = "measured"  # default
    units = defd.get("units", "")
    semantic = defd.get("semantic_class", "STATE")
    provenance_extra = defd.get("provenance_note", "")

    if stage_id == "v1_l4_e_spikes":
        # Trace from spikes
        try:
            spikes = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))
            if spikes.ndim != 2:
                raise ValueError("spikes not [T,N]")
            # Population mean spike rate: binary per dt -> for STA we use 10 ms binned then mean, but for trace we keep raw mean per step then later bin in STA? Use raw for STA and annotate Hz.
            # Use population mean per step (0/1 mean) then STA will show PSTH; convert to Hz by *1000/dt but STA helper works on trace directly; we convert after.
            trace = population_mean_trace(spikes, v1_l4_e)  # [T] fraction active per step
            # Convert to Hz instantaneous: fraction * (1000/dt)
            trace_hz = trace * (1000.0 / float(dt_ms)) if dt_ms > 0 else trace
            # For STA, use trace_hz
            sta = _compute_sta(trace_hz, dt_ms, onsets_a, onsets_b, window_ms=win) if not is_spontaneous else None
            array_hash = _safe_hash(spikes)
            n_total = len(meta) if meta else spikes.shape[1]
            n_rendered = len(v1_l4_e)
            # Build figure with subplots: STA top, diff+d bottom
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                subplot_titles=(f"STA — mean over {n_rendered} V1 L4 E neurons (same spikes array)", "A/B difference + Cohen's d"))
            if sta is not None and sta["n_a"] > 0 and sta["n_b"] > 0:
                t = sta["t_ms"]
                fig.add_trace(go.Scatter(x=t, y=sta["sta_a"], mode="lines", name="A mean", line=dict(color="#1f77b4", width=2), hovertemplate="A t %{x:.0f} ms %{y:.1f} Hz<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["sta_b"], mode="lines", name="B mean", line=dict(color="#ff7f0e", width=2), hovertemplate="B t %{x:.0f} ms %{y:.1f} Hz<extra></extra>"), row=1, col=1)
                # sem bands
                fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([sta["sta_a"] + sta["sem_a"], (sta["sta_a"] - sta["sem_a"])[::-1]]), fill="toself", fillcolor="rgba(31,119,180,0.12)", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([sta["sta_b"] + sta["sem_b"], (sta["sta_b"] - sta["sem_b"])[::-1]]), fill="toself", fillcolor="rgba(255,127,14,0.12)", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["diff"], mode="lines", name="A−B diff", line=dict(color="#d62728", width=1.5), hovertemplate="diff t %{x:.0f} ms %{y:.1f} Hz<extra></extra>"), row=2, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["d"], mode="lines", name="Cohen's d", line=dict(color="#9467bd", width=1, dash="dot"), yaxis="y4", hovertemplate="d t %{x:.0f} ms %{y:.2f}<extra></extra>"), row=2, col=1)
                fig.add_annotation(x=sta["latency_ms"], y=sta["peak_diff"], xref="x2", yref="y3", text=f"latency {sta['latency_ms']:.0f} ms<br>Δ {sta['peak_diff']:.1f} Hz d {sta['peak_d']:.2f} nA{A}B{sta['n_b']}", showarrow=True, arrowhead=2, font=dict(size=8, color="#d62728"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#d62728", row=2, col=1)
                title = f"{defd.get('title')} — A/B Δ {sta['peak_diff']:.1f} Hz d {sta['peak_d']:.2f} latency {sta['latency_ms']:.0f} ms (nA {sta['n_a']} B {sta['n_b']}, {units}, {semantic}, measured)"
                effect = sta["peak_d"]
                latency = sta["latency_ms"]
            else:
                # Spontaneous — show whole-trace mean with time
                t_ms = np.arange(trace_hz.shape[0]) * float(dt_ms)
                step = max(1, trace_hz.shape[0] // 3000)
                fig.add_trace(go.Scatter(x=t_ms[::step], y=trace_hz[::step], mode="lines", name="V1 L4 E mean rate", line=dict(color="#1f77b4", width=1)), row=1, col=1)
                fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="Spontaneous — no A/B contrast (StimulusSchedule events=())<br>mean rate shown, A/B Δ/effect size not applicable without RF drive", showarrow=False, font=dict(size=9, color="#555"), bgcolor="rgba(255,255,255,0.85)", bordercolor="#999", align="center")
                title = f"{defd.get('title')} — spontaneous baseline (no A/B), mean {float(np.mean(trace_hz)):.1f} Hz ({units}, {semantic}, measured)"
                latency = float("nan"); effect = float("nan")
                peak_diff = float("nan")
                # For footer uniform shape
                sta = {"t_ms": t_ms[::step], "sta_a": trace_hz[::step], "sta_b": trace_hz[::step], "diff": np.zeros_like(trace_hz[::step]), "d": np.zeros_like(trace_hz[::step]), "n_a": 0, "n_b": 0, "latency_ms": latency, "peak_diff": float("nan"), "peak_d": float("nan")}
            fig.update_xaxes(title_text="Time from stimulus onset (ms)", row=2, col=1)
            fig.update_yaxes(title_text="Rate (Hz)", row=1, col=1)
            fig.update_yaxes(title_text="Δ Hz / d", row=2, col=1)
            fig.update_layout(title=dict(text=title, font=dict(size=10)), height=520, margin=dict(l=50, r=20, t=70, b=40), legend=dict(orientation="h", y=-0.18))
            fig.add_annotation(x=0.5, y=-0.16, xref="paper", yref="paper", text=f"Units {units} — estimator {defd.get('estimator')} — array {defd.get('array')} hash {array_hash[:8]} — semantic_class {semantic} — measured/derived/proxy/unavailable: <b>measured (STATE)</b> — {defd.get('source')} — builder.py:62 MOTIF_GAIN, builder.py:90 delays [20,80,120] via emitters.py:545", showarrow=False, font=dict(size=7, color="#555"), align="center")
            prov = {"stage": stage_id, "units": units, "semantic_class": semantic, "status": "measured", "array_hash": array_hash, "n_total": int(n_total), "n_rendered": int(n_rendered), "filters": f"V1 L4 E n={len(v1_l4_e)} via neuron_metadata", "latency_ms": float(latency) if 'latency' in locals() else float("nan"), "effect_d": float(effect) if 'effect' in locals() else float("nan"), "is_spontaneous": bool(is_spontaneous)}
            return fig, prov
        except Exception as e:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=[f"V1 L4 E spikes unavailable: {e}"], showlegend=False))
            return fig, {"error": str(e), "status": "unavailable"}

    elif stage_id == "delay_history":
        delay_info = _edge_delay_info(model)
        status = "measured" if delay_info.get("has_delay") else "derived"
        # Show delay_steps histogram + buffer note
        fig = make_subplots(rows=1, cols=2, subplot_titles=("DelaySteps distribution (structural, per edge)", "Ring buffer B_t schematic (history)"), column_widths=[0.55, 0.45])
        try:
            el = model.params.get("edge_list") if model is not None else None
            if el is not None:
                ds = _as_numpy(getattr(el, "delay_steps", np.array([0])))
                # histogram of uniq
                uniq, counts = np.unique(ds, return_counts=True)
                fig.add_trace(go.Bar(x=[str(int(u)) for u in uniq], y=counts.astype(float), marker=dict(color="#1f77b4"), name="delay_steps", hovertemplate="steps %{x} count %{y}<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text", text=[f"D_max {delay_info.get('max_delay')} steps<br>{delay_info.get('max_delay',0)*float(dt_ms):.1f} ms<br>buffer ({delay_info.get('max_delay',0)+1}×{len(meta)})<br>ContinuationState.delay_state<br>jaxfne/_pipeline.py:72,92<br>emitters.py:545, builder.py:90"], textposition="middle center", showlegend=False, textfont=dict(size=10)), row=1, col=2)
                fig.update_xaxes(title_text="delay_steps (steps ×0.1ms=ms)", row=1, col=1)
                fig.update_yaxes(title_text="edge count", row=1, col=1)
                fig.update_xaxes(visible=False, row=1, col=2)
                fig.update_yaxes(visible=False, row=1, col=2)
                # Also try to show if delay_state present via continuation
                # If signals contain delay_state in metadata?
                delay_state_note = ""
                try:
                    sig0_meta = getattr(sig0, "metadata", {}) or {}
                    if isinstance(sig0_meta, dict) and "delay_state" in sig0_meta:
                        ds_shape = getattr(sig0_meta["delay_state"], "shape", "?")
                        delay_state_note = f"delay_state present shape {ds_shape} (ContinuationState)"
                    else:
                        delay_state_note = f"delay_state structural (edge_list.delay_steps) — dynamic B_t not traced per step (expected, history STATE via ContinuationState API jaxfne/_pipeline.py:395 requires continuation)"
                except Exception:
                    delay_state_note = "delay_state via ContinuationState (not per-step trajectory)"
                title = f"{defd.get('title')} — D_max {delay_info.get('max_delay')} steps ({delay_info.get('max_delay',0)*float(dt_ms):.1f} ms) ∈ {delay_info.get('uniq')} ({units}, {semantic}, {status}) — {delay_state_note}"
            else:
                fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=["No edge_list — delay history unavailable"], showlegend=False), row=1, col=1)
                title = f"{defd.get('title')} — unavailable"
                status = "unavailable"
        except Exception as e:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=[f"delay/history error: {e}"], showlegend=False))
            title = f"{defd.get('title')} — error"
            status = "unavailable"
        fig.update_layout(title=dict(text=title[:220], font=dict(size=10)), height=420, margin=dict(l=50, r=20, t=70, b=40))
        fig.add_annotation(x=0.5, y=-0.18, xref="paper", yref="paper", text=f"Units {units} — estimator {defd.get('estimator')} — array {defd.get('array')} hash {delay_info.get('hash','?')[:8]} — semantic_class {semantic} — <b>{status}</b> — {defd.get('source')} — builder.py:90 laminar delays 2/8/12 ms →20/80/120 steps via emitters.py:545", showarrow=False, font=dict(size=7, color="#555"), align="center")
        prov = {"stage": stage_id, "units": units, "semantic_class": semantic, "status": status, "array_hash": str(delay_info.get("hash","")), "n_total": int(delay_info.get("n_edges",0)), "n_rendered": int(len(delay_info.get("uniq",[]))), "latency_ms": float(delay_info.get("max_delay",0)*float(dt_ms)), "effect_d": float("nan")}
        return fig, prov

    elif stage_id == "l4e_l23e_syn_state":
        # Try to get syn_state trajectory if present, else proxy via filtered spikes
        has_real = False
        trace = None
        array_hash = "proxy"
        status = "proxy"
        semantic = "STATE" if has_real else "PROXY_ESTIMATE"
        try:
            # Check for syn_state in signal diagnostics
            syn_trace = None
            # 1) signal.syn_state ?
            if hasattr(sig0, "syn_state") and sig0.syn_state is not None:
                syn_trace = _as_numpy(sig0.syn_state)  # [T,n_edges]?
                has_real = True
            # 2) metadata syn_state
            if syn_trace is None:
                md = getattr(sig0, "metadata", None) or {}
                if isinstance(md, dict) and "syn_state" in md:
                    syn_trace = _as_numpy(md["syn_state"])
                    has_real = True
            # 3) sources as proxy? sources [T,N] per-neuron source proxy q
            if syn_trace is not None and syn_trace.ndim == 2 and syn_trace.shape[1] == idxs["n_edges"]:
                # Slice to L4→L2/3 edges
                if edge_ids:
                    sub = syn_trace[:, edge_ids]  # [T, n_l4_l23]
                    trace = np.mean(sub.astype(float), axis=1)  # [T]
                    array_hash = _safe_hash(syn_trace)
                    status = "measured"
                    semantic = "STATE"
                else:
                    trace = np.mean(syn_trace.astype(float), axis=1)
                    array_hash = _safe_hash(syn_trace)
                    status = "measured"
            else:
                # Proxy: filter V1 L4 E spikes via exponential decay tau 2ms (emitters.py:457)
                spikes = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))
                if spikes.ndim == 2 and v1_l4_e:
                    tau_ms = 2.0
                    decay = float(np.exp(-float(dt_ms) / tau_ms))
                    # Compute per-step syn proxy as filtered spike train (leaky integrator)
                    spike_mean = np.mean(spikes[:, v1_l4_e].astype(float), axis=1) if len(v1_l4_e) > 0 else np.zeros(spikes.shape[0])
                    # IIR filter
                    syn_proxy = np.zeros_like(spike_mean, dtype=float)
                    s = 0.0
                    for t in range(spike_mean.shape[0]):
                        s = s * decay + float(spike_mean[t])
                        syn_proxy[t] = s
                    trace = syn_proxy
                    array_hash = _safe_hash(spikes) + "_proxy_tau2"
                    status = "proxy"
                    semantic = "PROXY_ESTIMATE"
                else:
                    trace = np.zeros(100, dtype=float)
                    status = "unavailable"
        except Exception as e:
            trace = np.zeros(100, dtype=float)
            status = "unavailable"
            provenance_extra = str(e)
        # STA on trace
        if trace is not None and trace.size > 10:
            sta = _compute_sta(trace, dt_ms, onsets_a, onsets_b, window_ms=win) if not is_spontaneous else None
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], subplot_titles=(f"STA — L4E→L23E syn_state ({status}, {len(edge_ids)} edges, tau 2ms)" , "A/B diff + d"))
            if sta is not None and sta["n_a"] > 0 and sta["n_b"] > 0:
                t = sta["t_ms"]
                fig.add_trace(go.Scatter(x=t, y=sta["sta_a"], mode="lines", name="A", line=dict(color="#1f77b4", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["sta_b"], mode="lines", name="B", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["diff"], mode="lines", name="A−B", line=dict(color="#d62728", width=1.5)), row=2, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["d"], mode="lines", name="d", line=dict(color="#9467bd", width=1, dash="dot")), row=2, col=1)
                title = f"{defd.get('title')} — Δ {sta['peak_diff']:.3g} d {sta['peak_d']:.2f} latency {sta['latency_ms']:.0f} ms ({units}, {semantic}, {status})"
                latency = sta["latency_ms"]; effect = sta["peak_d"]
            else:
                # Spontaneous — show whole trace
                t_ms = np.arange(trace.shape[0]) * float(dt_ms)
                step = max(1, trace.shape[0] // 3000)
                fig.add_trace(go.Scatter(x=t_ms[::step], y=trace[::step], mode="lines", name="syn_state proxy", line=dict(color="#1f77b4", width=1)), row=1, col=1)
                fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=f"Spontaneous — no A/B contrast<br>{status} via {'recorded syn_state' if has_real else 'filtered spikes proxy tau 2 ms (emitters.py:457)'}<br>n edges {len(edge_ids)}", showarrow=False, font=dict(size=9, color="#555"), bgcolor="rgba(255,255,255,0.85)", bordercolor="#999")
                title = f"{defd.get('title')} — spontaneous baseline ({units}, {semantic}, {status})"
                latency = float("nan"); effect = float("nan")
            fig.update_xaxes(title_text="Time from stimulus (ms)", row=2, col=1)
            fig.update_yaxes(title_text="syn_state (a.u.)", row=1, col=1)
            fig.update_yaxes(title_text="Δ / d", row=2, col=1)
            fig.update_layout(title=dict(text=title[:220], font=dict(size=10)), height=520, margin=dict(l=50, r=20, t=70, b=40), legend=dict(orientation="h", y=-0.18))
            fig.add_annotation(x=0.5, y=-0.16, xref="paper", yref="paper", text=f"Units {units} — estimator {defd.get('estimator')} — array {defd.get('array')} hash {array_hash[:8]} — semantic_class {semantic} — <b>{status}</b> (proxy_status {status=='proxy'}) — {defd.get('source')} — jaxfne/_pipeline.py:48 syn_state, emitters.py:457 decay", showarrow=False, font=dict(size=7, color="#555"), align="center")
            prov = {"stage": stage_id, "units": units, "semantic_class": semantic, "status": status, "array_hash": array_hash, "n_total": int(idxs["n_edges"]), "n_rendered": int(len(edge_ids)), "latency_ms": float(latency), "effect_d": float(effect), "n_edges_l4_l23": int(len(edge_ids))}
            return fig, prov
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=["syn_state unavailable"], showlegend=False))
            return fig, {"status": "unavailable"}

    elif stage_id == "l4e_l23e_current":
        # Machine-derived check first
        if blocked_info.get("blocked") or blocked_info.get("has_delay"):
            # Check if edge_current_trace actually present despite block? If present, show measured; else NOT OBSERVABLE
            has_trace = False
            try:
                # Check model diagnostics
                if hasattr(model, "last_hdp_diagnostics"):
                    diag = model.last_hdp_diagnostics()
                    if diag is not None and diag.get("edge_current_trace") is not None:
                        has_trace = True
                # Also check signal field?
                if not has_trace:
                    md = getattr(sig0, "metadata", None) or {}
                    if isinstance(md, dict) and "edge_current_trace" in md:
                        has_trace = True
            except Exception:
                has_trace = False
            if not has_trace and blocked_info.get("has_delay"):
                # Render NOT OBSERVABLE machine-derived
                fig = _fig_not_observable_machine(blocked_info, defd)
                prov = {"stage": stage_id, "units": defd.get("units",""), "semantic_class": "PROXY_ESTIMATE", "status": "unavailable", "array_hash": "blocked_"+str(blocked_info.get("delay_info",{}).get("hash",""))[:8], "n_total": int(idxs["n_edges"]), "n_rendered": 0, "latency_ms": float("nan"), "effect_d": float("nan"), "blocked": True, "reason": blocked_info.get("reason","")}
                return fig, prov
        # If not blocked, try to render measured trace
        try:
            ec = None
            # Try model diagnostics
            try:
                if hasattr(model, "last_hdp_diagnostics"):
                    diag = model.last_hdp_diagnostics()  # type: ignore
                    if diag is not None and diag.get("edge_current_trace") is not None:
                        ec = _as_numpy(diag.get("edge_current_trace"))  # [T,n_edges]
            except Exception:
                ec = None
            if ec is None:
                md = getattr(sig0, "metadata", None) or {}
                if isinstance(md, dict) and "edge_current_trace" in md:
                    ec = _as_numpy(md["edge_current_trace"])
            if ec is None and hasattr(sig0, "edge_current_trace"):
                ec = _as_numpy(getattr(sig0, "edge_current_trace"))
            # If still none, treat as proxy via W·r·tau ? But we must not silently proxy; show unavailable with note
            if ec is not None and ec.ndim == 2 and ec.shape[0] > 10:
                # Slice to L4→L23 edges
                if edge_ids and max(edge_ids) < ec.shape[1]:
                    sub = ec[:, edge_ids]  # [T, n_l4_l23]
                    trace = np.mean(np.abs(sub).astype(float), axis=1) if sub.size else np.zeros(ec.shape[0])
                else:
                    trace = np.mean(np.abs(ec).astype(float), axis=1) if ec.size else np.zeros(ec.shape[0])
                array_hash = _safe_hash(ec)
                status = "measured"
                semantic = "READOUT"
                # Also compute partition for provenance if needed
                sta = _compute_sta(trace, dt_ms, onsets_a, onsets_b, window_ms=win) if not is_spontaneous else None
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], subplot_titles=(f"STA — L4E→L23E current mean |w·syn| over {len(edge_ids) or ec.shape[1]} edges (READOUT)", "A/B diff + d"))
                if sta is not None and sta["n_a"] > 0 and sta["n_b"] > 0:
                    t = sta["t_ms"]
                    fig.add_trace(go.Scatter(x=t, y=sta["sta_a"], mode="lines", name="A", line=dict(color="#1f77b4", width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=t, y=sta["sta_b"], mode="lines", name="B", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=t, y=sta["diff"], mode="lines", name="A−B", line=dict(color="#d62728", width=1.5)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=t, y=sta["d"], mode="lines", name="d", line=dict(color="#9467bd", width=1, dash="dot")), row=2, col=1)
                    title = f"{defd.get('title')} — Δ {sta['peak_diff']:.3g} a.u. d {sta['peak_d']:.2f} latency {sta['latency_ms']:.0f} ms ({units}, {semantic}, {status})"
                    latency = sta["latency_ms"]; effect = sta["peak_d"]
                else:
                    t_ms = np.arange(trace.shape[0]) * float(dt_ms)
                    step = max(1, trace.shape[0] // 3000)
                    fig.add_trace(go.Scatter(x=t_ms[::step], y=trace[::step], mode="lines", name="current |w·syn|", line=dict(color="#1f77b4", width=1)), row=1, col=1)
                    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="Spontaneous — mean |current| shown<br>measured READOUT via record_edge_current", showarrow=False, font=dict(size=9, color="#555"), bgcolor="rgba(255,255,255,0.85)", bordercolor="#999")
                    title = f"{defd.get('title')} — spontaneous baseline ({units}, {semantic}, {status})"
                    latency = float("nan"); effect = float("nan")
                fig.update_xaxes(title_text="Time from stimulus (ms)", row=2, col=1)
                fig.update_yaxes(title_text="current (a.u.)", row=1, col=1)
                fig.update_yaxes(title_text="Δ / d", row=2, col=1)
                fig.update_layout(title=dict(text=title[:220], font=dict(size=10)), height=520, margin=dict(l=50, r=20, t=70, b=40), legend=dict(orientation="h", y=-0.18))
                fig.add_annotation(x=0.5, y=-0.16, xref="paper", yref="paper", text=f"Units {units} — estimator partition_currents_by_motif (observables.py:35) — array {defd.get('array')} hash {array_hash[:8]} — semantic_class {semantic} — <b>{status}</b> — {defd.get('source')} — jaxfne/_model_simulate.py:280 HDP+delay guard, jaxfne/emitters.py:2846 record_edge_current", showarrow=False, font=dict(size=7, color="#555"), align="center")
                prov = {"stage": stage_id, "units": units, "semantic_class": semantic, "status": status, "array_hash": array_hash, "n_total": int(ec.shape[1]), "n_rendered": int(len(edge_ids) if edge_ids else ec.shape[1]), "latency_ms": float(latency), "effect_d": float(effect)}
                return fig, prov
            else:
                # No trace — show unavailable with machine context (has_delay true but not strictly blocked)
                fig = _fig_not_observable_machine(blocked_info, defd)
                prov = {"stage": stage_id, "units": units, "semantic_class": "PROXY_ESTIMATE", "status": "unavailable", "array_hash": "no_edge_current_trace", "n_total": int(idxs["n_edges"]), "n_rendered": 0, "latency_ms": float("nan"), "effect_d": float("nan"), "blocked": bool(blocked_info.get("has_delay"))}
                return fig, prov
        except Exception as e:
            fig = _fig_not_observable_machine(blocked_info, defd) if blocked_info.get("has_delay") else go.Figure()
            if not blocked_info.get("has_delay"):
                fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=[f"current error: {e}"], showlegend=False))
            return fig, {"error": str(e), "status": "unavailable", "blocked": bool(blocked_info.get("has_delay"))}

    elif stage_id in ("l23_e_vm", "l23_e_spikes"):
        is_vm = stage_id == "l23_e_vm"
        try:
            if is_vm:
                arr = _as_numpy(sig0.V_m if hasattr(sig0, "V_m") else sig0.get("V_m", np.zeros((1, 1))))
                if arr.ndim != 2:
                    raise ValueError("V_m not [T,N]")
                trace = population_mean_trace(arr, v1_l23_e)  # [T] mV mean over L2/3 E
                array_hash = _safe_hash(arr)
                units = "mV"
                semantic = "STATE"
                status = "measured"
                y_title = "Vm (mV)"
            else:
                arr = _as_numpy(sig0.spikes if hasattr(sig0, "spikes") else sig0.get("spikes", np.zeros((1, 1))))
                if arr.ndim != 2:
                    raise ValueError("spikes not [T,N]")
                trace_frac = population_mean_trace(arr, v1_l23_e)  # fraction per step
                trace = trace_frac * (1000.0 / float(dt_ms))  # Hz
                array_hash = _safe_hash(arr)
                units = "Hz"
                semantic = "STATE"
                status = "measured"
                y_title = "Rate (Hz)"
            sta = _compute_sta(trace, dt_ms, onsets_a, onsets_b, window_ms=win) if not is_spontaneous else None
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                subplot_titles=(f"STA — V1 L2/3 E {'Vm' if is_vm else 'spikes'} mean over {len(v1_l23_e)} neurons", "A/B diff + d"))
            if sta is not None and sta["n_a"] > 0 and sta["n_b"] > 0:
                t = sta["t_ms"]
                fig.add_trace(go.Scatter(x=t, y=sta["sta_a"], mode="lines", name="A", line=dict(color="#1f77b4", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["sta_b"], mode="lines", name="B", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["diff"], mode="lines", name="A−B", line=dict(color="#d62728", width=1.5)), row=2, col=1)
                fig.add_trace(go.Scatter(x=t, y=sta["d"], mode="lines", name="d", line=dict(color="#9467bd", width=1, dash="dot")), row=2, col=1)
                title = f"{defd.get('title')} — Δ {sta['peak_diff']:.3g} {units} d {sta['peak_d']:.2f} latency {sta['latency_ms']:.0f} ms ({units}, {semantic}, {status}, n={len(v1_l23_e)})"
                latency = sta["latency_ms"]; effect = sta["peak_d"]
            else:
                t_ms = np.arange(trace.shape[0]) * float(dt_ms)
                step = max(1, trace.shape[0] // 3000)
                fig.add_trace(go.Scatter(x=t_ms[::step], y=trace[::step], mode="lines", name=f"L2/3 E {'Vm' if is_vm else 'rate'}", line=dict(color="#1f77b4", width=1)), row=1, col=1)
                fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=f"Spontaneous — no A/B contrast<br>mean {'Vm' if is_vm else 'rate'} shown ({'mV' if is_vm else 'Hz'})<br>n {len(v1_l23_e)}", showarrow=False, font=dict(size=9, color="#555"), bgcolor="rgba(255,255,255,0.85)", bordercolor="#999")
                title = f"{defd.get('title')} — spontaneous baseline ({units}, {semantic}, {status}, n={len(v1_l23_e)})"
                latency = float("nan"); effect = float("nan")
            fig.update_xaxes(title_text="Time from stimulus (ms)", row=2, col=1)
            fig.update_yaxes(title_text=y_title, row=1, col=1)
            fig.update_yaxes(title_text="Δ / d", row=2, col=1)
            fig.update_layout(title=dict(text=title[:220], font=dict(size=10)), height=520, margin=dict(l=50, r=20, t=70, b=40), legend=dict(orientation="h", y=-0.18))
            fig.add_annotation(x=0.5, y=-0.16, xref="paper", yref="paper", text=f"Units {units} — estimator {defd.get('estimator')} — array {defd.get('array')} hash {array_hash[:8]} — semantic_class {semantic} — <b>{status}</b> — {defd.get('source')} — model_summary.py:822 observable_basis Vm STATE, spikes STATE/OUTPUT*", showarrow=False, font=dict(size=7, color="#555"), align="center")
            prov = {"stage": stage_id, "units": units, "semantic_class": semantic, "status": status, "array_hash": array_hash, "n_total": int(len(meta) if meta else 0), "n_rendered": int(len(v1_l23_e)), "latency_ms": float(latency), "effect_d": float(effect), "is_spontaneous": bool(is_spontaneous)}
            return fig, prov
        except Exception as e:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=[f"{stage_id} error: {e}"], showlegend=False))
            return fig, {"error": str(e), "status": "unavailable"}

    # Fallback
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=[f"Unknown stage {stage_id}"], showlegend=False))
    return fig, {"error": "unknown stage", "status": "unavailable"}


def transfer_figures(simulation_result: dict) -> tuple[dict[str, Any], dict[str, dict], dict]:
    """Build 6 transfer panels from simulation_result.

    Returns (figs, provenance_map, summary)
    - figs: dict stage_id -> go.Figure (6 panels)
    - provenance_map: dict stage_id -> provenance footer dict (figure_id, config_hash, code_sha, units, semantic_class, claim→estimator→array→hash, status)
    - summary: chain summary with latencies, effect sizes, observable/ unavailable counts
    """
    signals = _resolve_signals(simulation_result)
    model = simulation_result.get("model")
    if model is None and isinstance(signals[0], dict) and "model" in signals[0]:
        model = signals[0].get("model")
    meta = _neuron_metadata(model, signals) if model is not None else []
    dt_ms = float(simulation_result.get("dt_ms", simulation_result.get("dt", 0.1)))
    # Hashes for provenance
    config_hash = str(simulation_result.get("config_hash") or getattr(getattr(model, "cfg", None), "metadata", {}).get("config_hash", "unknown") if model is not None else "unknown")
    if config_hash == "unknown" and model is not None:
        try:
            from jaxfne.io import config_hash as jch  # type: ignore
            config_hash = str(jch(model.cfg))
        except Exception:
            try:
                config_hash = str(model.cfg.metadata.get("config_hash", "unknown"))
            except Exception:
                config_hash = "unknown"
    code_sha = _get_code_sha()
    run_id = str(simulation_result.get("run_id", "unknown"))
    # Build each stage
    figs: dict[str, Any] = {}
    prov_map: dict[str, dict] = {}
    for defd in _TRANSFER_STAGE_DEFS:
        sid = defd["id"]
        try:
            fig, prov = _fig_transfer_stage(sid, simulation_result, model, signals, meta, dt_ms)
            figs[sid] = fig
            # Build provenance footer per figure (figure_id, config_hash, code_sha, etc.)
            footer = {
                "figure_id": sid,
                "title": defd.get("title", sid),
                "label": defd.get("label", sid),
                "run_id": run_id,
                "config_hash": config_hash,
                "code_sha": code_sha,
                "seed": int(simulation_result.get("seed", 0)),
                "dt_ms": float(dt_ms),
                "claim": defd.get("claim", ""),
                "estimator": defd.get("estimator", ""),
                "estimator_version": defd.get("estimator_version", ""),
                "array": defd.get("array", ""),
                "array_hash": prov.get("array_hash", "unknown"),
                "source": defd.get("source", ""),
                "units": prov.get("units", defd.get("units", "")),
                "semantic_class": prov.get("semantic_class", defd.get("semantic_class", "")),
                "derived_from": defd.get("derived_from", ""),
                "proxy_status": bool(prov.get("status") in ("proxy",)),
                "status": prov.get("status", "unknown"),  # measured/derived/proxy/unavailable
                "n_total": int(prov.get("n_total", 0)),
                "n_rendered": int(prov.get("n_rendered", 0)),
                "latency_ms": prov.get("latency_ms", float("nan")),
                "effect_d": prov.get("effect_d", float("nan")),
                "owner": "generated-owner (same arrays as analyses, not recomputed prettier alternatives)",
                "provenance_note": defd.get("provenance_note", ""),
                "file_line_citations": {
                    "builder_motif": "jomission/network/builder.py:62 MOTIF_GAIN 16-gain pseudogenome",
                    "builder_delays": "jomission/network/builder.py:90 FF_LAYER_MAP/FB_LAYER_MAPS laminar delays 2/8/12 ms →20/80/120 steps",
                    "emitters_delay": "jaxfne/emitters.py:545 edge_list_with_delay_ms",
                    "model_simulate_guard": "jaxfne/_model_simulate.py:280,572,1063 HDP+delay guard",
                    "pipeline": "jaxfne/_pipeline.py:395 compile_step_fn record_edge_current (only kernel==hdp), :72 ContinuationState.delay_state",
                    "observables": "jomission/recording/observables.py:32 I_edge_current optional seam",
                    "area_local": "jomission/recording/area_local.py:52 field_by_area_from_signal linear partition",
                    "observable_basis": "jomission/visualization/model_summary.py:822 observable_basis() C_t→X_t→q_t→φ_t→y_t",
                },
                "is_spontaneous": bool(prov.get("is_spontaneous", False)),
            }
            prov_map[sid] = footer
        except Exception as e:
            figs[sid] = None
            prov_map[sid] = {"figure_id": sid, "error": str(e), "status": "unavailable"}
    # Chain summary
    summary = {
        "n_stages": len(_TRANSFER_STAGE_DEFS),
        "stages": [d["id"] for d in _TRANSFER_STAGE_DEFS],
        "chain": "V1 L4 E spikes → delay/history → L4E→L23E syn_state → L4E→L23E current → L2/3 E Vm → L2/3 E spikes",
        "status_by_stage": {sid: prov_map.get(sid, {}).get("status", "unknown") for sid in prov_map},
        "latencies_ms": {sid: prov_map.get(sid, {}).get("latency_ms") for sid in prov_map},
        "effect_ds": {sid: prov_map.get(sid, {}).get("effect_d") for sid in prov_map},
        "config_hash": config_hash,
        "code_sha": code_sha,
    }
    return figs, prov_map, summary


def summary_panel_fig(provenance_map: dict[str, dict], summary: dict) -> Any:
    """Chain-level summary figure: latency and effect size across stages."""
    if not _PLOTLY or go is None:
        return None
    stages = summary.get("stages", [d["id"] for d in _TRANSFER_STAGE_DEFS])
    labels = [ _STAGE_ID_TO_DEF[s].get("label", s) for s in stages ]
    latencies = [provenance_map.get(s, {}).get("latency_ms", float("nan")) for s in stages]
    effects = [provenance_map.get(s, {}).get("effect_d", float("nan")) for s in stages]
    statuses = [provenance_map.get(s, {}).get("status", "unknown") for s in stages]
    # Replace nan with 0 for plotting but annotate
    lat_plot = [float(v) if np.isfinite(v) else 0.0 for v in latencies]
    eff_plot = [float(v) if np.isfinite(v) else 0.0 for v in effects]
    colors = ["#d62728" if s == "unavailable" else ("#ff7f0e" if s == "proxy" else "#1f77b4") for s in statuses]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Latency to peak A−B (ms, NaN=spontaneous/no contrast)", "Effect size peak Cohen's d (A vs B)"), horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=labels, y=lat_plot, marker=dict(color=colors), name="latency", hovertemplate="%{x}<br>latency %{y:.0f} ms status %{text}<extra></extra>", text=statuses), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=eff_plot, marker=dict(color=colors), name="d", hovertemplate="%{x}<br>d %{y:.2f} status %{text}<extra></extra>", text=statuses), row=1, col=2)
    for i, s in enumerate(statuses):
        if s == "unavailable":
            fig.add_annotation(x=labels[i], y=lat_plot[i]+5, text="NOT<br>OBSERVABLE", showarrow=False, font=dict(size=7, color="#d62728"), row=1, col=1)
    fig.update_layout(title="Transfer chain summary — latency and effect size per stage (machine status: measured=blue proxy=orange unavailable=red)", height=420, margin=dict(l=50, r=20, t=70, b=120), showlegend=False)
    fig.update_xaxes(tickangle=30, row=1, col=1)
    fig.update_xaxes(tickangle=30, row=1, col=2)
    fig.update_yaxes(title_text="ms", row=1, col=1)
    fig.update_yaxes(title_text="Cohen's d", row=1, col=2)
    return fig


__all__ = ["transfer_figures", "summary_panel_fig", "_TRANSFER_STAGE_DEFS", "_is_edge_current_blocked", "_edge_delay_info"]


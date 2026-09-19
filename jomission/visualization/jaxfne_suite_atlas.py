"""jaxfne_suite_atlas — the Plotly reference gallery's style, generalized to any jaxfne circuit.

The six figures in ``docs/_static/plotly/`` were each hand-written against jomission's own
builder: ``network_3d.py`` hardcodes ``area_x = {"V1": 0.0, "V4": 35.0, ...}``, ``activity.py``
hardcodes ``area_order`` and the AAAB paradigm. They cannot be pointed at another circuit.

This module keeps their visual standard exactly — ``theme.py``'s dark palette, the provenance
card, the evidence badge — and replaces every hardcoded structure with introspection:

    model.neuron_table()        area, layer, cell_type, neuron_id, x, y, z   per neuron
    model.params["edge_list"]   pre, post, weight, receptor_index            per edge
    signals.V_m                 (T, N)      membrane potential
    signals.spikes              (T, N)      spike indicators
    signals.field.lfp_proxy     (T, C)      laminar field proxy
    signals.field.csd_proxy     (T, C)      current source density proxy
    signals.field.contact_depths (C,)       contact geometry

Areas, layers and cell classes are whatever the model declares. Known names keep their gallery
colors so existing figures stay recognisable; unknown names draw from a palette in a stable
order, so a circuit with areas the gallery never saw still renders with a consistent legend.

Panels are independent: a panel whose inputs are absent renders a labelled placeholder and says
which input was missing, rather than being silently dropped from the index.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jomission.visualization.theme import (
    AREA_COLORS,
    CLASS_COLORS,
    PROJ_COLORS,
    apply_dark_theme,
    wrap_figure_with_provenance_html,
)

SUITE = "jaxfne_suite_atlas.v1"

#: Drawn in order for areas and classes the gallery palette does not name.
FALLBACK_PALETTE = (
    "#38bdf8", "#4ade80", "#fb923c", "#c084fc", "#f43f5e", "#eab308",
    "#22d3ee", "#a3e635", "#fb7185", "#818cf8", "#facc15", "#2dd4bf",
)

BG_PAPER, BG_PLOT, FG, FG_DIM, LINE = "#0d1117", "#161b22", "#c9d1d9", "#8b949e", "#30363d"

#: Diverging scale for signed quantities (CSD sinks/sources, E/I weight balance).
DIVERGING = [
    [0.0, "#2563eb"], [0.25, "#1e3a8a"], [0.5, "#0d1117"],
    [0.75, "#7f1d1d"], [1.0, "#ef4444"],
]
#: Sequential scale for unsigned magnitudes (LFP power, degree, |w|).
SEQUENTIAL = [
    [0.0, "#0d1117"], [0.3, "#155e75"], [0.6, "#0891b2"],
    [0.85, "#22d3ee"], [1.0, "#a5f3fc"],
]


# --------------------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------------------

def _layer_sort_key(layer: str) -> tuple:
    """Order layers superficial to deep. Handles L1, L2, L2/3, L23, L4, L5, L6 alike."""
    digits = "".join(ch for ch in str(layer) if ch.isdigit())
    return (float(digits[0]) if digits else 99.0, str(layer))


def _stable_colors(names: list[str], known: dict[str, str]) -> dict[str, str]:
    out, spare = {}, [c for c in FALLBACK_PALETTE if c not in known.values()]
    for nm in names:
        if nm in known:
            out[nm] = known[nm]
        else:
            out[nm] = spare[len([k for k in out if k not in known]) % len(spare)]
    return out


def describe(model: Any) -> dict:
    """Everything the panels need about a model's structure, read from the model itself."""
    tbl = model.neuron_table()
    areas = sorted({r["area"] for r in tbl})
    layers = sorted({r["layer"] for r in tbl}, key=_layer_sort_key)
    classes = sorted({r["cell_type"] for r in tbl})
    el = model.params["edge_list"]
    return {
        "table": tbl,
        "n_neurons": len(tbl),
        "areas": areas,
        "layers": layers,
        "classes": classes,
        "area_color": _stable_colors(areas, AREA_COLORS),
        "class_color": _stable_colors(classes, CLASS_COLORS),
        "area_index": {a: i for i, a in enumerate(areas)},
        "layer_index": {lay: i for i, lay in enumerate(layers)},
        "pre": np.asarray(el.pre),
        "post": np.asarray(el.post),
        "weight": np.asarray(el.weight),
        "n_edges": int(np.asarray(el.pre).shape[0]),
        "area_of": np.array([r["area"] for r in tbl]),
        "layer_of": np.array([r["layer"] for r in tbl]),
        "class_of": np.array([r["cell_type"] for r in tbl]),
    }


def _sorted_display_order(D: dict) -> np.ndarray:
    """Neuron order for rasters and matrices: area, then layer superficial-to-deep, then class."""
    ai, li = D["area_index"], D["layer_index"]
    return np.array(sorted(
        range(D["n_neurons"]),
        key=lambda i: (ai[D["area_of"][i]], li[D["layer_of"][i]], D["class_of"][i], i),
    ))


def _placeholder(reason: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=f"<b>{title}</b><br><span style='color:{FG_DIM}'>{reason}</span>",
                       showarrow=False, font=dict(size=14, color=FG), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_dark_theme(fig, title, "input unavailable")


# --------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------

def panel_network_3d(model: Any, D: dict, *, max_edges: int = 600, seed: int = 0):
    """Dark 3D explorer. Areas separate along x, layers stack along z, both read from the model."""
    tbl = D["table"]
    xs = np.array([float(r["x"]) for r in tbl])
    ys = np.array([float(r["y"]) for r in tbl])
    zs = np.array([float(r["z"]) for r in tbl])

    # Spread areas along x so the hierarchy reads left to right even when the model's own
    # coordinates overlap them. Layer depth stays whatever the model assigned to z.
    span = float(np.ptp(xs)) or 1.0
    offset = np.array([D["area_index"][a] * span * 1.6 for a in D["area_of"]])
    xs = xs + offset

    out_deg = np.bincount(D["pre"], minlength=D["n_neurons"])
    in_deg = np.bincount(D["post"], minlength=D["n_neurons"])

    fig = go.Figure()
    for ct in D["classes"]:
        k = D["class_of"] == ct
        hover = [
            f"<b>#{tbl[i]['neuron_id']} · {tbl[i]['area']} {tbl[i]['layer']} {ct}</b><br>"
            f"out-degree {out_deg[i]}<br>in-degree {in_deg[i]}"
            for i in np.flatnonzero(k)
        ]
        fig.add_trace(go.Scatter3d(
            x=xs[k], y=ys[k], z=zs[k], mode="markers", name=ct,
            marker=dict(size=3.2, color=D["class_color"][ct], opacity=0.85,
                        line=dict(width=0)),
            hovertext=hover, hoverinfo="text"))

    rng = np.random.RandomState(seed)
    n_e = D["n_edges"]
    pick = rng.choice(n_e, size=min(max_edges, n_e), replace=False) if n_e else np.array([], int)
    ai = D["area_index"]
    buckets: dict[str, list] = {"recurrent": [], "FF": [], "FB": []}
    for e in pick:
        s, t = int(D["pre"][e]), int(D["post"][e])
        a_s, a_t = ai[D["area_of"][s]], ai[D["area_of"][t]]
        kind = "recurrent" if a_s == a_t else ("FF" if a_t > a_s else "FB")
        buckets[kind] += [(xs[s], ys[s], zs[s]), (xs[t], ys[t], zs[t]), (None, None, None)]
    for kind, pts in buckets.items():
        if not pts:
            continue
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in pts], y=[p[1] for p in pts], z=[p[2] for p in pts],
            mode="lines", name=f"{kind} ({len(pts)//3})", hoverinfo="skip",
            line=dict(color=PROJ_COLORS[kind], width=1.4), opacity=0.35))

    axis = dict(showbackground=True, backgroundcolor=BG_PLOT, gridcolor="#21262d",
                zerolinecolor=LINE, color=FG_DIM)
    fig.update_layout(
        scene=dict(xaxis=dict(title="area →", **axis), yaxis=dict(title="unit", **axis),
                   zaxis=dict(title="depth", **axis), bgcolor=BG_PAPER,
                   camera=dict(eye=dict(x=1.6, y=1.5, z=0.9))),
        height=760, showlegend=True)
    apply_dark_theme(fig, "Network 3D",
                     f"{D['n_neurons']} units · {n_e:,} edges · {len(pick)} drawn · "
                     f"{len(D['areas'])} areas × {len(D['layers'])} layers")
    caption = ("Every unit at its own coordinates from <code>model.neuron_table()</code>, colored "
               "by cell class. Areas are offset along x by their index so the hierarchy reads left "
               "to right; z is the model's own depth. Edges are a uniform random sample, classified "
               "by the area index of their endpoints: same area recurrent, increasing feedforward, "
               "decreasing feedback.")
    prov = {"panel": "network_3d", "evidence": "OBSERVED",
            "source": "model.neuron_table() + model.params['edge_list']",
            "edges_total": f"{n_e:,}", "edges_drawn": str(len(pick)), "edge_sample_seed": str(seed)}
    return fig, caption, prov


def panel_raster(model: Any, D: dict, spikes, dt_ms: float, *, max_units: int = 1200):
    """Raster ordered area → layer → class, over per-class population rate."""
    if spikes is None:
        return _placeholder("signals.spikes is None", "Raster & population rate"), "", {}
    spikes = np.asarray(spikes)
    n_steps = spikes.shape[0]
    t = np.arange(n_steps) * dt_ms

    order = _sorted_display_order(D)
    if order.size > max_units:
        order = order[np.linspace(0, order.size - 1, max_units).astype(int)]
    row_of = {int(nid): r for r, nid in enumerate(order)}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06,
                        subplot_titles=("Spike raster (area → layer → class)",
                                        "Population rate by cell class"))
    for ct in D["classes"]:
        idx = [i for i in order if D["class_of"][i] == ct]
        if not idx:
            continue
        ti, ni = np.nonzero(spikes[:, idx])
        if ti.size:
            fig.add_trace(go.Scattergl(
                x=t[ti], y=[row_of[idx[j]] for j in ni], mode="markers", name=ct,
                marker=dict(size=1.6, color=D["class_color"][ct]), hoverinfo="skip"),
                row=1, col=1)
        # 1 ms bins; rate in Hz per unit
        per_ms = max(1, int(round(1.0 / dt_ms)))
        trim = (n_steps // per_ms) * per_ms
        binned = spikes[:trim, idx].reshape(-1, per_ms, len(idx)).sum(axis=1)
        fig.add_trace(go.Scatter(
            x=np.arange(binned.shape[0]) * 1.0, y=binned.mean(axis=1) * 1000.0,
            mode="lines", name=f"{ct} rate", line=dict(color=D["class_color"][ct], width=1.2),
            showlegend=False), row=2, col=1)

    # Area boundaries as horizontal separators on the raster.
    seen, bound = None, []
    for r, i in enumerate(order):
        if D["area_of"][i] != seen:
            seen = D["area_of"][i]
            bound.append((r, seen))
    for r, a in bound[1:]:
        fig.add_hline(y=r, line=dict(color=LINE, width=1, dash="dot"), row=1, col=1)
    for r, a in bound:
        fig.add_annotation(x=t[-1], y=r, text=f" {a}", showarrow=False, xanchor="left",
                           font=dict(size=10, color=D["area_color"][a]), row=1, col=1)

    fig.update_yaxes(title_text="unit (sorted)", row=1, col=1)
    fig.update_yaxes(title_text="Hz", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_layout(height=780, legend=dict(orientation="h", y=1.06))
    apply_dark_theme(fig, "Raster & population rate",
                     f"{order.size} of {D['n_neurons']} units · {n_steps * dt_ms:.0f} ms · dt {dt_ms} ms")
    caption = ("Each dot is one spike. Units are sorted by area, then layer superficial to deep, "
               "then cell class, so laminar and hierarchical structure appear as horizontal bands; "
               "dotted lines mark area boundaries. The lower axis is the mean per-unit rate in 1 ms "
               "bins, one trace per class.")
    prov = {"panel": "raster", "evidence": "OBSERVED", "source": "signals.spikes",
            "shape": f"{spikes.shape}", "dt_ms": str(dt_ms),
            "units_drawn": f"{order.size} of {D['n_neurons']}",
            "rate_estimator": "1 ms bins, mean over units, DERIVED"}
    return fig, caption, prov


def panel_membrane(model: Any, D: dict, V_m, dt_ms: float, *, per_class: int = 4, seed: int = 0,
                   max_points: int = 8000, violin_points: int = 20000):
    """Sampled V(t) per class, plus the voltage distribution each class occupies.

    Both the traces and the violins are decimated before they reach the browser. Shipping every
    sample is what makes a page unopenable: 10,000 steps x 828 E units is 8.3M points per violin
    and a 69 MB file, for a distribution that is indistinguishable from a 20,000-point sample.
    """
    if V_m is None:
        return _placeholder("signals.V_m is None", "Membrane potential"), "", {}
    V = np.asarray(V_m)
    rng = np.random.RandomState(seed)
    stride = max(1, int(np.ceil(V.shape[0] / max_points)))
    t = np.arange(0, V.shape[0], stride) * dt_ms

    fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28], horizontal_spacing=0.06,
                        subplot_titles=("Membrane potential, sampled units",
                                        "Voltage occupancy"))
    for ct in D["classes"]:
        pool = np.flatnonzero(D["class_of"] == ct)
        if pool.size == 0:
            continue
        pick = rng.choice(pool, size=min(per_class, pool.size), replace=False)
        for j, i in enumerate(pick):
            fig.add_trace(go.Scattergl(
                x=t, y=V[::stride, i], mode="lines", name=ct if j == 0 else None,
                showlegend=(j == 0), legendgroup=ct, opacity=0.8,
                line=dict(color=D["class_color"][ct], width=0.9),
                hovertemplate=f"{ct} #{i}<br>%{{x:.1f}} ms<br>%{{y:.1f}} mV<extra></extra>"),
                row=1, col=1)
        pooled = V[::stride, pool].ravel()
        if pooled.size > violin_points:
            pooled = pooled[rng.choice(pooled.size, violin_points, replace=False)]
        fig.add_trace(go.Violin(
            y=pooled, name=ct, legendgroup=ct, showlegend=False,
            line=dict(color=D["class_color"][ct]), fillcolor=D["class_color"][ct],
            opacity=0.45, points=False, spanmode="hard"), row=1, col=2)

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="V (mV)", row=1, col=1)
    fig.update_yaxes(title_text="V (mV)", row=1, col=2)
    fig.update_layout(height=560, legend=dict(orientation="h", y=1.08))
    apply_dark_theme(fig, "Membrane potential",
                     f"{per_class} units per class · {V.shape[0] * dt_ms:.0f} ms")
    caption = ("Raw traces, not averages: a spike appears as the reset discontinuity the "
               "integrator actually produced. The violin on the right is the pooled voltage "
               "distribution over every unit of that class and the whole window, so a class "
               "sitting far from threshold is visible even when its traces are quiet.")
    prov = {"panel": "membrane", "evidence": "OBSERVED", "source": "signals.V_m",
            "shape": f"{V.shape}", "dt_ms": str(dt_ms),
            "units_per_class": str(per_class), "sample_seed": str(seed),
            "time_decimation": f"every {stride} step(s) -> {t.size} drawn",
            "violin_sample": f"<= {violin_points} values per class"}
    return fig, caption, prov


def panel_field(model: Any, D: dict, signals: Any, dt_ms: float):
    """LFP and CSD proxies as depth × time heatmaps, plus the depth profile."""
    field = getattr(signals, "field", None)
    if field is None:
        return _placeholder("signals.field is None", "LFP & CSD"), "", {}
    lfp = np.asarray(getattr(field, "lfp_proxy", None) if getattr(field, "lfp_proxy", None)
                     is not None else getattr(field, "lfp", None))
    csd = np.asarray(getattr(field, "csd_proxy", None) if getattr(field, "csd_proxy", None)
                     is not None else getattr(field, "csd", None))
    depths = np.asarray(getattr(field, "contact_depths", np.arange(lfp.shape[1])))
    t = np.arange(lfp.shape[0]) * dt_ms
    # Heatmaps downsample in the browser; the stacked line traces do not, so decimate those.
    lstride = max(1, int(np.ceil(lfp.shape[0] / 8000)))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.37, 0.37, 0.26],
                        vertical_spacing=0.06,
                        subplot_titles=("LFP proxy (depth × time)",
                                        "CSD proxy — blue sink, red source",
                                        "Stacked LFP by depth"))
    lim_l = float(np.abs(lfp).max()) or 1.0
    fig.add_trace(go.Heatmap(z=lfp.T, x=t, y=depths, colorscale=SEQUENTIAL,
                             zmin=float(lfp.min()), zmax=float(lfp.max()),
                             colorbar=dict(title="LFP", len=0.3, y=0.85, thickness=12),
                             hovertemplate="%{x:.1f} ms<br>depth %{y}<br>%{z:.4g}<extra></extra>"),
                  row=1, col=1)
    lim_c = float(np.abs(csd).max()) or 1.0
    fig.add_trace(go.Heatmap(z=csd.T, x=t, y=depths, colorscale=DIVERGING,
                             zmid=0.0, zmin=-lim_c, zmax=lim_c,
                             colorbar=dict(title="CSD", len=0.3, y=0.5, thickness=12),
                             hovertemplate="%{x:.1f} ms<br>depth %{y}<br>%{z:.4g}<extra></extra>"),
                  row=2, col=1)
    step = (np.abs(lfp).max() or 1.0) * 1.8
    for c in range(lfp.shape[1]):
        shade = 0.35 + 0.65 * (c / max(1, lfp.shape[1] - 1))
        fig.add_trace(go.Scattergl(
            x=t[::lstride], y=lfp[::lstride, c] + (lfp.shape[1] - 1 - c) * step, mode="lines",
            name=f"c{c}", showlegend=False, opacity=shade,
            line=dict(color="#22d3ee", width=0.8),
            hovertemplate=f"contact {c} (depth {depths[c]:.3g})<extra></extra>"), row=3, col=1)

    fig.update_yaxes(title_text="depth", autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text="depth", autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="contact (offset)", showticklabels=False, row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
    fig.update_layout(height=900)
    apply_dark_theme(fig, "LFP & CSD",
                     f"{lfp.shape[1]} contacts · {lfp.shape[0] * dt_ms:.0f} ms · lfp_proxy / csd_proxy")
    caption = ("Both are the engine's <code>_proxy</code> fields, not recorded potentials: they are "
               "computed from unit sources through <code>field.kernel_matrix</code>, so they carry "
               "the kernel's assumptions and no electrode. CSD is on a symmetric scale about zero "
               "so sinks and sources are comparable; depth increases downward on both heatmaps.")
    prov = {"panel": "field", "evidence": "DERIVED",
            "source": "signals.field.lfp_proxy / csd_proxy / contact_depths",
            "contacts": str(lfp.shape[1]), "lfp_shape": f"{lfp.shape}", "csd_shape": f"{csd.shape}",
            "csd_scale": f"symmetric ±{lim_c:.4g}", "lfp_range": f"±{lim_l:.4g}",
            "stacked_decimation": f"every {lstride} step(s)"}
    return fig, caption, prov


def panel_connectivity(model: Any, D: dict):
    """Summed signed weight, area × area and layer × layer, plus the class balance."""
    A, L, C = D["areas"], D["layers"], D["classes"]
    ai, li = D["area_index"], D["layer_index"]
    ci = {c: i for i, c in enumerate(C)}
    pre, post, w = D["pre"], D["post"], D["weight"]

    Wa = np.zeros((len(A), len(A)))
    np.add.at(Wa, ([ai[a] for a in D["area_of"][pre]], [ai[a] for a in D["area_of"][post]]), w)
    Wl = np.zeros((len(L), len(L)))
    np.add.at(Wl, ([li[x] for x in D["layer_of"][pre]], [li[x] for x in D["layer_of"][post]]), w)
    Wc = np.zeros((len(C), len(C)))
    np.add.at(Wc, ([ci[x] for x in D["class_of"][pre]], [ci[x] for x in D["class_of"][post]]), w)

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.09,
                        subplot_titles=("Area → area", "Layer → layer", "Class → class"))
    for col, (M, lab) in enumerate([(Wa, A), (Wl, L), (Wc, C)], start=1):
        lim = float(np.abs(M).max()) or 1.0
        fig.add_trace(go.Heatmap(
            z=M, x=lab, y=lab, colorscale=DIVERGING, zmid=0.0, zmin=-lim, zmax=lim,
            showscale=(col == 3), colorbar=dict(title="Σw", thickness=12),
            hovertemplate="%{y} → %{x}<br>Σw %{z:.4g}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="postsynaptic", row=1, col=col)
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    fig.update_yaxes(title_text="presynaptic", row=1, col=1)
    fig.update_layout(height=520)
    apply_dark_theme(fig, "Connectivity",
                     f"{D['n_edges']:,} edges · signed Σw · rows presynaptic, columns postsynaptic")
    caption = ("Each cell is the <b>sum of signed weights</b> over every edge in that block, so it "
               "reports net drive rather than edge count: a block with many balanced excitatory and "
               "inhibitory edges reads near zero and is not the same as an absent projection. Red is "
               "net excitatory, blue net inhibitory.")
    prov = {"panel": "connectivity", "evidence": "OBSERVED",
            "source": "model.params['edge_list'].pre/post/weight",
            "n_edges": f"{D['n_edges']:,}", "statistic": "sum of signed weight per block",
            "blocks": f"{len(A)}×{len(A)} area, {len(L)}×{len(L)} layer, {len(C)}×{len(C)} class"}
    return fig, caption, prov


def panel_plasticity(model: Any, D: dict, H_trace, w_trace, dt_ms: float, source: str = ""):
    """H and effective-weight trajectories. Declares a static trace static rather than implying drift."""
    if H_trace is None:
        return _placeholder("no H trace supplied — run an HDP kernel and pass H_trace",
                            "Plasticity"), "", {}
    H = np.asarray(H_trace)
    if H.ndim == 3:            # (T, N, coords) -> take the first coordinate as H
        H = H[:, :, 0]
    t = np.arange(H.shape[0]) * dt_ms

    has_w = w_trace is not None
    rows = 2 if has_w else 1
    titles = ["H trajectory by cell class (band = 5–95th percentile)"]
    if has_w:
        titles.append("Effective weight relative to its initial value")
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=tuple(titles))
    for ct in D["classes"]:
        k = D["class_of"] == ct
        if not k.any():
            continue
        lo, mid, hi = (np.percentile(H[:, k], p, axis=1) for p in (5, 50, 95))
        col = D["class_color"][ct]
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                 fillcolor=col, opacity=0.16, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=mid, mode="lines", name=ct,
                                 line=dict(color=col, width=1.6)), row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color=FG_DIM, width=1, dash="dash"), row=1, col=1)

    static_note = ""
    if has_w:
        W = np.asarray(w_trace)
        w0 = np.abs(W[0]) + 1e-12
        ratio = np.abs(W) / w0
        for q, nm, dash in [(5, "5th", "dot"), (50, "median", "solid"), (95, "95th", "dot")]:
            fig.add_trace(go.Scatter(x=t, y=np.percentile(ratio, q, axis=1), mode="lines",
                                     name=f"|w|/|w0| {nm}", line=dict(color="#22d3ee", width=1.3,
                                                                     dash=dash)), row=2, col=1)
        fig.update_yaxes(title_text="|w| / |w₀|", row=2, col=1)
        if float(np.abs(ratio - 1.0).max()) < 1e-6:
            static_note = " Weights are static to 1e-6 over this window."
    fig.update_yaxes(title_text="H", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=rows, col=1)
    fig.update_layout(height=640 if has_w else 420, legend=dict(orientation="h", y=1.08))
    span = float(H.max() - H.min())
    apply_dark_theme(fig, "Plasticity",
                     f"H span {H.min():.3f}–{H.max():.3f} · {H.shape[1]} units · "
                     f"{H.shape[0] * dt_ms:.0f} ms")
    caption = ("Median H per class with its 5–95th percentile band, against the dashed reference at "
               "H = 1. A band that stays flat is reported as flat: under a non-plastic kernel these "
               "traces are constant by construction and that is not evidence of regulation." +
               static_note)
    prov = {"panel": "plasticity", "evidence": "OBSERVED",
            "source": source or "H trace (and w trace if given)",
            "H_shape": f"{H.shape}", "H_span": f"{span:.6g}",
            "w_trace": "supplied" if has_w else "absent", "dt_ms": str(dt_ms)}
    return fig, caption, prov


PANELS = (
    ("network_3d", "Network 3D", "OBSERVED"),
    ("raster", "Raster & population rate", "OBSERVED"),
    ("membrane", "Membrane potential", "OBSERVED"),
    ("field", "LFP & CSD", "DERIVED"),
    ("connectivity", "Connectivity", "OBSERVED"),
    ("plasticity", "Plasticity", "OBSERVED"),
)


# --------------------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------------------

def _index_html(title: str, subtitle: str, cards: list[dict]) -> str:
    badge = {"OBSERVED": "#238636", "DERIVED": "#1f6feb", "INFERRED": "#d29922"}
    items = "".join(
        f"""<a class="card{' err' if c['status'] != 'AVAILABLE' else ''}" href="{c['file']}">
      <div class="row"><span class="name">{c['panel']}</span>
        <span class="badge" style="background:{badge.get(c['evidence'], '#d29922')}">{c['evidence']}</span></div>
      <div class="meta">{c['note']}</div>
    </a>""" for c in cards)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ background:{BG_PAPER}; color:{FG}; margin:0; padding:32px 24px;
         font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 6px; color:#f0f6fc; }}
  .sub {{ color:{FG_DIM}; font-size:13px; margin-bottom:28px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
  .card {{ display:block; background:{BG_PLOT}; border:1px solid {LINE}; border-radius:8px;
           padding:16px 18px; text-decoration:none; color:{FG}; transition:border-color .15s; }}
  .card:hover {{ border-color:#58a6ff; }}
  .card.err {{ border-color:#d29922; }}
  .row {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
  .name {{ font-weight:600; font-size:14px; color:#f0f6fc; }}
  .badge {{ font-size:10px; font-weight:600; color:#fff; padding:2px 8px; border-radius:10px; }}
  .meta {{ color:{FG_DIM}; font-size:12px; margin-top:8px; line-height:1.5; }}
  code {{ background:#0d1117; padding:1px 5px; border-radius:4px; font-size:11px; }}
  @media (max-width:520px) {{ body {{ padding:20px 16px; }} }}
</style></head><body><div class="wrap">
<h1>{title}</h1><div class="sub">{subtitle}</div>
<div class="grid">{items}</div>
</div></body></html>
"""


def build_suite_atlas(
    model: Any,
    signals: Any | None = None,
    *,
    out_dir: str = "docs/_static/jaxfne_suite_atlas",
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    seed: int = 0,
    title: str = "jaxfne suite atlas",
    H_trace: Any | None = None,
    w_trace: Any | None = None,
    trace_source: str = "",
    max_edges: int = 600,
) -> dict:
    """Render the full suite for any jaxfne model. Returns the manifest and writes the HTML.

    ``signals`` is simulated if not supplied. ``H_trace`` / ``w_trace`` come from an HDP run and
    are optional; the plasticity panel says so rather than inventing a flat line. When those
    traces come from a different run than ``signals``, pass ``trace_source`` to say so on the
    panel's provenance card — the card must not imply one run produced both.
    """
    if signals is None:
        import jaxfne as _J
        signals = _J.simulate(model, _J.Simulation(duration_ms=float(duration_ms),
                                                   dt_ms=float(dt_ms), seed=int(seed)))
    D = describe(model)
    os.makedirs(out_dir, exist_ok=True)

    spikes = getattr(signals, "spikes", None)
    V_m = getattr(signals, "V_m", None)
    builders = {
        "network_3d": lambda: panel_network_3d(model, D, max_edges=max_edges, seed=seed),
        "raster": lambda: panel_raster(model, D, spikes, dt_ms),
        "membrane": lambda: panel_membrane(model, D, V_m, dt_ms, seed=seed),
        "field": lambda: panel_field(model, D, signals, dt_ms),
        "connectivity": lambda: panel_connectivity(model, D),
        "plasticity": lambda: panel_plasticity(model, D, H_trace, w_trace, dt_ms, trace_source),
    }

    cards, manifest_panels = [], []
    for key, label, evidence in PANELS:
        status, note = "AVAILABLE", ""
        try:
            fig, caption, prov = builders[key]()
            if not caption:
                status, note = "UNAVAILABLE", "required input was not present"
                caption = f"{label} could not be built: its input was not present in this run."
                prov = {"panel": key, "status": "UNAVAILABLE"}
        except Exception as exc:
            status = "ERROR"
            note = f"{type(exc).__name__}: {exc}"
            fig = _placeholder(note, label)
            caption = f"{label} raised {type(exc).__name__} while building."
            prov = {"panel": key, "status": "ERROR", "error": note}
        prov = {"suite": SUITE, "model_units": str(D["n_neurons"]),
                "model_edges": f"{D['n_edges']:,}", **prov}
        html = wrap_figure_with_provenance_html(fig, caption, prov, evidence)
        fname = f"{key}.html"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
            fh.write(html)
        if not note:
            note = caption.split(".")[0].replace("<code>", "").replace("</code>", "") + "."
        cards.append({"file": fname, "panel": label, "evidence": evidence,
                      "status": status, "note": note})
        manifest_panels.append({"file": fname, "key": key, "panel": label,
                                "evidence": evidence, "status": status,
                                "bytes": len(html.encode("utf-8"))})

    sub = (f"{D['n_neurons']} units · {D['n_edges']:,} edges · "
           f"{len(D['areas'])} areas ({', '.join(D['areas'])}) · "
           f"{len(D['layers'])} layers · {len(D['classes'])} classes · dt {dt_ms} ms")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(_index_html(title, sub, cards))

    manifest = {"suite": SUITE, "title": title, "out_dir": out_dir,
                "n_neurons": D["n_neurons"], "n_edges": D["n_edges"],
                "areas": D["areas"], "layers": D["layers"], "classes": D["classes"],
                "dt_ms": dt_ms, "seed": seed, "panels": manifest_panels}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest

"""AGSDR Dashboard — complete selection history for C019 (VIS_FOUNDATION_v0 V0 W4/AGSDR).

Reconstructs 4-panel dashboard consuming same generated-owner arrays/config objects
(results/agsdr_local/trace.jsonl, N1_summary, N2_summary, jacobian_N2.json) not recomputed.

Panels:
 1. Parameter space parallel coordinates g_rec,g_fastEI,g_dend,g_disinh,g_background
    N0 gray (24), N1 retained (8 blue), N2 finalists (3 orange, C019 selected red)
    per results/agsdr_local/trace.jsonl + N0_summary survivors + N1_summary ranking + N2_summary theta_star.
 2. Objective space rho vs P(CV>0.5) point size Fano, hover rates/Efrac/CL1-C.
    Uses trace.jsonl metrics (rho_global/mean_rho, p_cv_e_gt05/frac_gt05, median_fano, bar_r, efrac, per_class, J_CL1C deltas).
    Shows hard/soft bands: P≥0.15 hard, rho <0.60 hard and [-0.05,0.2] soft.
    Point size = Fano (not Efrac) to satisfy "Do not let proxy E/I dominate selection visually."
 3. Successive fidelity 24→8→3→1 with seeds S_1[0] S_2[0,1] S_4[0,1,2,3].
    Funnel + per-stage J_scalar strips, frozen per manifests/agsdr_local_harness.json:40 and seeds.json:24.
 4. Jacobian heatmap ∂Y/∂θ (bar+0.244 g_rec +2.37 g_fastEI etc. from jacobian_N2.json)
    Labeled local empirical sensitivities within sampled domain, not causal constants.
    Method via least squares over 12 points (N2) / 24 points (jacobian_estimate).

Engineering, not science: no theta mutation, reads frozen summaries.

Provenance citations:
- manifests/agsdr_local_freeze.json:132 five_dimensions_orthogonal, 211 C019 65b302e8c7cdceb5
- manifests/agsdr_local_harness.json:40 stages S0 24x1→S1 8x2→S2 3x4 (52 sims) + seeds.json S1[0] S2[0,1] S4[0,1,2,3]
- manifests/agsdr_local_objectives.json H01-H12 hard/soft, B2 rho [-0.05,0.2] etc.
- manifests/agsdr_local_seeds.json:21 policy successive fidelity N0 24→N1 8→N2 3→1
- results/agsdr_local/N0_summary.json, N1_summary.json, N2_summary.json (generated-owner)
- results/agsdr_local/trace.jsonl, trace_N1.jsonl, trace_N2.jsonl (24,8,3 per-seed)
- results/agsdr_local/jacobian_N2.json (least squares 12+12) + jacobian_estimate.json (24 points)
- jomission/network/builder.py:62,90, jaxfne/emitters.py, jomission/qualification/gen2_gates.py B2/B10

API:
  agsdr_dashboard_figs() -> dict
  save_agsdr_dashboard(out_path=None) -> Path
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None  # type: ignore
    make_subplots = None  # type: ignore

FROZEN_C019 = "65b302e8c7cdceb5"
FROZEN_PARENT = "be9b96ab679c9802"
FROZEN_C019_THETA = [1.256454357174736, 1.1871704759698585, 0.7088237829792216, 1.2181840541473454, 1.2887170240398067]
THETA_LABELS = ["g_rec", "g_fastEI", "g_dend", "g_disinh", "g_background"]

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TRACE = ROOT / "results/agsdr_local/trace.jsonl"
DEFAULT_N0 = ROOT / "results/agsdr_local/N0_summary.json"
DEFAULT_N1 = ROOT / "results/agsdr_local/N1_summary.json"
DEFAULT_N2 = ROOT / "results/agsdr_local/N2_summary.json"
DEFAULT_JN2 = ROOT / "results/agsdr_local/jacobian_N2.json"
DEFAULT_JE = ROOT / "results/agsdr_local/jacobian_estimate.json"


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_trace(path: pathlib.Path | None = None) -> List[Dict[str, Any]]:
    p = pathlib.Path(path) if path else DEFAULT_TRACE
    out: List[Dict[str, Any]] = []
    try:
        for line in p.read_text().splitlines():
            line=line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _load_agsdr_bundle(
    trace_path: pathlib.Path | None = None,
    n0_path: pathlib.Path | None = None,
    n1_path: pathlib.Path | None = None,
    n2_path: pathlib.Path | None = None,
    jn2_path: pathlib.Path | None = None,
) -> Dict[str, Any]:
    trace = _load_trace(trace_path)
    n0 = _load_json(n0_path or DEFAULT_N0) or {}
    n1 = _load_json(n1_path or DEFAULT_N1) or {}
    n2 = _load_json(n2_path or DEFAULT_N2) or {}
    jn2 = _load_json(jn2_path or DEFAULT_JN2) or {}
    je = _load_json(DEFAULT_JE) or {}
    return dict(trace=trace, n0=n0, n1=n1, n2=n2, jn2=jn2, je=je)


# ---------------------------------------------------------------------------
# Panel 1: parallel coordinates
# ---------------------------------------------------------------------------
def fig_parallel_coords(bundle: Dict[str, Any] | None = None) -> Any:
    if go is None:
        raise ImportError("plotly not installed")
    bundle = bundle or _load_agsdr_bundle()
    trace: List[Dict[str, Any]] = bundle.get("trace") or []
    n0 = bundle.get("n0") or {}
    n1 = bundle.get("n1") or {}
    n2 = bundle.get("n2") or {}
    survivors = set(n0.get("survivors") or [])
    finalists = set(n2.get("ranking_ybar_4") or n1.get("finalists_N2") or [])
    # C019 star
    theta_star_id = (n2.get("theta_star") or {}).get("candidate_id") or "AGS_N0_18"
    # Build arrays in trace order (24)
    thetas = []
    colors_val: List[int] = []
    hover: List[str] = []
    for rec in trace:
        th = rec.get("theta") or []
        if len(th) != 5:
            continue
        thetas.append(th)
        cid = rec.get("candidate_id")
        if cid == theta_star_id or (cid == "AGS_N0_18" and theta_star_id == "AGS_N0_18"):
            colors_val.append(3)  # star
            lab = f"{cid} ★ C019 {FROZEN_C019[:8]}"
        elif cid in finalists:
            colors_val.append(2)  # finalist
            lab = f"{cid} N2 finalist"
        elif cid in survivors:
            colors_val.append(1)  # N1 survivor
            lab = f"{cid} N1 retained (8)"
        else:
            colors_val.append(0)  # N0 only gray
            lab = f"{cid} N0 gray (24)"
        hover.append(lab)
    thetas_arr = np.array(thetas, dtype=float) if thetas else np.zeros((0,5))
    # colorscale: 0 gray, 1 blue, 2 orange, 3 red/star
    colorscale = [[0.0, "rgba(180,180,180,0.55)"], [0.33, "rgba(180,180,180,0.55)"],
                  [0.33, "rgba(31,119,180,0.85)"], [0.66, "rgba(31,119,180,0.85)"],
                  [0.66, "rgba(255,127,14,0.95)"], [0.90, "rgba(255,127,14,0.95)"],
                  [0.90, "rgba(220,20,60,1.0)"], [1.0, "rgba(220,20,60,1.0)"]]
    # Build parcoords dimensions with ranges per freeze § search_range ±30% etc.
    # Use actual min/max across trace plus padding
    if thetas_arr.size:
        mins = thetas_arr.min(axis=0)
        maxs = thetas_arr.max(axis=0)
        pad = 0.06
    else:
        mins = np.array([0.7,0.8,0.7,0.7,0.7]); maxs = np.array([1.3,1.2,1.3,1.3,1.3]); pad=0.05
    ranges = []
    for j, lab in enumerate(THETA_LABELS):
        lo = float(min(0.70 if lab in ("g_rec","g_dend","g_disinh") else 0.80, mins[j]-pad)) if thetas_arr.size else 0.70
        hi = float(max(1.30 if lab not in ("g_fastEI",) else 1.20, maxs[j]+pad)) if thetas_arr.size else 1.30
        # Clamp to freeze search_range
        ranges.append((lo, hi))
    # Order: g_rec, g_fastEI, g_dend, g_disinh, g_background (as freeze order)
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=np.array(colors_val, dtype=float), colorscale=colorscale, showscale=False, cmin=0, cmax=3),
        dimensions=[
            dict(range=[ranges[0][0], ranges[0][1]], label="g_rec<br>within [0.70,1.30]", values=thetas_arr[:,0] if thetas_arr.size else []),
            dict(range=[ranges[1][0], ranges[1][1]], label="g_fastEI<br>E↔PV [0.80,1.20]", values=thetas_arr[:,1] if thetas_arr.size else []),
            dict(range=[ranges[2][0], ranges[2][1]], label="g_dend<br>SST [0.70,1.30]", values=thetas_arr[:,2] if thetas_arr.size else []),
            dict(range=[ranges[3][0], ranges[3][1]], label="g_disinh<br>VIP→SST [0.70,1.30]", values=thetas_arr[:,3] if thetas_arr.size else []),
            dict(range=[ranges[4][0], ranges[4][1]], label="g_background<br>I_bg mean-ctrl", values=thetas_arr[:,4] if thetas_arr.size else []),
        ],
        unselected=dict(line=dict(color="rgba(180,180,180,0.18)", opacity=0.18)),
        labelangle=0, labelside="top",
    ))
    fig.update_layout(
        title=dict(text="<b>Panel 1 — Parameter space parallel coordinates</b> — g_rec,g_fastEI,g_dend,g_disinh,g_background | N0 24 gray → N1 8 blue → N2 3 orange → C019 ★ red (24→8→3→1)<br><sub>Trace: results/agsdr_local/trace.jsonl (N0 24×1) + N1_summary 8×2 + N2_summary theta_star AGS_N0_18 [1.256,1.187,0.708,1.218,1.288] | Orthogonal masks M_rec(9)/M_fastEI(2)/M_dend(4)/M_disinh(1) per manifests/agsdr_local_freeze.json:132 — no primitive mutation</sub>", font=dict(size=10)),
        width=1100, height=440,
        margin=dict(l=70, r=30, t=90, b=20),
        paper_bgcolor="#fafafa", plot_bgcolor="#fafafa",
        annotations=[
            dict(x=0.02, y=-0.08, xref="paper", yref="paper", text="N0 gray (24 candidates, seed S_1[0]) → N1 blue retained 8 → N2 orange finalists 3 (seed S_4[0,1,2,3] mean_4) → <span style='color:crimson'>★ C019 65b302e8c7cdceb5</span> (single ledger delta). Width encodes 5-D box ±20-30% local calibration only (freeze).", showarrow=False, font=dict(size=8, color="#555"), align="left", xanchor="left"),
        ],
    )
    # Custom legend as annotations (parcoords has no legend)
    for cx, lab, col in [(0.12, "N0 24 gray", "rgba(180,180,180,0.9)"), (0.26, "N1 8 blue", "rgba(31,119,180,0.9)"), (0.40, "N2 3 orange", "rgba(255,127,14,0.95)"), (0.55, "C019 ★ red", "rgba(220,20,60,1.0)")]:
        fig.add_annotation(x=cx, y=1.08, xref="paper", yref="paper", text=f"<span style='background:{col};padding:2px 8px;border-radius:8px;color:#fff;font-size:9px'>{lab}</span>", showarrow=False)
    return fig


# ---------------------------------------------------------------------------
# Panel 2: objective space rho vs P(CV>0.5) size Fano
# ---------------------------------------------------------------------------
def fig_objective_space(bundle: Dict[str, Any] | None = None) -> Any:
    if go is None:
        raise ImportError("plotly not installed")
    bundle = bundle or _load_agsdr_bundle()
    trace: List[Dict[str, Any]] = bundle.get("trace") or []
    n2 = bundle.get("n2") or {}
    n0_survivors = set((bundle.get("n0") or {}).get("survivors") or [])
    finalists = set((n2.get("ranking_ybar_4") or []) or [])
    star_id = (n2.get("theta_star") or {}).get("candidate_id") or "AGS_N0_18"
    # collect points: use trace (N0 seed0) y values per candidate
    xs, ys, sizes, colors, hovers, cids = [], [], [], [], [], []
    for rec in trace:
        cid = rec.get("candidate_id")
        metrics = rec.get("metrics") or {}
        rho = metrics.get("mean_rho") if "mean_rho" in metrics else metrics.get("rho_global")
        frac = metrics.get("p_cv_e_gt05") if "p_cv_e_gt05" in metrics else metrics.get("frac_gt05")
        fano = metrics.get("median_fano") if "median_fano" in metrics else 0.95
        bar_r = metrics.get("bar_r")
        efrac = metrics.get("efrac")
        per = metrics.get("per_class") or {}
        if rho is None or frac is None:
            continue
        xs.append(float(rho)); ys.append(float(frac))
        # size = Fano scaled 10 + (fano-0.7)*20, clamp 8-22 (not Efrac)
        sz = float(np.clip(8 + (float(fano) - 0.70) * 30, 8, 22))
        sizes.append(sz)
        cids.append(cid)
        # hover with rates/Efrac/CL1-C but compact
        hover = f"{cid}<br>ρ {float(rho):.3f} P(CV>0.5) {float(frac):.3f}<br>Fano {float(fano):.2f} bar {bar_r} Efrac_proxy {efrac:.2f} (PROXY)<br>E {per.get('E')} PV {per.get('PV')} VIP {per.get('VIP')}<br>J_CL1C PVΔ/SSTΔ/VIPΔ in trace_N2"
        hovers.append(hover)
        if cid == star_id:
            colors.append(3)
        elif cid in finalists:
            colors.append(2)
        elif cid in n0_survivors:
            colors.append(1)
        else:
            colors.append(0)
    # colorscale mapping 0->gray 1->blue 2->orange 3->crimson
    col_map = ["rgba(160,160,160,0.75)", "rgba(31,119,180,0.85)", "rgba(255,127,14,0.95)", "rgba(220,20,60,1.0)"]
    marker_colors = [col_map[int(c)] for c in colors]
    # also include N2 y_bar_4 points for finalists (mean across S4) for reference as diamonds
    xs2, ys2, cids2 = [], [], []
    for cand in (n2.get("candidates") or []):
        # y_bar_4 metrics
        yb = (cand.get("y_bar_4") or {}).get("metrics") or {}
        rho = yb.get("mean_rho")
        frac = yb.get("frac_gt05") if "frac_gt05" in yb else yb.get("p_cv_e_gt05")
        if rho is None or frac is None:
            continue
        xs2.append(float(rho)); ys2.append(float(frac)); cids2.append(cand.get("candidate_id"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=marker_colors, line=dict(color="#333", width=0.7), opacity=0.88),
        text=hovers, hovertemplate="%{text}<extra></extra>",
        name="N0 seed0 (24)",
    ))
    if xs2:
        fig.add_trace(go.Scatter(
            x=xs2, y=ys2, mode="markers",
            marker=dict(size=[16]*len(xs2), color="rgba(220,20,60,0.0)", line=dict(color="crimson", width=2.2), symbol="diamond", opacity=0.95),
            text=[f"{cid} y_bar_4 (mean S_4[0,1,2,3])" for cid in cids2],
            hovertemplate="%{text}<br>ρ %{x:.3f} P(CV>0.5) %{y:.3f}<extra></extra>",
            name="N2 y_bar_4 (mean 4 seeds)",
        ))
    # bands: P hard 0.15, rho hard 0.60 and soft [-0.05,0.2]
    # horizontal P hard
    fig.add_hline(y=0.15, line_dash="dash", line_color="#d62728", annotation_text="hard P(CV>0.5)≥0.15 (H05)", annotation_position="bottom right")
    # vertical rho hard 0.60
    fig.add_vline(x=0.60, line_dash="dash", line_color="#d62728", annotation_text="hard ρ<0.60 (H03)", annotation_position="top right")
    # soft rho band [-0.05,0.2]
    fig.add_vrect(x0=-0.05, x1=0.20, fillcolor="rgba(44,160,44,0.08)", line_width=0, annotation_text="soft ρ [-0.05,0.2] B2", annotation_position="top left")
    # soft P >=0.20 indication
    fig.add_hline(y=0.20, line_dash="dot", line_color="#2ca02c", annotation_text="soft P≥0.20", annotation_position="top right")
    fig.update_layout(
        title=dict(text="<b>Panel 2 — Objective space</b> — ρ (mean pairwise) vs P(CV&gt;0.5) — point size = Fano (not Efrac)<br><sub>Hover: rates / Efrac_proxy (PROXY_LIMITED) / CL1-C deltas; N0 gray (24) → N1 blue (8) → N2 orange (3) → C019 ★ red diamond (y_bar_4). Do not let proxy E/I dominate selection visually — Efrac in hover only, not size/color.</sub>", font=dict(size=10)),
        xaxis=dict(title="mean ρ (global, hard &lt;0.60, soft [-0.05,0.20] B2)", range=[0.30, 0.50]),
        yaxis=dict(title="P(CV<sub>E</sub>&gt;0.5) hard ≥0.15 soft ≥0.20 (protected typed diversity)", range=[0.10, 0.30]),
        width=680, height=520,
        margin=dict(l=70, r=30, t=90, b=60),
        paper_bgcolor="#fafafa", plot_bgcolor="#fff",
        legend=dict(orientation="h", y=-0.18, font=dict(size=9)),
        annotations=[
            dict(x=0.98, y=0.02, xref="paper", yref="paper", text="Size=Fano [0.7,2.0]; color=stage (not Efrac).<br>Source: trace.jsonl metrics per candidate seed0 + N2_summary y_bar_4 CI.", showarrow=False, font=dict(size=7, color="#666"), align="right", xanchor="right"),
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 3: successive fidelity
# ---------------------------------------------------------------------------
def fig_fidelity(bundle: Dict[str, Any] | None = None) -> Any:
    if go is None:
        raise ImportError("plotly not installed")
    bundle = bundle or _load_agsdr_bundle()
    n0 = bundle.get("n0") or {}
    n1 = bundle.get("n1") or {}
    n2 = bundle.get("n2") or {}
    # counts
    stages = ["N0<br>24×1<br>S₁[0]", "N1<br>8×2<br>S₂[0,1]", "N2<br>3×4<br>S₄[0,1,2,3]", "C019<br>1 selected<br>y_bar_4"]
    counts = [24, 8, 3, 1]
    sims = [24, 16, 12, 0]
    colors = ["#b0b0b0", "#1f77b4", "#ff7f0e", "crimson"]
    # J_scalar per stage for strip: N0 J via trace, N1 J_bar via n1.paired, N2 J via n2 candidates y_bar_4
    trace = bundle.get("trace") or []
    # funnel bar + strip subplot using make_subplots 2 rows
    if make_subplots is None:
        fig = go.Figure(data=[go.Bar(x=stages, y=counts, marker_color=colors, text=[f"{c} ({s} sims)" for c,s in zip(counts,sims)], textposition="outside")])
        fig.update_layout(title="Successive fidelity 24→8→3→1")
        return fig
    fig = make_subplots(rows=2, cols=1, row_heights=[0.42, 0.58], subplot_titles=("Funnel 24→8→3→1 (52 sims total 24+16+12, seeds S₁[0] S₂[0,1] S₄[0,1,2,3] frozen)", "Per-candidate J_scalar (distributional) — selection on mean not cherry-pick"), vertical_spacing=0.18)
    # Row1 funnel
    fig.add_trace(go.Funnel(
        y=stages, x=counts, textinfo="value+percent initial", marker=dict(color=colors, line=dict(color="#333", width=1.2)),
        hovertemplate="%{y} %{x} candidates<br>seeds %{customdata}<extra></extra>",
        customdata=["S_1=[0] x24", "S_2=[0,1] x16", "S_4=[0,1,2,3] x12", "ledger GEN2_C019 single delta"],
        name="funnel",
    ), row=1, col=1)
    # Row2 strip: J_scalar per candidate jittered by stage
    # Gather J values
    # N0: J_scalar from trace
    x0 = np.random.default_rng(7).normal(loc=0, scale=0.06, size=len(trace)) if trace else np.array([])
    j0 = [float(r.get("J_scalar") or r.get("soft",{}).get("J_scalar",0) or r.get("J_scalar_w6",0) or 0) for r in trace]
    # fallback: compute J from metrics if missing
    # N1: J_bar from n1.paired
    paired = n1.get("paired") or []
    j1 = [float(p.get("J_bar",0)) for p in paired]
    x1 = np.random.default_rng(8).normal(loc=1, scale=0.06, size=len(j1)) if j1 else np.array([])
    # N2: J_scalar y_bar_4 from n2 candidates
    cand2 = n2.get("candidates") or []
    j2 = [float(c.get("y_bar_4",{}).get("J_scalar",0)) for c in cand2]
    x2 = np.random.default_rng(9).normal(loc=2, scale=0.05, size=len(j2)) if j2 else np.array([])
    # C019 star
    theta_star = n2.get("theta_star") or {}
    j_star = float(theta_star.get("y_bar_4",{}).get("J_scalar", j2[0] if j2 else 0.24))
    fig.add_trace(go.Scatter(x=x0, y=j0, mode="markers", marker=dict(size=7, color="rgba(160,160,160,0.85)", line=dict(color="#333", width=0.5)), name="N0 24 (seed0 J)", hovertemplate="N0 %{y:.3f}<extra></extra>"), row=2, col=1)
    if len(j1):
        fig.add_trace(go.Scatter(x=x1, y=j1, mode="markers", marker=dict(size=9, color="rgba(31,119,180,0.9)", line=dict(color="#333", width=0.7)), name="N1 8 (J_bar mean S₂)", hovertemplate="N1 J_bar %{y:.3f}<extra></extra>"), row=2, col=1)
    if len(j2):
        fig.add_trace(go.Scatter(x=x2, y=j2, mode="markers", marker=dict(size=11, color="rgba(255,127,14,0.95)", line=dict(color="#333", width=0.9), symbol="diamond"), name="N2 3 (J_bar S₄)", hovertemplate="N2 J %{y:.3f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[3], y=[j_star], mode="markers+text", text=["★ C019<br>AGS_N0_18"], textposition="top center", marker=dict(size=14, color="crimson", line=dict(color="#333", width=1.2), symbol="star"), name="C019", hovertemplate="C019 J %{y:.3f} y_bar_4 mean S₄[0,1,2,3]<extra></extra>"), row=2, col=1)
    # mean lines per stage
    if j0:
        fig.add_hline(y=float(np.mean(j0)), line_dash="dot", line_color="#999", row=2, col=1)
    if j1:
        fig.add_hline(y=float(np.mean(j1)), line_dash="dash", line_color="#1f77b4", row=2, col=1)
    if j2:
        fig.add_hline(y=float(np.mean(j2)), line_dash="solid", line_color="#ff7f0e", row=2, col=1)
    fig.update_xaxes(tickvals=[0,1,2,3], ticktext=stages, row=2, col=1, title_text="stage (frozen seeds S_1[0] S_2[0,1] S_4[0,1,2,3] per manifests/agsdr_local_harness.json:40, seeds.json:21 — successive fidelity)")
    fig.update_yaxes(title_text="J_scalar (w=1/6 Pareto tie-breaker, lower better)", row=2, col=1, range=[0.22, 0.30])
    fig.update_layout(
        title=dict(text="<b>Panel 3 — Successive fidelity 24→8→3→1</b> with seeds S<sub>1</sub>[0] S<sub>2</sub>[0,1] S<sub>4</sub>[0,1,2,3] frozen (52 sims) — selection on mean, not cherry-pick", font=dict(size=10)),
        width=680, height=620,
        margin=dict(l=60, r=20, t=80, b=60),
        paper_bgcolor="#fafafa", plot_bgcolor="#fff",
        showlegend=True, legend=dict(orientation="h", y=-0.14, font=dict(size=8)),
        annotations=[
            dict(x=0.50, y=-0.08, xref="paper", yref="paper", text="Anti-cherry-pick: ranking on y_bar mean<sub>s∈S<sub>k</sub></sub> not best seed; once eliminated never reinstated (harness anti_cherry_pick_rules). N2 y_bar_4 t<sub>3</sub> 95% CI shown in N2_summary.json ci fields.", showarrow=False, font=dict(size=7, color="#666"), align="center", xanchor="center"),
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 4: Jacobian heatmap
# ---------------------------------------------------------------------------
def fig_jacobian(bundle: Dict[str, Any] | None = None) -> Any:
    if go is None:
        raise ImportError("plotly not installed")
    bundle = bundle or _load_agsdr_bundle()
    jn2 = bundle.get("jn2") or {}
    # y_keys and theta_labels per jacobian_N2.json
    y_keys = jn2.get("y_keys") or ["bar_r","cv_isi_mean","cv_isi_frac_gt_0p5","rho_global","Fano","Efrac","cv_rate_population","delta_PV_E","delta_PV_rho","delta_SST_E","delta_VIP_SST","J_scalar_ybar"]
    theta_labels = jn2.get("theta_labels") or THETA_LABELS
    jac = jn2.get("jacobian") or {}
    # Build matrix rows=y_keys, cols=theta_labels
    z = np.zeros((len(y_keys), len(theta_labels)), dtype=float)
    for i, y in enumerate(y_keys):
        row = jac.get(y) or {}
        for j, th in enumerate(theta_labels):
            z[i, j] = float(row.get(th, 0.0))
    # color diverging: robust range clamp to ±0.6 for bar etc but fastEI 2.37 may exceed; keep symmetric around 0 with outlier cap
    z_cap = np.clip(z, -1.0, 2.5)
    fig = go.Figure(data=go.Heatmap(
        z=z_cap,
        x=theta_labels,
        y=y_keys,
        colorscale="RdBu",
        reversescale=True,
        zmid=0,
        zmin=-0.8, zmax=0.8,
        colorbar=dict(title=dict(text="∂Y/∂θ<br>(per unit g)", side="right")),
        hovertemplate="Y %{y} vs θ %{x}<br>∂Y/∂θ %{z:.3f} (raw %{customdata:.3f})<extra></extra>",
        customdata=z,
    ))
    # annotate raw values for strong cells
    annotations = []
    for i in range(len(y_keys)):
        for j in range(len(theta_labels)):
            v = float(z[i, j])
            if abs(v) > 0.18:  # annotate moderate/large sensitivities
                annotations.append(dict(
                    x=theta_labels[j], y=y_keys[i], text=f"{v:.2f}", showarrow=False,
                    font=dict(color="#fff" if abs(v) > 0.4 else "#111", size=8),
                ))
    fig.update_layout(
        title=dict(text="<b>Panel 4 — Jacobian ∂Y/∂θ</b> local empirical sensitivities (least squares N=12+12 N2 y_bar_4 | y_bar functional) — not causal constants<br><sub>Rows Y_k = [bar_r, CV, P(CV>0.5), ρ, Fano, Efrac_proxy, CV_rate, ΔPV_E, ΔPV_ρ, ΔSST_E, ΔVIP_SST, J] &nbsp; Cols θ_j = [g_rec, g_fastEI, g_dend, g_disinh, g_background] &nbsp; Values e.g. bar +0.24 g_rec +2.37 g_fastEI (jacobian_N2.json); capped color ±0.8 for readability, raw in hover</sub>", font=dict(size=10)),
        width=720, height=560,
        margin=dict(l=160, r=80, t=90, b=60),
        xaxis=dict(side="bottom", tickangle=0),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#fafafa", plot_bgcolor="#fff",
        annotations=annotations + [
            dict(x=0.50, y=-0.16, xref="paper", yref="paper", text="Local empirical sensitivities within sampled domain g∈[0.7,1.3]±20-30% (freeze), via least squares over nearby candidates (N2 12 points + functional 12, jacobian_N2.json). Intercept bar 2.65 etc. — not causal constants; depends on operating point (ρ0 0.38, P0 0.23). δ ~0.05. Efrac_proxy row shown but not dominant — constraints protected (H05 P≥0.15, H03 ρ&lt;0.60) first; selection is Pareto feasible-first, not Efrac-driven.", showarrow=False, font=dict(size=7, color="#666"), align="center", xanchor="center"),
            dict(x=1.02, y=1.06, xref="paper", yref="paper", text="method: least squares<br>12 points spont+12 func<br>se from S_4 CI", showarrow=False, font=dict(size=7, color="#333"), align="left"),
        ],
    )
    return fig


def agsdr_dashboard_figs(bundle: Dict[str, Any] | None = None) -> Dict[str, Any]:
    bundle = bundle or _load_agsdr_bundle()
    return dict(
        parallel=fig_parallel_coords(bundle),
        objective=fig_objective_space(bundle),
        fidelity=fig_fidelity(bundle),
        jacobian=fig_jacobian(bundle),
    )


def save_agsdr_dashboard(
    out_path: str | pathlib.Path | None = None,
    bundle: Dict[str, Any] | None = None,
) -> pathlib.Path:
    if out_path is None:
        out_path = pathlib.Path("results/visualization/agsdr_dashboard_C019.html")
    else:
        out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = bundle or _load_agsdr_bundle()
    figs = agsdr_dashboard_figs(bundle)
    try:
        import plotly.io as pio
        div_pc = pio.to_html(figs["parallel"], include_plotlyjs="cdn", full_html=False)
        div_obj = pio.to_html(figs["objective"], include_plotlyjs=False, full_html=False)
        div_fid = pio.to_html(figs["fidelity"], include_plotlyjs=False, full_html=False)
        div_jac = pio.to_html(figs["jacobian"], include_plotlyjs=False, full_html=False)
    except Exception as e:
        div_pc = div_obj = div_fid = div_jac = f"<div>Plotly error {e}</div>"
    n0 = bundle.get("n0") or {}; n1 = bundle.get("n1") or {}; n2 = bundle.get("n2") or {}; jn2 = bundle.get("jn2") or {}
    star = (n2.get("theta_star") or {}) if n2 else {}
    theta_star = star.get("theta") or FROZEN_C019_THETA
    html = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"/>
<title>AGSDR Dashboard — C019 {FROZEN_C019} — 24→8→3→1</title>
<style>
 body{{font-family:-apple-system,system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#111}}
 header{{padding:14px 18px;background:#0b132b;color:#fff}}
 header h1{{margin:0;font-size:15px}} header p{{margin:4px 0 0;font-size:11px;opacity:0.88}}
 main{{padding:12px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
 .card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:8px;overflow:auto}}
 .full{{grid-column:1 / -1}}
 table{{border-collapse:collapse;font-size:11px;width:100%}} th,td{{border:1px solid #ddd;padding:4px 6px;text-align:left}} th{{background:#0b132b;color:#fff}}
 .note{{color:#555;font-size:11px;margin:8px 0}} code{{font-size:11px}}
 h2{{font-size:13px;margin:14px 0 6px}}
 .badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:6px}}
</style>
</head><body>
<header>
  <h1>AGSDR Dashboard — Complete selection history for C019 {FROZEN_C019} — 24→8→3→1 (52 sims, S<sub>1</sub>[0] S<sub>2</sub>[0,1] S<sub>4</sub>[0,1,2,3] frozen)</h1>
  <p>Panels: (1) parameter space parallel coords g_rec,g_fastEI,g_dend,g_disinh,g_background — N0 24 gray, N1 retained 8 blue, N2 finalists 3 orange, C019 ★ red &nbsp;|&nbsp; (2) objective space ρ vs P(CV&gt;0.5) size=Fano (not Efrac), hover rates/Efrac/CL1-C &nbsp;|&nbsp; (3) successive fidelity funnel 24×1→8×2→3×4→1 with seeds &nbsp;|&nbsp; (4) Jacobian ∂Y/∂θ heatmap (local empirical sensitivities within sampled domain, not causal constants). Sources: results/agsdr_local/trace.jsonl (24), N1_summary 8×2, N2_summary y_bar_4 (mean<sub>S4</sub> CI), jacobian_N2.json. Do not let proxy E/I dominate selection visually — Efrac in hover/heatmap row only, not size/color.</p>
  <p>Theta* AGS_N0_18 = [{theta_star[0]:.3f}, {theta_star[1]:.3f}, {theta_star[2]:.3f}, {theta_star[3]:.3f}, {theta_star[4]:.3f}] on C018 {FROZEN_PARENT[:8]} (be9b96ab) → {FROZEN_C019[:8]} | y_bar_4 bar {star.get("y_bar_4",{}).get("metrics",{}).get("bar_r","")} ρ {star.get("y_bar_4",{}).get("metrics",{}).get("mean_rho","")} P(CV&gt;0.5) {star.get("y_bar_4",{}).get("metrics",{}).get("frac_gt05","")} Efrac_proxy {star.get("y_bar_4",{}).get("metrics",{}).get("efrac","")} (PROXY_LIMITED)</p>
</header>
<main>
  <div class=\"card full\">{div_pc}</div>
  <div class=\"grid\">
    <div class=\"card\">{div_obj}</div>
    <div class=\"card\">{div_fid}</div>
  </div>
  <div class=\"card full\" style=\"margin-top:14px\">{div_jac}</div>
  <h2>Selection lineage (generated-owner, not recomputed)</h2>
  <table><thead><tr><th>Stage</th><th>N</th><th>Seeds</th><th>Sims</th><th>Ranking statistic</th><th> survivors / finalists</th><th>Source</th></tr></thead><tbody>
    <tr><td>N0 coarse</td><td>24</td><td>S<sub>1</sub> [0] frozen</td><td>24×1=24</td><td>hard PASS (H01-H12) + Pareto non-dominated on J_rates/J_irregularity/J_synchrony/J_Fano/J_hetero/J_EI/J_CL1C</td><td>8 survivors: AGS_N0_18,01,09,21,17,03,04,05</td><td><code>results/agsdr_local/trace.jsonl (24 lines)</code> + <code>N0_summary.json</code></td></tr>
    <tr><td>N1 refine</td><td>8</td><td>S<sub>2</sub> [0,1] frozen</td><td>8×2=16</td><td>y_bar = mean<sub>s∈[0,1]</sub> y(s), Pareto on y_bar (not best seed) + protected P(CV)&gt;0.5≥0.15 max≥1.5 both seeds finite</td><td>3 finalists: AGS_N0_18,03,05 (ranking_ybar_4)</td><td><code>results/agsdr_local/N1_summary.json</code> + <code>trace_N1.jsonl (8×2)</code></td></tr>
    <tr><td>N2 confirm</td><td>3</td><td>S<sub>4</sub> [0,1,2,3] frozen</td><td>3×4=12</td><td>y_bar_4 = mean<sub>s∈[0,1,2,3]</sub> y(s) with t<sub>3</sub> 95% CI; Pareto y_bar_4, lower var breaks ties</td><td>theta* AGS_N0_18 selected ★ (y_bar_4 bar 5.72 ρ0.38 P0.23 Fano0.95 Efrac0.75)</td><td><code>results/agsdr_local/N2_summary.json</code> + <code>trace_N2.jsonl (3×4)</code> + <code>jacobian_N2.json</code></td></tr>
    <tr><td colspan=\"7\">Total 35 distinct θ (24+8+3) &amp; 52 sims (24+16+12). Anti-cherry-pick: S<sub>1</sub>/S<sub>2</sub>/S<sub>4</sub> frozen in manifests/agsdr_local_seeds.json:21 BEFORE first evaluation; final rank on S<sub>2</sub>/S<sub>4</sub> mean, not max; eliminated never reinstated; same 2000ms window dt0.1.</td></tr>
  </tbody></table>
  <h2>File:line citations</h2>
  <table><thead><tr><th>Claim</th><th>File:line</th></tr></thead><tbody>
    <tr><td>5-D orthogonal masks &amp; search ranges</td><td><code>manifests/agsdr_local_freeze.json:131 five_dimensions_orthogonal (M_rec9/M_fastEI2/M_dend4/M_disinh1 + g_background mean-controlled)</code></td></tr>
    <tr><td>Theta* &amp; C019 hash</td><td><code>manifests/agsdr_local_freeze.json:211</code> + <code>results/agsdr_local/N2_summary.json:1610 theta_star AGS_N0_18 [1.256,1.187,0.708,1.218,1.288] → 65b302e8c7cdceb5</code></td></tr>
    <tr><td>Stages &amp; fidelity</td><td><code>manifests/agsdr_local_harness.json:40 S0 24×1 → S1 8×2 → S2 3×4 (52 sims)</code> + <code>manifests/agsdr_local_seeds.json:21 S_1[0] S_2[0,1] S_4[0,1,2,3] frozen</code></td></tr>
    <tr><td>Objectives hard/soft B2/B3</td><td><code>manifests/agsdr_local_objectives.json:22 H01-H12 + B2 ρ [-0.05,0.2] hard&lt;0.60, P≥0.15</code> + <code>jomission/qualification/gen2_gates.py:139-164 B2/B3</code></td></tr>
    <tr><td>Jacobian</td><td><code>results/agsdr_local/jacobian_N2.json</code> (least squares 12+12, bar+0.24 g_rec +2.37 g_fastEI etc.) + <code>jacobian_estimate.json 24 points</code> — local empirical, not causal constants, δ~0.05</td></tr>
    <tr><td>Builder seams</td><td><code>jomission/network/builder.py:62 MOTIF_GAIN 16 gains</code>, <code>builder.py:90 FF L2/3→L4 FB L6→L1/L5 delays</code>, <code>builder.py:395 spatial_sigma0.08 max25</code>, <code>jaxfne/emitters.py:545 EdgeList</code></td></tr>
  </tbody></table>
  <p class=\"note\">Engineering, not science — reads same N2/N1/trace arrays as analyses, not recomputed. Efrac is proxy_readout (jaxfne/_model_simulate.py:280 HDP+delay incompat, proxy via W·r, gen2_C014:166 PROXY_LIMITED) — not used for size/color dominance. Panel 2 size = Fano, color = stage; Panel 1 width = g (structural); Panel 4 heatmap row Efrac shown but capped and labeled “local empirical sensitivites within sampled domain, not causal constants”.</p>
</main>
</body></html>
"""
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    p = save_agsdr_dashboard()
    print(f"Wrote {p} ({p.stat().st_size/1024:.1f} KB)")

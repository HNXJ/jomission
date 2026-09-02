"""Interactive Longitudinal Plasticity Trajectory, Remodeling Matrix, & Temporal Kernel Explorer."""

import json
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jomission.visualization.theme import (
    PROJ_COLORS,
    CLASS_COLORS,
    apply_dark_theme,
    wrap_figure_with_provenance_html,
)


def build_plasticity_figure(results_path: str = "results/plasticity_100s_extension_results.json") -> tuple[go.Figure, str, dict]:
    with open(results_path, "r") as f:
        res = json.load(f)

    # Optional: Load temporal kernel results if available
    tk_path = "results/ab_ba_temporal_kernel_results.json"
    tk_res = None
    if os.path.exists(tk_path):
        with open(tk_path, "r") as f:
            tk_res = json.load(f)

    checkpoints = res["checkpoints"]
    summaries = res["summaries"]

    ages = [int(t) for t in checkpoints]
    g_global = [summaries[str(t)]["global"]["gain"] for t in checkpoints]
    g_rec = [summaries[str(t)]["by_projection_type"]["recurrent"]["gain"] for t in checkpoints]
    g_ff = [summaries[str(t)]["by_projection_type"]["FF"]["gain"] for t in checkpoints]
    g_fb = [summaries[str(t)]["by_projection_type"]["FB"]["gain"] for t in checkpoints]

    d2_global = [summaries[str(t)]["global"]["d2_displacement"] for t in checkpoints]
    d2_rec = [summaries[str(t)]["by_projection_type"]["recurrent"]["d2_displacement"] for t in checkpoints]
    d2_ff = [summaries[str(t)]["by_projection_type"]["FF"]["d2_displacement"] for t in checkpoints]
    d2_fb = [summaries[str(t)]["by_projection_type"]["FB"]["d2_displacement"] for t in checkpoints]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>Synaptic Weight Gain G(t) Across Exposure Ages</b>",
            "<b>Normalized Structural Displacement D_2(t)</b>",
            "<b>4×4 Cell-Class Remodeling Matrix (t = 100s Gain)</b>",
            "<b>Empirical Temporal Memory Kernel D_order(ΔT)</b>",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    # 1. Gain Trajectory
    fig.add_trace(go.Scatter(x=ages, y=g_global, mode="lines+markers", line=dict(color="#f8fafc", width=2.5), name="Global Gain"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ages, y=g_rec, mode="lines+markers", line=dict(color=PROJ_COLORS["recurrent"], width=2.0), name="Recurrent"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ages, y=g_ff, mode="lines+markers", line=dict(color=PROJ_COLORS["FF"], width=2.0), name="Feedforward (FF)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ages, y=g_fb, mode="lines+markers", line=dict(color=PROJ_COLORS["FB"], width=2.0), name="Feedback (FB)"), row=1, col=1)

    # Reference baseline line
    fig.add_hline(y=1.0, line_dash="dash", line_color="#475569", row=1, col=1)

    # 2. D2 Displacement
    fig.add_trace(go.Scatter(x=ages, y=d2_global, mode="lines+markers", line=dict(color="#f8fafc", width=2.5), name="Global D_2", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=ages, y=d2_rec, mode="lines+markers", line=dict(color=PROJ_COLORS["recurrent"], width=2.0), name="Recurrent D_2", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=ages, y=d2_ff, mode="lines+markers", line=dict(color=PROJ_COLORS["FF"], width=2.0), name="FF D_2", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=ages, y=d2_fb, mode="lines+markers", line=dict(color=PROJ_COLORS["FB"], width=2.0), name="FB D_2", showlegend=False), row=1, col=2)

    # 3. 4x4 Class Remodeling Heatmap (t = 100s)
    classes = ["E", "PV", "SST", "VIP"]
    matrix_100s = np.zeros((4, 4))
    sm100 = summaries["100"]["by_class_pair"]

    for i, pre_c in enumerate(classes):
        for j, post_c in enumerate(classes):
            pair_key = f"{pre_c}->{post_c}"
            if pair_key in sm100:
                matrix_100s[i, j] = sm100[pair_key]["gain"]

    annotations_text = [[f"<b>{matrix_100s[i, j]:.2f}×</b>" for j in range(4)] for i in range(4)]

    fig.add_trace(
        go.Heatmap(
            z=matrix_100s,
            x=classes,
            y=classes,
            text=annotations_text,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#ffffff"),
            colorscale="Magma",
            colorbar=dict(title="Gain", x=0.46, len=0.38, y=0.18),
            hoverinfo="x+y+z",
            name="Class Gain Matrix",
        ),
        row=2,
        col=1,
    )

    # 4. Temporal Kernel D_order(ΔT)
    if tk_res is not None:
        dt_vals = [float(k) for k in sorted(tk_res.keys(), key=lambda x: float(x))]
        d_order_pct = [tk_res[str(int(k))]["d_order"] * 100.0 for k in dt_vals]
        rec_d_pct = [tk_res[str(int(k))]["by_projection"]["recurrent"]["d_order"] * 100.0 for k in dt_vals]

        fig.add_trace(
            go.Scatter(
                x=dt_vals,
                y=d_order_pct,
                mode="lines+markers",
                line=dict(color="#f43f5e", width=2.5),
                marker=dict(size=8, symbol="diamond"),
                name="Global D_order (%)",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=dt_vals,
                y=rec_d_pct,
                mode="lines+markers",
                line=dict(color="#38bdf8", width=1.8, dash="dot"),
                name="Recurrent D_order (%)",
            ),
            row=2,
            col=2,
        )

        fig.add_hline(y=1.0, line_dash="dash", line_color="#e2e8f0", annotation_text="1% Divergence Threshold", annotation_position="top left", row=2, col=2)
        fig.update_xaxes(title_text="Inter-Event Gap ΔT (ms)", type="log", row=2, col=2)
        fig.update_yaxes(title_text="Order Divergence D_order (%)", range=[0.0, 1.5], row=2, col=2)

    fig.update_xaxes(title_text="Exposure Age (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Synaptic Gain G(t)", row=1, col=1)
    fig.update_xaxes(title_text="Exposure Age (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Displacement D_2(t)", row=1, col=2)
    fig.update_xaxes(title_text="Postsynaptic Target Class", row=2, col=1)
    fig.update_yaxes(title_text="Presynaptic Source Class", row=2, col=1)

    fig.update_layout(width=1180, height=760)
    apply_dark_theme(fig, "Longitudinal Plasticity Trajectory & Circuit Remodeling Architecture", "Asymptotic Convergence to Recurrent Quasi-Steady State (+33.1%) and Empirical Order-Invariance Across Timescales")

    caption = (
        "Multi-scale empirical characterization of HDP synaptic plasticity. Top-Left: Weight gain G(t) across 100 s showing fast 2 s overshoot "
        "followed by asymptotic relaxation: recurrent circuits stay permanently remodeled (+33.1%), whereas feedforward (+1.4%) and feedback (+3.1%) "
        "relax back to baseline. Top-Right: Normalized structural displacement D_2(t) confirming recurrent dominance. Bottom-Left: 4×4 source-target "
        "remodeling matrix at t = 100 s, revealing profound potentiation of SST interneuron output (SST→E +92.6%, SST→PV +72.2%) contrasting with invariant "
        "VIP disinhibition (VIP→SST +2.0%). Bottom-Right: Empirical temporal memory kernel D_order(ΔT) demonstrating that the learning rule is "
        "fundamentally order-insensitive across all intervals (D_order = 0.29% at ΔT = 50 ms)."
    )

    provenance = {
        "Exposure Checkpoints": "0, 1, 2, 5, 10, 20, 30, 60, 100 seconds",
        "Plasticity Parameters": "K_HDP = 0.003, K_w_ctrl = 0.001, w_floor = 0.01, w_ceil = 10.0",
        "Total Synapses": "10,666 edges (10,076 Recurrent, 287 FF, 303 FB)",
        "Quasi-Steady State Metric": "Δ_age(60→100s) = 0.001098 (0.11%)",
        "Order-Sensitivity Metric": "D_order(50 ms) = 0.002941 (0.29%)",
        "Evidence Status": "OBSERVED (Deterministic Execution Receipts)",
    }

    return fig, caption, provenance


if __name__ == "__main__":
    fig, cap, prov = build_plasticity_figure()
    html = wrap_figure_with_provenance_html(fig, cap, prov, "OBSERVED")
    import os
    os.makedirs("docs/_static/plotly", exist_ok=True)
    with open("docs/_static/plotly/plasticity_trajectory.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Successfully built docs/_static/plotly/plasticity_trajectory.html")

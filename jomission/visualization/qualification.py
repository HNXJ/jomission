"""Interactive B1-B3 Qualification & Root-Cause Diagnostic Dashboard."""

import json
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jomission.visualization.theme import (
    CLASS_COLORS,
    apply_dark_theme,
    wrap_figure_with_provenance_html,
)


def build_qualification_figure(results_path: str = "results/b1b2b3_root_cause_closure_results.json") -> tuple[go.Figure, str, dict]:
    with open(results_path, "r") as f:
        res = json.load(f)

    phase_data = res["phase_randomization_test"]
    op_data = res["operating_point_map"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>B2 Root Cause: Pairwise Correlation ρ Across Time Windows</b>",
            "<b>B1/B3 Operating-Point Deficit: I_native vs Rheobase I_rh</b>",
            "<b>Observed Firing Rates vs Fluctuation-Driven Firing</b>",
            "<b>Recurrent Synaptic Decoupling (I_syn vs I_ext)</b>",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    # 1. B2 Rho Across Time Windows
    arms = ["Arm A (Canonical)", "Arm B (Subthreshold Jitter)", "Arm C (Full Dynamic Dispersion)"]
    windows = ["Full (0-2000ms)", "Early (0-500ms)", "Late (500-2000ms)"]
    win_colors = {"Full (0-2000ms)": "#38bdf8", "Early (0-500ms)": "#f43f5e", "Late (500-2000ms)": "#4ade80"}

    for w in windows:
        rhos = [phase_data[arm][w]["mean_rho"] for arm in arms]
        fig.add_trace(
            go.Bar(
                x=["Canonical", "Subthreshold Jitter", "Full Dispersion"],
                y=rhos,
                name=f"Window: {w.split()[0]}",
                marker=dict(color=win_colors[w]),
            ),
            row=1,
            col=1,
        )

    # Add B2 soft gate boundary [-0.05, 0.20]
    fig.add_hline(y=0.20, line_dash="dash", line_color="#f59e0b", annotation_text="B2 Ceiling (0.20)", annotation_position="top left", row=1, col=1)
    fig.add_hline(y=0.00, line_color="#64748b", row=1, col=1)

    # 2. Operating Point Map: I_native vs Rheobase
    classes = ["E", "PV", "SST", "VIP"]
    i_native = [op_data[c]["mean_i_native"] for c in classes]
    i_rh = [op_data[c]["i_rheobase"] for c in classes]
    deficit = [op_data[c]["distance_to_rheobase"] for c in classes]

    fig.add_trace(
        go.Bar(
            x=classes,
            y=i_native,
            name="Executed I_native",
            marker=dict(color=[CLASS_COLORS[c] for c in classes]),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=classes,
            y=i_rh,
            mode="markers",
            marker=dict(size=14, symbol="line-ew-open", line=dict(color="#ffffff", width=3.0)),
            name="Rheobase I_rh",
        ),
        row=1,
        col=2,
    )

    for i, c in enumerate(classes):
        fig.add_annotation(
            x=c,
            y=i_native[i] + 0.25,
            text=f"<b>Δ={deficit[i]:+.2f}</b>",
            showarrow=False,
            font=dict(color="#f87171" if deficit[i] < -0.5 else "#38bdf8", size=11),
            row=1,
            col=2,
        )

    # 3. Observed Firing Rates & Silence Fraction
    r_obs = [op_data[c]["observed_rate_mean"] for c in classes]
    silence = [op_data[c]["silence_percent"] for c in classes]

    fig.add_trace(
        go.Bar(
            x=classes,
            y=r_obs,
            name="Observed Rate (Hz)",
            marker=dict(color=[CLASS_COLORS[c] for c in classes]),
            text=[f"{r:.1f} Hz<br>({s:.0f}% silent)" for r, s in zip(r_obs, silence)],
            textposition="auto",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Target ranges (E: 5-8 Hz, PV: 12-25 Hz, SST: 6-12 Hz, VIP: 8-15 Hz)
    target_mid = {"E": 6.5, "PV": 18.5, "SST": 9.0, "VIP": 11.5}
    fig.add_trace(
        go.Scatter(
            x=classes,
            y=[target_mid[c] for c in classes],
            mode="markers",
            marker=dict(size=10, symbol="diamond", color="#facc15"),
            name="Target Midpoint (MODEL_ASSUMPTION)",
        ),
        row=2,
        col=1,
    )

    # 4. Recurrent Synaptic Decoupling (I_syn vs I_ext)
    i_e = [op_data[c]["mean_i_e"] for c in classes]
    i_i = [op_data[c]["mean_i_i"] for c in classes]
    i_ext = [op_data[c]["mean_i_ext"] for c in classes]

    fig.add_trace(go.Bar(x=classes, y=i_ext, name="External I_ext", marker=dict(color="#3b82f6")), row=2, col=2)
    fig.add_trace(go.Bar(x=classes, y=[e * 100.0 for e in i_e], name="Recurrent I_E (×100)", marker=dict(color="#10b981")), row=2, col=2)
    fig.add_trace(go.Bar(x=classes, y=[abs(i) * 100.0 for i in i_i], name="|Recurrent I_I| (×100)", marker=dict(color="#ef4444")), row=2, col=2)

    fig.update_xaxes(title_text="Initialization Condition", row=1, col=1)
    fig.update_yaxes(title_text="Mean Pairwise Correlation ρ", row=1, col=1)
    fig.update_xaxes(title_text="Cell Class", row=1, col=2)
    fig.update_yaxes(title_text="Membrane Current (Native Units)", row=1, col=2)
    fig.update_xaxes(title_text="Cell Class", row=2, col=1)
    fig.update_yaxes(title_text="Firing Rate (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Cell Class", row=2, col=2)
    fig.update_yaxes(title_text="Current Magnitude", row=2, col=2)

    fig.update_layout(width=1180, height=760, barmode="group")
    apply_dark_theme(fig, "B1-B3 Qualification & Mechanistic Root-Cause Dashboard", "Quantitative Operating-Point Analysis: Startup Transient Resolution, Rheobase Deficits, and Synaptic Decoupling")

    caption = (
        "Root-cause diagnostic decomposition of cortical qualification failures (B1, B2, B3). Top-Left: High spike correlation (ρ) in B2 is resolved "
        "as a startup transient artifact (Early window ρ = 0.530); in the late steady-state window (500-2000 ms), ρ is naturally zero (mean ρ = -0.0008), "
        "and subthreshold phase jitter immediately lowers full-window ρ to 0.189 (<0.20 ceiling). Top-Right: All populations operate subthreshold "
        "(I_native < I_rh), but VIP suffers a severe -1.56 current unit deficit below rheobase (I_native = 2.19 vs I_rh = 3.75). Bottom-Left: Subthreshold "
        "positioning collapses VIP firing rate to 0.39 Hz (23.4% silence). Bottom-Right: Recurrent synaptic currents (I_E, I_I < 0.005) represent "
        "<0.15% of external drive (I_ext ≈ 3.41), proving that the spontaneous cortical baseline is currently operating in a recurrently decoupled regime."
    )

    provenance = {
        "Qualification Target": "B1 (Coverage >= 0.60), B2 (CV_ISI in [0.5, 1.5], rho <= 0.20), B3 (E_frac <= 0.60)",
        "Diagnosed Cause B2": "Transient startup volley (steady-state rho = -0.0008; CV_ISI = 0.31-0.35 intrinsic)",
        "Diagnosed Cause B1": "VIP subthreshold deficit (-1.559 current units below rheobase)",
        "Diagnosed Cause B3": "Recurrent synaptic decoupling (|I_syn| < 0.005 vs I_ext = 3.41)",
        "Evidence Status": "OBSERVED (Empirical Root-Cause Verification)",
    }

    return fig, caption, provenance


if __name__ == "__main__":
    fig, cap, prov = build_qualification_figure()
    html = wrap_figure_with_provenance_html(fig, cap, prov, "OBSERVED")
    import os
    os.makedirs("docs/_static/plotly", exist_ok=True)
    with open("docs/_static/plotly/b1_b2_b3_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Successfully built docs/_static/plotly/b1_b2_b3_dashboard.html")

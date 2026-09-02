"""Interactive Spike Raster & Population Rate Dynamics Explorer."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jomission.network.builder import build_jomission_model, simulation_with_background_poisson
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule, SLOT_ONSET_MS, SLOT_DURATION_MS
from jomission.visualization.theme import (
    CLASS_COLORS,
    AREA_COLORS,
    apply_dark_theme,
    wrap_figure_with_provenance_html,
)
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig


def build_activity_figure(model=None, spikes=None, dt_ms: float = 0.1) -> tuple[go.Figure, str, dict]:
    if model is None:
        model = build_jomission_model(n_per_area=100, seed=0, dt_ms=dt_ms)

    tbl = model.neuron_table()

    # Simulate canonical 2000 ms with background Poisson if spikes not provided
    if spikes is None:
        dur_ms = 2000.0
        cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AAAB"][0]
        sched = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)
        sim_p = simulation_with_background_poisson(model.cfg, duration_ms=dur_ms, dt_ms=dt_ms, seed=0)
        sim = Simulation(
            duration_ms=dur_ms,
            dt_ms=dt_ms,
            seed=0,
            runtime=RuntimeConfig(recurrent_backend="edge_list"),
            poisson_drive=sim_p.poisson_drive,
        )
        sig = jtfne.simulate(model, sim, paradigm=sched)
        spikes = np.asarray(sig.spikes)
    else:
        dur_ms = float(spikes.shape[0] * dt_ms)

    n_steps, n_neurons = spikes.shape
    time_ms = np.arange(n_steps) * dt_ms

    # Sort neurons by Area and Layer for structured raster visualization
    area_order = {"V1": 0, "V4": 1, "FEF": 2, "PFC": 3}
    layer_order = {"L1": 0, "L2/3": 1, "L4": 2, "L5": 3, "L6": 4}
    sorted_neuron_indices = sorted(
        range(n_neurons),
        key=lambda i: (area_order[tbl[i]['area']], layer_order[tbl[i]['layer']], tbl[i]['cell_type'])
    )

    y_pos_map = {orig_i: plot_y for plot_y, orig_i in enumerate(sorted_neuron_indices)}

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "<b>Spike Raster: 400 Cortical Neurons Sorted by Area & Laminar Depth</b>",
            "<b>Population Firing Rate Dynamics (10 ms Bins) by Cell Class</b>",
        ),
    )

    # 1. Raster Plot by Cell Class
    for ct in ("E", "PV", "SST", "VIP"):
        ct_idx = [i for i, r in enumerate(tbl) if r['cell_type'] == ct]
        sub_spikes = spikes[:, ct_idx]
        t_indices, n_indices = np.where(sub_spikes > 0.5)

        if len(t_indices) > 0:
            actual_spike_times = t_indices * dt_ms
            plot_y_positions = [y_pos_map[ct_idx[ni]] for ni in n_indices]

            fig.add_trace(
                go.Scattergl(
                    x=actual_spike_times,
                    y=plot_y_positions,
                    mode="markers",
                    marker=dict(size=3.0, color=CLASS_COLORS[ct], opacity=0.85),
                    name=f"Class {ct} Spikes",
                    hoverinfo="x+text",
                    text=[f"Neuron {ct_idx[ni]} ({tbl[ct_idx[ni]]['area']} {tbl[ct_idx[ni]]['layer']})" for ni in n_indices],
                ),
                row=1,
                col=1,
            )

    # Add stimulus epoch vertical bands (p1, p2, p3, p4)
    epoch_colors = {"p1": "#0284c7", "p2": "#0284c7", "p3": "#0284c7", "p4": "#e11d48"}
    for slot in ("p1", "p2", "p3", "p4"):
        onset = SLOT_ONSET_MS.get(slot, 0.0)
        dur = SLOT_DURATION_MS.get(slot, 531.0)
        if onset < dur_ms:
            end_t = min(onset + dur, dur_ms)
            fig.add_vrect(
                x0=onset,
                x1=end_t,
                fillcolor=epoch_colors.get(slot, "#38bdf8"),
                opacity=0.12,
                layer="below",
                line_width=1,
                line_color="#475569",
                annotation_text=f"<b>{slot.upper()}</b>",
                annotation_position="top left",
                annotation_font_color="#94a3b8",
                row=1,
                col=1,
            )

    # 2. Binned Population Rate Dynamics (Bottom Panel)
    bin_ms = 10.0
    bin_steps = int(bin_ms / dt_ms)
    n_bins = n_steps // bin_steps
    bin_centers = (np.arange(n_bins) + 0.5) * bin_ms

    for ct in ("E", "PV", "SST", "VIP"):
        ct_idx = [i for i, r in enumerate(tbl) if r['cell_type'] == ct]
        ct_spikes = spikes[:, ct_idx]
        binned_rates = np.zeros(n_bins)
        for b in range(n_bins):
            binned_rates[b] = ct_spikes[b * bin_steps : (b + 1) * bin_steps, :].mean() * (1000.0 / dt_ms)

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=binned_rates,
                mode="lines",
                line=dict(color=CLASS_COLORS[ct], width=2.0),
                name=f"Rate: {ct}",
            ),
            row=2,
            col=1,
        )

    # Add Area horizontal separator lines to raster plot
    for sep_idx in (100, 200, 300):
        fig.add_hline(y=sep_idx - 0.5, line_width=1, line_dash="dash", line_color="#334155", row=1, col=1)

    fig.add_annotation(x=10, y=50, text="<b>V1</b>", showarrow=False, font=dict(color=AREA_COLORS["V1"], size=13), row=1, col=1)
    fig.add_annotation(x=10, y=150, text="<b>V4</b>", showarrow=False, font=dict(color=AREA_COLORS["V4"], size=13), row=1, col=1)
    fig.add_annotation(x=10, y=250, text="<b>FEF</b>", showarrow=False, font=dict(color=AREA_COLORS["FEF"], size=13), row=1, col=1)
    fig.add_annotation(x=10, y=350, text="<b>PFC</b>", showarrow=False, font=dict(color=AREA_COLORS["PFC"], size=13), row=1, col=1)

    fig.update_xaxes(title_text="Simulation Time (ms)", range=[0, dur_ms], row=2, col=1)
    fig.update_yaxes(title_text="Hierarchical Neuron Index (0-399)", range=[-5, 405], row=1, col=1)
    fig.update_yaxes(title_text="Firing Rate (Hz)", row=2, col=1)

    fig.update_layout(width=1180, height=720)
    apply_dark_theme(fig, "Population Activity & Rate Dynamics Explorer", "Time-Resolved Spiking Raster and Cell-Class Firing Curves Across Sensory Epochs (dt = 0.1 ms)")

    caption = (
        "Interactive population spiking and time-resolved rate dynamics during structured exposure. Top: Full raster plot of all 400 cortical neurons "
        "sorted by hierarchy (V1, V4, FEF, PFC) and laminar depth, color-coded by cell class (E: Cyan, PV: Crimson, SST: Amber, VIP: Violet). "
        "Shaded vertical regions mark stimulus presentation epochs (p1, p2, p3 in blue; p4 in red). Bottom: Binned population firing rate trajectories "
        "(10 ms sliding window) showing fast, transient recruitment of PV interneurons alongside sustained excitatory pyramidal drive."
    )

    provenance = {
        "Duration": f"{dur_ms:.1f} ms ({n_steps:,} steps at dt=0.1 ms)",
        "Stimulus Paradigm": "AAAB Structured Protocol (drive amp=5.0)",
        "Background Poisson": "2000.0 Hz, amp=2.0 private per-neuron",
        "Global Mean Rate": f"{spikes.mean() * 10000.0:.2f} Hz",
        "Total Spikes": f"{int(spikes.sum()):,}",
        "Backend": "JaxFNE edge_list baseline kernel",
    }

    return fig, caption, provenance


if __name__ == "__main__":
    fig, cap, prov = build_activity_figure()
    html = wrap_figure_with_provenance_html(fig, cap, prov, "OBSERVED")
    import os
    os.makedirs("docs/_static/plotly", exist_ok=True)
    with open("docs/_static/plotly/raster_population.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Successfully built docs/_static/plotly/raster_population.html")

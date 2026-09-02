"""Interactive Spectral Response & Spectrolaminar Frequency Explorer."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch, spectrogram

from jomission.network.builder import build_jomission_model, simulation_with_background_poisson
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule
from jomission.visualization.theme import (
    AREA_COLORS,
    CLASS_COLORS,
    apply_dark_theme,
    wrap_figure_with_provenance_html,
)
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig


def build_spectral_figure(model=None, V_m=None, dt_ms: float = 0.1) -> tuple[go.Figure, str, dict]:
    if model is None:
        model = build_jomission_model(n_per_area=100, seed=0, dt_ms=dt_ms)

    tbl = model.neuron_table()

    if V_m is None:
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
        V_m = np.asarray(sig.V_m)
    else:
        dur_ms = float(V_m.shape[0] * dt_ms)

    fs = 1000.0 / dt_ms  # Sampling frequency: 10,000 Hz

    # Subsample V_m to 1000 Hz for spectral analysis (cleaner, faster, highly accurate up to 500 Hz Nyquist)
    ds_factor = int(1.0 / (dt_ms / 1.0))
    v_ds = V_m[::ds_factor, :]
    fs_ds = fs / ds_factor  # 1,000 Hz

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Power Spectral Density (PSD) by Cortical Area</b>",
            "<b>V1 Spectrogram: Time-Frequency Dynamics (0-100 Hz)</b>",
        ),
        horizontal_spacing=0.12,
    )

    # 1. PSD across Areas (Welch method)
    freqs = None
    for area in ("V1", "V4", "FEF", "PFC"):
        area_idx = [i for i, r in enumerate(tbl) if r['area'] == area]
        # Average LFP / membrane potential proxy for area
        area_lfp = v_ds[:, area_idx].mean(axis=1)
        f, pxx = welch(area_lfp - area_lfp.mean(), fs=fs_ds, nperseg=int(fs_ds * 0.5), noverlap=int(fs_ds * 0.25))

        # Restrict to 1-100 Hz
        valid_idx = np.where((f >= 1.0) & (f <= 100.0))[0]
        freqs = f[valid_idx]
        pxx_valid = pxx[valid_idx]

        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=10.0 * np.log10(pxx_valid + 1e-12),
                mode="lines",
                line=dict(color=AREA_COLORS[area], width=2.2),
                name=f"Area {area}",
            ),
            row=1,
            col=1,
        )

    # Add canonical frequency band shading
    bands = [
        ("Theta (4-8 Hz)", 4.0, 8.0, "rgba(56, 189, 248, 0.08)"),
        ("Alpha (8-12 Hz)", 8.0, 12.0, "rgba(74, 222, 128, 0.08)"),
        ("Beta (15-30 Hz)", 15.0, 30.0, "rgba(251, 146, 60, 0.08)"),
        ("Gamma (30-80 Hz)", 30.0, 80.0, "rgba(192, 132, 252, 0.08)"),
    ]
    for b_name, f_lo, f_hi, b_col in bands:
        fig.add_vrect(
            x0=f_lo,
            x1=f_hi,
            fillcolor=b_col,
            layer="below",
            line_width=1,
            line_dash="dot",
            line_color="#475569",
            annotation_text=f"<b>{b_name.split()[0]}</b>",
            annotation_position="top left",
            annotation_font_color="#94a3b8",
            row=1,
            col=1,
        )

    # 2. Time-Frequency Spectrogram for Area V1
    v1_idx = [i for i, r in enumerate(tbl) if r['area'] == "V1"]
    v1_lfp = v_ds[:, v1_idx].mean(axis=1)

    f_spec, t_spec, sxx = spectrogram(
        v1_lfp - v1_lfp.mean(),
        fs=fs_ds,
        nperseg=int(fs_ds * 0.2),  # 200 ms windows
        noverlap=int(fs_ds * 0.18),  # 180 ms overlap
    )

    valid_f_spec = np.where((f_spec >= 1.0) & (f_spec <= 100.0))[0]
    sxx_log = 10.0 * np.log10(sxx[valid_f_spec, :] + 1e-12)

    fig.add_trace(
        go.Heatmap(
            x=t_spec * 1000.0,
            y=f_spec[valid_f_spec],
            z=sxx_log,
            colorscale="Viridis",
            colorbar=dict(title="dB/Hz", x=1.02, len=0.8),
            name="V1 Spectrogram",
            hoverinfo="x+y+z",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Frequency (Hz)", range=[1, 100], row=1, col=1)
    fig.update_yaxes(title_text="Power Spectral Density (dB/Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", range=[0, dur_ms], row=1, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", range=[1, 100], row=1, col=2)

    fig.update_layout(width=1180, height=580)
    apply_dark_theme(fig, "Spectral Response & Time-Frequency Explorer", "Area-Resolved Power Spectral Densities and Spectrolaminar Dynamics Across Canonical Oscillatory Bands")

    caption = (
        "Interactive spectral decomposition and time-frequency dynamics of the cortical hierarchy. Left: Area-resolved Power Spectral Density "
        "(Welch PSD estimate) for V1 (Blue), V4 (Green), FEF (Orange), and PFC (Purple), highlighting canonical physiological bands (Theta 4-8 Hz, "
        "Alpha 8-12 Hz, Beta 15-30 Hz, Gamma 30-80 Hz). Right: Spectrogram of V1 population potential across the 2000 ms trial displaying "
        "spectral power variations across sensory drive slots."
    )

    provenance = {
        "Sampling Rate": f"{fs:.0f} Hz (downsampled to {fs_ds:.0f} Hz for Welch/STFT)",
        "Spectral Estimator": "Welch PSD (500 ms Hanning window, 50% overlap)",
        "STFT Window": "200 ms segment, 90% overlap",
        "Frequency Range": "1.0 - 100.0 Hz",
        "Target Signals": "LFP proxy (population-averaged somatic V_m; not extracellular LFP)",
    }

    return fig, caption, provenance


if __name__ == "__main__":
    fig, cap, prov = build_spectral_figure()
    html = wrap_figure_with_provenance_html(fig, cap, prov, "DERIVED")
    import os
    os.makedirs("docs/_static/plotly", exist_ok=True)
    with open("docs/_static/plotly/spectral_response.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Successfully built docs/_static/plotly/spectral_response.html")

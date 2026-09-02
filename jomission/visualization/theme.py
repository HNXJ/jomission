"""Shared Plotly theme and visual standard for Jomission interactive figures."""

import plotly.graph_objects as go

# Color Encodings
AREA_COLORS = {
    "V1": "#38bdf8",   # Sky blue
    "V4": "#4ade80",   # Green
    "FEF": "#fb923c",  # Orange
    "PFC": "#c084fc",  # Purple
}

CLASS_COLORS = {
    "E": "#06b6d4",    # Cyan / Excitatory
    "PV": "#f43f5e",   # Crimson / Fast-spiking
    "SST": "#eab308",  # Amber / Low-threshold
    "VIP": "#a855f7",  # Violet / Disinhibitory
}

PROJ_COLORS = {
    "recurrent": "#64748b",  # Slate
    "FF": "#22c55e",         # Green
    "FB": "#f97316",         # Orange
}

DARK_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=12),
        xaxis=dict(
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            tickcolor="#8b949e",
            linecolor="#30363d",
        ),
        yaxis=dict(
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            tickcolor="#8b949e",
            linecolor="#30363d",
        ),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=40, t=70, b=60),
    )
)


def apply_dark_theme(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    """Apply consistent dark styling, title, and fonts to a Plotly figure."""
    full_title = f"<b>{title}</b>"
    if subtitle:
        full_title += f"<br><span style='font-size:12px; color:#8b949e; font-weight:normal'>{subtitle}</span>"

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=full_title, x=0.04, y=0.96, xanchor="left", yanchor="top"),
    )
    return fig


def wrap_figure_with_provenance_html(
    fig: go.Figure,
    caption: str,
    provenance: dict[str, str],
    evidence_level: str = "OBSERVED",
) -> str:
    """Generate standalone HTML document containing the Plotly figure, caption, and provenance card."""
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    badge_color = "#238636" if evidence_level == "OBSERVED" else ("#1f6feb" if evidence_level == "DERIVED" else "#d29922")

    provenance_rows = "".join(
        f"<tr><td style='color:#8b949e; padding:3px 12px 3px 0;'><b>{k}</b></td><td style='color:#c9d1d9; font-family:monospace;'>{v}</td></tr>"
        for k, v in provenance.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{fig.layout.title.text if fig.layout.title else 'Jomission Interactive Visualization'}</title>
  <style>
    body {{
      background-color: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .container {{
      width: 100%;
      max-width: 1280px;
    }}
    .caption-box {{
      background-color: #161b22;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 16px 20px;
      margin-top: 16px;
      font-size: 13px;
      line-height: 1.6;
    }}
    .provenance-card {{
      background-color: #161b22;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 14px 20px;
      margin-top: 12px;
      font-size: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
      color: #ffffff;
      background-color: {badge_color};
      margin-bottom: 8px;
    }}
    table {{
      border-collapse: collapse;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="container">
    {fig_html}
    <div class="caption-box">
      <span class="badge">{evidence_level}</span>
      <div>{caption}</div>
    </div>
    <div class="provenance-card">
      <b style="color:#f0f6fc;">Provenance & Execution Audit</b>
      <table>
        {provenance_rows}
      </table>
    </div>
  </div>
</body>
</html>
"""
    return html

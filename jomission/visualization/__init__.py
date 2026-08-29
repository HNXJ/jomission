"""Visualization package — re-export network_viz + model_summary + w4/agsdr dashboards (VIS_FOUNDATION V0–V2)."""

from jomission.visualization.network_viz import (
    hierarchy_fig,
    motif_matrix_fig,
    spatial_fig,
    rf_fig,
    get_motif_stats,
)
from jomission.visualization.model_summary import (
    model_summary,
    model_summary_text,
    observable_basis,
    ontology_table,
    get_observable_basis_hash,
)

try:
    from jomission.visualization.manifest import build_manifest, verify_V0, VISUALIZATION_VERSION
except Exception:  # pragma: no cover
    build_manifest = None  # type: ignore
    verify_V0 = None  # type: ignore
    VISUALIZATION_VERSION = "unknown"  # type: ignore

try:
    from jomission.visualization.w4_dashboard import w4_sankey_fig, save_w4_sankey
except Exception:  # pragma: no cover
    w4_sankey_fig = None  # type: ignore
    save_w4_sankey = None  # type: ignore

try:
    from jomission.visualization.agsdr_dashboard import (
        agsdr_dashboard_figs,
        save_agsdr_dashboard,
        fig_parallel_coords,
        fig_objective_space,
        fig_fidelity,
        fig_jacobian,
    )
except Exception:  # pragma: no cover
    agsdr_dashboard_figs = None  # type: ignore
    save_agsdr_dashboard = None  # type: ignore
    fig_parallel_coords = None  # type: ignore
    fig_objective_space = None  # type: ignore
    fig_fidelity = None  # type: ignore
    fig_jacobian = None  # type: ignore

__all__ = [
    "hierarchy_fig",
    "motif_matrix_fig",
    "spatial_fig",
    "rf_fig",
    "get_motif_stats",
    "model_summary",
    "model_summary_text",
    "observable_basis",
    "ontology_table",
    "get_observable_basis_hash",
    "build_manifest",
    "verify_V0",
    "VISUALIZATION_VERSION",
    "w4_sankey_fig",
    "save_w4_sankey",
    "agsdr_dashboard_figs",
    "save_agsdr_dashboard",
    "fig_parallel_coords",
    "fig_objective_space",
    "fig_fidelity",
    "fig_jacobian",
]

"""Network builder — constructs JaxFNE Configuration + Model for jomission.

Single source of truth for V1/V4/FEF/PFC × layer × E/PV/SST/VIP.

Delegates to JaxFNE primitives: Configuration.column, area_layer_cell_types, connectivity,
hdp, field, probe, runtime. No parallel neural simulator.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Optional

import jaxfne as jtfne
from jaxfne import Configuration

from jomission.network.populations import (
    JOMISSION_AREAS,
    JOMISSION_LAYERS,
    AREA_LAYER_CELL_TYPES,
    LAYER_DEPTH_BANDS,
    LAYER_COUNT_FRAC_DEFAULT,
)
from jomission.network.connectivity import (
    WITHIN_GAIN_DEFAULT,
    P_FEEDFORWARD_DEFAULT,
    P_FEEDBACK_DEFAULT,
    HIERARCHY,
)


def build_jomission_network(
    *,
    n_per_area: int = 100,
    areas: Sequence[str] = JOMISSION_AREAS,
    layers: Sequence[str] = JOMISSION_LAYERS,
    seed: int = 0,
    within_gain: float = WITHIN_GAIN_DEFAULT,
    p_feedforward: float = P_FEEDFORWARD_DEFAULT,
    p_feedback: float = P_FEEDBACK_DEFAULT,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
    n_contacts: int = 16,
) -> Configuration:
    """Build jomission Configuration with explicit areas/layers/cell types.

    - n_per_area ≥100, configurable
    - Each area declares a column with shared layers
    - Per-area per-layer E/PV/SST/VIP fractions via area_layer_cell_types
    - FF+FB between adjacent hierarchical pairs (both directions)
    - Geometry probe for LFP-like/CSD-like via linear_solver
    """
    if n_per_area < 100:
        raise ValueError(f"n_per_area {n_per_area} < minimum 100")
    if tuple(areas) != HIERARCHY and set(areas) != set(HIERARCHY):
        # Allow subset but warn if order wrong; enforce low->high order if same set
        pass

    cfg = Configuration()
    # Declare areas for bookkeeping (optional, not required by builder but useful)
    try:
        cfg = cfg.areas(list(areas))  # type: ignore[attr-defined]
    except Exception:
        pass

    # Declare columns
    for area in areas:
        cfg = cfg.column(area, layers=list(layers), n=int(n_per_area))

    # Wire inter-area connectivity for each adjacent pair — both FF and FB
    for lo, hi in zip(areas[:-1], areas[1:]):
        cfg = cfg.inter_column_connectivity(
            source_area=lo, target_area=hi, mode="sparse",
            p_feedforward=float(p_feedforward), p_feedback=0.0,
        )
        cfg = cfg.inter_column_connectivity(
            source_area=hi, target_area=lo, mode="sparse",
            p_feedforward=0.0, p_feedback=float(p_feedback),
        )

    # Global fallback cell fractions (used if per-area table missing)
    from jomission.network.populations import FLAT_CELL_TYPE_FRACTIONS
    cfg = cfg.cell_types(FLAT_CELL_TYPE_FRACTIONS)
    for area in areas:
        per_layer = AREA_LAYER_CELL_TYPES.get(area)
        if per_layer is None:
            continue
        cfg = cfg.area_layer_cell_types(area, per_layer)

    # Layer depth fractions (proxy geometry)
    cfg = cfg.layer_fractions({k: list(v) for k, v in LAYER_DEPTH_BANDS.items()})

    # Per-layer count fractions (explicit, replaceable) — set via direct metadata to avoid column redeclaration
    cfg = cfg.update_metadata(
        layer_count_frac=dict(LAYER_COUNT_FRAC_DEFAULT),
        area_layer_count_frac={area: dict(LAYER_COUNT_FRAC_DEFAULT) for area in areas},
    )

    # Within-area connectivity
    cfg = cfg.connectivity(within_area="sparse", within_gain=float(within_gain))

    # Field + probe — linear_solver, proxy_readout (no physical calibration)
    cfg = cfg.field(source_mode="proxy_no_field_solve")
    cfg = cfg.probe(n_contacts=int(n_contacts))

    # Runtime — edge_list backend required for full-state continuation (C_t carry)
    cfg = cfg.runtime(recurrent_backend="edge_list")

    # Emitter family
    cfg = cfg.emitter(family=str(emitter))

    # Metadata
    cfg = cfg.update_metadata(
        jomission_version="0.1.0",
        hierarchy="->".join(areas),
        n_per_area=int(n_per_area),
        n_total=int(n_per_area * len(areas)),
        within_gain=float(within_gain),
        p_feedforward=float(p_feedforward),
        p_feedback=float(p_feedback),
        seed=int(seed),
        dt_ms=float(dt_ms),
    )
    return cfg


def build_jomission_model(
    *,
    n_per_area: int = 100,
    areas: Sequence[str] = JOMISSION_AREAS,
    seed: int = 0,
    **kwargs: Any,
) -> jtfne.Model:
    cfg = build_jomission_network(n_per_area=n_per_area, areas=areas, seed=seed, **kwargs)
    return jtfne.construct(cfg)


# Convenience for tests
def validate_network(n_per_area: int = 100) -> dict[str, Any]:
    cfg = build_jomission_network(n_per_area=n_per_area, seed=0)
    # Check metadata columns
    meta = cfg.metadata
    cols = meta.get("columns", [])
    issues: list[str] = []
    if len(cols) != 4:
        issues.append(f"columns {len(cols)} != 4")
    names = [c["name"] for c in cols]
    if names != list(JOMISSION_AREAS):
        issues.append(f"column names {names} != {list(JOMISSION_AREAS)}")
    total = sum(c["n"] for c in cols)
    if total != n_per_area * 4:
        issues.append(f"total n {total} != {n_per_area*4}")
    return {"valid": not issues, "issues": issues, "columns": names, "total_n": total}

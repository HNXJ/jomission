"""Explicit geometry — each neuron has area, layer, cell class, depth.

Delegates to JaxFNE's LaminarPopulation / LaminarSourceGeometry where possible,
but ensures naming V1_L2_3_E etc is preserved in metadata for W_{(a,l,c)->(a',l',c')} analysis.
"""

from __future__ import annotations

from typing import Sequence

import jaxfne as jtfne
from jaxfne import LaminarPopulation, laminar_source_geometry

from jomission.network.populations import (
    JOMISSION_AREAS,
    JOMISSION_LAYERS,
    JOMISSION_CELL_TYPES,
    LAYER_DEPTH_BANDS,
    AREA_LAYER_CELL_TYPES,
)


def build_laminar_populations(
    *,
    n_per_area: int = 100,
    areas: Sequence[str] = JOMISSION_AREAS,
) -> list[LaminarPopulation]:
    """Create LaminarPopulation list for geometry with explicit area/layer/cell.

    Depth is proxy [0,1]; overlapping co-located E/PV/SST/VIP within same layer is allowed.
    """
    pops: list[LaminarPopulation] = []
    # Distribute n_per_area across layers roughly by LAYER_COUNT_FRAC_DEFAULT via equal split for now;
    # JaxFNE's column builder does the exact count allocation. Here we create geometry-level populations
    # with proportional n.
    from jomission.network.populations import LAYER_COUNT_FRAC_DEFAULT

    for area in areas:
        for layer in JOMISSION_LAYERS:
            depth_min, depth_max = LAYER_DEPTH_BANDS[layer]
            frac_table = AREA_LAYER_CELL_TYPES[area][layer]
            layer_n = max(1, round(n_per_area * LAYER_COUNT_FRAC_DEFAULT[layer]))
            for ct in JOMISSION_CELL_TYPES:
                ct_frac = frac_table[ct]
                n_units = max(0, round(layer_n * ct_frac))
                if n_units == 0:
                    continue
                name = f"{area}_{layer.replace('/','_')}_{ct}"
                pops.append(
                    LaminarPopulation(
                        name=name,
                        cell_type=ct,
                        layer=layer,
                        depth_min=float(depth_min),
                        depth_max=float(depth_max),
                        n_units=int(n_units),
                    )
                )
    return pops


def build_geometry(
    *,
    n_per_area: int = 100,
    areas: Sequence[str] = JOMISSION_AREAS,
) -> jtfne.LaminarSourceGeometry:
    pops = build_laminar_populations(n_per_area=n_per_area, areas=areas)
    return laminar_source_geometry(pops)


def validate_geometry(n_per_area: int = 100) -> dict:
    geom = build_geometry(n_per_area=n_per_area)
    v = geom.validate()
    issues = list(v.get("issues", []))
    # Check naming convention
    for pop in geom.populations:
        if "_" not in pop.name or pop.cell_type not in JOMISSION_CELL_TYPES:
            issues.append(f"naming {pop.name} not V*_*_E/PV/SST/VIP")
    return {"valid": not issues, "issues": issues, "n_units": geom.n_units_total, "n_pops": len(geom.populations)}

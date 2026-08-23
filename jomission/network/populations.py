"""Population definitions — explicit E/PV/SST/VIP per layer per area.

Mechanically readable names: V1_L2_3_E, V1_L2_3_PV, etc.
Fractions are explicit tables, not assumed identical across areas.
"""

from __future__ import annotations

from typing import Mapping

# Canonical areas, layers, cell types
JOMISSION_AREAS: tuple[str, ...] = ("V1", "V4", "FEF", "PFC")
JOMISSION_LAYERS: tuple[str, ...] = ("L1", "L2/3", "L4", "L5", "L6")
JOMISSION_CELL_TYPES: tuple[str, ...] = ("E", "PV", "SST", "VIP")

# Minimum per area
N_PER_AREA_MIN: int = 100

# Depth bands (proxy laminar depth in [0,1], superficial=0)
# Aligned to JaxFNE's canonical_z_bands but re-declared for auditability
LAYER_DEPTH_BANDS: dict[str, tuple[float, float]] = {
    "L1": (0.00, 0.10),
    "L2/3": (0.10, 0.45),
    "L4": (0.45, 0.55),
    "L5": (0.55, 0.80),
    "L6": (0.80, 1.00),
}

# Layer thickness fractions (count budget, not thickness coupling)
# If area_layer_count_frac not set, JaxFNE falls back to thickness-proportional.
LAYER_COUNT_FRAC_DEFAULT: dict[str, float] = {
    "L1": 0.08,
    "L2/3": 0.30,
    "L4": 0.15,
    "L5": 0.27,
    "L6": 0.20,
}

# Area-specific per-layer cell-type composition
# Literature placeholders — intentionally explicit and replaceable.
# Initial values approximate: superficial E-rich, L4 PV-rich, deep mixed.
# FEF/PFC have slightly higher VIP/SST in superficial and deep per frontal data.

V1_LAYER_CELL_TYPES: dict[str, dict[str, float]] = {
    "L1": {"E": 0.00, "PV": 0.00, "SST": 0.00, "VIP": 1.00},  # L1 mostly interneurons; E=0 handled by JaxFNE sampling
    "L2/3": {"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07},
    "L4": {"E": 0.60, "PV": 0.20, "SST": 0.10, "VIP": 0.10},
    "L5": {"E": 0.70, "PV": 0.12, "SST": 0.10, "VIP": 0.08},
    "L6": {"E": 0.70, "PV": 0.12, "SST": 0.10, "VIP": 0.08},
}

V4_LAYER_CELL_TYPES: dict[str, dict[str, float]] = {
    "L1": {"E": 0.00, "PV": 0.00, "SST": 0.00, "VIP": 1.00},
    "L2/3": {"E": 0.73, "PV": 0.12, "SST": 0.08, "VIP": 0.07},
    "L4": {"E": 0.55, "PV": 0.25, "SST": 0.10, "VIP": 0.10},
    "L5": {"E": 0.68, "PV": 0.14, "SST": 0.10, "VIP": 0.08},
    "L6": {"E": 0.68, "PV": 0.14, "SST": 0.10, "VIP": 0.08},
}

FEF_LAYER_CELL_TYPES: dict[str, dict[str, float]] = {
    "L1": {"E": 0.00, "PV": 0.00, "SST": 0.00, "VIP": 1.00},
    "L2/3": {"E": 0.70, "PV": 0.10, "SST": 0.10, "VIP": 0.10},
    "L4": {"E": 0.50, "PV": 0.20, "SST": 0.15, "VIP": 0.15},  # FEF L4 thin; still represented
    "L5": {"E": 0.65, "PV": 0.13, "SST": 0.12, "VIP": 0.10},
    "L6": {"E": 0.65, "PV": 0.13, "SST": 0.12, "VIP": 0.10},
}

PFC_LAYER_CELL_TYPES: dict[str, dict[str, float]] = {
    "L1": {"E": 0.00, "PV": 0.00, "SST": 0.00, "VIP": 1.00},
    "L2/3": {"E": 0.68, "PV": 0.10, "SST": 0.12, "VIP": 0.10},
    "L4": {"E": 0.45, "PV": 0.20, "SST": 0.18, "VIP": 0.17},
    "L5": {"E": 0.62, "PV": 0.13, "SST": 0.13, "VIP": 0.12},
    "L6": {"E": 0.62, "PV": 0.13, "SST": 0.13, "VIP": 0.12},
}

# Full area map
AREA_LAYER_CELL_TYPES: dict[str, dict[str, dict[str, float]]] = {
    "V1": V1_LAYER_CELL_TYPES,
    "V4": V4_LAYER_CELL_TYPES,
    "FEF": FEF_LAYER_CELL_TYPES,
    "PFC": PFC_LAYER_CELL_TYPES,
}

# Flat fallback (used only if area-specific table missing)
FLAT_CELL_TYPE_FRACTIONS: dict[str, float] = {"E": 0.70, "PV": 0.12, "SST": 0.10, "VIP": 0.08}

# Validation
def validate_populations() -> dict:
    issues: list[str] = []
    for area, table in AREA_LAYER_CELL_TYPES.items():
        if set(table.keys()) != set(JOMISSION_LAYERS):
            issues.append(f"{area} layers {set(table.keys())} != {set(JOMISSION_LAYERS)}")
        for layer, fracs in table.items():
            if set(fracs.keys()) != set(JOMISSION_CELL_TYPES):
                issues.append(f"{area} {layer} cell types {set(fracs.keys())} != {set(JOMISSION_CELL_TYPES)}")
            total = sum(fracs.values())
            if abs(total - 1.0) > 1e-6:
                issues.append(f"{area} {layer} fractions sum {total} != 1.0")
            for ct, v in fracs.items():
                if not (0 <= v <= 1):
                    issues.append(f"{area} {layer} {ct} fraction {v} out of [0,1]")
    # Depth bands
    for layer in JOMISSION_LAYERS:
        if layer not in LAYER_DEPTH_BANDS:
            issues.append(f"depth band missing for {layer}")
    return {"valid": not issues, "issues": issues, "n_areas": len(JOMISSION_AREAS), "n_layers": len(JOMISSION_LAYERS)}

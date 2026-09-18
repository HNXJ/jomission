"""RETINA[jomission_v0]: the structural input interface x := Retina^1024, G[Retina] = 32 x 32.

DRAFT, pending reviewer approval. R4 established where retinal input enters the architecture and
deliberately established nothing else, so every field here is new and each carries provenance:

    OBSERVED_CURRENT   read from the current qualified JaxFNE construction
    DERIVED            follows from TFNE/2 or from an OBSERVED_CURRENT field
    DEFINITION_CHOICE  chosen for this diagnostic, and therefore open to revision

The four material choices were put to the reviewer before anything was written: hemifield
retinotopy, peak efficacy at 1.0x the E tonic, a deterministic static spot, and the baseline
kernel. What follows is the smallest interface that makes those decidable.

Two properties are enforced rather than assumed, because the column builder supplies both and
neither belongs to an input interface:

    Retina^1024 declares no connectivity, so intra-retinal edges are removed after construction
    retinal units are silent absent stimulus, so their tonic drive is zeroed

Both are verified in the diagnostic receipt rather than trusted.
"""

from __future__ import annotations

NAME = "RETINA[jomission_v0]"
N = 1024
GRID = (32, 32)                 # (rows, cols); unit index = row * 32 + col
AREA = "Retina"
LAYER = "L4"
CLASS = "E"

#: Column boundary between the two hemifields. Left half drives V1.1, right half V1.2.
HEMIFIELD_SPLIT_COL = 16

#: Where each hemifield projects. Matches the frontier R4 qualified: in({V1^2 X[lat]}).
HEMIFIELD_TARGET = {"left": ("V1", "1"), "right": ("V1", "2")}

#: Summed synaptic weight of one target cell's retinal afferents, as a multiple of the E tonic.
#: The claim this makes is exact and checkable: sum over a cell's retinal edges of w equals
#: PEAK_EFFICACY_X_TONIC * tonic_E. It is NOT a claim about realized postsynaptic current, which
#: depends on retinal firing and AMPA filtering and is recorded as an observation instead.
PEAK_EFFICACY_X_TONIC = 1.0

MECHANISM = "AMPA"

#: Stimulus drive added to a lit retinal unit, in the emitter's native units. Equal to the
#: cortical E tonic, a value this model family already realizes as spiking.
STIMULUS_DRIVE = 5.0

#: The static localized spot: rows 12-19, cols 4-11 of the 32 x 32 field. 64 units, entirely
#: inside the left hemifield, so V1.2 is reached only through X[lat] and the V4 level.
SPOT = {"row0": 12, "row1": 20, "col0": 4, "col1": 12}

PROVENANCE = {
 "N": "DEFINITION_CHOICE: the checkpoint architecture declares Retina^1024",
 "GRID": "DEFINITION_CHOICE: G[Retina] = 32 x 32, from the sealed architecture",
 "indexing": "DEFINITION_CHOICE: row-major, unit = row * 32 + col; deterministic, no RNG",
 "hemifield": ("DEFINITION_CHOICE approved by the reviewer 2026-09-18: columns 0-15 to V1.1 and "
               "16-31 to V1.2, so the two V1 instances are not redundant"),
 "targets": "DERIVED: in({V1^2 X[lat]}) = V1.1.L4.E, V1.2.L4.E, as qualified in R4",
 "receptive_fields": ("DEFINITION_CHOICE: contiguous equal-sized blocks in hemifield raster order, "
                      "one block per target cell, assigned by floor(u * n_cells / n_units). "
                      "Deterministic, non-overlapping, and exhaustive"),
 "efficacy": ("DEFINITION_CHOICE approved by the reviewer 2026-09-18: summed afferent weight per "
              "target cell equals 1.0x the E tonic"),
 "mechanism": "DERIVED: AMPA, the only mechanism the architecture declares",
 "stimulus": ("DEFINITION_CHOICE approved by the reviewer 2026-09-18: a static deterministic spot, "
              "no stochastic activation, so propagation is attributable to the stimulus"),
 "stimulus_drive": ("DEFINITION_CHOICE: 5.0, equal to the cortical E tonic. Realized retinal firing "
                    "rate is recorded as an observation, not asserted here"),
 "tonic": ("DEFINITION_CHOICE: retinal tonic is zeroed, so the retina is silent absent stimulus and "
           "the drive additivity invariant reads I_executed = 0 + I_schedule on these units"),
 "no_intra_retinal_edges": ("DERIVED: TFNE/2 gives A^n n instances and no connectivity, and Retina "
                            "is not a CTX instance, so it inherits no D4d local connectivity"),
}


def unit_index(row, col):
    """Deterministic 32 x 32 indexing. Raises rather than wrapping."""
    if not (0 <= row < GRID[0] and 0 <= col < GRID[1]):
        raise ValueError(f"E_ADDRESS_UNRESOLVED: retinal unit ({row}, {col}) is outside {GRID}")
    return row * GRID[1] + col


def unit_rowcol(unit):
    return divmod(int(unit), GRID[1])


def hemifield(unit):
    return "left" if unit_rowcol(unit)[1] < HEMIFIELD_SPLIT_COL else "right"


def hemifield_units(side):
    """Units of one hemifield in raster order. Order is the receptive-field tiling order."""
    return [u for u in range(N) if hemifield(u) == side]


def receptive_fields(side, n_cells):
    """Contiguous equal blocks of one hemifield, one per target cell.

    Returns a list of n_cells unit lists. Exhaustive and non-overlapping by construction, so
    every retinal unit projects to exactly one target cell and no cell is left without input.
    """
    if n_cells <= 0:
        raise ValueError("E_ALLOCATION_POLICY_MISSING: no target cells to tile onto")
    units = hemifield_units(side)
    out = [[] for _ in range(n_cells)]
    for position, unit in enumerate(units):
        out[position * n_cells // len(units)].append(unit)
    empty = [k for k, rf in enumerate(out) if not rf]
    if empty:
        raise ValueError(f"E_ALLOCATION_POLICY_MISSING: cells {empty} received no retinal units")
    return out


def edge_weight(rf_size, tonic_e):
    """Per-edge weight such that a cell's summed retinal afferent weight is the declared peak."""
    if rf_size <= 0:
        raise ValueError("E_ALLOCATION_POLICY_MISSING: empty receptive field has no weight")
    return PEAK_EFFICACY_X_TONIC * float(tonic_e) / float(rf_size)


def spot_units():
    """The lit units of the static localized spot, in raster order."""
    return [unit_index(r, c) for r in range(SPOT["row0"], SPOT["row1"])
            for c in range(SPOT["col0"], SPOT["col1"])]


def declaration(tonic_e=None):
    """The full interface declaration, for the pre-execution seal."""
    rf_left = receptive_fields("left", 14)
    rf_right = receptive_fields("right", 14)
    lit = spot_units()
    return {
     "definition": NAME,
     "draft": "pending reviewer approval; no execution may consume this until approved",
     "N": N, "grid": {"rows": GRID[0], "cols": GRID[1]}, "indexing": "unit = row * 32 + col",
     "area": AREA, "layer": LAYER, "cell_type": CLASS,
     "hemifield_split_col": HEMIFIELD_SPLIT_COL,
     "targets": {side: ".".join([*path, LAYER, CLASS]) for side, path in HEMIFIELD_TARGET.items()},
     "units_per_hemifield": {"left": len(hemifield_units("left")), "right": len(hemifield_units("right"))},
     "receptive_field_sizes": {"left": sorted({len(rf) for rf in rf_left}),
                               "right": sorted({len(rf) for rf in rf_right})},
     "mechanism": MECHANISM,
     "peak_efficacy_x_tonic": PEAK_EFFICACY_X_TONIC,
     "edge_weight": (None if tonic_e is None else
                     {"left": sorted({round(edge_weight(len(rf), tonic_e), 8) for rf in rf_left}),
                      "right": sorted({round(edge_weight(len(rf), tonic_e), 8) for rf in rf_right}),
                      "tonic_e": float(tonic_e)}),
     "stimulus": {"kind": "static localized spot, deterministic", **SPOT,
                  "n_lit": len(lit), "drive": STIMULUS_DRIVE,
                  "hemifields_lit": sorted({hemifield(u) for u in lit}),
                  "note": ("the spot lies entirely in the left hemifield, so V1.2 is reached only "
                           "through X[lat] and the V4 level")},
     "tonic": 0.0,
     "enforced_after_construction": [
      "intra-retinal edges removed: Retina^1024 declares no connectivity",
      "retinal tonic zeroed: the retina is silent absent stimulus"],
     "provenance": PROVENANCE,
    }

"""CTX[jomission_v0]: the provisional cortical-unit definition for the whole-system checkpoint.

Explicitly provisional. The name carries `jomission_v0` so that a checkpoint implementation
cannot silently become canonical cortical truth: the eventual biological `CTX` problem is not
solved here. Every D4 field carries provenance:

    OBSERVED_CURRENT   read from the current qualified JaxFNE construction
    DERIVED            follows from TFNE/2 or from an OBSERVED_CURRENT field
    DEFINITION_CHOICE  chosen for this checkpoint, and therefore open to revision

D4a structural layers and realized populations, including absent ones
D4b N, P, R with exact integer allocation
D4c in / out interfaces
D4d local connectivity, inherited from the qualified column construction
D4e neuron and HDP realization
D4f construction geometry
"""

from __future__ import annotations

import jaxfne as jtfne
from jaxfne._config import _counts_from_fractions as _engine_counts
from jaxfne._construct_population import _SUITE2_LAYER_FRACTIONS as _ENGINE_LAYER_BANDS

NAME = "CTX[jomission_v0]"
LAYERS = tuple(jtfne.CANONICAL_LAYERS_6L)
CLASSES = ("E", "PV", "SST", "VIP")
N_PER_INSTANCE = 200


def layer_population_fractions():
    """D4b source: how the engine splits N across layers -- z-band width, not thickness policy."""
    return {lay: float(_ENGINE_LAYER_BANDS[lay][1] - _ENGINE_LAYER_BANDS[lay][0]) for lay in LAYERS}


def layer_cell_type_fractions():
    """D4a/D4b source: the engine's canonical laminar composition, read not invented."""
    return {lay: dict(jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS[lay]) for lay in LAYERS}


def allocate(total, proportions):
    """R := R_JaxFNE 0.4.24, the realization policy this definition observes rather than chooses.

    CTX[jomission_v0] is the TFNE definition of the current JaxFNE cortical realization, so R is
    OBSERVED_CURRENT: int(round(total*frac)) in declaration order, each clamped to the remaining
    budget, with the last key assigned the remainder. Delegating to the engine's own function
    keeps the definition honest -- a reimplementation could drift from what is realized.

    TFNE/2's general semantics are unchanged: it requires a declared deterministic R that
    preserves exact cardinality, and prescribes no particular policy. The independent per-class
    formula int(round(N_l*P)) is a different function that violates that invariant in 5031 of
    12006 audited cases, and is not R (results/ctx_allocation_audit.json).
    """
    return _engine_counts(int(total), proportions)


def declaration(n=N_PER_INSTANCE):
    """The D4a-D4f content of one CTX[jomission_v0] instance.

    TFNE owns the instance total N and the within-layer class proportions P. It does not own the
    depth profile: the engine builder exposes no parameter for it, so claiming one here would be
    inventing authority. The receipt therefore checks what TFNE does own -- per-instance totals,
    declared-absent populations empty, declared-present populations non-empty, and the realized
    class counts inside each layer equal to allocate(layer_total, P[layer]).
    """
    fracs = layer_cell_type_fractions()
    absent = [f"{lay}.{c}" for lay in LAYERS for c in CLASSES if fracs[lay].get(c, 0.0) == 0.0]
    return {
     "definition": NAME,
     "provisional": "CTX[jomission_v0] is a checkpoint definition, never canonical cortical truth",
     "layers": list(LAYERS),
     "classes": list(CLASSES),
     "N": n,
     "P": fracs,
     "R": "R_JaxFNE 0.4.24: int(round(total*frac)) in declaration order, clamped to the remaining budget, last key takes the remainder",
     "rho": "none: the allocation is deterministic and needs no realization RNG",
     "absent_populations": absent,
     "layer_profile": ("inherited from the current JaxFNE column construction and not owned by TFNE here: "
                       "the builder exposes no layer-depth parameter, so the split across L1-L6 is read from "
                       "the realized model rather than declared"),
     "P_layer": layer_population_fractions(),
     # D4c interfaces. `in` and `out` are the feedforward frontier; fb_in and fb_out are the
     # feedback frontier, declared so the O[fffb] rule can name them without guessing.
     "in": ["L4.E"],
     "out": ["L2.E", "L3.E"],
     "fb_in": ["L1.E"],
     "fb_out": ["L6.E"],
     "model": "izhikevich / cortical_eig, as built by jomission.qualification.cmin",
     "provenance": {
      "D4a_layers": "OBSERVED_CURRENT: jaxfne.CANONICAL_LAYERS_6L",
      "D4a_classes": "OBSERVED_CURRENT: jaxfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS keys",
      "D4a_absent": "DERIVED: L6 declares PV 0.0 and VIP 0.0, so those populations are absent by declaration",
      "D4b_P": "OBSERVED_CURRENT: the engine's canonical laminar composition",
      "D4b_layer_split": "OBSERVED_CURRENT: the layer depth profile is the engine builder's, which exposes no parameter for it. TFNE declares the within-layer class composition and the instance total, never the depth profile",
      "D4b_R": "OBSERVED_CURRENT: jaxfne 0.4.24 _counts_from_fractions, the policy the realization actually applies. Audited exhaustively for exact cardinality in results/ctx_allocation_audit.json",
      "D4b_N": "DEFINITION_CHOICE: 200 per instance, sized for a first diagnostic rather than for anatomy",
      "D4c_in": "DEFINITION_CHOICE: L4.E as the feedforward target",
      "D4c_out": "DEFINITION_CHOICE: supragranular L2.E and L3.E as the feedforward source",
      "D4c_fb": "DEFINITION_CHOICE: L6.E to L1.E as the feedback frontier. L1.E is anatomically unusual and exists here because the engine's canonical L1 composition declares E 0.5; it is recorded, not endorsed",
      "D4d_local": "OBSERVED_CURRENT: within-column connectivity from the qualified column construction, unchanged",
      "D4e_model": "OBSERVED_CURRENT: izhikevich cortical_eig emitter, as in the qualified column",
      "D4f_geometry": "OBSERVED_CURRENT: laminar geometry, which preserves per-neuron layer labels",
     },
    }


def expected_class_counts(layer_total, layer, n=None):
    """What TFNE requires inside one realized layer: allocate(layer_total, P[layer])."""
    return allocate(int(layer_total), layer_cell_type_fractions()[layer])

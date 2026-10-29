"""Omission-Model-lite: the two-column proof of concept from the dissertation deck (Chapter 3).

Two single-layer columns, V1 (layer L2/3) and FEF_PFC (layer L4), with the cell counts the
deck lists for the lite model: E 18, PV 4, SST 2 (V1) or 1 (FEF/PFC), VIP 2 (V1) or 1
(FEF/PFC). The counts sum to 26 and 24 (50 neurons in all), not the deck's "N=100". They
are used as listed. The deck gives no layers; the layer per column is this module's choice,
because JaxFNE treats a target layer of L4 as feedforward and any other target as feedback.
Structure only: no visual drive, fixation neuron or plasticity; the deck's proof-of-concept
stage is "tools validated", so this module builds and runs the graph.
"""

from __future__ import annotations

import jaxfne as jtfne
from jaxfne import Configuration

from jomission.network.connectivity import WITHIN_GAIN_DEFAULT

LITE_AREAS: tuple[str, ...] = ("V1", "FEF_PFC")
LITE_LAYERS: dict[str, str] = {"V1": "L2/3", "FEF_PFC": "L4"}
LITE_CELL_COUNTS: dict[str, dict[str, int]] = {
    "V1": {"E": 18, "PV": 4, "SST": 2, "VIP": 2},
    "FEF_PFC": {"E": 18, "PV": 4, "SST": 1, "VIP": 1},
}
LITE_P_FEEDFORWARD: float = 0.30
LITE_P_FEEDBACK: float = 0.20


def build_lite_network(
    *,
    seed: int = 0,
    within_gain: float = WITHIN_GAIN_DEFAULT,
    n_contacts: int = 4,
) -> Configuration:
    """Jomission lite Configuration: V1 -> FEF_PFC feedforward, FEF_PFC -> V1 feedback."""
    cfg = Configuration()
    for area in LITE_AREAS:
        counts = LITE_CELL_COUNTS[area]
        layer = LITE_LAYERS[area]
        n = sum(counts.values())
        cfg = cfg.column(area, layers=[layer], n=n)
        cfg = cfg.area_layer_cell_types(area, {layer: {ct: c / n for ct, c in counts.items()}})

    # Feedforward L2/3 -> L4 (p_feedforward); feedback L4 -> L2/3 (p_feedback).
    cfg = cfg.inter_column_connectivity(
        source_area="V1",
        target_area="FEF_PFC",
        mode="sparse",
        layer_to_layer_map={LITE_LAYERS["V1"]: LITE_LAYERS["FEF_PFC"]},
        p_feedforward=LITE_P_FEEDFORWARD,
        p_feedback=0.0,
        feedforward_weight_range=(0.5, 2.0),
        feedback_weight_range=(0.3, 1.5),
    )
    cfg = cfg.inter_column_connectivity(
        source_area="FEF_PFC",
        target_area="V1",
        mode="sparse",
        layer_to_layer_map={LITE_LAYERS["FEF_PFC"]: LITE_LAYERS["V1"]},
        p_feedforward=0.0,
        p_feedback=LITE_P_FEEDBACK,
        feedforward_weight_range=(0.5, 2.0),
        feedback_weight_range=(0.3, 1.5),
    )
    cfg = cfg.connectivity(
        within_area="all_to_all_uniform_random",
        within_gain=float(within_gain),
        p_connect=None,
    )
    # Same field declaration as the builder: the engine's default, written out.
    cfg = cfg.field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",
        gauge="mean_zero",
    )
    cfg = cfg.probe(n_contacts=int(n_contacts))
    cfg = cfg.runtime(recurrent_backend="edge_list")
    cfg = cfg.emitter(family="izhikevich")
    cfg = cfg.update_metadata(
        jomission_version="0.1.0",
        model="omission-model-lite",
        areas=list(LITE_AREAS),
        cell_counts=LITE_CELL_COUNTS,
        n_total=sum(sum(c.values()) for c in LITE_CELL_COUNTS.values()),
        p_feedforward=LITE_P_FEEDFORWARD,
        p_feedback=LITE_P_FEEDBACK,
        within_gain=float(within_gain),
        seed=int(seed),
        field_readout_mode="proxy_no_field_solve",
    )
    return cfg


def build_lite_model(**kwargs) -> jtfne.Model:
    """Construct the lite model. Keyword arguments as in build_lite_network."""
    return jtfne.construct(build_lite_network(**kwargs))

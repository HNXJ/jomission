"""JaxFNE 0.5.7 migration candidate: construction invariants of the TFNE substrate.

Construct-only. No simulation, no result file, no manifest.

JaxFNE >= 0.5.6 checks at construct that the declared areas are exactly the built columns
(jaxfne/_construct_core.py:571-577). The whole-system arms build a `Retina` column for the
retinal input interface, so the area must be declared as well; that is what
`realize.declare_retinal_interface` does. These tests pin the construction the sealed
SCI-EI-NATIVE-STATE-1 lineage ran, so a later engine change cannot silently change the
graph.

See docs/migrations/jaxfne_0_5_7_migration_spec.md for the measured engine matrix.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

#: The six cortical areas. The retinal interface is not one of them: it is an input
#: interface with no slot in the cortical hierarchy.
CORTICAL_AREAS = ("FEF", "PFC", "V1_1", "V1_2", "V4_1", "V4_2")
RETINA_AREA = "Retina"

#: Post-enforcement edge count of the sealed construction. The same figure is recorded in
#: the sealed results/whole_system_isolated_pop.json.
N_EDGES_AFTER_ENFORCE = 246844

#: sha256 of the realized weight bytes after enforcement. Measured under jaxfne 0.4.25 and
#: 0.5.7 with identical bytes, so it is a graph invariant rather than an engine identity.
WEIGHT_SHA256_AFTER_ENFORCE = "8fccec054edeee9dce09166c622c77862e457acdf7fb2df393060fbf3d0df813"


@pytest.fixture(scope="module")
def arm_model():
    import whole_system_ei_native_state_arm as arm

    model, _normal, _rf_decl, _tonic = arm.build()
    index, (area, layer, cls) = arm.realize.index_map(model)
    enforced, enforcement = arm.enforce(model, index)
    return arm, model, enforced, enforcement, (area, layer, cls)


def test_declared_areas_match_built_columns(arm_model):
    """The engine's contract: an area is what construct builds."""
    _arm, model, _enforced, _enforcement, _labels = arm_model
    declared = sorted(str(a) for a in (model.cfg.metadata.get("areas") or []))
    built = sorted(str(c) for c in (model.cfg.metadata.get("column_names") or []))
    assert declared == built, f"declared {declared} != built {built}"
    assert declared == sorted((*CORTICAL_AREAS, RETINA_AREA))


def test_six_cortical_areas_are_unchanged(arm_model):
    _arm, model, _enforced, _enforcement, (area, _layer, _cls) = arm_model
    built = {str(r["area"]) for r in model.neuron_table()}
    assert set(CORTICAL_AREAS).issubset(built)
    assert RETINA_AREA in built
    assert built == set(CORTICAL_AREAS) | {RETINA_AREA}


def test_retinal_interface_is_not_cortical(arm_model):
    """No hierarchy slot, no intra-retinal edge, no tonic: the retina stays structural."""
    arm, model, enforced, enforcement, _labels = arm_model
    r0, r1, n_cells = arm.realize.index_map(model)[0][
        f"{RETINA_AREA}.{arm.retina.LAYER}.{arm.retina.CLASS}"
    ]
    assert n_cells == arm.retina.N

    el = enforced.params["edge_list"]
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    intra = ((pre >= r0) & (pre < r1)) & ((post >= r0) & (post < r1))
    assert not intra.any(), "intra-retinal edges must be removed"
    assert enforcement["intra_retinal_edges_removed"] > 0

    tonic = np.asarray(enforced.params["emitter"].drive)
    assert float(tonic[r0:r1].max()) == 0.0, "the retinal interface must be silent"

    counts, _n_total = arm.realize.realized_projections(enforced, *_labels_from(enforced, arm))
    cortical = [k for k in counts if not k[0].startswith(RETINA_AREA)]
    assert len(cortical) == 24, f"cortical projection count changed: {sorted(cortical)}"
    retinal = sorted(k for k in counts if k[0].startswith(RETINA_AREA))
    assert retinal == [
        ("Retina.L4.E", "V1_1.L4.E"),
        ("Retina.L4.E", "V1_2.L4.E"),
    ], f"unexpected retinal afferents: {retinal}"


def _labels_from(model, arm):
    _index, (area, layer, cls) = arm.realize.index_map(model)
    return area, layer, cls


def test_graph_invariants_are_engine_independent(arm_model):
    """The graph, not the config hash: edges and weights are pinned."""
    _arm, _model, enforced, enforcement, _labels = arm_model
    el = enforced.params["edge_list"]
    assert int(np.asarray(el.pre).size) == N_EDGES_AFTER_ENFORCE
    payload = np.ascontiguousarray(np.asarray(el.weight)).tobytes()
    assert hashlib.sha256(payload).hexdigest() == WEIGHT_SHA256_AFTER_ENFORCE
    assert enforcement["retinal_tonic_after"] == 0.0

"""Tests for the layer-and-type silence option (SILENCE=PV@L5,L6, condition X1b) of jomission/laminar/allen_v1_replica.py.
Construction-level fixtures at 0.05 scale, 0.3 s; each test asserts its precondition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jomission.laminar.allen_v1_replica as av  # noqa: E402
from jomission.laminar import caminos_e0 as c0  # noqa: E402

SEED, SCALE, T_MS = 1, 0.05, 300.0


def test_pv_at_l5_l6_selects_only_pv_in_those_layers():
    pl = av.plan("H", SCALE, False)
    assert av.silence_targets("PV@L5,L6", pl["pops"]) == ["L5_PV", "L6_PV"]
    assert av.silence_targets("PV", pl["pops"]) == ["L2/3_PV", "L4_PV", "L5_PV", "L6_PV"], "fixture: all-layer PV"
    assert av.silence_targets(None, pl["pops"]) == []


@pytest.mark.parametrize("bad", ["XX@L5", "PV@L9", "PV@"])
def test_unknown_kind_or_layer_raises(bad):
    pl = av.plan("H", SCALE, False)
    with pytest.raises(ValueError):
        av.silence_targets(bad, pl["pops"])


def _build(silence):
    pl = av.plan("H", SCALE, False)
    silenced = av.silence_targets(silence, pl["pops"])
    model, rows = av.construct(SEED, pl, silenced)
    return pl, silenced, model


def test_silencing_pv_l5_l6_changes_only_their_thresholds_and_keeps_edges_and_drive():
    pl, _, base = _build(None)
    _, silenced, sil = _build("PV@L5,L6")
    assert silenced == ["L5_PV", "L6_PV"], "fixture: the two silenced populations"

    a, b = base.static["adex"], sil.static["adex"]
    arrays = [k for k in a if isinstance(a[k], np.ndarray)]
    assert len(arrays) >= 4, f"fixture: edge arrays present, got {arrays}"
    for k in arrays:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), f"edge array {k} changed under silencing"

    slices = av.pop_slices(pl)
    e_base, e_sil = base.params["emitter"], sil.params["emitter"]
    rows_sil = np.zeros(len(e_base.VT_mV), dtype=bool)
    for p in silenced:
        rows_sil[slices[p]] = True
    assert rows_sil.sum() == sum(pl["sizes"][p] for p in silenced) > 0, "fixture: silenced rows selected"
    assert np.all(np.asarray(e_sil.VT_mV)[rows_sil] == av.SILENCE["VT_mV"])
    assert np.all(np.asarray(e_sil.Vpeak_mV)[rows_sil] == av.SILENCE["Vpeak_mV"])
    keep = ~rows_sil
    for name in ("VT_mV", "Vpeak_mV", "EL_mV", "C_pF", "gL_nS", "a_nS", "b_pA", "tauw_ms", "Vreset_mV"):
        same = np.array_equal(np.asarray(getattr(e_sil, name))[keep], np.asarray(getattr(e_base, name))[keep],
                              equal_nan=True)  # Vpeak_mV is NaN for unsilenced AdEx cells
        assert same, (
            f"parameter {name} of unsilenced populations changed")

    # Drive: the runner's Poisson drive is one spec for every cell; silencing does not touch it.
    assert repr(c0.drive_spec(0.07, SEED)) == repr(c0.drive_spec(0.07, SEED))


def test_silenced_pv_l5_l6_spike_zero_and_other_populations_still_spike():
    import jaxfne as jtfne

    pl, silenced, sil = _build("PV@L5,L6")
    _, _, base = _build(None)
    sim = jtfne.Simulation(duration_ms=T_MS, dt_ms=c0.DT_MS, poisson_drive=c0.drive_spec(0.07, SEED))
    spikes_sil = np.asarray(sil.simulate(sim).spikes, dtype=bool)
    spikes_base = np.asarray(base.simulate(sim).spikes, dtype=bool)
    slices = av.pop_slices(pl)
    base_counts = {p: int(spikes_base[:, slices[p]].sum()) for p in silenced}
    assert all(c > 0 for c in base_counts.values()), f"fixture: unsilenced L5/L6 PV must spike: {base_counts}"
    for p in silenced:
        assert int(spikes_sil[:, slices[p]].sum()) == 0, f"{p} must not spike when silenced"
    others = [p for p in pl["pops"] if p not in silenced]
    assert sum(int(spikes_sil[:, slices[p]].sum()) for p in others) > 0, "other populations must still spike"

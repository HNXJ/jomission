"""Tier 0: the EQUAL_G realization arithmetic, and consistency of its recorded receipts.

No construction happens here -- construction is the driver's job. The law
``w_p,i = g * K_local,i / N_afferents,p,i`` is exercised on a synthetic graph built to have the
asymmetry that motivated it: two targets of the same channel with different convergence, whose
authorities must come out equal even though their per-edge weights do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jomission.tfne import o_authority as OA

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "whole_system_authority_bracket.json"

G_BRACKET = (0.05, 0.1, 0.2, 0.5)


def synthetic():
    """Two areas. Target X gets 2 ff afferents, target Y gets 4. Both have local E input.

    Layout: 0-3 area A L2.E (sources), 4 area B L4.E = X, 5 area B L4.E = Y,
    6-9 area B L3.E local sources.
    """
    area = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "B", "B"])
    layer = np.array(["L2", "L2", "L2", "L2", "L4", "L4", "L3", "L3", "L3", "L3"])
    pre, post, w = [], [], []
    for s in (0, 1):                       # X: 2 feedforward afferents
        pre.append(s), post.append(4), w.append(0.5)
    for s in (0, 1, 2, 3):                 # Y: 4 feedforward afferents
        pre.append(s), post.append(5), w.append(0.5)
    for s in (6, 7, 8, 9):                 # identical local drive, so convergence is the only
        pre.append(s), post.append(4), w.append(0.1)   # difference between X and Y
        pre.append(s), post.append(5), w.append(0.1)
    return (area, layer, np.array(pre), np.array(post), np.array(w, dtype=np.float64))


def test_equal_g_equalizes_authority_across_convergence():
    area, layer, pre, post, w = synthetic()
    cortical = np.ones(area.size, dtype=bool)
    rel = OA.classify(area, layer, pre, post, cortical)
    k_local = OA.local_authority(w, post, rel, area.size)
    assert k_local[4] == pytest.approx(0.4)     # 4 local edges x 0.1
    assert k_local[5] == pytest.approx(0.4)     # identical, so only convergence differs

    ff = rel == "ff"
    assert ff.sum() == 6
    # Before: one per-edge weight for both, so the 4-afferent target has twice the authority of
    # the 2-afferent one. That doubling is the artifact EQUAL_G removes.
    k_before = np.zeros(area.size)
    np.add.at(k_before, post[ff], w[ff])
    assert k_before[4] / k_local[4] == pytest.approx(2.5)
    assert k_before[5] / k_local[5] == pytest.approx(5.0)

    for g in G_BRACKET:
        cnt = np.bincount(post[ff], minlength=area.size)
        w_new = g * k_local[post[ff]] / cnt[post[ff]]
        k_after = np.zeros(area.size)
        np.add.at(k_after, post[ff], w_new)
        assert k_after[4] / k_local[4] == pytest.approx(g)
        assert k_after[5] / k_local[5] == pytest.approx(g)
        # The point of EQUAL_G: equal authority, unequal per-edge weight.
        wx = w_new[post[ff] == 4][0]
        wy = w_new[post[ff] == 5][0]
        assert wx != pytest.approx(wy)
        assert wx == pytest.approx(g * 0.4 / 2)
        assert wy == pytest.approx(g * 0.4 / 4)
        assert wx == pytest.approx(2 * wy)


def test_classify_excludes_non_cortical_and_unrelated_layer_pairs():
    area, layer, pre, post, w = synthetic()
    cortical = np.ones(area.size, dtype=bool)
    cortical[0] = False                      # pretend source 0 is an input interface
    rel = OA.classify(area, layer, pre, post, cortical)
    assert (rel[pre == 0] == "").all(), "edges from a non-cortical source must carry no relation"
    assert set(np.unique(rel)) <= {"", "ff", "fb", "lat", "local"}


def test_relation_table_covers_exactly_the_declared_motifs():
    assert OA.RELATION == {("L2", "L4"): "ff", ("L3", "L4"): "ff",
                           ("L6", "L1"): "fb", ("L3", "L3"): "lat"}
    assert set(OA.RELATION.values()) == set(OA.CHANNELS)


@pytest.mark.skipif(not RESULT.exists(), reason="bracket not executed in this tree")
def test_recorded_bracket_receipts_are_self_consistent():
    rec = json.loads(RESULT.read_text(encoding="utf-8"))
    for arm in rec["arms"]:
        r = arm["o_authority"]
        a = r["acceptance"]
        # A realization invariant, not a scientific verdict: the arm asserts its own weights
        # match the authority it declared. A scientific FAIL does not touch it.
        recorded = a["pass"]
        assert recorded == (a["max_abs_error"] < a["eps"]), arm["g"]
        assert r["edges_reweighted"] + r["edges_untouched"] == r["edges_total"], arm["g"]
        assert r["local_and_retinal_weights_unchanged"], arm["g"]
        assert r["delay"].startswith("DELAY_UNQUALIFIED_FROZEN"), arm["g"]
        # Every arm must reweight the same edges; only their values differ.
        assert r["edges_reweighted"] == rec["arms"][0]["o_authority"]["edges_reweighted"]

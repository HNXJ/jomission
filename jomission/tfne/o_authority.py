"""EQUAL_G realization of TFNE O and X coupling.

Per-edge weight is the wrong primitive for long-range coupling strength when convergence
differs between targets. This module makes the dimensionless transfer authority the
architectural parameter and derives the weight from it, per receiving neuron:

    g_p,i = K_p,i / K_local,i                                  specification
    w_p,i = g_p * K_local,i / N_afferents,p,i                  realization

so a target reached by two upstream areas gets a smaller per-edge weight rather than twice
the authority. K_local,i is neuron i's summed local excitatory afferent weight, which is what
the long-range input is being measured against.

Channel, not motif, is the unit of authority: g_FF names the pooled L2.E->L4.E and L3.E->L4.E
input a cell receives. Within a channel the existing relative contribution of each motif is
preserved, which for the current realization means one per-edge weight per (channel, target),
since every motif in a channel starts at the same weight.

Delay is untouched. It remains DELAY_UNQUALIFIED_FROZEN at the inherited 0 steps so that
coupling gain and temporal coupling are not calibrated on the same lineage.
"""

from __future__ import annotations

import dataclasses

import numpy as np

#: TFNE relation per (pre layer, post layer) for the realized cortical projections.
#: ff and fb come from the O[fffb] clauses, lat from X[lat].
RELATION = {("L2", "L4"): "ff", ("L3", "L4"): "ff", ("L6", "L1"): "fb", ("L3", "L3"): "lat"}

CHANNELS = ("ff", "fb", "lat")


def classify(area, layer, pre, post, cortical):
    """Per-edge channel label: 'ff', 'fb', 'lat', 'local', or '' for anything else.

    Only edges with both endpoints cortical are labelled. Retinal afferents are an input
    interface, not a TFNE relation, and are left out of both the numerator and the denominator.
    """
    both = cortical[pre] & cortical[post]
    same = area[pre] == area[post]
    rel = np.where(both & same, "local", "")
    cross = np.flatnonzero(both & ~same)
    lab = np.array([RELATION.get((layer[a], layer[b]), "") for a, b in zip(pre[cross], post[cross])])
    rel[cross] = lab
    return rel


def local_authority(weight, post, rel, n_neurons):
    """K_local,i: summed local excitatory afferent weight per neuron."""
    m = (rel == "local") & (weight > 0)
    k = np.zeros(n_neurons)
    np.add.at(k, post[m], np.abs(weight[m]))
    return k


def realize_equal_g(model, index, area, layer, cls, g, *, eps=1e-6):
    """Return (model, receipt) with every long-range cortical weight set by EQUAL_G at authority g.

    Nothing but the weight of the 24 declared cortical projections changes: connectivity, delay,
    tau, receptor, retinal edges, local weights, drive and every other parameter are untouched.
    """
    el = model.params["edge_list"]
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    w = np.asarray(el.weight).astype(np.float64).copy()
    n = int(max(post.max(), pre.max())) + 1

    cortical = np.zeros(n, dtype=bool)
    for name, (lo, hi, _) in index.items():
        if not name.startswith("Retina"):
            cortical[lo:hi] = True

    rel = classify(np.asarray(area), np.asarray(layer), pre, post, cortical)
    k_local = local_authority(w, post, rel, n)

    before = {}
    after = {}
    per_channel = {}
    touched = np.zeros(w.size, dtype=bool)
    for ch in CHANNELS:
        m = rel == ch
        if not m.any():
            continue
        cnt = np.bincount(post[m], minlength=n)
        k_before = np.zeros(n)
        np.add.at(k_before, post[m], np.abs(w[m]))
        recv = np.flatnonzero(cnt > 0)
        if not (k_local[recv] > 0).all():
            raise ValueError(f"{ch}: a target has no local excitatory afferents; g is undefined there")
        # w_p,i = g * K_local,i / N_p,i, applied to every edge of this channel by its target.
        w_target = np.zeros(n)
        w_target[recv] = g * k_local[recv] / cnt[recv]
        sign = np.sign(w[m])
        if not (sign > 0).all():
            raise ValueError(f"{ch}: expected excitatory long-range edges only")
        w[m] = w_target[post[m]]
        touched |= m
        k_after = np.zeros(n)
        np.add.at(k_after, post[m], np.abs(w[m]))
        before[ch] = float(k_before[recv].mean())
        after[ch] = float(k_after[recv].mean())
        per_channel[ch] = {
            "n_edges": int(m.sum()),
            "n_target_cells": int(recv.size),
            "convergence_classes": sorted({int(c) for c in cnt[recv]}),
            "w_before_mean": float(np.abs(np.asarray(el.weight))[m].mean()),
            "w_after_mean": float(np.abs(w[m]).mean()),
            "w_after_by_convergence": {str(int(c)): float(w_target[recv[cnt[recv] == c]].mean())
                                       for c in sorted({int(x) for x in cnt[recv]})},
            "g_before_mean": float((k_before[recv] / k_local[recv]).mean()),
            "g_after_max_abs_error": float(np.abs(k_after[recv] / k_local[recv] - g).max()),
        }

    # Acceptance, evaluated on every cell that receives any long-range input at all, and on the
    # float32 weights the kernel will actually read rather than the float64 intermediate.
    w32 = np.asarray(w, dtype=np.float32)
    m_long = np.isin(rel, CHANNELS)
    k_long = np.zeros(n)
    np.add.at(k_long, post[m_long], np.abs(w32[m_long]).astype(np.float64))
    recv_any = np.flatnonzero(np.bincount(post[m_long], minlength=n) > 0)
    err = np.abs(k_long[recv_any] / k_local[recv_any] - g)
    multi = [c for c in recv_any if sum(int((rel == ch)[post == c].any()) for ch in CHANNELS) > 1]

    params = dict(model.params)
    params["edge_list"] = dataclasses.replace(el, weight=w32)
    receipt = {
        "policy": "EQUAL_G",
        "g": g,
        "definition": "w_p,i = g * K_local,i / N_afferents,p,i, per receiving neuron",
        "channels_held_equal": "g_FF = g_FB = g_X = g, one principal delta",
        "delay": "DELAY_UNQUALIFIED_FROZEN: inherited uniform 0 steps, untouched",
        "edges_reweighted": int(touched.sum()),
        "edges_total": int(w.size),
        "edges_untouched": int((~touched).sum()),
        "local_and_retinal_weights_unchanged": bool(
            np.array_equal(np.asarray(el.weight)[~touched], w32[~touched])),
        "K_local_mean_over_targets": float(k_local[recv_any].mean()),
        "per_channel": per_channel,
        "cells_receiving_two_channels": len(multi),
        "acceptance": {
            "rule": "max_i |K_long,i / K_local,i - g| < eps",
            "eps": eps,
            "n_cells_checked": int(recv_any.size),
            "max_abs_error": float(err.max()),
            "max_rel_error": float((err / g).max()),
            "measured_on": "float32 weights as stored, not the float64 intermediate",
            "pass": bool(err.max() < eps),
        },
    }
    return dataclasses.replace(model, params=params), receipt

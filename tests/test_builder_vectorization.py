"""Tier 1: builder mask/gain helpers match their scalar references bit-for-bit.

Each test replicates the pre-vectorization per-edge loop inline (the reference)
and compares against the module helper (the candidate) across asymmetric,
boundary, inhibitory/excitatory, empty/small, and out-of-range/negative-index
cases. No simulation; no model needed.
"""

import numpy as np

from jomission.network.builder import (
    DELAY_FB_MS,
    DELAY_FF_MS,
    DELAY_WITHIN_MS,
    _laminar_delay_ms,
    _motif_gains,
    _sst_mask,
    _vertical_motif_gains,
)


def ref_sst_mask(cell_types, pre_np):
    n_edges = int(pre_np.shape[0])
    is_sst = np.zeros(n_edges, dtype=bool)
    for i in range(n_edges):
        try:
            if cell_types[int(pre_np[i])] == "SST":
                is_sst[i] = True
        except Exception:
            pass
    return is_sst


def ref_motif_gains(cell_types, pre_np, post_np, gain_map, w_len):
    gains = np.ones(w_len, dtype=np.float64)
    for i in range(len(pre_np)):
        try:
            pc = cell_types[int(pre_np[i])]
            qc = cell_types[int(post_np[i])]
            g = gain_map.get((pc, qc), 1.0)
            gains[i] = float(g)
        except Exception:
            gains[i] = 1.0
    return gains


def ref_laminar_delay_ms(area_labels, layer_labels, pre_np, post_np):
    n_edges = int(pre_np.shape[0])
    delay_ms = np.zeros(n_edges, dtype=np.float64)
    for k in range(n_edges):
        try:
            a_pre = area_labels[int(pre_np[k])]
            a_post = area_labels[int(post_np[k])]
            if a_pre == a_post:
                delay_ms[k] = float(DELAY_WITHIN_MS)
            else:
                post_layer = layer_labels[int(post_np[k])]
                is_ff = post_layer == "L4"
                delay_ms[k] = float(DELAY_FF_MS) if is_ff else float(DELAY_FB_MS)
        except Exception:
            delay_ms[k] = float(DELAY_WITHIN_MS)
    return delay_ms


def ref_vertical_motif_gains(area_labels, layer_labels, cell_types, pre_np, post_np, gain_map):
    n_edges = int(pre_np.shape[0])
    gains = np.ones(n_edges, dtype=np.float64)
    vertical_pairs = {("L4", "L2/3"), ("L2/3", "L5")}
    for i in range(n_edges):
        try:
            pre_id = int(pre_np[i])
            post_id = int(post_np[i])
            if area_labels[pre_id] != area_labels[post_id]:
                continue
            pre_l = layer_labels[pre_id]
            post_l = layer_labels[post_id]
            if (pre_l, post_l) not in vertical_pairs:
                continue
            pre_ct = cell_types[pre_id]
            post_ct = cell_types[post_id]
            key = (pre_l, pre_ct, post_l, post_ct)
            g = gain_map.get(key, 1.0)
            gains[i] = float(g)
        except Exception:
            gains[i] = 1.0
    return gains


LABELS = {
    "area": ["V1", "V1", "V4", "FEF", "V1"],
    "layer": ["L4", "L2/3", "L5", "L4", "L1"],
    "cell": ["E", "PV", "SST", "VIP", "E"],
}
GAIN_MAP = {("E", "E"): 1.5, ("E", "PV"): 0.7}
VERTICAL_MAP = {("L4", "E", "L2/3", "E"): 2.0, ("L2/3", "E", "L5", "E"): 1.5}


def edge_cases(n):
    base = [
        np.arange(n, dtype=np.int64),
        np.arange(n - 1, -1, -1, dtype=np.int64),
        np.full(n, 2, dtype=np.int64),
        np.array([n + 5, -n - 1, 0, 1, -1][:n], dtype=np.int64),
    ]
    if n == 0:
        return [np.array([], dtype=np.int64)]
    return base


def test_sst_mask_matches_scalar():
    for n in (0, 1, 5):
        for pre in edge_cases(n):
            np.testing.assert_array_equal(
                _sst_mask(LABELS["cell"][: max(n, 1)], pre),
                ref_sst_mask(LABELS["cell"][: max(n, 1)], pre),
            )


def test_motif_gains_matches_scalar():
    for n in (0, 1, 5):
        ct = LABELS["cell"][: max(n, 1)]
        for pre in edge_cases(n):
            for post in edge_cases(n):
                np.testing.assert_array_equal(
                    _motif_gains(ct, pre, post, GAIN_MAP),
                    ref_motif_gains(ct, pre, post, GAIN_MAP, n),
                )
    assert (_motif_gains(["E"], np.array([0]), np.array([0]), {}) == 1.0).all()


def test_laminar_delay_ms_matches_scalar():
    for n in (0, 1, 5):
        a = LABELS["area"][: max(n, 1)]
        la = LABELS["layer"][: max(n, 1)]
        for pre in edge_cases(n):
            for post in edge_cases(n):
                np.testing.assert_array_equal(
                    _laminar_delay_ms(a, la, pre, post), ref_laminar_delay_ms(a, la, pre, post)
                )


def test_vertical_motif_gains_matches_scalar():
    for n in (0, 1, 5):
        a = LABELS["area"][: max(n, 1)]
        la = LABELS["layer"][: max(n, 1)]
        ct = LABELS["cell"][: max(n, 1)]
        for pre in edge_cases(n):
            for post in edge_cases(n):
                np.testing.assert_array_equal(
                    _vertical_motif_gains(a, la, ct, pre, post, VERTICAL_MAP),
                    ref_vertical_motif_gains(a, la, ct, pre, post, VERTICAL_MAP),
                )

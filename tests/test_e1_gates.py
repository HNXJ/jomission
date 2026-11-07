"""E1 seal gates (docs/laminar_motif/E1_SEAL.md, Gates before scoring).

1. Realized sizes: each construct has exactly the declared N_SST, N_PV, N_E;
   zero PV-to-SST and SST-to-SST edges; each block's edge count within 4 SD of p * n_pre * n_post.
2. E0 tests pass unchanged on the parameterized build (default arguments equal E0 constants).
3. Positive control: the rho = 1 build equals the E0 build (byte-equal on seed 1).
4. Null: the permutation statistic is computed on a synthetic case where f_pk is independent of rho;
   it must not pass.
"""
from pathlib import Path

import numpy as np
import pytest

from jomission.laminar import caminos_e0 as c
from jomission.laminar import e1_verdict as v

P = 0.07


def test_gate1_realized_sizes_and_connectivity_across_all_rhos():
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_kernel_args

    for rho in v.RHOS:
        n_sst = c.rho_to_nsst(rho)
        sz = c.sizes(n_sst)
        assert sz["E"] == 8000
        assert sz["PV"] + sz["SOM"] == 1500
        assert sz["SOM"] == v.N_SST[rho]
        assert sz["PV"] == v.N_PV[rho]

        model = jtfne.construct(c.build_spec(P, 1, n_sst=n_sst))
        kw = adex_kernel_args(model, c.DT_MS)
        labels = tuple(model.params["emitter"].labels)
        assert labels.count("E") == sz["E"]
        assert labels.count("PV") == sz["PV"]
        assert labels.count("SOM") == sz["SOM"]
        assert labels == ("E",) * sz["E"] + ("PV",) * sz["PV"] + ("SOM",) * sz["SOM"]

        pre, post = np.asarray(kw["edge_pre"]), np.asarray(kw["edge_post"])
        cls_edges = np.array([sz["E"], sz["E"] + sz["PV"], c.N])
        pcls = np.searchsorted(cls_edges, pre, side="right")
        qcls = np.searchsorted(cls_edges, post, side="right")
        names = ("E", "PV", "SOM")
        declared = {(a, b) for _, a, b, _, _ in c.RULES}
        total = 0
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                sel = (pcls == i) & (qcls == j)
                n = int(sel.sum())
                total += n
                if (a, b) not in declared:
                    assert n == 0, f"{a}->{b} must have no edges for rho={rho}"
                else:
                    pairs = sz[a] * sz[b] - (sz[a] if a == b else 0)
                    mean = P * pairs
                    sd = (pairs * P * (1.0 - P)) ** 0.5
                    assert abs(n - mean) < 4 * sd, f"{a}->{b} n={n} outside 4 SD of {mean}"
        assert total == len(pre)


def test_gate2_default_arguments_equal_e0_constants():
    assert c.sizes() == {"E": c.NE, "PV": c.NPV, "SOM": c.NSST}
    assert c.rho_to_nsst(1.0) == c.NSST


def test_gate3_positive_control_rho1_equals_e0():
    """Verify rho = 1 construct produces identical connectivity to E0 build."""
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_kernel_args

    m_e0 = jtfne.construct(c.build_spec(P, 1))
    m_rho1 = jtfne.construct(c.build_spec(P, 1, n_sst=750))
    kw_e0 = adex_kernel_args(m_e0, c.DT_MS)
    kw_rho1 = adex_kernel_args(m_rho1, c.DT_MS)

    for key in ("edge_pre", "edge_post", "edge_weight_nS", "edge_delay_steps", "edge_receptor"):
        assert np.array_equal(kw_e0[key], kw_rho1[key]), f"Mismatch in {key}"


def test_gate4_null_hypothesis_fails_permutation_test():
    """Gate 4: Synthetic case where f_pk is independent of rho must not pass."""
    null_runs = {}
    for s in v.SEEDS:
        for rho in v.RHOS:
            null_runs[(rho, s)] = {
                "rho": rho,
                "seed": s,
                "steps": v.STEPS_EXPECTED,
                "analysis": {
                    "rates_Hz": {"E": 1.1, "PV": 11.0, "SOM": 7.0},
                    "peak_freq_E_Hz": 34.0,
                    "peak_excess_E": 15.0,
                    "oscillates": True,
                    "undefined": [],
                },
            }
    res = v.verdict(null_runs)
    assert res["verdict"] == "FAIL"
    assert res["conditions"]["c4_permutation_test"]["status"] == "FAIL"

"""E1b arm-B builder (docs/laminar_motif/E1B_SEAL.md): N_PV fixed at 750, N_SST = round(750 rho).

The default build must stay the E0/E1 build; arm B must realize its declared sizes and blocks."""

import numpy as np
import pytest

from jomission.laminar import caminos_e0 as c

RHOS = (0.25, 0.5, 1.0, 2.0, 4.0)
ARM_B_NSST = {0.25: 188, 0.5: 375, 1.0: 750, 2.0: 1500, 4.0: 3000}


def test_default_sizes_unchanged():
    assert c.sizes() == {"E": 8000, "PV": 750, "SOM": 750}
    for rho, n_sst in zip(RHOS, (300, 500, 750, 1000, 1200)):
        assert c.sizes(c.rho_to_nsst(rho)) == {"E": 8000, "PV": 1500 - n_sst, "SOM": n_sst}


def test_arm_b_sizes():
    for rho in RHOS:
        n_sst = int(round(750 * rho))
        assert n_sst == ARM_B_NSST[rho]
        assert c.sizes(n_sst, n_pv=750) == {"E": 8000, "PV": 750, "SOM": n_sst}
    with pytest.raises(ValueError):
        c.sizes(10, n_pv=-1)


def _edges(model):
    from jaxfne._construct_adex import adex_kernel_args

    kw = adex_kernel_args(model, c.DT_MS)
    return {k: np.asarray(kw[k]) for k in ("edge_pre", "edge_post", "edge_weight_nS", "edge_receptor")}


def test_arm_b_at_rho1_equals_default_build():
    import jaxfne as jtfne

    a = _edges(jtfne.construct(c.build_spec(0.07, 1)))
    b = _edges(jtfne.construct(c.build_spec(0.07, 1, n_sst=750, n_pv=750)))
    assert a["edge_pre"].size > 0, "fixture must have edges"
    for k in a:
        assert np.array_equal(a[k], b[k]), k


@pytest.mark.parametrize("rho", (0.25, 4.0))
def test_arm_b_realized_sizes_and_blocks(rho):
    import jaxfne as jtfne

    n_sst = ARM_B_NSST[rho]
    model = jtfne.construct(c.build_spec(0.07, 1, n_sst=n_sst, n_pv=750))
    labels = tuple(model.params["emitter"].labels)
    assert labels == ("E",) * 8000 + ("PV",) * 750 + ("SOM",) * n_sst
    e = _edges(model)
    size = {"E": 8000, "PV": 750, "SOM": n_sst}
    bounds = np.array([8000, 8750, 8750 + n_sst])
    pc = np.searchsorted(bounds, e["edge_pre"], side="right")
    qc = np.searchsorted(bounds, e["edge_post"], side="right")
    declared = {(a, b) for _, a, b, _, _ in c.RULES}
    for i, a in enumerate(("E", "PV", "SOM")):
        for j, b in enumerate(("E", "PV", "SOM")):
            n = int(((pc == i) & (qc == j)).sum())
            if (a, b) not in declared:
                assert n == 0, f"{a}->{b} must have no edges"
                continue
            pairs = size[a] * size[b] - (size[a] if a == b else 0)
            mean, sd = 0.07 * pairs, (pairs * 0.07 * 0.93) ** 0.5
            assert abs(n - mean) < 4 * sd, (a, b, n, mean, sd)

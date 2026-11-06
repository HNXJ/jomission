"""E0 gates (docs/laminar_motif/E0_SEAL.md): parameter identity, realized connectivity, estimators with a known positive and a null.

Short JAX (one 9500-neuron construct), no simulation, writes nothing.
"""

import numpy as np
import pytest

from jomission.laminar import caminos_analysis as an
from jomission.laminar import caminos_e0 as c

DT = 1e-4
NE, NPV, NSST = 8000, 750, 750

# JaxFNE PORT_SPEC Table 1 (E, PV, SOM), read from artifacts/etudes/papers/caminos/PORT_SPEC.md.
TABLE1 = {
    "VT_mV": (-50.0, -50.0, -50.0),
    "Delta_mV": (2.0, 0.5, 2.0),
    "Tref_ms": (5.0, 5.0, 5.0),
    "tauw_ms": (500.0, 500.0, 500.0),
    "a_nS": (4.0, 0.0, 4.0),
    "b_pA": (50.0, 0.0, 80.0),
    "C_pF": (200.0, 200.0, 200.0),
    "gL_nS": (10.0, 10.0, 10.0),
    "EL_mV": (-65.0, -65.0, -55.0),
    "Vreset_mV": (-65.0, -65.0, -65.0),
}


def _synthetic(seed, modulate, sst_delay_steps=70, T_s=12.0, f_hz=35.0):
    """Population counts [steps, 3]; with modulate, all rates follow a 35 Hz cycle and SST lags PV."""
    rng = np.random.default_rng(seed)
    n = int(round(T_s / DT))
    t = np.arange(n) * DT
    amp = 0.8 if modulate else 0.0
    base = {"E": 1.1, "PV": 11.0, "SOM": 7.0}
    size = {"E": NE, "PV": NPV, "SOM": NSST}

    def counts(pop, shift):
        tt = t - shift * DT
        rate = base[pop] * (1.0 + amp * np.sin(2 * np.pi * f_hz * tt))
        return rng.poisson(size[pop] * rate * DT)

    return np.stack([counts("E", 0), counts("PV", 0), counts("SOM", sst_delay_steps)], axis=1).astype(np.int32)


SIZES = {"E": NE, "PV": NPV, "SOM": NSST}


def test_estimators_known_positive():
    out = an.analyze_run(_synthetic(1, True), DT, SIZES)
    assert out["oscillates"] is True
    assert 33.0 < out["peak_freq_E_Hz"] < 37.0
    assert 6.0 < out["lag_PV_SOM_ms"] < 8.0, "SST built 7 ms after PV"
    assert abs(out["rates_Hz"]["E"] - 1.1) < 0.1


def test_estimators_null_rate_matched_poisson():
    pos = _synthetic(1, True)
    null = _synthetic(1, False)
    assert null[:, 0].sum() > 0 and abs(null[:, 0].mean() / pos[:, 0].mean() - 1) < 0.05, "null must match the rate"
    assert an.analyze_run(null, DT, SIZES)["oscillates"] is False


def test_zero_spike_population_is_undefined():
    cnt = _synthetic(1, True)
    cnt[:, 2] = 0
    out = an.analyze_run(cnt, DT, SIZES)
    assert "lag" in out["undefined"] and out["lag_PV_SOM_ms"] is None


def test_spikes_to_counts_known_answer():
    rng = np.random.default_rng(0)
    sp = rng.random((50, c.N)) < 0.01
    cnt = c.spikes_to_counts(sp, SIZES)
    assert cnt.shape == (50, 3) and cnt.dtype == np.int32
    assert (cnt.sum(axis=1) == sp.sum(axis=1)).all() and sp.sum() > 0
    assert (cnt[:, 1] == sp[:, NE : NE + NPV].sum(axis=1)).all()


def test_spec_constants_and_rules():
    assert (c.NE, c.NPV, c.NSST, c.DT_MS, c.DELAY_MS) == (NE, NPV, NSST, 0.1, 2.0)
    assert (c.Q_E_E, c.Q_E_INH, c.Q_INH, c.Q_EXT) == (2.0, 3.0, 5.0, 2.5)
    assert (c.TAU_E_MS, c.TAU_I_MS, c.REV_E_MV, c.REV_I_MV) == (2.0, 8.0, 0.0, -80.0)
    assert c.ext_source_count(0.07) == 560
    d = c.drive_spec(0.07, 3)
    assert d["n_sources"] == 560 and d["rate_hz"] == 4.0 and d["weight_nS"] == 2.5
    assert d["target_cell_types"] == ["E", "PV"] and d["seed"] == 3
    pairs = {(pre, post) for _, pre, post, _, _ in c.RULES}
    assert len(c.RULES) == 7 and len(pairs) == 7
    assert ("PV", "SOM") not in pairs and ("SOM", "SOM") not in pairs
    assert ("SOM", "PV") in pairs, "strong SST to PV"


@pytest.fixture(scope="module")
def realized():
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_kernel_args

    model = jtfne.construct(c.build_spec(0.07, 1))
    return model, adex_kernel_args(model, c.DT_MS)


def test_realized_cell_parameters_equal_table1(realized):
    model, _ = realized
    em = model.params["emitter"]
    labels = tuple(em.labels)
    assert labels.count("E") == NE and labels.count("PV") == NPV and labels.count("SOM") == NSST
    assert labels == ("E",) * NE + ("PV",) * NPV + ("SOM",) * NSST, "population order is E, PV, SOM"
    for name, expected in TABLE1.items():
        arr = np.asarray(getattr(em, name), dtype=float)
        for cls, sl in zip(expected, (slice(0, NE), slice(NE, NE + NPV), slice(NE + NPV, None))):
            assert (arr[sl] == cls).all(), (name, cls)


def test_realized_edges_match_declared_blocks(realized):
    _, kw = realized
    pre, post = np.asarray(kw["edge_pre"]), np.asarray(kw["edge_post"])
    w, rec, dly = np.asarray(kw["edge_weight_nS"]), np.asarray(kw["edge_receptor"]), np.asarray(kw["edge_delay_steps"])
    cls_edges = np.array([NE, NE + NPV, c.N])
    pcls, qcls = np.searchsorted(cls_edges, pre, side="right"), np.searchsorted(cls_edges, post, side="right")
    names = ("E", "PV", "SOM")
    size = {"E": NE, "PV": NPV, "SOM": NSST}
    assert (pre != post).all(), "no autapses"
    assert (dly == int(round(c.DELAY_MS / c.DT_MS))).all()
    declared = {(a, b): (q, m) for _, a, b, q, m in c.RULES}
    receptor = {"ampa": 0, "gaba": 1}
    total = 0
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            sel = (pcls == i) & (qcls == j)
            n = int(sel.sum())
            total += n
            if (a, b) not in declared:
                assert n == 0, f"{a}->{b} must have no edges"
                continue
            q, m = declared[(a, b)]
            pairs = size[a] * size[b] - (size[a] if a == b else 0)
            mean, sd = 0.07 * pairs, (pairs * 0.07 * 0.93) ** 0.5
            assert abs(n - mean) < 4 * sd, (a, b, n, mean, sd)
            assert (w[sel] == q).all() and (rec[sel] == receptor[m]).all(), (a, b)
    assert total == len(pre), "every edge belongs to a declared or checked-empty block"

"""Sparse within-area requests are realized: p_connect lowers the realized edge count.

Restores the sparse-path coverage that tests/test_runtime_regression.py lost when its
within_area='sparse' input was refused by jaxfne 0.5.8 (realized form: p_connect).
"""

import numpy as np

import jaxfne as jtfne

N = 200
DENSE = N * (N - 1)


def _model(p_connect):
    cfg = (
        jtfne.Configuration()
        .runtime(seed=1, dtype="float32", duration_ms=20.0, dt_ms=0.5)
        .column(name="c", layers=["L4"], n=N)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45, p_connect=p_connect)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    return jtfne.construct(cfg)


def _n_edges(model):
    el = model.params["edge_list"]
    return int(el.n_edges) if hasattr(el, "n_edges") else int(np.asarray(el.pre).shape[0])


def test_sparse_p_connect_realizes_fewer_edges_than_dense():
    sparse = _n_edges(_model(0.05))
    denser = _n_edges(_model(0.5))
    assert 0 < sparse < denser, (sparse, denser)
    assert sparse < 0.2 * DENSE, (sparse, DENSE)
    # Realized density tracks the request within sampling noise (generous bound).
    assert abs(sparse / DENSE - 0.05) < 0.02, sparse / DENSE

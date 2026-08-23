"""Audit that L1 has no E/PV/SST — only VIP — per provenance."""

from collections import Counter
from jomission.network.builder import build_jomission_model


def test_l1_no_excitatory():
    model = build_jomission_model(n_per_area=100, seed=0)
    meta = model.static.get("neuron_metadata") or []
    counter = Counter((r["area"], r["layer"], r["cell_type"]) for r in meta)
    for area in ("V1", "V4", "FEF", "PFC"):
        assert (area, "L1", "VIP") in counter
        assert counter[(area, "L1", "VIP")] == 8
        assert (area, "L1", "E") not in counter, f"L1_E exists in {area}"
        assert (area, "L1", "PV") not in counter
        assert (area, "L1", "SST") not in counter
    # Total L1 is 32
    l1_total = sum(v for k, v in counter.items() if k[1] == "L1")
    assert l1_total == 32

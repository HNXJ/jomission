"""Per-unit SST->E decay (sstNN, sstdNN) in the CAMINOS stack: parse and edge-level wiring."""
import numpy as np
import pytest

from jomission.laminar import caminos_stack as cs

SIZES = {"S": {"E": 40, "PV": 6, "SOM": 4}, "M": {"E": 40, "PV": 5, "SOM": 5}, "D": {"E": 40, "PV": 4, "SOM": 6}}


def test_sst_taus_parse():
    assert cs.sst_taus("e2_3unit", SIZES) is None
    assert cs.sst_taus("e2_3unit_sst15", SIZES) == {"S": 15.0, "M": 15.0, "D": 15.0}
    assert cs.sst_taus("e2_3unit_sst15_sstd60", SIZES) == {"S": 15.0, "M": 15.0, "D": 60.0}
    assert cs.sst_taus("e2_3unit_sstd30", SIZES) == {"D": 30.0}
    assert cs.sst_taus("e2_3unit_sstd30", {"C": SIZES["S"], "E": SIZES["M"]}) is None  # not area/type names


def test_rhod_sets_deep_ratio_only():
    base, _, _, _ = cs.layout("e2_3unit_n4000_dens")
    sizes, rho, _, _ = cs.layout("e2_3unit_n4000_dens_rhod8")
    assert rho == {**cs.RHO, "D": 8.0}
    assert sizes["S"] == base["S"] and sizes["M"] == base["M"]
    d = sizes["D"]
    assert d["E"] == base["D"]["E"] and d["PV"] + d["SOM"] == base["D"]["PV"] + base["D"]["SOM"]
    assert abs(d["SOM"] / d["PV"] - 8.0) < 0.2 and d["SOM"] > base["D"]["SOM"]


@pytest.mark.parametrize("lay", ["e2_3unit_sst15_sstd60", "e2_3unit_sstd30"])
def test_sstd_wires_deep_only(lay):
    pytest.importorskip("jaxfne")
    p_local = {m: 0.5 for m in SIZES}  # small fixture: K_E / 40 would exceed 1 (refused since W1)
    model, rows = cs.construct(0, SIZES, 0.01, p_local, tau_sst_e_ms=cs.sst_taus(lay, SIZES))
    st = model.static["adex"]
    pre, post = np.asarray(st["pre"]), np.asarray(st["post"])
    rec, names = np.asarray(st["receptor"]), list(st["receptor_names"])
    tau = np.asarray(st["receptor_tau_ms"], dtype=float)
    lay_ = np.array([r["layer"] for r in rows])
    typ = np.array([r["cell_type"] for r in rows])
    taus = cs.sst_taus(lay, SIZES)
    for m in SIZES:
        sel = (typ[pre] == "SOM") & (typ[post] == "E") & (lay_[post] == m) & (lay_[pre] == m)
        assert sel.sum() > 0, "fixture must have SST->E edges"
        got = set(tau[rec[sel]].tolist())
        assert got == {taus.get(m, 8.0)}, (m, got, names)

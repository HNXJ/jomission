"""rhouUX: per-unit SOM/PV ratio for data-derived composition (OPEN-38). Units not given keep RHO; the inhibitory
count per unit is unchanged; conflicting ratio modifiers are refused."""
import numpy as np
import pytest

from jomission.laminar import caminos_stack as cs

REF = "e2_3unit_n4000_dens_k1_kd1_sst15"


def test_rhou_sets_named_units_only_and_keeps_inhibitory_count():
    s, rho = cs.layout(REF)[:2]
    u, rho_u = cs.layout(REF + "_rhouS0p6_rhouD1p8")[:2]
    assert rho["S"] != 0.6 and rho["D"] != 1.8, "fixture targets must differ from RHO"
    assert rho_u == {"S": 0.6, "M": rho["M"], "D": 1.8}
    for m in cs.MODULES:
        assert u[m]["E"] == s[m]["E"] and u[m]["PV"] + u[m]["SOM"] == s[m]["PV"] + s[m]["SOM"], m
        np.testing.assert_allclose(u[m]["SOM"] / u[m]["PV"], rho_u[m], rtol=0.02, err_msg=m)


def test_rhou_default_layout_unchanged():
    assert cs.layout("e2_3unit")[1] == cs.RHO


@pytest.mark.parametrize("bad", ["e2_3unit_rhouS1_rhod2", "e2_3unit_rhouS1_rhof1", "e2_3unit_rhouS1_rhor",
                                 "e2_3unit_rhouS1_rhos2", "e2_3unit_rhouS1_rhouS2", "e2_3unit_rhouX1",
                                 "e2_3unit_rhouS0", "e2_3unit_rhouD-1"])
def test_rhou_refusals(bad):
    with pytest.raises(ValueError, match="cannot be combined|declared twice|unit must be|must be > 0"):
        cs.variant(bad)


def test_rhou_refused_outside_3unit_stack():
    with pytest.raises(ValueError, match="3-unit stack"):
        cs.layout("e2_21bin_rhouS1")

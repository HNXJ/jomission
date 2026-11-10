"""Density and SOM/PV-ratio control layouts (seal docs/laminar_motif/SEAL_T8.md): densr exchanges the S and D
densities for unit size and kd in-degree; rhofX sets a flat ratio; rhor reverses the ramp. Each test builds its case
and asserts the case's precondition."""
import numpy as np
import pytest

from jomission.laminar import caminos_stack as cs

REF = "e2_3unit_n4000_dens_k1_kd1_sst15"


def test_densr_exchanges_s_and_d_sizes_and_indegree():
    s, rho, _, _ = cs.layout(REF)
    r, rho_r, _, _ = cs.layout("e2_3unit_n4000_densr_k1_kd1_sst15")
    assert s["S"]["E"] != s["D"]["E"], "fixture must have unequal S and D sizes"
    assert (r["S"]["E"], r["M"]["E"], r["D"]["E"]) == (s["D"]["E"], s["M"]["E"], s["S"]["E"])
    assert rho_r == rho, "densr must not touch the ratio"
    p, pr = cs.local_p(REF, s), cs.local_p("e2_3unit_n4000_densr_k1_kd1_sst15", r)
    k = {m: p[m] * s[m]["E"] for m in s}
    kr = {m: pr[m] * r[m]["E"] for m in r}
    assert k["S"] != pytest.approx(k["D"]), "fixture must have unequal S and D in-degrees"
    np.testing.assert_allclose([kr["S"], kr["M"], kr["D"]], [k["D"], k["M"], k["S"]], rtol=1e-12)


def test_flat_density_keeps_total_e_and_mean_indegree():
    s = cs.layout(REF)[0]
    f = cs.layout("e2_3unit_n4000_k1_sst15")[0]
    assert len({f[m]["E"] for m in f}) == 1, "flat control must have equal unit sizes"
    assert abs(sum(f[m]["E"] for m in f) - sum(s[m]["E"] for m in s)) <= 2, "total E kept up to rounding"
    assert cs.local_p("e2_3unit_n4000_k1_sst15", f) is None, "no kd: one in-degree rule for every unit"


def test_rhof_sets_flat_ratio_with_inhibitory_count_kept():
    s, rho, _, _ = cs.layout(REF)
    f, rho_f, _, _ = cs.layout("e2_3unit_n4000_dens_k1_kd1_sst15_rhof0p794")
    assert len(set(RHO := rho.values())) == 3, f"fixture must have a ramp, got {RHO}"
    assert rho_f == {m: 0.794 for m in cs.MODULES}
    for m in cs.MODULES:
        assert f[m]["E"] == s[m]["E"] and f[m]["PV"] + f[m]["SOM"] == s[m]["PV"] + s[m]["SOM"], m
        np.testing.assert_allclose(f[m]["SOM"] / f[m]["PV"], 0.794, rtol=0.02, err_msg=m)


def test_rhor_reverses_ramp():
    rho = cs.layout(REF)[1]
    r, rho_r, _, _ = cs.layout("e2_3unit_n4000_dens_k1_kd1_sst15_rhor")
    assert rho["S"] < rho["D"], "fixture ramp must rise with depth"
    assert rho_r == {"S": rho["D"], "M": rho["M"], "D": rho["S"]}
    assert r["D"]["SOM"] < r["D"]["PV"] and r["S"]["SOM"] > r["S"]["PV"]


@pytest.mark.parametrize("bad", ["e2_3unit_dens_densr", "e2_3unit_rhof1_rhod2", "e2_3unit_rhof1_rhor",
                                 "e2_3unit_rhor_rhos2", "e2_3unit_rhof1_rhos2", "e2_3unit_rhor_rhod2",
                                 "e2_3unit_densr1", "e2_3unit_rhor2", "e2_3unit_rhof0", "e2_3unit_rhof-1"])
def test_control_modifier_refusals(bad):
    with pytest.raises(ValueError, match="cannot be combined|takes no value|must be > 0"):
        cs.variant(bad)

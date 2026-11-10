"""Connection probabilities refuse to clip at 1.0 (caminos_stack._prob); layouts below the clip are unchanged."""
import numpy as np
import pytest

from jomission.laminar import caminos_stack as cs

# cs.local_p on e2_3unit_n4000_dens_k1_kd1, baseline code (min(1.0, x) era), repr of the floats
N4000_KD1_LOCAL_P = {"S": 0.13999660483390391, "M": 0.13999990985175864, "D": 0.14000784126408145}


def test_prob_passes_up_to_one_and_refuses_above():
    assert cs._prob(0.5, "x") == 0.5
    assert cs._prob(1.0, "x") == 1.0
    with pytest.raises(ValueError, match=r"local p of unit S = 1\.4 > 1"):
        cs._prob(1.4, "local p of unit S")


def test_n270_local_p_refuses_clip():
    name = "e2_3unit_n270_dens_kd1_sst15"
    sizes, _, _, _ = cs.layout(name)
    assert min(s["E"] for s in sizes.values()) < cs.K_E, "fixture must clip: K_E cells cannot fit in a unit"
    with pytest.raises(ValueError, match=r"local p \(kd\) of unit [SMD] = [\d.]+ > 1"):
        cs.local_p(name, sizes)


def test_n270_default_local_p_refuses_clip():
    pytest.importorskip("jaxfne")
    sizes, _, _, _ = cs.layout("e2_3unit_n270_dens")
    assert min(s["E"] for s in sizes.values()) < cs.K_E, "fixture must clip: K_E cells cannot fit in a unit"
    with pytest.raises(ValueError, match=r"local p of unit [SMD] = [\d.]+ > 1"):
        cs.build_config(0, sizes, cs.P_INTER, None)


def test_n1000_config_builds():
    pytest.importorskip("jaxfne")
    name = "e2_3unit_n1000_dens_k1_kd1_sst15"
    sizes, _, p_inter, _ = cs.layout(name)
    mods = cs.variant(name)[1]
    cfg = cs.build_config(0, sizes, p_inter, cs.local_p(name, sizes), tau_sst_e_ms=cs.sst_taus(name, sizes),
                          k_scale=mods.get("k", 1.0))
    assert cfg is not None


def test_n4000_local_p_unchanged():
    name = "e2_3unit_n4000_dens_k1_kd1"
    sizes, _, _, _ = cs.layout(name)
    p = cs.local_p(name, sizes)
    assert set(p) == set(N4000_KD1_LOCAL_P)
    for m, v in N4000_KD1_LOCAL_P.items():
        np.testing.assert_allclose(p[m], v, rtol=1e-12)

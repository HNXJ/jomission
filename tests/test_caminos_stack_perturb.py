"""Layout perturbations of the CAMINOS stack: giX (GABA weights), rhosX (rho of every unit), ipvUV and isomUV
(constant current on PV or SOM cells). Each test builds its case and asserts the case's precondition."""
import hashlib

import numpy as np
import pytest

from jomission.laminar import caminos_e0 as c0
from jomission.laminar import caminos_stack as cs

SIZES = {"S": {"E": 40, "PV": 6, "SOM": 4}, "M": {"E": 40, "PV": 5, "SOM": 5}, "D": {"E": 40, "PV": 4, "SOM": 6}}
P_LOCAL = {m: 0.5 for m in SIZES}
# sha256 of (pre, post, rule_id, receptor, weight_nS) of e2_3unit_sst15 on SIZES, seed 0, p_inter 0.01,
# p_local 0.5: computed on the code before the perturbation modifiers (bf50b09), 3762 edges
BASELINE_SST15_SHA256 = "39051ce101956cbfbf0e20ff2afb6c35c383f3b0be7bcd0960cfab7fd26450c0"


def _model(name: str, sizes=SIZES):
    """(model, rows) of a layout name on the test fixture, with the arguments run_e2_stack passes."""
    mods = cs.variant(name)[1]
    return cs.construct(0, sizes, 0.01, P_LOCAL, tau_sst_e_ms=cs.sst_taus(name, sizes), k_scale=mods.get("k", 1.0),
                        gaba_scale=mods.get("gi", 1.0), triple_drive=cs.drive_entries(name, sizes),
                        w_scales=mods.get("w"))


def _edge_sha(st) -> str:
    h = hashlib.sha256()
    for key, dt in (("pre", "<i8"), ("post", "<i8"), ("rule_id", "<i8"), ("receptor", "<i8"), ("weight_nS", "<f8")):
        h.update(np.ascontiguousarray(np.asarray(st[key]), dtype=dt).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("base", ["e2_3unit", "e2_3unit_sst15"])
def test_gi2_doubles_gaba_weights_only(base):
    pytest.importorskip("jaxfne")
    name = base + "_gi2"
    assert cs.variant(name)[1]["gi"] == 2.0, "fixture must carry gi2"
    a = _model(base)[0].static["adex"]
    b = _model(name)[0].static["adex"]
    names = list(a["receptor_names"])
    assert names == list(b["receptor_names"])
    for key in ("pre", "post", "rule_id", "receptor"):
        np.testing.assert_array_equal(a[key], b[key])  # same seed: identical edges
    gaba = np.array([names[r].startswith("gaba") for r in np.asarray(a["receptor"])])
    assert gaba.sum() > 0 and (~gaba).sum() > 0, "fixture must have GABA and AMPA edges"
    wa, wb = np.asarray(a["weight_nS"]), np.asarray(b["weight_nS"])
    np.testing.assert_array_equal(wb[gaba], 2.0 * wa[gaba])
    np.testing.assert_array_equal(wb[~gaba], wa[~gaba])


def test_rhos2_doubles_som_over_pv_per_unit():
    base_s, base_rho, _, _ = cs.layout("e2_3unit")
    s, rho, _, _ = cs.layout("e2_3unit_rhos2")
    assert rho == {m: 2 * base_rho[m] for m in base_rho}
    assert cs.layout("e2_3unit_rhod8_rhos0p5")[1] == {"S": 0.125, "M": 0.5, "D": 4.0}, "rhos applies after rhod"
    for m in cs.MODULES:
        b, d = base_s[m], s[m]
        assert b["PV"] > 0 and d["PV"] > 0, "fixture must have PV cells"
        assert d["E"] == b["E"] and d["PV"] + d["SOM"] == b["PV"] + b["SOM"], m
        np.testing.assert_allclose(d["SOM"] / d["PV"], 2 * b["SOM"] / b["PV"], rtol=0.01, err_msg=m)


def test_ipvam200_drives_every_pv_cell_minus_200():
    pytest.importorskip("jaxfne")
    model, rows = _model("e2_3unit_ipvam200")
    typ = np.array([r["cell_type"] for r in rows])
    assert (typ == "PV").sum() > 0, "fixture must have PV cells"
    drv = np.asarray(model.static["adex"]["triple_drive_pA"])
    np.testing.assert_array_equal(drv, np.where(typ == "PV", -200.0, 0.0))


def test_isomD50_drives_deep_som_plus_50_only():
    pytest.importorskip("jaxfne")
    model, rows = _model("e2_3unit_isomD50")
    typ = np.array([r["cell_type"] for r in rows])
    lay = np.array([r["layer"] for r in rows])
    want = np.where((typ == "SOM") & (lay == "D"), 50.0, 0.0)
    assert want.sum() > 0, "fixture must have deep SOM cells"
    np.testing.assert_array_equal(np.asarray(model.static["adex"]["triple_drive_pA"]), want)


def test_no_drive_modifier_keeps_baseline_edges():
    pytest.importorskip("jaxfne")
    name = "e2_3unit_sst15"
    model, _ = _model(name)
    st = model.static["adex"]
    assert st["triple_drive_pA"] is None
    assert cs.drive_entries(name, SIZES) is None
    assert len(st["pre"]) == 3762
    assert _edge_sha(st) == BASELINE_SST15_SHA256


def _pv_spikes(name: str) -> int:
    import jaxfne as jtfne

    model, rows = _model(name)
    sim = jtfne.Simulation(duration_ms=200.0, dt_ms=c0.DT_MS, poisson_drive=c0.drive_spec(0.07, 0))
    spikes = np.asarray(model.simulate(sim).spikes, dtype=bool)
    assert spikes.shape[0] == 2000, spikes.shape  # 0.2 s at dt 0.1 ms
    pv = np.array([r["cell_type"] == "PV" for r in rows])
    return int(spikes[:, pv].sum())


def test_pv_current_moves_pv_spikes_with_its_sign():
    # The silencing dose is T4's to find (log sweep); this checks that the current reaches the kernel.
    pytest.importorskip("jaxfne")
    base = _pv_spikes("e2_3unit")
    assert base > 0, "fixture must spike: baseline PV spikes = 0"
    lo, hi = _pv_spikes("e2_3unit_ipvam400"), _pv_spikes("e2_3unit_ipva400")
    assert lo < base < hi, (lo, base, hi)


@pytest.mark.parametrize("tok,pre,post,unit,x", [("weeS2", "E", "E", "S", 2.0), ("wseD0p5", "SOM", "E", "D", 0.5)])
def test_w_scales_one_rule_in_one_unit(tok, pre, post, unit, x):
    pytest.importorskip("jaxfne")
    base = "e2_3unit_sst15"  # sst15: SOM->E carries the charge rescale, which w must compose with
    a, rows = _model(base)
    b, _ = _model(f"{base}_{tok}")
    a, b = a.static["adex"], b.static["adex"]
    for key in ("pre", "post", "rule_id", "receptor"):
        np.testing.assert_array_equal(a[key], b[key])  # same seed: identical edges
    typ = np.array([r["cell_type"] for r in rows])
    lay = np.array([r["layer"] for r in rows])
    pr, po = np.asarray(a["pre"]), np.asarray(a["post"])
    hit = (typ[pr] == pre) & (typ[po] == post) & (lay[pr] == unit) & (lay[po] == unit)
    assert hit.sum() > 0 and (~hit).sum() > 0, "fixture must have target and other edges"
    wa, wb = np.asarray(a["weight_nS"]), np.asarray(b["weight_nS"])
    np.testing.assert_array_equal(wb[hit], x * wa[hit])
    np.testing.assert_array_equal(wb[~hit], wa[~hit])


def test_w_refusals():
    assert cs.variant("e2_3unit_weeS2_wpeM0p5")[1]["w"] == {"e_e:S": 2.0, "pv_e:M": 0.5}
    with pytest.raises(ValueError, match="declared twice"):
        cs.variant("e2_3unit_weeS2_weeS3")
    with pytest.raises(ValueError, match="want w"):
        cs.variant("e2_3unit_wxxS2")
    with pytest.raises(ValueError, match="want w"):
        cs.variant("e2_3unit_weeQ2")
    with pytest.raises(ValueError, match="not a number"):  # no minus sign in w: a weight factor is >= 0
        cs.variant("e2_3unit_weeSm1")
    with pytest.raises(ValueError, match="finite"):
        cs.variant("e2_3unit_weeSinf")


def test_drive_and_rhos_refusals():
    with pytest.raises(ValueError, match="declared twice"):
        cs.variant("e2_3unit_ipvam200_ipvam50")
    with pytest.raises(ValueError, match="unit must be one of"):
        cs.variant("e2_3unit_ipvX50")
    with pytest.raises(ValueError, match="not a number"):
        cs.variant("e2_3unit_isomD5x")
    with pytest.raises(ValueError, match="driven twice"):  # a covers S PV; ipvS50 would add a second entry
        cs.drive_entries("e2_3unit_ipvam200_ipvS50", SIZES)
    with pytest.raises(ValueError, match="declared twice"):
        cs.variant("e2_3unit_gi2_gi3")
    with pytest.raises(ValueError, match="must be finite"):
        cs.variant("e2_3unit_rhosnan")
    with pytest.raises(ValueError, match="3-unit stack"):
        cs.layout("unit100_rhos2")

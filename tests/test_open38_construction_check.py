"""Construction-gate level (owner ruling 2026-11-17, option B): the stage-2 failure (shape s1 e_pv_S, 421 511 edges)
fails at the frozen 99.9 % per-test level and passes at 5 % family-wise over 780 tests; real construction errors of
the size the gate must catch still fail."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "cc", Path(__file__).resolve().parents[1] / "scripts" / "open38_construction_check.py")
cc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cc)

P = 0.139996605  # local E->PV p in unit S (independent review 2026-11-17)
N_E, N_PV, N_SOM = 5785, 518, 567  # shape arm, unit S
COUNT = 421511  # o38s_s2_attempt1_construction.json, shape s1 e_pv_S


def test_family_level():
    assert cc.family_level(780) == pytest.approx(1 - 0.05 / 780, rel=1e-15)
    with pytest.raises(ValueError):
        cc.family_level(0)


def test_frozen_failure_reproduced_then_passes_under_ruling():
    lo, hi = cc.rule_interval([N_E * N_PV], P, 0.999)
    assert (lo, hi) == (417543.0, 421496.0), "must reproduce the frozen gate's interval"
    assert not lo <= COUNT <= hi
    lo, hi = cc.rule_interval([N_E * N_PV], P, cc.family_level(780))
    assert lo <= COUNT <= hi


def test_ruling_still_catches_construction_errors():
    lo, hi = cc.rule_interval([N_E * N_PV], P, cc.family_level(780))
    mu, sd = P * N_E * N_PV, math.sqrt(N_E * N_PV * P * (1 - P))
    assert not lo <= mu + 4.5 * sd <= hi and not lo <= mu - 4.5 * sd <= hi
    assert not lo <= round(P * N_E * N_SOM) <= hi, "PV/SOM swap must fail"
    assert not lo <= round(1.01 * mu) <= hi, "a 1 % error in p must fail"


def _fake(count: int) -> dict:
    n = count
    return {"sizes": {"S": {"E": N_E, "PV": N_PV, "SOM": N_SOM}}, "rho": {"S": N_SOM / N_PV},
            "edges": {"e_e_S": {k: np.zeros(3) for k in ("pre", "post", "weight_nS", "delay_ms", "receptor")},
                      "e_pv_S": {"pre": np.zeros(n, np.int32), "post": np.zeros(n, np.int32),
                                 "weight_nS": np.ones(n), "delay_ms": np.ones(n), "receptor": np.zeros(n)}},
            "e_par": {"C_pF": np.ones(3)}, "p": {"e_pv_S": P}, "depth": {"S": 0.0}, "gcols": [], "gtau": np.ones(1),
            "grev": np.ones(1), "drive": {"seed": 1}}


def test_compare_applies_the_level_it_is_given():
    ref, arm = _fake(COUNT), _fake(COUNT)
    assert not cc.compare(ref, arm, 0.999)["inh_rules"]["ok"]
    assert cc.compare(ref, arm, cc.family_level(780))["inh_rules"]["ok"]
    assert sum(cc.is_inh(r) for r in ref["edges"]) == 1

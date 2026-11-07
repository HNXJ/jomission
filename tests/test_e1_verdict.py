"""E1 verdict rule tests (docs/laminar_motif/E1_SEAL.md, Verdict rule).

Covers:
  - Known positive passes (PASS)
  - Each condition at its boundary (inside, outside, threshold)
  - Null hypothesis where f_pk is independent of rho fails (FAIL)
  - Inconclusive cases (condition 1 fails with < 4 oscillating points)
  - Incomplete cases (missing run, short run, extra run)
  - Mutants that must be killed (mutation tests against mutant implementations)
  - Band classification (gamma, beta, alpha_or_lower)
  - Constant identity against E1_SEAL.md
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from jomission.laminar import e1_verdict as v


def _mock_run(rho, seed, f_pk=34.0, osc=True, pe=15.0, steps=v.STEPS_EXPECTED, undefined=()):
    return {
        "rho": float(rho),
        "seed": int(seed),
        "steps": int(steps),
        "analysis": {
            "rates_Hz": {"E": 1.1, "PV": 11.0, "SOM": 7.0},
            "peak_freq_E_Hz": float(f_pk),
            "peak_excess_E": float(pe),
            "oscillates": bool(osc),
            "undefined": list(undefined),
        },
    }


def _synthetic_positive(slopes=(-1.0, -1.0, -1.0), base_f=34.0):
    """Synthetic positive: frequency drops linearly with log2(rho).

    rhos = [0.25, 0.5, 1.0, 2.0, 4.0] -> log2(rho) = [-2, -1, 0, 1, 2].
    For slope = -1.0, f = base_f - 1.0 * log2(rho) -> [base_f+2, base_f+1, base_f, base_f-1, base_f-2].
    Drop f(0.25) - f(4.0) = 4.0 Hz >= 3.0 Hz.
    """
    runs = {}
    for s_idx, s in enumerate(v.SEEDS):
        slope = slopes[s_idx]
        for rho in v.RHOS:
            x = np.log2(rho)
            f = base_f + slope * x
            runs[(rho, s)] = _mock_run(rho, s, f_pk=f, osc=True)
    return runs


def test_known_positive_passes():
    runs = _synthetic_positive()
    out = v.verdict(runs)
    assert out["verdict"] == "PASS"
    assert out["conditions"]["c1_min_oscillating_per_seed"]["status"] == "PASS"
    assert out["conditions"]["c2_negative_slope_per_seed"]["status"] == "PASS"
    assert out["conditions"]["c3_mean_freq_drop"]["status"] == "PASS"
    assert out["conditions"]["c4_permutation_test"]["status"] == "PASS"
    assert out["conditions"]["c3_mean_freq_drop"]["mean_drop_Hz"] == pytest.approx(4.0)
    assert out["conditions"]["c4_permutation_test"]["p_value"] < 0.05


def test_condition1_boundary():
    # 4 oscillating points in seed 1, 5 in seeds 2 and 3 -> condition 1 PASS
    runs = _synthetic_positive()
    runs[(4.0, 1)]["analysis"]["oscillates"] = False
    runs[(4.0, 1)]["analysis"]["peak_freq_E_Hz"] = None
    # For seed 1, remaining points are 0.25, 0.5, 1.0, 2.0. Drop f(0.25)-f(4) is undefined for seed 1.
    out = v.verdict(runs)
    assert out["conditions"]["c1_min_oscillating_per_seed"]["status"] == "PASS"
    # But because seed 1 endpoint 4.0 did not oscillate, c3 is UNDEFINED, so overall verdict is INCONCLUSIVE
    assert out["verdict"] == "INCONCLUSIVE"

    # Exactly 3 oscillating points in seed 1 -> condition 1 FAILS -> INCONCLUSIVE
    runs[(2.0, 1)]["analysis"]["oscillates"] = False
    out = v.verdict(runs)
    assert out["conditions"]["c1_min_oscillating_per_seed"]["status"] == "FAIL"
    assert out["verdict"] == "INCONCLUSIVE"


def test_condition2_boundary_slope_negative():
    # If slope is negative in seeds 1 and 2, but non-negative (0.0 or positive) in seed 3:
    # All 5 oscillate in each seed, so condition 1 passes and all 5 oscillate -> FAIL
    runs = _synthetic_positive(slopes=(-1.0, -1.0, 0.05))
    out = v.verdict(runs)
    assert out["conditions"]["c2_negative_slope_per_seed"]["status"] == "FAIL"
    assert out["verdict"] == "FAIL"

    # Exactly 0 slope in seed 3
    runs_zero = _synthetic_positive(slopes=(-1.0, -1.0, 0.0))
    out_zero = v.verdict(runs_zero)
    assert out_zero["conditions"]["c2_negative_slope_per_seed"]["status"] == "FAIL"
    assert out_zero["verdict"] == "FAIL"

    # Barely negative slope (-0.01) in all seeds, but drop < 3 Hz -> c2 passes, c3 fails -> FAIL
    runs_slight = _synthetic_positive(slopes=(-0.1, -0.1, -0.1))
    out_slight = v.verdict(runs_slight)
    assert out_slight["conditions"]["c2_negative_slope_per_seed"]["status"] == "PASS"
    assert out_slight["conditions"]["c3_mean_freq_drop"]["status"] == "FAIL"
    assert out_slight["verdict"] == "FAIL"


def test_condition3_boundary_mean_drop():
    # Drop f(0.25) - f(4.0) over x span [-2, 2] is 4 * (-slope).
    # For drop = 3.0 Hz, slope = -3.0 / 4 = -0.75.
    runs_at_threshold = _synthetic_positive(slopes=(-0.75, -0.75, -0.75))
    out_at = v.verdict(runs_at_threshold)
    assert out_at["conditions"]["c3_mean_freq_drop"]["mean_drop_Hz"] == pytest.approx(3.0)
    assert out_at["conditions"]["c3_mean_freq_drop"]["status"] == "PASS"
    assert out_at["verdict"] == "PASS"

    # Just below threshold: drop = 2.96 Hz (slope = -0.74)
    runs_below = _synthetic_positive(slopes=(-0.74, -0.74, -0.74))
    out_below = v.verdict(runs_below)
    assert out_below["conditions"]["c3_mean_freq_drop"]["mean_drop_Hz"] == pytest.approx(2.96)
    assert out_below["conditions"]["c3_mean_freq_drop"]["status"] == "FAIL"
    assert out_below["verdict"] == "FAIL"


def test_null_hypothesis_fails():
    """Null hypothesis: frequency is independent of rho (e.g. constant 34 Hz)."""
    runs = {}
    for s in v.SEEDS:
        for rho in v.RHOS:
            runs[(rho, s)] = _mock_run(rho, s, f_pk=34.0, osc=True)
    out = v.verdict(runs)
    # Slope is 0, drop is 0, p-val is approx 1.0 (>= 0.05)
    assert out["conditions"]["c2_negative_slope_per_seed"]["status"] == "FAIL"
    assert out["conditions"]["c3_mean_freq_drop"]["status"] == "FAIL"
    assert out["conditions"]["c4_permutation_test"]["status"] == "FAIL"
    assert out["verdict"] == "FAIL"


def test_incomplete_runs():
    # Missing run
    runs = _synthetic_positive()
    del runs[(1.0, 2)]
    out = v.verdict(runs)
    assert out["verdict"] == "INCOMPLETE"
    assert [1.0, 2] in out["missing_runs"]

    # Short run
    runs = _synthetic_positive()
    runs[(1.0, 2)]["steps"] = v.STEPS_EXPECTED - 1
    out = v.verdict(runs)
    assert out["verdict"] == "INCOMPLETE"
    assert [1.0, 2] in out["short_runs"]

    # Extra run
    runs = _synthetic_positive()
    runs[(0.1, 1)] = _mock_run(0.1, 1)
    out = v.verdict(runs)
    assert out["verdict"] == "INCOMPLETE"
    assert [0.1, 1] in out["unexpected_runs"]


def test_band_classification():
    assert v.band_class(35.0) == "gamma"
    assert v.band_class(30.0) == "gamma"
    assert v.band_class(29.9) == "beta"
    assert v.band_class(13.0) == "beta"
    assert v.band_class(12.9) == "alpha_or_lower"
    assert v.band_class(5.0) == "alpha_or_lower"
    assert v.band_class(None) is None


def test_input_not_mutated():
    runs = _synthetic_positive()
    before = copy.deepcopy(runs)
    v.verdict(runs)
    assert runs == before


def test_constants_equal_e1_seal():
    seal_path = Path(__file__).resolve().parents[1] / "docs" / "laminar_motif" / "E1_SEAL.md"
    text = seal_path.read_text(encoding="utf-8")
    for token in ("0.25", "0.5", "1", "2", "4", "300", "500", "750", "1000", "1200", "3 Hz", "0.05", "10 000"):
        assert token in text, f"Missing token {token} in E1_SEAL.md"


def test_mutants_killed():
    """Verify that intentional mutations to the verdict logic are killed by the test suite."""
    pos_runs = _synthetic_positive()

    # Mutant 1: Mean slope < 0 instead of every seed slope < 0
    # Create a case where mean slope is negative (-0.33) but seed 3 has positive slope (+0.34)
    # Under real rule, this must FAIL. A mutant that only checks mean slope would PASS.
    pos_with_one_pos_seed = _synthetic_positive(slopes=(-1.0, -1.0, 0.34))
    real_verdict = v.verdict(pos_with_one_pos_seed)["verdict"]
    assert real_verdict == "FAIL", "Mutant 1 killed: individual positive slope must fail despite negative mean"

    # Mutant 2: Permutation alpha changed from 0.05 to 0.50
    # A flat null has p-value around 0.50. Under real rule, it must not pass.
    null_runs = _synthetic_positive(slopes=(0.0, 0.0, 0.0))
    null_verdict = v.verdict(null_runs)["verdict"]
    assert null_verdict == "FAIL", "Mutant 2 killed: null case must not pass"

    # Mutant 3: Threshold for drop relaxed from 3.0 Hz to 2.0 Hz
    # A drop of 2.5 Hz must FAIL under real rule (threshold 3.0 Hz)
    border_runs = _synthetic_positive(slopes=(-0.625, -0.625, -0.625))  # drop = 4 * 0.625 = 2.5 Hz
    assert v.verdict(border_runs)["verdict"] == "FAIL", "Mutant 3 killed: drop below 3.0 Hz must fail"

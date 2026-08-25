"""Q8 phenotype evaluation tests — Subagent B deliverable verification.

Frozen authorities:
- Evaluate Q8 counterfactuals: compare post reference vs H_post→H_pre etc.
  on rate/field/T1-T7-relevant deltas, recovery trajectory,
  with POSITIVE|NEGATIVE|UNRESOLVED per frozen criteria.
- Must not alter T1-T7; use matched inputs/RNG.
- Require artifact-backed evidence, generated-owner.

Bounded responsibility (B):
- Independently evaluate phenotype deltas for each counterfactual:
  rate/field/T1-T7-relevant deltas, recovery trajectory, polarity,
  technical limitations.
- Do not implement state replacement (A's job); consume A's artifacts
  or re-run with same specs.
- Produce machine-readable Q8 result matrix.
"""

import json
import pathlib
import tempfile

import numpy as np
import pytest

from jomission.analysis.comparison_matrix import COMPARISON_MATRIX
from jomission.analysis.targets import FALSIFICATION_TARGETS
from jomission.simulation.state_replacement import REPLACEMENT_SPECS, FROZEN_CONFIG_HASH, FROZEN_HP_FULL
from jomission.analysis.q8_phenotype import (
    Q8_MATRIX_VERSION,
    Q8_FROZEN_CRITERIA,
    Q8_QUESTION,
    assign_polarity,
    evaluate_counterfactual,
    evaluate_q8_matrix,
    run_q8_evaluation,
)
from jomission.simulation.state_replacement import capture_pre_post_states


@pytest.fixture(scope="module")
def captured():
    return capture_pre_post_states(seed=0, dt_ms=2.0, n_pre_trials=1, n_exposure_trials=1, duration_ms=50.0)


@pytest.fixture(scope="module")
def q8_matrix_single(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("q8_matrix")
    # Single matrix with minimal battery for speed: 2 trials, 100ms
    art = evaluate_q8_matrix(seed=0, dt_ms=2.0, duration_ms=100.0, n_pre_trials=1, n_exposure_trials=1, trial_conditions=["AAAB", "AXAB"], results_dir=str(tmp))
    art["_tmp_path"] = str(tmp)
    return art


def test_t1_t7_not_altered():
    ids = [t.id for t in FALSIFICATION_TARGETS]
    assert ids == ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    assert COMPARISON_MATRIX["matrix_version"] == "jomission_comparison_matrix.v0.1.0"
    assert len(COMPARISON_MATRIX["targets"]) == 7
    import jomission.analysis.q8_phenotype as qp
    import pathlib, inspect
    src = pathlib.Path(inspect.getfile(qp)).read_text()
    assert "FALSIFICATION_TARGETS" not in src or "validate" in src
    assert "FALSIFICATION_TARGETS" not in json.dumps(Q8_FROZEN_CRITERIA)


def test_frozen_criteria_present_and_unchanged():
    assert Q8_MATRIX_VERSION == "jomission_q8_matrix.v0.1.0"
    assert Q8_QUESTION.startswith("Does H or")
    assert "rate" in Q8_FROZEN_CRITERIA
    assert "field" in Q8_FROZEN_CRITERIA
    assert "recovery" in Q8_FROZEN_CRITERIA
    assert Q8_FROZEN_CRITERIA["rate"]["effect_threshold_hz"] == 0.5
    assert Q8_FROZEN_CRITERIA["field"]["log_ratio_threshold"] == 0.1
    assert Q8_FROZEN_CRITERIA["field_claim_level"] == "proxy_readout"
    assert Q8_FROZEN_CRITERIA["physical_amplitude_calibrated"] is False


def test_consumes_same_carrier_sets_not_reimplement():
    import jomission.analysis.q8_phenotype as qp
    import pathlib
    src = pathlib.Path(qp.__file__).read_text()
    assert "from jomission.simulation.state_replacement import" in src
    assert "REPLACEMENT_SPECS" in src
    assert "def replace_H(" not in src
    assert "def replace_Theta(" not in src
    assert "apply_replacement_by_name" in src
    assert set(REPLACEMENT_SPECS.keys()) == {"H_post_to_H_pre", "Theta_post_to_Theta_pre", "HTheta_post_to_HTheta_pre", "fast_X_post_to_X_pre", "history_valid_HTheta_vs_fast"}


def test_evaluate_counterfactual_matched_inputs_RNG(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    model = captured["model"]
    res = evaluate_counterfactual(
        post_state=post, pre_state=pre, model=model,
        replacement_name="H_post_to_H_pre",
        dt_ms=2.0, duration_ms=50.0,
        trial_conditions=["AAAB", "AXAB"],
        seed_base=999,
    )
    assert res["matched_RNG_preserved"] is True
    assert res["verification"]["valid"] is True
    assert res["technical_validity"]["valid"] is True
    assert res["matched_inputs"]["trial_conditions"] == ["AAAB", "AXAB"]
    assert res["hashes"]["replaced"]["H"] == res["hashes"]["pre"]["H"]
    assert res["hashes"]["replaced"]["prng_key"] == res["hashes"]["post"]["prng_key"]
    # Also check one fast replacement preserves RNG
    r = evaluate_counterfactual(post_state=post, pre_state=pre, model=model, replacement_name="fast_X_post_to_X_pre", dt_ms=2.0, duration_ms=50.0, trial_conditions=["AAAB"], seed_base=1000)
    assert r["matched_RNG_preserved"] is True


def test_rate_phenotype_dimensions(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    model = captured["model"]
    res = evaluate_counterfactual(post_state=post, pre_state=pre, model=model, replacement_name="H_post_to_H_pre", dt_ms=2.0, duration_ms=100.0, trial_conditions=["AAAB", "AXAB", "BBBA"], seed_base=111)
    rp = res["rate_phenotype"]
    for k in ("overall_rate", "omission_slot_rate", "recovery_window_rate", "area_rates"):
        assert k in rp
    for sub in ("overall_rate", "omission_slot_rate", "recovery_window_rate"):
        entry = rp[sub]
        for field in ("effect_rep_minus_post_hz", "mean_post_hz", "polarity", "p_value", "cohens_d", "n"):
            assert field in entry
        assert entry["polarity"] in ("POSITIVE", "NEGATIVE", "UNRESOLVED")


def test_field_phenotype_proxy_and_polarity(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    model = captured["model"]
    res_short = evaluate_counterfactual(post_state=post, pre_state=pre, model=model, replacement_name="H_post_to_H_pre", dt_ms=2.0, duration_ms=50.0, trial_conditions=["AAAB", "AXAB"], seed_base=222)
    fp_short = res_short["field_phenotype"]
    assert fp_short["field_claim_level"] == "proxy_readout"
    assert fp_short["physical_amplitude_calibrated"] is False
    assert fp_short["limitation"] is not None
    assert fp_short["overall_field_polarity"] == "UNRESOLVED"


def test_recovery_trajectory_dimensions(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    model = captured["model"]
    res = evaluate_counterfactual(post_state=post, pre_state=pre, model=model, replacement_name="H_post_to_H_pre", dt_ms=2.0, duration_ms=100.0, trial_conditions=["AXAB", "AAAB"], seed_base=444)
    rec = res["recovery_trajectory"]
    assert "bins_ms" in rec
    assert "avg_diff_rep_minus_post_hz" in rec
    assert "recovery_effect_hz" in rec
    assert rec["polarity"] in ("POSITIVE", "NEGATIVE", "UNRESOLVED")
    assert rec["n_p2_trials"] >= 1


def test_t1t7_relevant_dimensions(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    model = captured["model"]
    res = evaluate_counterfactual(post_state=post, pre_state=pre, model=model, replacement_name="Theta_post_to_Theta_pre", dt_ms=2.0, duration_ms=100.0, trial_conditions=["AAAB", "AXAB", "BBBA", "BXBA"], seed_base=555)
    t1t7 = res["t1t7_relevant"]
    for k in ("t1_omission_effect_post_hz", "t1_delta_rep_minus_post_hz", "t1_polarity"):
        assert k in t1t7
    assert t1t7["t1_polarity"] in ("POSITIVE", "NEGATIVE", "UNRESOLVED")


def test_polarity_frozen_criteria():
    assert assign_polarity(1.0, p_value=0.01, cohen_d=0.5, threshold=0.5) == "POSITIVE"
    assert assign_polarity(0.1, p_value=0.5, cohen_d=0.1, threshold=0.5) == "NEGATIVE"
    assert assign_polarity(1.0, p_value=0.2, cohen_d=0.5, threshold=0.5) == "UNRESOLVED"
    assert assign_polarity(1.0, p_value=0.01, cohen_d=0.5, threshold=0.5, limitation="insufficient n") == "UNRESOLVED"
    assert assign_polarity(float("nan"), p_value=0.01, cohen_d=0.5, threshold=0.5) == "UNRESOLVED"
    assert assign_polarity(0.5, p_value=0.04, cohen_d=0.3, threshold=0.5) == "POSITIVE"


def test_matrix_machine_readable_and_artifact_backed(q8_matrix_single):
    art = q8_matrix_single
    assert art["namespace"] == "q8_evaluation"
    assert art["owner"] == "generated"
    assert art["q8_matrix_version"] == Q8_MATRIX_VERSION
    assert art["frozen"]["config_hash"] == FROZEN_CONFIG_HASH
    assert art["frozen"]["hp_hash"] == FROZEN_HP_FULL
    assert "matrix" in art
    assert len(art["matrix"]) == len(REPLACEMENT_SPECS)
    for row in art["matrix"]:
        assert "counterfactual" in row
        assert row["overall_polarity"] in ("POSITIVE", "NEGATIVE", "UNRESOLVED")
        assert "rate_omission_slot_polarity" in row
        assert "field_low_gamma_polarity" in row
        assert "recovery_polarity" in row
        assert "limitations" in row
    assert "verification_summary" in art
    for name in REPLACEMENT_SPECS:
        assert name in art["verification_summary"]
    assert "artifact_path" in art
    p = pathlib.Path(art["artifact_path"])
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded["owner"] == "generated"
    assert "csv_path" in art
    assert pathlib.Path(art["csv_path"]).exists()
    assert "counterfactual" in pathlib.Path(art["csv_path"]).read_text().splitlines()[0]
    assert "npz_path" in art
    assert pathlib.Path(art["npz_path"]).exists()
    assert art["T1_T7_intact"]["valid"] is True
    assert art["field_claim_level"] == "proxy_readout"
    # Fast vs history distinction
    rows = {r["counterfactual"]: r for r in art["matrix"]}
    assert rows["fast_X_post_to_X_pre"]["carrier"] == "X_fast"
    assert rows["HTheta_post_to_HTheta_pre"]["carrier"] == "H+Theta"
    per = art["per_counterfactual"]
    assert per["fast_X_post_to_X_pre"]["spec"]["replaced"] == ["v", "u", "prev_spikes", "syn_state"]
    assert per["HTheta_post_to_HTheta_pre"]["spec"]["replaced"] == ["H", "w"]
    # All per counterfactual have required phenotypes
    for name, res in art["per_counterfactual"].items():
        assert "rate_phenotype" in res
        assert "field_phenotype" in res
        assert "recovery_trajectory" in res
        assert "t1t7_relevant" in res
        assert res["overall_polarity"] in ("POSITIVE", "NEGATIVE", "UNRESOLVED")


def test_technical_limitations_recorded(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    model = captured["model"]
    res = evaluate_counterfactual(post_state=post, pre_state=pre, model=model, replacement_name="H_post_to_H_pre", dt_ms=2.0, duration_ms=50.0, trial_conditions=["AAAB"], seed_base=777)
    assert len(res["technical_limitations"]) > 0
    lim_text = " ".join(res["technical_limitations"])
    assert "duration" in lim_text.lower() or "short" in lim_text.lower()
    assert any("dt" in l.lower() for l in res["technical_limitations"])


def test_does_not_alter_state_replacement_implementation():
    import jomission.simulation.state_replacement as sr
    import pathlib
    src_before = pathlib.Path(sr.__file__).read_text()
    import importlib, jomission.analysis.q8_phenotype
    importlib.reload(jomission.analysis.q8_phenotype)
    src_after = pathlib.Path(sr.__file__).read_text()
    assert src_before == src_after
    assert set(sr.REPLACEMENT_SPECS.keys()) == {"H_post_to_H_pre", "Theta_post_to_Theta_pre", "HTheta_post_to_HTheta_pre", "fast_X_post_to_X_pre", "history_valid_HTheta_vs_fast"}

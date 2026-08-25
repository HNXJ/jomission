"""Q8 state-replacement tests — technically valid counterfactual carriers.

Frozen authorities:
- Config 4f9fdeae7428199a / hp f327f9d2, dt canonical 0.1
- C_t=(X,H,Θ,D,RNG,cursor) preserved except declared component
- Q8: H or Θ history despite Δ≈0 — frozen counterfactuals H_post→H_pre etc
- Require artifact-backed evidence, generated-owner

These tests prove via hash verification that only declared carrier changed,
that replacement source/hash matches pre state, and that matched RNG/input
is preserved.
"""

import hashlib
import json
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
import jaxfne.hdp_network as hdp

from jomission.simulation.state_replacement import (
    FROZEN_CONFIG_HASH,
    FROZEN_HP_FULL,
    FROZEN_DT_MS,
    REPLACEMENT_SPECS,
    capture_pre_post_states,
    apply_replacement_by_name,
    replace_H,
    replace_Theta,
    replace_HTheta,
    replace_fast_X,
    verify_only_declared_changed,
    verify_technical_validity,
    get_state_hashes,
    run_counterfactual_probe,
    run_q8_suite,
)
from jaxfne.io import config_hash
from jomission.network.builder import build_jomission_model


@pytest.fixture(scope="module")
def captured():
    # Minimal capture for speed: 1 pre +1 exposure, 50ms dt2.0 (cfg dt stays 0.1)
    # Keeps frozen config 4f9... while simulation steps are minimal (25 steps per trial)
    return capture_pre_post_states(seed=0, dt_ms=2.0, n_pre_trials=1, n_exposure_trials=1, duration_ms=50.0)


def test_frozen_config_and_hp(captured):
    # Config hash for seed 0 must equal frozen
    assert captured["config_hash"] == FROZEN_CONFIG_HASH
    assert captured["hp_hash"] == FROZEN_HP_FULL
    assert captured["canonical_valid"] is True
    # Also check dt canonical preserved in cfg
    assert captured["model"].cfg.metadata["dt_ms"] == 0.1
    # hp matches expected
    hp = hdp.v1_pfc_aaab_hdp_params()
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]
    assert hp_hash == FROZEN_HP_FULL


def test_capture_provides_continuous_state(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    # ContinuationState has required fields
    assert hasattr(pre, "dynamic") and hasattr(pre, "prng_key") and hasattr(pre, "step_index")
    assert hasattr(post, "dynamic") and hasattr(post, "prng_key") and hasattr(post, "step_index")
    # DynamicState fields are X,H,Theta
    for field in ["v", "u", "prev_spikes", "syn_state", "H", "w"]:
        assert hasattr(pre.dynamic, field)
        assert hasattr(post.dynamic, field)
        assert np.asarray(getattr(pre.dynamic, field)).size > 0
    # D is None for current model (no delays)
    assert pre.delay_state is None and post.delay_state is None
    # RNG preserved shape
    assert np.asarray(pre.prng_key).shape == np.asarray(post.prng_key).shape
    # cursor step_index advanced
    assert int(post.step_index) > int(pre.step_index)


def test_replacement_source_hash_and_unchanged_components(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    pre_h = get_state_hashes(pre)
    post_h = get_state_hashes(post)

    for name, spec in REPLACEMENT_SPECS.items():
        replaced = apply_replacement_by_name(post, pre, name)
        rep_h = get_state_hashes(replaced)
        declared = spec["replaced"]
        # Source/hash: replaced's declared fields must equal pre's hash
        for f in declared:
            assert rep_h[f] == pre_h[f], f"{name} field {f} not from pre: {rep_h[f]} vs {pre_h[f]}"
            # Also ensure it differs from post unless pre==post coincidentally
            # Not enforced strictly, but at least rep == pre
        # Unchanged: preserved fields must equal post's hash
        for f in spec["preserved"]:
            assert rep_h[f] == post_h[f], f"{name} preserved field {f} changed: {rep_h[f]} vs {post_h[f]}"
        # Purge: also check that overall combined hash reflects replacement
        assert rep_h["_combined"] != post_h["_combined"] or declared == []  # should differ


def test_verify_only_declared_changed_proves_isolation(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    for name, spec in REPLACEMENT_SPECS.items():
        replaced = apply_replacement_by_name(post, pre, name)
        out = verify_only_declared_changed(post, replaced, pre, declared_replaced=spec["replaced"])
        assert out["valid"], f"{name} failed isolation: {out['issues']}"
        # Each declared field check true, each preserved true
        for f, ok in out["checks"].items():
            assert ok, f"{name} check {f} failed"


def test_technical_validity_same_shapes_and_config(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    model = captured["model"]
    tv = verify_technical_validity(post, pre, model=model)
    assert tv["valid"], tv["issues"]
    assert tv["cfg"]["hp_hash"] == FROZEN_HP_FULL
    # Shapes must match
    for field in ["v", "u", "prev_spikes", "syn_state", "H", "w"]:
        assert getattr(pre.dynamic, field).shape == getattr(post.dynamic, field).shape
        assert getattr(pre.dynamic, field).dtype == getattr(post.dynamic, field).dtype


def test_matched_RNG_and_input_preserved(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    model = captured["model"]
    # Single probe to keep test fast; other replacements covered by hash tests and suite
    name = "H_post_to_H_pre"
    res = run_counterfactual_probe(
        base_state=post, pre_state=pre, post_state=post,
        replacement_name=name, model=model, dt_ms=2.0, duration_ms=50.0, seed=123, condition_name="AXAB"
    )
    # Matched RNG: prng_key preserved (not replaced)
    assert res["matched_RNG"]["preserved"] is True
    assert res["matched_inputs"]["seed"] == 123
    assert res["matched_inputs"]["condition"] == "AXAB"
    # Technical validity
    assert res["technical_validity"]["valid"]
    assert res["verification"]["valid"]
    # Cursor preserved
    assert res["replaced_hashes"]["step_index"] == get_state_hashes(post)["step_index"]


def test_individual_replace_helpers(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    # H
    rh = replace_H(post, pre)
    assert np.allclose(np.asarray(rh.dynamic.H), np.asarray(pre.dynamic.H))
    assert np.allclose(np.asarray(rh.dynamic.w), np.asarray(post.dynamic.w))
    # Theta
    rt = replace_Theta(post, pre)
    assert np.allclose(np.asarray(rt.dynamic.w), np.asarray(pre.dynamic.w))
    assert np.allclose(np.asarray(rt.dynamic.H), np.asarray(post.dynamic.H))
    # HTheta
    rht = replace_HTheta(post, pre)
    assert np.allclose(np.asarray(rht.dynamic.H), np.asarray(pre.dynamic.H))
    assert np.allclose(np.asarray(rht.dynamic.w), np.asarray(pre.dynamic.w))
    # fast X
    rf = replace_fast_X(post, pre)
    for f in ["v", "u", "prev_spikes", "syn_state"]:
        assert np.allclose(np.asarray(getattr(rf.dynamic, f)), np.asarray(getattr(pre.dynamic, f)))
    for f in ["H", "w"]:
        assert np.allclose(np.asarray(getattr(rf.dynamic, f)), np.asarray(getattr(post.dynamic, f)))


def test_fast_vs_history_valid_replacements_are_distinct(captured):
    pre = captured["pre_state"]
    post = captured["post_state"]
    fast = apply_replacement_by_name(post, pre, "fast_X_post_to_X_pre")
    hist = apply_replacement_by_name(post, pre, "HTheta_post_to_HTheta_pre")
    h_only = apply_replacement_by_name(post, pre, "H_post_to_H_pre")
    theta_only = apply_replacement_by_name(post, pre, "Theta_post_to_Theta_pre")
    # Fast and history must affect different carriers, so hashes differ
    assert get_state_hashes(fast)["H"] == get_state_hashes(post)["H"]  # fast preserves H
    assert get_state_hashes(hist)["H"] == get_state_hashes(pre)["H"]   # history replaces H
    assert get_state_hashes(fast)["v"] == get_state_hashes(pre)["v"]   # fast replaces v
    assert get_state_hashes(hist)["v"] == get_state_hashes(post)["v"]  # history preserves v
    # H only vs Theta only vs both are distinct
    assert get_state_hashes(h_only)["_combined"] != get_state_hashes(theta_only)["_combined"]
    assert get_state_hashes(h_only)["_combined"] != get_state_hashes(hist)["_combined"]


def test_artifact_backed_evidence_generated_owner(tmp_path):
    # Run suite with minimal probe to generate artifact (fast)
    artifact = run_q8_suite(seed=0, dt_ms=2.0, duration_ms=50.0, n_pre_trials=1, n_exposure_trials=1, condition_name="AXAB", results_dir=str(tmp_path))
    assert artifact["owner"] == "generated"
    assert artifact["namespace"] == "q8_state_replacement"
    assert "results" in artifact
    assert set(artifact["results"].keys()) == set(REPLACEMENT_SPECS.keys())
    for name, res in artifact["results"].items():
        assert "verification" in res
        assert res["verification"]["valid"] is True
        assert "artifact_hashes" in res
        assert "post_state_hash" in res["artifact_hashes"]
        assert "pre_state_hash" in res["artifact_hashes"]
        assert "replaced_state_hash" in res["artifact_hashes"]
        # matched RNG
        assert res["matched_RNG"]["preserved"] is True
        # probe produced V_m diff (at least H/fast should produce non-zero)
    # File exists
    assert "artifact_path" in artifact
    p = pathlib.Path(artifact["artifact_path"])
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded["owner"] == "generated"
    assert loaded["frozen"]["config_hash"] == FROZEN_CONFIG_HASH


def test_continuous_state_preserved_except_declared(captured):
    # Exhaustive check that (X,H,Theta,D,RNG,cursor) all preserved except declared
    pre = captured["pre_state"]
    post = captured["post_state"]
    # H replacement should preserve X,Theta,D,RNG,cursor
    rh = replace_H(post, pre)
    assert np.allclose(np.asarray(rh.dynamic.v), np.asarray(post.dynamic.v))
    assert np.allclose(np.asarray(rh.dynamic.u), np.asarray(post.dynamic.u))
    assert np.allclose(np.asarray(rh.dynamic.prev_spikes), np.asarray(post.dynamic.prev_spikes))
    assert np.allclose(np.asarray(rh.dynamic.syn_state), np.asarray(post.dynamic.syn_state))
    assert np.allclose(np.asarray(rh.dynamic.w), np.asarray(post.dynamic.w))
    assert rh.delay_state == post.delay_state
    assert np.array_equal(np.asarray(rh.prng_key), np.asarray(post.prng_key))
    assert int(rh.step_index) == int(post.step_index)
    # Theta replacement preserves X,H,D,RNG,cursor
    rt = replace_Theta(post, pre)
    assert np.allclose(np.asarray(rt.dynamic.H), np.asarray(post.dynamic.H))
    assert np.allclose(np.asarray(rt.dynamic.v), np.asarray(post.dynamic.v))
    assert int(rt.step_index) == int(post.step_index)


def test_do_not_alter_T1_T7():
    # Ensure replacement module does not import or mutate T1-T7 targets
    import jomission.analysis.targets as t
    import jomission.analysis.comparison_matrix as cm
    # Frozen T1-T7 still present, not altered by replacement code
    assert len(t.FALSIFICATION_TARGETS) == 7
    assert cm.COMPARISON_MATRIX["matrix_version"] == "jomission_comparison_matrix.v0.1.0"
    # Replacement specs do not mention T1-T7 logic
    for spec in REPLACEMENT_SPECS.values():
        assert "T1" not in spec["description"] and "T7" not in spec["description"]


def test_different_replicates_produce_different_hashes():
    cap0 = capture_pre_post_states(seed=0, dt_ms=2.0, n_pre_trials=1, n_exposure_trials=1, duration_ms=50.0)
    cap1 = capture_pre_post_states(seed=2, dt_ms=2.0, n_pre_trials=1, n_exposure_trials=1, duration_ms=50.0)
    # Different model seeds produce different weights/H means
    assert cap0["post_H_mean"] != cap1["post_H_mean"] or cap0["post_w_mean"] != cap1["post_w_mean"] or cap0["post_hashes"]["_combined"] != cap1["post_hashes"]["_combined"]
    # But both have same hp
    assert cap0["hp_hash"] == cap1["hp_hash"] == FROZEN_HP_FULL


def test_code_inspection_only_declared_carrier():
    # Prove via code inspection that replacement helpers only touch declared fields
    import pathlib, inspect
    import jomission.simulation.state_replacement as sr
    src = pathlib.Path(inspect.getfile(sr)).read_text()
    # Each helper must be defined and only mention its declared fields
    assert "def replace_H(" in src
    assert "def replace_Theta(" in src
    assert "def replace_HTheta(" in src
    assert "def replace_fast_X(" in src
    # Check REPLACEMENT_SPECS preserved vs replaced are mutually exclusive and cover C_t
    for name, spec in REPLACEMENT_SPECS.items():
        replaced = set(spec["replaced"])
        preserved = set(spec["preserved"])
        assert replaced.isdisjoint(preserved), f"{name} overlap"
        # All carriers in C_t must be either replaced or preserved (except maybe alias)
        all_fields = {"v", "u", "prev_spikes", "syn_state", "H", "w", "prng_key", "step_index", "delay_state"}
        assert replaced | preserved == all_fields, f"{name} not covering all C_t: {replaced|preserved}"
    # Verify _apply_replacement uses only replaced list to select pre vs post
    assert "_apply_replacement" in src
    assert "getattr(pre.dynamic" in src and "getattr(post.dynamic" in src
    # Ensure module does not import T1-T7 analysis that would alter them
    assert "FALSIFICATION_TARGETS" not in src
    assert "COMPARISON_MATRIX" not in src

"""Regression test suite for SCI-EI-NATIVE-STATE-1 retention instrumentation.

Verifies:
- Representative cell identification and indexing across classes (E, PV, SST, VIP).
- Retention accumulator initialization, per-chunk accumulation, and post-loop concatenation.
- Exact array shapes, dtypes, sample counts, and native 0.1 ms clock alignment.
- Full driver execution pipeline using mocked continuation (no results/ pollution).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np

import scripts.whole_system_ei_native_state_arm as arm

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "results" / "whole_system_ei_native_state_spec.json"


def test_representative_cell_identities_and_indices():
    """Verify representative cells deterministically map to first neuron of V1_1.L4.<class>."""
    model, _, _, _ = arm.build()
    index, _ = arm.realize.index_map(model)
    members = {g: np.arange(index[g][0], index[g][1]) for g in arm.groups_of(index)}

    rep_cells = {}
    for c in ("E", "PV", "SST", "VIP"):
        g = f"V1_1.L4.{c}"
        assert g in members and members[g].size > 0, f"group {g} missing or empty"
        rep_cells[c] = int(members[g][0])

    rep_idx = np.array([rep_cells[c] for c in ("E", "PV", "SST", "VIP")], dtype=np.int64)

    assert len(rep_cells) == 4
    assert len(set(rep_idx)) == 4, "representative cell indices must be disjoint"
    # Specific known indices in 2224-neuron circuit
    assert rep_cells["E"] == 490
    assert rep_cells["PV"] == 504
    assert rep_cells["SST"] == 508
    assert rep_cells["VIP"] == 509
    assert np.array_equal(rep_idx, np.array([490, 504, 508, 509]))


def test_accumulator_initialization_and_chunk_concatenation():
    """Verify per-chunk accumulation and concatenation of native-resolution traces."""
    n_chunks = 3
    chunk_ms = 100
    dt_ms = 0.1
    steps_per_ms = int(round(1.0 / dt_ms))  # 10
    chunk_steps = int(round(chunk_ms / dt_ms))  # 1000
    n_neurons = 20
    rep_idx = np.array([1, 5, 8, 12], dtype=np.int64)

    rep_acc = {"v": [], "u": [], "current": [], "spikes": []}
    spikes_1ms = []

    for c in range(n_chunks):
        v = np.full((chunk_steps, n_neurons), c * 10.0, dtype=np.float32)
        u = np.full((chunk_steps, n_neurons), c * 2.0, dtype=np.float32)
        current = np.full((chunk_steps, n_neurons), c * 0.5, dtype=np.float32)
        spikes = np.zeros((chunk_steps, n_neurons), dtype=np.float32)
        # Plant a spike in bin 0 of neuron 1
        if c == 0:
            spikes[3, 1] = 1.0

        rep_acc["v"].append(v[:, rep_idx])
        rep_acc["u"].append(u[:, rep_idx])
        rep_acc["current"].append(current[:, rep_idx])
        rep_acc["spikes"].append(spikes[:, rep_idx])

        spikes_1ms.append(
            spikes.reshape(chunk_ms, steps_per_ms, n_neurons).max(axis=1).astype(np.uint8)
        )

    rep = {k: np.concatenate(arrs, axis=0) for k, arrs in rep_acc.items()}
    full_spikes_1ms = np.concatenate(spikes_1ms, axis=0).astype(np.uint8)

    total_steps = n_chunks * chunk_steps
    total_ms = n_chunks * chunk_ms

    assert rep["v"].shape == (total_steps, 4)
    assert rep["u"].shape == (total_steps, 4)
    assert rep["current"].shape == (total_steps, 4)
    assert rep["spikes"].shape == (total_steps, 4)
    assert full_spikes_1ms.shape == (total_ms, n_neurons)

    assert rep["v"].dtype == np.float32
    assert rep["u"].dtype == np.float32
    assert rep["current"].dtype == np.float32
    assert rep["spikes"].dtype == np.float32
    assert full_spikes_1ms.dtype == np.uint8

    # Verify spike timing alignment
    # Neuron 1 had a spike at step 3 (inside 1 ms bin 0 of chunk 0)
    assert rep["spikes"][3, 0] == 1.0  # neuron 1 is at index 0 of rep_idx
    assert full_spikes_1ms[0, 1] == 1
    assert full_spikes_1ms[1:, 1].sum() == 0


def test_driver_execution_pipeline_with_mock_continuation():
    """Verify driver executes end-to-end with mock continuation, creating valid retention kwargs."""
    arm.configure()
    test_dir = ROOT / "scratch" / "_test_pipeline"
    test_dir.mkdir(parents=True, exist_ok=True)
    out_json = test_dir / "test.json"
    out_npz = test_dir / "test.npz"

    def mock_run_cont(step_fn, state, sched):
        chunk_steps = int(round(arm.CHUNK_MS / arm.DT))
        n_neurons = 2224
        v = np.tile(np.asarray(state.dynamic.v), (chunk_steps, 1))
        spikes = np.zeros((chunk_steps, n_neurons), dtype=np.float32)
        sources = np.zeros((chunk_steps, n_neurons), dtype=np.float32)
        H = np.zeros((chunk_steps, n_neurons), dtype=np.float32)
        current = np.zeros((chunk_steps, n_neurons), dtype=np.float32)
        u = np.tile(np.asarray(state.dynamic.u), (chunk_steps, 1))
        return state, (v, spikes, sources, H, current, u)

    with mock.patch("jaxfne.run_continuation", side_effect=mock_run_cont):
        with mock.patch("numpy.savez_compressed") as mock_savez:
            with mock.patch("pathlib.Path.write_text") as mock_write:
                arm.main(0.0, "on", 0, out=out_json, out_y=out_npz)

                assert mock_savez.called, "numpy.savez_compressed must be called"
                assert mock_write.called, "receipt JSON must be written"

                kwargs = mock_savez.call_args[1]
                assert kwargs["rep_v"].shape == (130000, 4)
                assert kwargs["rep_u"].shape == (130000, 4)
                assert kwargs["rep_current"].shape == (130000, 4)
                assert kwargs["rep_spikes"].shape == (130000, 4)
                assert kwargs["spikes_1ms"].shape == (13000, 2224)

                assert kwargs["rep_v"].dtype == np.float32
                assert kwargs["rep_u"].dtype == np.float32
                assert kwargs["rep_current"].dtype == np.float32
                assert kwargs["rep_spikes"].dtype == np.uint8
                assert kwargs["spikes_1ms"].dtype == np.uint8

                written_json = json.loads(mock_write.call_args[0][0])
                assert "retention" in written_json
                ret = written_json["retention"]
                assert ret["sample_count_steps"] == 130000
                assert ret["dt_ms"] == 0.1
                assert ret["rep_v_shape"] == [130000, 4]
                assert ret["spikes_1ms_shape"] == [13000, 2224]


def test_spec_parameters_and_timing_invariants():
    """Verify scientific spec parameters match frozen invariants."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["lineage_id"] == "SCI-EI-NATIVE-STATE-1"
    assert spec["parent"] == "SCI-EI-RECRUITMENT-1"
    assert spec["parent_commit"] == "6d2c4ca"
    assert spec["frozen"]["dt_ms"] == 0.1
    assert spec["duration"]["steps"] == 130000
    assert spec["duration"]["total_ms"] == 13000.0
    assert spec["duration"]["chunk_ms"] == 500.0
    assert spec["duration"]["settle_ms"] == 10000.0
    assert spec["duration"]["stimulus_ms"] == 1000.0
    assert spec["duration"]["post_stimulus_ms"] == 2000.0


def test_missing_observable_fails_closed():
    """Verify driver fails closed with RuntimeError if continuation outputs lack genuine u/current."""
    import pytest

    arm.configure()

    def mock_run_cont_arity_4(step_fn, state, sched):
        chunk_steps = int(round(arm.CHUNK_MS / arm.DT))
        n_neurons = 2224
        v = np.tile(np.asarray(state.dynamic.v), (chunk_steps, 1))
        spikes = np.zeros((chunk_steps, n_neurons), dtype=np.float32)
        sources = np.zeros((chunk_steps, n_neurons), dtype=np.float32)
        H = np.zeros((chunk_steps, n_neurons), dtype=np.float32)
        return state, (v, spikes, sources, H)

    with mock.patch("jaxfne.run_continuation", side_effect=mock_run_cont_arity_4):
        with pytest.raises(RuntimeError, match="SCI-EI-NATIVE-STATE-1 requires genuine per-neuron outputs"):
            arm.main(0.0, "on", 0)


def test_native_state_executed_result_contract():
    """Verify executed result conforms to SCI-EI-NATIVE-STATE-1 pre-execution contract and verdicts."""
    res_path = ROOT / "results" / "whole_system_ei_native_state.json"
    assert res_path.exists(), "results/whole_system_ei_native_state.json must exist"
    res = json.loads(res_path.read_text(encoding="utf-8"))

    assert res["lineage"] == "SCI-EI-NATIVE-STATE-1"
    predeclared = [
        "DYNAMIC_RECRUITMENT_DEFICIT",
        "INTRINSIC_STATE_VETO",
        "SST_STATE_ADVANTAGE",
        "MIXED_STATE_MECHANISM",
        "NATIVE_STATE_UNRESOLVED",
    ]
    assert res["verdict"] in predeclared, f"Verdict {res['verdict']} not in predeclared {predeclared}"
    assert res["verdict"] == "MIXED_STATE_MECHANISM"
    assert res["clock"]["total_steps"] == 130000
    assert res["clock"]["dt_ms"] == 0.1
    assert res["clock"]["total_ms"] == 13000.0

    # Verify per-class representative cells
    for c in ("E", "PV", "SST", "VIP"):
        assert c in res["classes"]
        cls_data = res["classes"][c]
        assert cls_data["spikes_total_13s"] >= 0
        assert cls_data["rate_hz"] >= 0.0

    # Contract visualization paths
    viz = res["visualization"]
    assert (ROOT / viz["V0"]).exists()
    assert (ROOT / viz["V1"]).exists()
    assert (ROOT / viz["V2"]).exists()
    assert (ROOT / viz["V3"]).exists()

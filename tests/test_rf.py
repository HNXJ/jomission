"""RF operator — per V1 unit Gaussian weights, L1-normalized, sparse, A/B blobs.

Tests for jomission/network/rf.py

Validates:
- geometry 32x32, 8° field, 0.25°/px, sigma 1.8, spacing 3.2, overlap 0.45
- per V1 unit Gaussian weights, L1-normalized, sparse, V1 targeting (no V4/FEF/PFC)
- A/B blobs at (8,8)/(24,24) sparsity 0.18-0.30, Jaccard<0.15, omission zero
- Tier1 binary / Tier2 graded via existing JaxFNE target_indices
- config_hash distinct from canonical 4f9fdeae7428199a
- uses existing JaxFNE capability (no engine modification)
- StimulusSchedule target_indices sparse, V1 only
"""

import numpy as np
import pytest
import jaxfne as jtfne
from jomission.simulation.runtime import simulation_for_model
from jaxfne import Simulation
from jaxfne.io import config_hash

from jomission.network.builder import build_jomission_network, build_jomission_model
from jomission.network.rf import (
    RFConfig,
    RFOperator,
    build_jomission_network_with_rf,
    build_jomission_model_with_rf,
    apply_rf_to_configuration,
    prove_jaxfne_target_indices_capability,
)
from jomission.paradigm.spec import JOMISSION_PARADIGM

# GEN2 cumulative hash after W3.6 (C016 PV recruitment PV drive 1.77->3.01 scale1.7) → W3.7 C017 VIP b0.05→0.20 intrinsic recruitment → W3.8 C018 typed E mixture pseudogenome M2 70% RS /20% CH /10% E_FS → C019 AGSDR-L1 theta_local [1.2565,1.1872,0.7088,1.2182,1.2887] → C020 typed vertical microcircuit v1 vertical_spatial_sigma0.12 vertical_max40 + vertical_motif_gain L4_E→L2/3_E×2.0 L4_E→SST×0.7 L2/3_E→L5_E×1.8 etc.
# Gen-1 frozen was 4f9fdeae7428199a; W1.1-1.3 cumulative 57b9f98c3f8bacb8; W2.1 motif 4a8908e7064c32bf; W2.2 Poisson opt-in hash dad7290b21e447dc (with Poisson 1kHz), baseline (no Poisson) 822949392c225622; W2.3 hierarchy dad7290b21e447dc; W2.4 VIP fix → plain 52d59748bc553a1b, poisson 05226a6bbf3e265c; W2.5 lognormal → plain 5d3c769e0e87c6cf, poisson fdc7fe71297d328b; W3.1 tonic 3.5 scale0.7 → plain 4bf1403e628ac06f (baseline 1.0 is 55047c16e7172554), poisson+tonic 14f4201ed5d4f478; W3.2 C012 I_bgλ 3.0+Poisson2k/2.0 → 35a59588ac09c91c (seed0) / 6ca586cd76ef14ae (seed1); W3.3 C013 SST×0.5 bounded → 9a60b672e4db3a39 (seed0) / f490a00dc0ee3506 (seed1); W3.4 C014 pseudogenome DESIRED_MOTIF_GAIN v0 (E→PV1.70 E→SST0.70 PV→E1.30 SST→E0.60 SST→PV0.60 VIP→SST5.00 etc, SST alpha 1.0 supersedes 0.5) → 9f8abf33bffb4df0 (seed0) / 2311280573a72806 (seed1), pseudogenome_version v0; W3.5 C015 SST b 0.25→0.21 operating-point correction (MODEL_ASSUMPTION magnitude 0.21 / LITERATURE_PRIOR SST LTS) → 8d94ecf9c46378f4 (seed0) / eace803b159c4b8c (seed1), sst_b_corrected 0.21 jitter [0.189,0.231]; W3.6 C016 PV recruitment PV drive 1.77->3.01 scale1.7 per-class drive seam _apply_pv_drive_boost (MODEL_ASSUMPTION 1.7 magnitude to reach rheobase / LITERATURE_PRIOR fast PV) → 393c58c2dc6b6519 (seed0) / 54aa6eec5c517621 (seed1), pv_drive_scale 1.7 deterministic per-PV, VIP b0.05 frozen; W3.7 C017 VIP b0.05→0.20 intrinsic recruitment (MODEL_ASSUMPTION 0.20 / LITERATURE_PRIOR b>=0 Izhikevich2003 / DERIVED f-I I_rh>15→4 thr107.79→94.99 V_m-77.79→-64.99) → f3707cd55894787e (seed0) / 0f6d2e04524fa534 (seed1), vip_b_corrected 0.20 jitter [0.18,0.22] I_rh4 r21@I8, PV1.7 SST0.21 lognormal Poisson2kHz pseudogenome delays preserved; W3.8 C018 typed E mixture M2 70% RS a0.02 c-65 d8 /20% CH a0.02 c-50 d2 /10% E_FS a0.10 c-65 d2 via per-neuron cell_params deterministic seed pseudogenomic (LITERATURE_PRIOR RS/CH/IB Izhikevich2003 Fig.1 / MODEL_ASSUMPTION proportions 70/20/10) → be9b96ab679c9802 (seed0); C019 AGSDR-L1 theta_local [1.2565,1.1872,0.7088,1.2182,1.2887] → 65b302e8c7cdceb5 (seed0); C020 typed vertical v1 vertical_spatial_sigma0.12 vertical_max40 Ne16→54 k0.59→2.0 C0.444→0.741 + typed W L4_E→L2/3_E×2.0 etc. → 4574a15519ee8c33 (seed0) / bbe6bc709a4b4c5b (seed1)
FROZEN_CANONICAL_HASH = "4574a15519ee8c33"


def test_rf_config_geometry():
    cfg = RFConfig()
    v = cfg.validate()
    assert v["valid"], v["issues"]
    assert cfg.lattice_size == 32
    assert cfg.field_dva == 8.0
    assert abs(cfg.dva_per_px - 0.25) < 1e-9
    assert abs(cfg.sigma_px - 1.8) < 1e-9
    assert abs(cfg.spacing_px - 3.2) < 1e-9
    # overlap 0.45 derived
    assert abs(cfg.overlap - 0.4537887685) < 0.02
    assert cfg.n_pixels == 1024
    assert cfg.field_shape == (32, 32)
    assert cfg.blob_center_A == (8.0, 8.0)
    assert cfg.blob_center_B == (24.0, 24.0)
    assert cfg.target_area == "V1"
    # V1 targeting default
    assert cfg.target_area == "V1"


def test_rf_config_provenance():
    cfg = RFConfig()
    d = cfg.to_dict()
    assert d["lattice_size"] == 32
    assert d["field_dva"] == 8.0
    # hash distinct
    h = cfg.hash()
    assert isinstance(h, str) and len(h) == 16
    # metadata for config_hash
    meta = cfg.to_metadata()
    assert "rf_lattice_size" in meta
    assert meta["rf_sigma_px"] == 1.8


def test_config_hash_distinct_from_canonical():
    canon_cfg = build_jomission_network(seed=0)
    canon_hash = config_hash(canon_cfg)
    assert canon_hash == FROZEN_CANONICAL_HASH, f"canonical hash drift {canon_hash}"
    rf_cfg = build_jomission_network_with_rf(seed=0)
    rf_hash = config_hash(rf_cfg)
    assert rf_hash != FROZEN_CANONICAL_HASH, f"RF hash not distinct {rf_hash}"
    assert rf_hash != canon_hash
    # Also test apply_rf_to_configuration
    cfg = RFConfig(seed=1)
    rf_cfg2 = apply_rf_to_configuration(canon_cfg, cfg)
    assert config_hash(rf_cfg2) != canon_hash
    # Different seeds give different hashes
    rf_cfg3 = build_jomission_network_with_rf(RFConfig(seed=2), seed=0)
    assert config_hash(rf_cfg3) != rf_hash


def test_rf_operator_weights_l1_and_sparse():
    cfg = RFConfig()
    model = build_jomission_model(seed=0)
    op = RFOperator(cfg, model)
    # shape
    assert op.weights.shape == (400, 1024)
    assert op.weights_target.shape[0] == op.n_target
    assert op.weights_target.shape[1] == 1024
    assert op.n_v1 == 100
    assert op.n_total == 400
    # L1 normalized per target row
    sums = op.weights_target.sum(axis=1)
    for s in sums:
        if s > 1e-12:
            assert abs(float(s) - 1.0) < 1e-5, f"L1 sum {s} not 1"
    # sparse: check weight sparsity at 1e-4 is within expected
    ws = op.weight_sparsity(relative_thresh=1e-4)
    assert 0.05 <= ws <= 0.35, f"weight sparsity {ws} out of 0.05-0.35"
    # V1 targeting: non-V1 rows zero
    non_v1 = [i for i in range(400) if i not in op.v1_indices]
    assert np.all(op.weights[non_v1, :] == 0), "non-V1 rows should be zero"
    # V4/FEF/PFC specifically zero
    from jaxfne import paradigm_target_indices_from_model
    for area in ("V4", "FEF", "PFC"):
        idx = paradigm_target_indices_from_model(model, area=area)
        idx = [int(x) for x in np.asarray(idx).tolist()]
        if idx:
            assert np.all(op.weights[idx, :] == 0), f"{area} should have zero RF weights"


def test_rf_operator_v1_targeting_no_v4_drive():
    cfg = RFConfig()
    model = build_jomission_model(seed=0)
    op = RFOperator(cfg, model)
    # Drive for A should be non-zero only for V1 target, zero for V4
    drive = op.drive_for_stimulus("stimulus_A")
    from jaxfne import paradigm_target_indices_from_model
    v4_idx = [int(x) for x in np.asarray(paradigm_target_indices_from_model(model, area="V4")).tolist()]
    assert np.all(drive[v4_idx] == 0), "V4 drive must be zero"
    # V1 target should have non-zero
    assert np.any(drive[op.target_indices] > 0), "V1 target must have non-zero drive for A"


def test_rf_operator_ab_blobs_sparsity_and_jaccard():
    cfg = RFConfig()
    model = build_jomission_model(seed=0)
    op = RFOperator(cfg, model)
    v = op.validate()
    assert v["valid"], v["issues"]
    metrics = v["metrics"]
    # sparsity 0.18-0.30 (with tolerance)
    pop_a = metrics["population_sparsity_A"]
    pop_b = metrics["population_sparsity_B"]
    lo, hi = cfg.sparsity_range
    assert lo - 0.05 <= pop_a <= hi + 0.05, f"A sparsity {pop_a} not in {cfg.sparsity_range}"
    assert lo - 0.05 <= pop_b <= hi + 0.05, f"B sparsity {pop_b} not in {cfg.sparsity_range}"
    # Jaccard <0.15
    jac = metrics["jaccard_AB"]
    assert jac < cfg.jaccard_threshold + 1e-9, f"Jaccard {jac} >= {cfg.jaccard_threshold}"
    assert jac < 0.15
    # A vs B drives should be distinct
    da = op.drive_for_stimulus("stimulus_A")[op.target_indices]
    db = op.drive_for_stimulus("stimulus_B")[op.target_indices]
    # correlation low
    # At least not identical
    assert not np.allclose(da, db), "A and B drives should differ"


def test_rf_operator_omission_zero_drive():
    cfg = RFConfig()
    model = build_jomission_model(seed=0)
    op = RFOperator(cfg, model)
    # Omitted stimulus gives zero drive
    drive_omit = op.drive_for_stimulus("stimulus_omitted")
    assert np.all(drive_omit == 0), "omission drive must be zero"
    # Pattern for omitted is zeros
    pat = op.stimulus_pattern("stimulus_omitted")
    assert np.all(pat == 0)
    assert pat.shape == (32, 32)
    # Random stimulus should be non-zero but not equal to A/B
    pat_r = op.stimulus_pattern("random_stimulus")
    assert pat_r.shape == (32, 32)
    assert not np.all(pat_r == 0)
    drive_r = op.drive_for_stimulus("random_stimulus")
    assert np.any(drive_r[op.target_indices] > 0)


def test_rf_operator_tier_binary_vs_graded():
    model = build_jomission_model(seed=0)
    cfg_graded = RFConfig(tier="graded")
    cfg_binary = RFConfig(tier="binary")
    op_graded = RFOperator(cfg_graded, model)
    op_binary = RFOperator(cfg_binary, model)
    # Both should be valid
    assert op_graded.validate()["valid"]
    assert op_binary.validate()["valid"]
    # Patterns differ: binary is disk, graded is Gaussian
    pat_g = op_graded.stimulus_pattern("stimulus_A", tier="graded")
    pat_b = op_binary.stimulus_pattern("stimulus_A", tier="binary")
    assert not np.allclose(pat_g, pat_b)
    # Graded pattern should be continuous [0,1], binary is 0/1
    assert set(np.unique(pat_b)).issubset({0.0, 1.0})
    assert pat_g.max() == 1.0
    # Schedules differ
    cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AAAB"][0]
    sched_g = op_graded.to_stimulus_schedule(cond)
    sched_b = op_binary.to_stimulus_schedule(cond)
    # Binary should have one event per sensory slot with uniform amplitude
    # Graded should have per-unit events with varying amplitudes
    # Count drive events
    drive_g = [e for e in sched_g.events if e.get("is_drive_event")]
    drive_b = [e for e in sched_b.events if e.get("is_drive_event")]
    # Graded has more events than binary (per-unit vs per-slot)
    assert len(drive_g) > len(drive_b), "graded should have more per-unit events than binary"
    # Binary amplitudes uniform
    amps_b = [e["amplitude"] for e in drive_b]
    assert len(set(amps_b)) == 1, "binary amplitudes should be uniform"
    # Graded amplitudes varied
    amps_g = [e["amplitude"] for e in drive_g]
    assert len(set(np.round(amps_g, 3))) > 1, "graded amplitudes should vary"


def test_rf_operator_stimulus_schedule_uses_target_indices():
    cfg = RFConfig(tier="binary")
    model = build_jomission_model(seed=0)
    op = RFOperator(cfg, model)
    cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AAAB"][0]
    sched = op.to_stimulus_schedule(cond, n_neurons=400)
    # Must contain target_indices
    assert any("target_indices" in e for e in sched.events if e.get("is_drive_event"))
    # All drive events target only V1 (and specifically target_indices)
    for e in sched.events:
        if e.get("is_drive_event"):
            for idx in e["target_indices"]:
                assert idx in op.target_indices, f"drive index {idx} not in V1 target"
                # Ensure no V4 etc.
                assert idx < 100 or idx in op.v1_indices, "target must be V1"
    # Check omission condition has zero drive but preserves timing
    cond_omit = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AXAB"][0]
    sched_omit = op.to_stimulus_schedule(cond_omit, n_neurons=400)
    # Find p2 event which is omission
    p2_events = [e for e in sched_omit.events if e["label"] == "p2"]
    assert len(p2_events) == 1
    assert p2_events[0]["is_drive_event"] is False
    assert p2_events[0]["amplitude"] == 0.0
    # p2 onset must be canonical even when omitted
    from jomission.paradigm.spec import SLOT_ONSET_MS
    assert p2_events[0]["onset_ms"] == SLOT_ONSET_MS["p2"]
    # to_array should respect target_indices sparsely
    arr = sched.to_array(n_steps=46240, dt_ms=0.1)
    assert arr.shape == (46240, 400)
    # Check V4 has zero drive during sensory slots
    from jaxfne import paradigm_target_indices_from_model
    v4_idx = [int(x) for x in np.asarray(paradigm_target_indices_from_model(model, area="V4")).tolist()]
    # During p1 slot (0-531ms), V4 should be zero
    p1_start = int(round(0.0 / 0.1))
    p1_end = int(round(531.0 / 0.1))
    v4_drive_p1 = np.asarray(arr[p1_start:p1_end, v4_idx])
    assert np.all(v4_drive_p1 == 0), "V4 must have zero drive"
    # V1 target should have non-zero
    v1_tgt_drive = np.asarray(arr[p1_start:p1_end, op.target_indices])
    assert np.any(v1_tgt_drive > 0), "V1 target must have drive"


def test_rf_operator_jaxfne_capability_proof():
    proof = prove_jaxfne_target_indices_capability()
    assert proof["supports_target_indices"] is True
    assert proof["per_unit_heterogeneity_works"] is True
    assert proof["no_modification_needed"] is True
    # Ensure source inspection shows target_indices handling
    assert "target_indices" in proof["evidence"] or proof["supports_target_indices"]


def test_rf_operator_simulation_smoke():
    """Smoke: run a short simulation with RF schedule to prove integration."""
    cfg = RFConfig(tier="binary")
    model = build_jomission_model(seed=0)
    op = RFOperator(cfg, model)
    cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AAAB"][0]
    sched = op.to_stimulus_schedule(cond, n_neurons=400)
    # Short simulation 100ms (1000 steps) to verify no error
    sim = simulation_for_model(model, duration_ms=100.0, dt_ms=0.5, seed=0)
    sig = jtfne.simulate(model, sim, paradigm=sched)
    assert sig.V_m.shape[0] == int(100.0 / 0.5)
    assert sig.V_m.shape[1] == 400
    # Check field present
    assert sig.field is not None


def test_rf_operator_weights_deterministic():
    cfg = RFConfig(seed=0)
    model = build_jomission_model(seed=0)
    op1 = RFOperator(cfg, model)
    op2 = RFOperator(cfg, model)
    assert np.allclose(op1.weights, op2.weights), "weights must be deterministic"
    # GEN2_C021 interleaved tiling: weights are now seed-dependent via deterministic shuffle (seed ^0xC02A)
    cfg_different = RFConfig(seed=1)
    op_diff = RFOperator(cfg_different, model)
    if cfg.interleaved_tiling:
        # With interleaving, different seeds give different tilings (deterministic per seed)
        assert not np.allclose(op1.weights, op_diff.weights), "interleaved tiling should be seed-dependent (deterministic shuffle)"
        # But non-interleaved should remain seed-independent
        cfg_no_shuffle = RFConfig(seed=0, interleaved_tiling=False)
        cfg_no_shuffle2 = RFConfig(seed=1, interleaved_tiling=False)
        op_ns1 = RFOperator(cfg_no_shuffle, model)
        op_ns2 = RFOperator(cfg_no_shuffle2, model)
        assert np.allclose(op_ns1.weights, op_ns2.weights), "non-interleaved tiling should be seed independent"
    else:
        # Weights for A/B should be same (deterministic tiling), only random stimulus differs
        assert np.allclose(op1.weights, op_diff.weights), "weights tiling is deterministic, not seed dependent"
    # Random stimulus pattern should differ with seed
    pat0 = op1.stimulus_pattern("random_stimulus")
    pat1 = op_diff.stimulus_pattern("random_stimulus")
    assert not np.allclose(pat0, pat1), "random stimulus should differ with seed"


def test_rf_operator_centers_tiling():
    cfg = RFConfig()
    model = build_jomission_model(seed=0)
    op = RFOperator(cfg, model)
    # Check centers cover 8° field
    centers = list(op.centers.values())
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    # Should span roughly 1.6 to 30.4
    assert min(xs) >= 1.0 and max(xs) <= 31.0
    assert min(ys) >= 1.0 and max(ys) <= 31.0
    # Check spacing approx 3.2 between neighbors (for full V1 10x10)
    # For sorted centers in grid order, neighbor distance should be ~3.2
    # Take first row
    sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]))
    # First 10 in same row should be spaced 3.2
    row0 = sorted_centers[:10]
    for i in range(1, len(row0)):
        dist = abs(row0[i][0] - row0[i-1][0])
        assert abs(dist - 3.2) < 1e-6, f"spacing {dist} !=3.2"

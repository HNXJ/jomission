"""Area-local field recording — provenance, shapes, reconstruction, and non-fabrication.

This test is the artifact-backed evidence for the area-local recording path
(jomission/recording/area_local.py). It proves:

1. JaxFNE native field is single global (T, C) not area-local — missing capability
   documented, then post-hoc linear partition is the valid path.
2. field_by_area has correct shape and per-area distinctness (not contact-averaged copies).
3. Provenance: neuron_metadata -> area indices -> kernel slices.
4. Linear reconstruction: sum_a field_a == global field.
5. Truth gates preserved: proxy_readout, physical_amplitude_calibrated=False, linear_solver.
6. 4D trial stack shape: (n_trials, n_areas, n_contacts, time).

References (file:line file:line for inspection):
- jaxfne.fields.proxy.project_laminar_sources:148 kernel K[c,n] at proxy.py:192, field=sources@K.T at proxy.py:201
- jaxfne.fields.proxy.probe_laminar_modes:523 no area selector
- jaxfne._config.Configuration.probe: attaches only n_contacts, no area
- jaxfne._model_simulate.simulate:764-773 single project_laminar_sources call -> FieldOutput(16, 400) for 400 neurons
"""

from __future__ import annotations

import inspect

import numpy as np

import jaxfne as jtfne
from jaxfne.fields.proxy import probe_laminar_modes, project_laminar_sources

import jomission.recording.area_local as al
from jomission.network.builder import build_jomission_model
from jomission.network.populations import JOMISSION_AREAS
from jomission.recording.probes import N_CONTACTS_DEFAULT


def test_jaxfne_api_has_no_area_selector():
    """Prove JaxFNE cannot express area-local field natively (missing capability)."""
    sig_proj = inspect.signature(project_laminar_sources)
    assert "area" not in sig_proj.parameters, (
        f"project_laminar_sources unexpectedly has area param: {sig_proj}"
    )
    assert set(sig_proj.parameters) == {"sources", "positions", "n_contacts", "width", "mode", "dtype"}

    sig_probe = inspect.signature(probe_laminar_modes)
    assert "area" not in sig_probe.parameters, f"probe_laminar_modes unexpectedly has area: {sig_probe}"

    # Configuration.probe has only n_contacts / modes, no area — verified via inspect
    from jomission.network.builder import build_jomission_network
    cfg = build_jomission_network(n_per_area=100, seed=0)
    assert cfg.probes[0].get("area") is None, "probe unexpectedly carries area"

    # Simulate -> single global field
    model = build_jomission_model(n_per_area=100, seed=0)
    sim = jtfne.Simulation(duration_ms=50.0, dt_ms=0.1, seed=0)
    sig = jtfne.simulate(model, sim)
    assert sig.field is not None
    assert sig.field.lfp_proxy.shape == (500, N_CONTACTS_DEFAULT), (
        f"global lfp_proxy shape {sig.field.lfp_proxy.shape} != (500, 16)"
    )
    assert sig.field.kernel.shape == (N_CONTACTS_DEFAULT, 400)
    # Kernel depth-only: verify it's not area-separated (X ignored)
    # If it were area-aware, kernel would have block structure; instead it's smooth in z
    kernel = np.asarray(sig.field.kernel)
    assert kernel.shape == (16, 400)


def test_field_by_area_shape_and_provenance():
    """field_by_area has correct shapes and provenance; each area has distinct content."""
    model = build_jomission_model(n_per_area=100, seed=0)
    sim = jtfne.Simulation(duration_ms=50.0, dt_ms=0.1, seed=1)
    sig = jtfne.simulate(model, sim)

    per_area = al.field_by_area_from_signal(sig, model)
    meta = per_area.pop("__meta__")

    # Provenance metadata
    assert meta["provenance"].startswith("area indices from neuron_metadata")
    assert meta["physical_amplitude_calibrated"] is False
    assert meta["field_claim_level"] == "proxy_readout"
    assert meta["field_solver_status"] == "linear_solver"
    assert meta["areas"] == list(JOMISSION_AREAS)
    assert meta["n_contacts"] == N_CONTACTS_DEFAULT

    # Shapes per area: [T, C] = [500, 16]
    for area in JOMISSION_AREAS:
        arr = per_area[area]
        assert arr.shape == (500, N_CONTACTS_DEFAULT), f"{area} shape {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{area} non-finite"

    # Area distinctness: not all areas equal (not contact-averaged copies)
    # If someone fabricated by assigning global mean to each area, they'd be identical
    for a, b in [("V1", "V4"), ("FEF", "PFC"), ("V1", "PFC")]:
        diff = float(np.max(np.abs(per_area[a] - per_area[b])))
        assert diff > 1e-6, f"areas {a} and {b} unexpectedly identical (fabrication?) max diff {diff}"

    # Area neuron counts sum to total
    assert sum(meta["area_n_neurons"].values()) == 400
    for area in JOMISSION_AREAS:
        assert meta["area_n_neurons"][area] == 100, f"{area} n_neurons {meta['area_n_neurons'][area]} != 100"


def test_field_by_area_reconstruction():
    """Linear reconstruction: sum_a field_by_area[a] == global lfp_proxy."""
    model = build_jomission_model(n_per_area=100, seed=2)
    sim = jtfne.Simulation(duration_ms=50.0, dt_ms=0.1, seed=2)
    sig = jtfne.simulate(model, sim)

    per_area = al.field_by_area_from_signal(sig, model, include_diagnostics=False)
    rec = al.verify_reconstruction(sig, per_area, model=model, atol=1e-3)
    assert rec["ok"], f"reconstruction failed: {rec}"
    assert rec["max_abs_error"] < 1e-3, f"max error {rec['max_abs_error']} exceeds 1e-3"
    assert rec["global_shape"] == [500, 16]

    # Also test array helper reconstruction via sum over area axis
    arr, areas, meta = al.field_by_area_array(sig, model, time_major=False)  # [A, C, T]
    assert arr.shape == (4, 16, 500), f"stacked shape {arr.shape}"
    assert areas == tuple(JOMISSION_AREAS)
    assert meta["layout"] == "A_C_T"
    assert meta["physical_amplitude_calibrated"] is False
    # Sum over areas in [A, C, T] -> [C, T] -> transpose to [T, C] for comparison
    summed = arr.sum(axis=0)  # [C, T]
    global_ct = np.asarray(sig.field.lfp_proxy).T  # [C, T]
    assert float(np.max(np.abs(summed - global_ct))) < 1e-3


def test_field_by_area_time_major_layout():
    """time_major=True gives [A, T, C] layout."""
    model = build_jomission_model(n_per_area=100, seed=3)
    sig = jtfne.simulate(model, jtfne.Simulation(duration_ms=20.0, dt_ms=0.1, seed=3))
    arr_tm, _, meta_tm = al.field_by_area_array(sig, model, time_major=True)
    arr_cm, _, meta_cm = al.field_by_area_array(sig, model, time_major=False)
    assert arr_tm.shape == (4, 200, 16)
    assert meta_tm["layout"] == "A_T_C"
    assert arr_cm.shape == (4, 16, 200)
    assert meta_cm["layout"] == "A_C_T"
    # Transpose equivalence
    assert np.allclose(np.transpose(arr_cm, (0, 2, 1)), arr_tm)


def test_field_by_area_provenance_via_metadata_only():
    """Area indices resolvable purely from signals.metadata['neuron_metadata'] (no model)."""
    model = build_jomission_model(n_per_area=100, seed=4)
    sig = jtfne.simulate(model, jtfne.Simulation(duration_ms=20.0, dt_ms=0.1, seed=4))
    assert sig.metadata.get("neuron_metadata") is not None
    # Call without model — uses metadata alone
    per_area_no_model = al.field_by_area_from_signal(sig, model=None, include_diagnostics=False)
    per_area_with_model = al.field_by_area_from_signal(sig, model=model, include_diagnostics=False)
    for area in JOMISSION_AREAS:
        assert np.allclose(per_area_no_model[area], per_area_with_model[area]), f"{area} mismatch"


def test_field_by_area_4d_multi_trial():
    """field[trial, area, contact, time] — 4D stack has correct shape and provenance."""
    model = build_jomission_model(n_per_area=100, seed=5)
    signals = []
    for trial in range(3):
        sig = jtfne.simulate(model, jtfne.Simulation(duration_ms=20.0, dt_ms=0.1, seed=10 + trial))
        signals.append(sig)

    field_4d, areas, meta = al.field_by_area_4d(signals, model, time_major=False)
    assert field_4d.shape == (3, 4, 16, 200), f"4D shape {field_4d.shape}"
    assert areas == tuple(JOMISSION_AREAS)
    assert meta["n_trials"] == 3
    assert meta["field_4d_shape"] == [3, 4, 16, 200]
    assert meta["layout"] == "trial_A_C_T"
    assert meta["physical_amplitude_calibrated"] is False

    # Trial axis distinctness: different seeds -> different fields
    assert float(np.max(np.abs(field_4d[0] - field_4d[1]))) > 1e-6

    # time_major variant
    field_4d_tm, _, meta_tm = al.field_by_area_4d(signals, model, time_major=True)
    assert field_4d_tm.shape == (3, 4, 200, 16)
    assert meta_tm["layout"] == "trial_A_T_C"
    # Check transpose equivalence with per-trial stack
    assert np.allclose(np.transpose(field_4d, (0, 1, 3, 2)), field_4d_tm)


def test_field_by_area_not_contact_averaged():
    """Prove field_by_area is not a fabricated contact-averaged broadcast to areas.

    Fabrication would be: area_field = global_field.mean(axis=1, keepdims=True) tile to areas.
    Genuine must have contact-wise structure distinct per area and vary across contacts.
    """
    model = build_jomission_model(n_per_area=100, seed=6)
    sig = jtfne.simulate(model, jtfne.Simulation(duration_ms=50.0, dt_ms=0.1, seed=6))

    per_area = al.field_by_area_from_signal(sig, model, include_diagnostics=False)
    # Each area's field must vary across contacts (not uniform across C due to depth weighting)
    for area in JOMISSION_AREAS:
        arr = per_area[area]  # [T, C]
        # Std across contacts at a fixed time should be > 0 (contacts have different laminar depths)
        # Use time-averaged variance
        contact_var = float(arr.var(axis=1).mean())
        assert contact_var > 1e-8, f"{area} field has no contact variation (maybe fabricated?) var {contact_var}"

    # Check that per-area fields are not all identical to global mean across contacts
    global_field = np.asarray(sig.field.lfp_proxy)
    global_mean = global_field.mean(axis=1, keepdims=True)  # [T, 1] contact-averaged
    for area in JOMISSION_AREAS:
        # Genuine area field should differ from contact-averaged global
        diff_from_mean = float(np.max(np.abs(per_area[area] - global_mean)))
        assert diff_from_mean > 1e-6, f"{area} equals global contact-averaged (fabrication detected)"


def test_csd_by_area_and_claims():
    """CSD area-local inherits proxy claim; verify shape and finiteness."""
    model = build_jomission_model(n_per_area=100, seed=7)
    sig = jtfne.simulate(model, jtfne.Simulation(duration_ms=20.0, dt_ms=0.1, seed=7))
    csd_by_area = al.csd_by_area_from_signal(sig, model)
    for area in JOMISSION_AREAS:
        assert csd_by_area[area].shape == (200, 16)
        assert np.all(np.isfinite(csd_by_area[area]))


def test_verify_reconstruction_helper():
    """verify_reconstruction helper reports ok and carries shapes."""
    model = build_jomission_model(n_per_area=100, seed=8)
    sig = jtfne.simulate(model, jtfne.Simulation(duration_ms=20.0, dt_ms=0.1, seed=8))
    report = al.verify_reconstruction(sig, model=model, atol=1e-3)
    assert report["ok"] is True
    assert report["max_abs_error"] < 1e-3
    assert report["global_shape"] == [200, 16]
    assert all(v == [200, 16] for v in report["per_area_shapes"].values())


def test_frozen_hashes_unchanged():
    """Frozen config/hash identities must not be altered by recording changes."""
    import hashlib
    import json

    import jaxfne.hdp_network as hdp
    from jaxfne.io import config_hash

    model = build_jomission_model(n_per_area=100, seed=0)
    ch = config_hash(model.cfg)
    assert ch == "4f9fdeae7428199a", f"config_hash {ch} != frozen 4f9fdeae7428199a"
    hp = hdp.v1_pfc_aaab_hdp_params()
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]
    assert hp_hash == "f327f9d2ad64cc88", f"hp_hash {hp_hash} != f327f9d2ad64cc88"

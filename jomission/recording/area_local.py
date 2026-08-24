"""Area-local field recording — C→q→φ→y provenance-preserving partition.

Frozen authorities:
- JaxFNE 0.4.17 is the simulation engine; we do not reimplement neurons/synapses/HDP/source/field/probe.
- C→q→φ→y remains distinct: source (q) → field (φ via proxy_readout) → probe/readout (y).
  Field is proxy_readout, physical_amplitude_calibrated=False.
- No fabrication: area-local fields are linear partitions of the JaxFNE laminar
  projection, not contact-averaged copies broadcast to areas.

JaxFNE inspection (file:line evidence):
- jaxfne.fields.proxy.project_laminar_sources:148 defines
    project_laminar_sources(sources [T,N], positions [N,3], n_contacts, width, mode, dtype) -> FieldOutput
  with kernel K[c,n] = exp(-0.5*((z_contact[c]-z_n)/width)^2)  (proxy.py:192)
  and field = sources @ K.T                                   (proxy.py:201)
  Kernel depends ONLY on depth z (positions[:,2]), not on area label or x/y.
  No area selector exists (verified via inspect.signature).
- jaxfne.fields.proxy.probe_laminar_modes:523 extracts from an existing FieldOutput;
  modes=("source","phi_e","CSD","LFP") — no area parameter.
- jaxfne._config.Configuration.probe: attaches probe metadata with only n_contacts;
  no area-specific probe declaration.
- jaxfne._model_simulate.simulate:764-773 single project_laminar_sources call over
  ALL neurons → one global FieldOutput with lfp_proxy shape (T, n_contacts) e.g. (500, 16).
  No per-area field is produced natively.

Scientific validity of post-hoc partition:
- Linearity: field = sum_a sources_a @ K_a.T where K_a = K[:, idx_a] and idx_a
  are neuron indices for area a from neuron_metadata (model.neuron_table() /
  signals.metadata["neuron_metadata"]). Verified: sum_a field_a == global field
  within float32 accumulation error (max diff ~7e-04 for T=500, N=400 on test).
- Semantics documented: each area's field is the contribution of its neurons to
  the SHARED laminar contacts spanning z in [0,1]. Contacts are depth-defined, not
  xy-defined; area separation (x offset) is not in the projection. This is
  source-ownership partitioning, not a separate physical probe per area.
- Provenance: neuron_metadata row → area label → index set → kernel slice → field slice.
  All steps are traceable; no heuristic reassignment.
- Truth gates preserved: proxy_readout, physical_amplitude_calibrated=False,
  field_solver_status=linear_solver (inherited from FieldOutput.diagnostics).

Tensor shapes:
- Per-signal call:  field_by_area  dict[area -> (T, n_contacts)] and
  stacked array with shape (n_areas, n_contacts, T) or (n_areas, T, n_contacts)
  depending on convention. We expose both plus a canonical 4D for trials:
  field_by_area_4d shape (n_trials, n_areas, n_contacts, T).
- Single trial (no trial axis in core Signals): caller provides trial batch
  as list[Signals] → stacked to (n_trials, n_areas, n_contacts, T).

Usage:
    from jomission.recording.area_local import field_by_area_from_signal, field_by_area_4d

    # Single trial
    per_area = field_by_area_from_signal(sig, model)  # dict area -> (T, C)
    arr, areas, meta = field_by_area_array(sig, model)  # (n_areas, C, T) or (n_areas, T, C)

    # Multi-trial
    field_4d, areas, meta = field_by_area_4d(signals_list, model)  # (n_trials, n_areas, C, T)
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

AREAS_CANONICAL: tuple[str, ...] = ("V1", "V4", "FEF", "PFC")


def _resolve_area_indices(
    *,
    neuron_metadata: list[dict[str, Any]] | None,
    model: Any | None = None,
    areas: tuple[str, ...] = AREAS_CANONICAL,
) -> dict[str, np.ndarray]:
    """Resolve {area -> int[] neuron indices} from neuron_metadata or model.neuron_table().

    Prefers signals.metadata["neuron_metadata"] (the per-trial provenance record);
    falls back to model.neuron_table() when metadata is absent (still area-faithful
    because model construction is deterministic for frozen config_hash 4f9fdeae7428199a).

    Raises ValueError if neither source provides area labels.
    """
    table: list[dict[str, Any]] | None = None
    if neuron_metadata is not None and len(neuron_metadata) > 0:
        table = neuron_metadata
    elif model is not None and hasattr(model, "neuron_table"):
        try:
            table = model.neuron_table()
        except Exception:
            table = None
    if table is None or len(table) == 0:
        raise ValueError(
            "Cannot resolve area indices: signals.metadata['neuron_metadata'] is None/empty "
            "and model.neuron_table() unavailable. Build model via build_jomission_model() "
            "so geometry_meta -> neuron_metadata is present, or pass neuron_metadata explicitly."
        )
    # neuron_id in table is the positional index in the flattened neuron array
    # (0..N-1 contiguous per _construct_population). Use row order, not neuron_id value,
    # to tolerate any future reordering — but current code uses contiguous ids anyway.
    area_to_idx: dict[str, list[int]] = {a: [] for a in areas}
    for pos, row in enumerate(table):
        area = str(row.get("area", ""))
        if area in area_to_idx:
            # Use positional index pos, which matches array axis for V_m/spikes/sources
            area_to_idx[area].append(pos)
    # Validate coverage
    missing = [a for a, idx in area_to_idx.items() if len(idx) == 0]
    if missing:
        available = sorted({str(r.get("area", "")) for r in table})
        raise ValueError(
            f"No neurons for area(s) {missing}; available areas in table: {available}. "
            f"Requested areas={list(areas)}."
        )
    return {a: np.asarray(idx, dtype=np.int32) for a, idx in area_to_idx.items()}


def _kernel_and_sources(
    signal: Any,
    model: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (sources [T,N], kernel [C,N], contact_depths [C]) as numpy arrays.

    Sources are required (used for the partition); kernel/contact_depths come
    from signal.field. Raises with provenance if absent.
    """
    if signal.field is None:
        raise ValueError(
            "signal.field is None (record_fields=False or not wired). "
            "Run simulate with Simulation(record_fields=True) (default) so field is present."
        )
    if signal.sources is None:
        # Fallback: reconstruct per-neuron source proxy from diagnostics if ever needed,
        # but current jomission always records sources (record_sources=True by default).
        raise ValueError(
            "signal.sources is None (record_sources=False). Field partition requires sources [T,N]; "
            "run with record_sources=True to preserve C->q provenance."
        )
    sources = np.asarray(signal.sources)  # [T, N]
    kernel = np.asarray(signal.field.kernel)  # [C, N]
    contact_depths = np.asarray(signal.field.contact_depths)  # [C]
    if sources.ndim != 2:
        raise ValueError(f"sources must be 2D [T,N], got shape {sources.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D [C,N], got shape {kernel.shape}")
    if sources.shape[1] != kernel.shape[1]:
        raise ValueError(
            f"sources width {sources.shape[1]} != kernel width {kernel.shape[1]}; "
            "model/positions mismatch (frozen config_hash 4f9fdeae7428199a expects N=400 for n_per_area=100)."
        )
    return sources, kernel, contact_depths


def field_by_area_from_signal(
    signal: Any,
    model: Any | None = None,
    *,
    areas: tuple[str, ...] = AREAS_CANONICAL,
    include_diagnostics: bool = True,
) -> dict[str, np.ndarray]:
    """Partition a single Signals field into area-local contributions.

    Each area's field is sources[:, idx_a] @ kernel[:, idx_a].T  (linear partition
    of jaxfne.fields.proxy.project_laminar_sources at proxy.py:201).

    Parameters
    ----------
    signal: jaxfne Signals with .field and .sources
    model: optional Model for fallback neuron_table() lookup
    areas: ordered area tuple; default canonical V1/V4/FEF/PFC
    include_diagnostics: attach provenance in returned dict under "__meta__" (not an area)

    Returns
    -------
    dict mapping area -> ndarray [T, C] (time x contacts), plus optional "__meta__".
    C = n_contacts (default 16), T = n_steps (e.g. 46240 for 4624 ms at dt 0.1).

    Truth gates: proxy_readout, physical_amplitude_calibrated=False inherited; no escalation.
    """
    sources, kernel, contact_depths = _kernel_and_sources(signal, model)
    neuron_meta = signal.metadata.get("neuron_metadata") if hasattr(signal, "metadata") else None
    area_idx = _resolve_area_indices(neuron_metadata=neuron_meta, model=model, areas=areas)

    out: dict[str, np.ndarray] = {}
    for area in areas:
        idx = area_idx[area]
        # Partition: sources[:, idx] @ kernel[:, idx].T  — preserves linearity/reconstruction
        k_a = kernel[:, idx]  # [C, N_a]
        s_a = sources[:, idx]  # [T, N_a]
        field_a = s_a @ k_a.T  # [T, C]
        out[area] = field_a

    if include_diagnostics:
        # Use signal.field diagnostics region as provenance anchor
        diag = dict(getattr(signal.field, "diagnostics", {}) or {})
        out["__meta__"] = {  # type: ignore[assignment]
            "method": "linear_partition_of_laminar_proxy (project_laminar_sources @ proxy.py:192,201)",
            "semantics": (
                "contribution of area's neurons to SHARED laminar contacts spanning z in [0,1]; "
                "contacts are depth-defined, not xy-defined; area separation (x offset) not in projection. "
                "Source-ownership partitioning, not separate physical probes."
            ),
            "provenance": "area indices from neuron_metadata (signals.metadata['neuron_metadata']) or model.neuron_table()",
            "kernel_shape": list(kernel.shape),
            "contact_depths": contact_depths.tolist(),
            "n_contacts": int(kernel.shape[0]),
            "areas": list(areas),
            "area_n_neurons": {a: int(len(area_idx[a])) for a in areas},
            "source_proxy_shape": list(sources.shape),
            "field_solver_status": diag.get("field_solver_status", "linear_solver"),
            "field_claim_level": diag.get("field_claim_level", "proxy_readout"),
            "physical_amplitude_calibrated": False,
            "claim_level": "proxy_readout",
            "reconstruction_note": "sum_a field_by_area[a] == global field up to float32 accumulation order (~7e-04 for T=500,N=400)",
        }
    return out


def field_by_area_array(
    signal: Any,
    model: Any | None = None,
    *,
    areas: tuple[str, ...] = AREAS_CANONICAL,
    time_major: bool = False,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Stacked array form of field_by_area_from_signal.

    Parameters
    ----------
    time_major: if False (default), returns [n_areas, n_contacts, T] (contact-major, convenient for
                spectral analysis per contact). If True, returns [n_areas, T, n_contacts].

    Returns
    -------
    (array, areas, meta) where array shape is (A, C, T) or (A, T, C) and meta is provenance dict.
    """
    per_area = field_by_area_from_signal(signal, model, areas=areas, include_diagnostics=True)
    meta = per_area.pop("__meta__", {})  # type: ignore
    # Each per_area[area] is [T, C]; stack to desired layout
    stacked_tc = np.stack([per_area[a] for a in areas], axis=0)  # [A, T, C]
    if time_major:
        arr = stacked_tc  # [A, T, C]
    else:
        arr = np.transpose(stacked_tc, (0, 2, 1))  # [A, C, T]
    meta["stacked_shape"] = list(arr.shape)
    meta["layout"] = "A_T_C" if time_major else "A_C_T"
    return arr, areas, meta


def field_by_area_4d(
    signals: list[Any],
    model: Any | None = None,
    *,
    areas: tuple[str, ...] = AREAS_CANONICAL,
    time_major: bool = False,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Multi-trial stack: field[trial, area, contact, time].

    Parameters
    ----------
    signals: list of Signals, one per trial (all must share same N,C and area table)
    model: optional for area resolution fallback
    time_major: False -> [n_trials, A, C, T]; True -> [n_trials, A, T, C]

    Returns
    -------
    (field_4d, areas, meta) with field_4d shape (n_trials, A, C, T) or (n_trials, A, T, C).
    """
    if not signals:
        raise ValueError("signals list is empty")
    per_trial = []
    metas = []
    for sig in signals:
        arr, _, meta = field_by_area_array(sig, model, areas=areas, time_major=time_major)
        per_trial.append(arr)
        metas.append(meta)
    # Validate shape consistency
    shapes = {tuple(a.shape) for a in per_trial}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent per-trial field shapes: {shapes}")
    field_4d = np.stack(per_trial, axis=0)  # [n_trials, A, C, T] or [n_trials, A, T, C]
    agg_meta: dict[str, Any] = {
        "n_trials": len(signals),
        "areas": list(areas),
        "field_4d_shape": list(field_4d.shape),
        "layout": "trial_A_C_T" if not time_major else "trial_A_T_C",
        "per_trial_meta_sample": metas[0] if metas else {},
        "field_solver_status": "linear_solver",
        "physical_amplitude_calibrated": False,
        "claim_level": "proxy_readout",
        "method": "stack of linear partitions (field_by_area_from_signal per trial)",
    }
    return field_4d, areas, agg_meta


def verify_reconstruction(
    signal: Any,
    per_area: dict[str, np.ndarray] | None = None,
    *,
    model: Any | None = None,
    areas: tuple[str, ...] = AREAS_CANONICAL,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """Verify sum_a field_by_area[a] reconstructs the global field.

    Returns {ok, max_abs_error, mean_abs_error, atol, global_shape, per_area_shapes}.
    """
    global_field = np.asarray(signal.field.lfp_proxy)  # [T, C]
    if per_area is None:
        per_area_full = field_by_area_from_signal(signal, model, areas=areas, include_diagnostics=False)
    else:
        per_area_full = {k: v for k, v in per_area.items() if not k.startswith("__")}
    summed = sum(per_area_full[a] for a in areas)  # [T, C]
    diff = summed - global_field
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    return {
        "ok": bool(max_abs <= atol),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "atol": float(atol),
        "global_shape": list(global_field.shape),
        "per_area_shapes": {a: list(per_area_full[a].shape) for a in areas},
        "criterion": f"max_abs_error <= atol ({atol})",
    }


# ---------------------------------------------------------------------------
# CSD / phi_e area-local variants (optional; same partition logic)
# ---------------------------------------------------------------------------

def csd_by_area_from_signal(
    signal: Any,
    model: Any | None = None,
    *,
    areas: tuple[str, ...] = AREAS_CANONICAL,
) -> dict[str, np.ndarray]:
    """Area-local CSD proxy, partitioned from lfp-derived CSD.

    JaxFNE computes CSD as spatial second derivative of phi_e/lfp (csd_tensor at
    proxy.py:1028): csd = -(phi[c+1]-2*phi[c]+phi[c-1])/dz^2. Since both operations
    are linear (field projection + CSD stencil), per-area CSD is the CSD of per-area phi.

    Returns dict area -> [T, C] CSD proxy per area.
    """
    from jaxfne.fields.proxy import csd_tensor

    per_area_lfp = field_by_area_from_signal(signal, model, areas=areas, include_diagnostics=False)
    contact_depths = np.asarray(signal.field.contact_depths)
    dz = float(contact_depths[1] - contact_depths[0]) if len(contact_depths) > 1 else 1.0
    out: dict[str, np.ndarray] = {}
    for area in areas:
        lfp_a = per_area_lfp[area]  # [T, C]
        csd_a = np.asarray(csd_tensor(jnp.asarray(lfp_a), dz))
        out[area] = csd_a
    return out

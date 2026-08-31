"""Centralized runtime authority for Jomission models.

Delayed Jomission models carry nonzero edge_list.delay_steps (builder.py:908
delays [20,80,120] via edge_list_with_delay_ms). JaxFNE >=0.4.18 correctly
rejects dense+delay (silent loss) and HDP+delay (no kernel). This module is
the single source of truth for Simulation runtime selection so delay semantics
cannot be accidentally executed through dense recurrence.

Rule:
- If model has any delay_steps >0 → recurrent_backend must be "edge_list".
  Explicit caller request for dense with delayed model → ValueError (fail loudly).
  Omitted runtime → edge_list is supplied.
- If no delays → caller choice honored; omitted → JaxFNE default (dense) is fine,
  but edge_list also works.
- HDP flag and other RuntimeConfig fields are preserved; this helper only
  enforces backend compatibility, not scientific choice.

References:
- jaxfne._model_simulate:_simulate_arrays:152-167 dense+delay guard
- jaxfne._model_simulate:_simulate_continuation_arrays:587-595 HDP+delay guard
- jaxfne._pipeline:model_requires_delay_state:330
- jomission/network/builder.py:460 cfg.runtime(edge_list) + :908 _apply_laminar_delays
"""

from __future__ import annotations

from typing import Any

import numpy as np


def model_requires_edge_list(model: Any) -> bool:
    """True if model edge_list has any nonzero delay_steps."""
    try:
        el = model.params.get("edge_list")
        if el is None:
            return False
        ds = np.asarray(getattr(el, "delay_steps", 0))
        return bool(np.any(ds != 0))
    except Exception:
        return False


def resolve_runtime_for_model(
    model: Any,
    requested: Any | None = None,
) -> Any:
    """Resolve RuntimeConfig for model, enforcing delay→edge_list.

    requested: None or jaxfne.RuntimeConfig. If None, supplies edge_list when
    required, else leaves JaxFNE default. If explicit, validates compatibility.
    """
    from jaxfne import RuntimeConfig

    requires_edge = model_requires_edge_list(model)
    if requested is None:
        if requires_edge:
            return RuntimeConfig(recurrent_backend="edge_list")
        return None  # let Simulation use default
    # explicit requested
    backend = getattr(requested, "recurrent_backend", None)
    if requires_edge and backend == "dense":
        raise ValueError(
            "Jomission delayed model (delay_steps [20,80,120] via "
            "builder.py:908 + edge_list_with_delay_ms) requires "
            "recurrent_backend='edge_list'; dense has no finite-delay path "
            "(_model_simulate.py:162). Use edge_list or remove delays."
        )
    # HDP+delay incompatibility is left to JaxFNE guard (_model_simulate.py:296)
    # so caller gets the authoritative ValueError there; we do not silently
    # disable HDP.
    return requested


def simulation_for_model(
    model: Any,
    *,
    duration_ms: float,
    dt_ms: float = 0.1,
    seed: int = 0,
    runtime: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Create jaxfne.Simulation with runtime resolved for model's delays.

    Centralized factory — prefer over direct Simulation(...) for delayed models.
    """
    from jaxfne import Simulation

    resolved = resolve_runtime_for_model(model, requested=runtime)
    if resolved is None:
        return Simulation(duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=int(seed), **kwargs)
    return Simulation(duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=int(seed), runtime=resolved, **kwargs)

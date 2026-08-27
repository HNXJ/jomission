"""Plasticity wiring — exposes HDP/RBD via JaxFNE RuntimeConfig.

No parallel simulator; this is a thin declarative layer.
"""

from __future__ import annotations

from typing import Any

import jaxfne as jtfne
from jaxfne import RuntimeConfig


def make_runtime(
    *,
    h_enabled: bool = False,
    hdp_enabled: bool = False,
    h_state_dim: int = 5,
    seed: int = 0,
    record_edge_current: bool = False,
    record_dH_components: bool = False,
    record_weight_trace: bool = True,
) -> RuntimeConfig:
    """Create RuntimeConfig for H/HDP ablation.

    GEN2_C004 (B3 E/I currents): opt-in observability for per-edge currents.
    When record_edge_current=True, hdp_params carries the flag through to
    jaxfne.compile_step_fn(record_edge_current=True) via the existing seam
    (jaxfne/emitters.py:2846, _pipeline.py:395). Default False preserves
    performance and existing callers (factorial_v0p2_fast.py still pops for
    speed; canonical path now CAN expose currents when requested).
    """
    if h_enabled and hdp_enabled:
        # H is carried as RBD state; HDP is adaptive Theta; they are mutually controlled via enable_homeostasis vs enable_hdp
        # For milestone: enable_hdp for HDP, enable_homeostasis for H-like trace (if needed)
        # Keep them exclusive per JaxFNE constraint enable_homeostasis and enable_hdp mutually exclusive
        # So we choose HDP when both requested, else H via homeostasis
        pass
    if hdp_enabled:
        hp: dict[str, object] = {"h_state_dim": int(h_state_dim), "h_state_locality": "node"}
        # GEN2_C004 opt-in current observability — forwarded to compile_step_fn/simulate
        if record_edge_current:
            hp["record_edge_current"] = True
        if record_dH_components:
            hp["record_dH_components"] = True
        if not record_weight_trace:
            hp["record_weight_trace"] = False
        return RuntimeConfig(enable_hdp=True, hdp_params=hp, recurrent_backend="edge_list", seed=seed)
    if h_enabled:
        return RuntimeConfig(enable_homeostasis=True, homeostasis_params={"tau_r_ms": 1000.0 * h_state_dim}, recurrent_backend="edge_list", seed=seed)
    # Even without HDP, honor record flags if caller explicitly asks for observability
    # (no effect without HDP kernel, but keeps API uniform)
    if record_edge_current or record_dH_components:
        hp2: dict[str, object] = {}
        if record_edge_current:
            hp2["record_edge_current"] = True
        if record_dH_components:
            hp2["record_dH_components"] = True
        return RuntimeConfig(recurrent_backend="edge_list", seed=seed, enable_hdp=False, hdp_params=hp2)
    return RuntimeConfig(recurrent_backend="edge_list", seed=seed)

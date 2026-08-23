"""Plasticity wiring — exposes HDP/RBD via JaxFNE RuntimeConfig.

No parallel simulator; this is a thin declarative layer.
"""

from __future__ import annotations

from typing import Any

import jaxfne as jtfne
from jaxfne import RuntimeConfig


def make_runtime(*, h_enabled: bool = False, hdp_enabled: bool = False, h_state_dim: int = 5, seed: int = 0) -> RuntimeConfig:
    """Create RuntimeConfig for H/HDP ablation."""
    if h_enabled and hdp_enabled:
        # H is carried as RBD state; HDP is adaptive Theta; they are mutually controlled via enable_homeostasis vs enable_hdp
        # For milestone: enable_hdp for HDP, enable_homeostasis for H-like trace (if needed)
        # Keep them exclusive per JaxFNE constraint enable_homeostasis and enable_hdp mutually exclusive
        # So we choose HDP when both requested, else H via homeostasis
        pass
    if hdp_enabled:
        return RuntimeConfig(enable_hdp=True, hdp_params={"h_state_dim": int(h_state_dim), "h_state_locality": "node"}, recurrent_backend="edge_list", seed=seed)
    if h_enabled:
        return RuntimeConfig(enable_homeostasis=True, homeostasis_params={"tau_r_ms": 1000.0 * h_state_dim}, recurrent_backend="edge_list", seed=seed)
    return RuntimeConfig(recurrent_backend="edge_list", seed=seed)

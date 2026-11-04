"""Activity-balance H rule plus a population-synchrony term (spec: docs/designs/H_SYNC_TEST_spec.md).

Balance part: identical to activity_balance (reused, not copied). Added term (time in ms, rates in Hz):
  r_fast_i   : fast per-neuron rate trace, tau_f dr_fast/dt = r_inst - r_fast        (aux "rate_fast")
  P_f        = mean_j r_fast_j          P_s = mean_j r_j  (slow trace, tau_r)
  sigma      = max(0, P_f - (1 + g) P_s) / max(P_s, eps)    (relative burst excess)
  c_i        = r_fast_i / max(P_f, eps)                      (share of the burst; mean_i c_i = 1)
  dH_i      += -kappa_s * sigma * c_i / 1000                 (per ms)
  db_i/dt    = kappa_I * dH_i/dt                             (same drive_bias channel)
Parameters PROPOSED AND UNTUNED; fixed in the sealed spec.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from jaxfne.hdp_rule import (
    HDPRuleContext,
    HDPRuleDescriptor,
    HDPRuleUpdate,
    is_registered_hdp_rule,
    register_hdp_rule,
)

from jomission.dynamics.activity_balance import DEFAULT_PARAMS as BALANCE_PARAMS
from jomission.dynamics.activity_balance import _activity_balance_step

RULE_NAME = "activity_balance_sync"

DEFAULT_PARAMS = {
    **BALANCE_PARAMS,
    "tau_f": 10.0,      # ms, fast trace
    "g": 0.5,           # burst threshold above the slow baseline (dimensionless)
    "kappa_s": 0.5,     # 1/s per unit sigma (proposed, untuned)
    "eps": 1e-3,        # Hz, floor in divisions
}


def sync_terms(r, r_fast, g, eps):
    """sigma (scalar) and c_i (per neuron) from slow and fast traces."""
    p_f = jnp.mean(r_fast)
    p_s = jnp.mean(r)
    sigma = jnp.maximum(0.0, p_f - (1.0 + g) * p_s) / jnp.maximum(p_s, eps)
    c = r_fast / jnp.maximum(p_f, eps)
    return sigma, c


def _step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    p = ctx.rule_params
    dtype = ctx.H.dtype
    r, r_fast = ctx.aux[:, 0], ctx.aux[:, 1]
    base = _activity_balance_step(dataclasses.replace(ctx, aux=r))
    g = jnp.asarray(p["g"], dtype=dtype)
    eps = jnp.asarray(p["eps"], dtype=dtype)
    kappa_s = jnp.asarray(p["kappa_s"], dtype=dtype)
    kappa_I = jnp.asarray(p["kappa_I"], dtype=dtype)
    tau_f = jnp.asarray(p["tau_f"], dtype=dtype)
    sigma, c = sync_terms(r, r_fast, g, eps)
    dH = base.dH - kappa_s * sigma * c * (1.0 / 1000.0)
    r_inst = ctx.spikes.astype(dtype) * (1000.0 / ctx.dt)
    d_aux = jnp.stack([base.d_aux, (r_inst - r_fast) / tau_f], axis=1)
    return HDPRuleUpdate(dH=dH, d_aux=d_aux, d_theta={"drive_bias": kappa_I * dH})


def register() -> None:
    """Idempotent registration."""
    if is_registered_hdp_rule(RULE_NAME):
        return
    register_hdp_rule(
        HDPRuleDescriptor(
            name=RULE_NAME,
            h_coords=("H",),
            theta_targets=("drive_bias",),
            aux_coords=("rate", "rate_fast"),
            aux_layout="per_neuron",
            scope="node",
            h_bounds=(0.1, 10.0),
            b_bounds=(-50.0, 50.0),
            default_params=dict(DEFAULT_PARAMS),
        ),
        _step,
    )


register()

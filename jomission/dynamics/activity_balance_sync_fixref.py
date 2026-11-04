"""Probe variant of activity_balance_sync with a fixed sigma denominator (spec: docs/designs/H_SYNC_MECH_spec.md).

Identical to activity_balance_sync except
  sigma = max(0, P_f - (1 + g) P_s) / max(p_ref, eps)     (p_ref: constant rule parameter, Hz)
so the burst signal no longer blows up when the lagged slow rate P_s is small. Tests the
ratio-blowup alternative to the relaxation-oscillation hypothesis. Probe only; not a candidate rule.
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

from jomission.dynamics.activity_balance import _activity_balance_step
from jomission.dynamics.activity_balance_sync import DEFAULT_PARAMS as SYNC_PARAMS

RULE_NAME = "activity_balance_sync_fixref"
DEFAULT_PARAMS = {**SYNC_PARAMS, "p_ref": 10.0}  # Hz; set per run from the baseline population rate


def _step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    p = ctx.rule_params
    dtype = ctx.H.dtype
    r, r_fast = ctx.aux[:, 0], ctx.aux[:, 1]
    base = _activity_balance_step(dataclasses.replace(ctx, aux=r))
    g, eps = jnp.asarray(p["g"], dtype=dtype), jnp.asarray(p["eps"], dtype=dtype)
    kappa_s, kappa_I = jnp.asarray(p["kappa_s"], dtype=dtype), jnp.asarray(p["kappa_I"], dtype=dtype)
    tau_f, p_ref = jnp.asarray(p["tau_f"], dtype=dtype), jnp.asarray(p["p_ref"], dtype=dtype)
    p_f, p_s = jnp.mean(r_fast), jnp.mean(r)
    sigma = jnp.maximum(0.0, p_f - (1.0 + g) * p_s) / jnp.maximum(p_ref, eps)
    c = r_fast / jnp.maximum(p_f, eps)
    dH = base.dH - kappa_s * sigma * c * (1.0 / 1000.0)
    r_inst = ctx.spikes.astype(dtype) * (1000.0 / ctx.dt)
    d_aux = jnp.stack([base.d_aux, (r_inst - r_fast) / tau_f], axis=1)
    return HDPRuleUpdate(dH=dH, d_aux=d_aux, d_theta={"drive_bias": kappa_I * dH})


def register() -> None:
    if is_registered_hdp_rule(RULE_NAME):
        return
    register_hdp_rule(
        HDPRuleDescriptor(
            name=RULE_NAME, h_coords=("H",), theta_targets=("drive_bias",),
            aux_coords=("rate", "rate_fast"), aux_layout="per_neuron", scope="node",
            h_bounds=(0.1, 10.0), b_bounds=(-50.0, 50.0), default_params=dict(DEFAULT_PARAMS),
        ),
        _step,
    )


register()

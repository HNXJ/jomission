"""Activity-balance H rule (registered with jaxfne's HDP rule surface).

Physical idea (owner, 2026-11-04): H is a decaying history of activity. Near baseline it stays
about 1.0. Outside the 1 to 10 Hz band it moves further from 1.0, and the feedback grows with
the distance from the band. H acts as an additive input current (drive_bias), so hypoactivity
receives extra input and hyperactivity is suppressed. The connectome is not changed.

Equations (time in ms, rates in Hz; dt is the kernel step):
  r_inst(t)   = spikes_t * 1000 / dt                       (impulse rate)
  tau_r  dr/dt = r_inst - r                                 (activity trace, aux "rate")
  delta       = max(0, r - r_hi) - max(0, r_lo - r)        (signed deviation outside the band)
  tau_H  dH/dt = (1 - H) - kappa * delta * (tau_H / 1000)   (H history with restoring term)
  db/dt       = kappa_I * dH/dt                             (drive_bias; b = kappa_I (H - 1) in exact arithmetic; float32 drift <= 5.7e-3 over 2000 ms)

Units of kappa: 1/(Hz * s). With the defaults, a neuron at 22 Hz settles near H = 0.4 and
a silent neuron near H = 1.05. Parameters are PROPOSED AND UNTUNED; they are fixed in the spec
that is sealed before any run.
"""
from __future__ import annotations

import jax.numpy as jnp

from jaxfne.hdp_rule import (
    HDPRuleContext,
    HDPRuleDescriptor,
    HDPRuleUpdate,
    is_registered_hdp_rule,
    register_hdp_rule,
)

RULE_NAME = "activity_balance"

DEFAULT_PARAMS = {
    "r_lo": 1.0,        # Hz, lower edge of the balance band (owner)
    "r_hi": 10.0,       # Hz, upper edge of the balance band (owner)
    "tau_r": 1000.0,    # ms, activity trace
    "tau_H": 1000.0,    # ms, H history
    "kappa": 0.05,      # 1/(Hz s), gain of the deviation term (proposed, untuned)
    "kappa_I": 5.0,     # current units per unit H (proposed, untuned)
}


def _activity_balance_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    p = ctx.rule_params
    r_lo = jnp.asarray(p["r_lo"], dtype=ctx.H.dtype)
    r_hi = jnp.asarray(p["r_hi"], dtype=ctx.H.dtype)
    tau_r = jnp.asarray(p["tau_r"], dtype=ctx.H.dtype)
    tau_H = jnp.asarray(p["tau_H"], dtype=ctx.H.dtype)
    kappa = jnp.asarray(p["kappa"], dtype=ctx.H.dtype)
    kappa_I = jnp.asarray(p["kappa_I"], dtype=ctx.H.dtype)

    r = ctx.aux                                              # Hz, per neuron
    r_inst = ctx.spikes.astype(ctx.H.dtype) * (1000.0 / ctx.dt)
    d_r = (r_inst - r) / tau_r                               # per ms
    delta = jnp.maximum(0.0, r - r_hi) - jnp.maximum(0.0, r_lo - r)
    dH = (1.0 - ctx.H) / tau_H - kappa * delta * (1.0 / 1000.0)   # per ms
    d_b = kappa_I * dH                                       # b = kappa_I (H - 1) in exact arithmetic
    return HDPRuleUpdate(dH=dH, d_aux=d_r, d_theta={"drive_bias": d_b})


def register() -> None:
    """Idempotent registration (safe to call from several modules)."""
    if is_registered_hdp_rule(RULE_NAME):
        return
    register_hdp_rule(
        HDPRuleDescriptor(
            name=RULE_NAME,
            h_coords=("H",),
            theta_targets=("drive_bias",),
            aux_coords=("rate",),
            aux_layout="per_neuron",
            scope="node",
            h_bounds=(0.1, 10.0),
            b_bounds=(-50.0, 50.0),
            default_params=dict(DEFAULT_PARAMS),
        ),
        _activity_balance_step,
    )


register()

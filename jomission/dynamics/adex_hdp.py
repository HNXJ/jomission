"""AdEx-like spike-frequency adaptation as a registered JaxFNE HDP rule.

AdEx (Tahvili 2025 eq. 1-2, 2026 eq. 15-16):
    tau_w dw/dt = a (V - E_L) - w + b sum_j delta(t - t_j),   I_adapt = -w.
The rule carries w as the per-neuron additive current ``drive_bias`` (native
current units, theta target of the registrable kernel), bias = -w:
    d bias/dt = -(a_s (v - E_L) + bias) / tau_w      (rate, dt in ms)
    step jump = -b per spike (written as rate -b*spike/dt; 0.5.8 has no jumps)
so the kernel's ``current = drive + sched + bias + syn + noise`` receives -w.
Per-type adaptation (PV none, E and SST on) is set with per-neuron arrays in
``a_s`` and ``b`` (zero entries switch adaptation off).

Units: v, E_L in mV, tau_w in ms; ``a_s`` and ``b`` in native current units
(``a_s`` per mV). The native-to-pA factor is a design choice recorded by the
caller; nothing here fixes it.

The rule touches only ``drive_bias``: no H, no edge weights.
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

RULE_NAME = "adex_adaptation_drive_bias"
# Wide enough that the bias never clips in the operating range; the clip is
# still a hard bound, checked by tests (item 9e).
B_BOUNDS = (-200.0, 200.0)


def _adex_adaptation_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    p = ctx.rule_params
    dtype = ctx.H.dtype
    a_s = jnp.asarray(p["a_s"], dtype=dtype)
    b = jnp.asarray(p["b"], dtype=dtype)
    tau_w = jnp.asarray(p["tau_w_ms"], dtype=dtype)
    e_l = jnp.asarray(p["E_L"], dtype=dtype)
    # The bias is not in the rule context; it is recovered by the kernel as the
    # integral of d_theta. The ODE needs the current bias, carried in aux.
    bias = ctx.aux
    d_bias = -(a_s * (ctx.v - e_l) + bias) / tau_w
    # JaxFNE 0.5.8 has no per-event jumps (jH/j_theta arrived in 0.5.9). The
    # spike term is a rate -b * spike / dt: the kernel integrates dt * rate, so
    # the step adds exactly -b per spike at every dt (same result as a jump).
    spike_rate = -b * ctx.spikes.astype(dtype) / ctx.dt
    d_total = d_bias + spike_rate
    return HDPRuleUpdate(
        dH=jnp.zeros_like(ctx.H),
        d_aux=d_total,
        d_theta={"drive_bias": d_total},
    )


def register() -> str:
    """Register the rule once; return its name."""
    if not is_registered_hdp_rule(RULE_NAME):
        register_hdp_rule(
            HDPRuleDescriptor(
                name=RULE_NAME,
                h_coords=("H",),
                theta_targets=("drive_bias",),
                aux_layout="per_neuron",
                aux_coords=("bias_mirror",),
                scope="node",
                b_bounds=B_BOUNDS,
                default_params={"a_s": 0.0, "b": 0.0, "tau_w_ms": 500.0, "E_L": -65.0},
            ),
            _adex_adaptation_step,
        )
    return RULE_NAME

"""Phase-2 native HDP transformation bridge: minimum Jomission-owned rule.

Rule ``jomission_sat_ee_v0`` (public ADVANCED registration surface only):

- per-edge activity trace ``act`` (aux, event-driven, decay ``tau_act_ms``);
- slow H carried near idle (``k_h=0``, ``gamma_h`` relaxes to 1; H_post gates
  the drive in form, ~1 in practice by predeclared default params);
- reversible saturating EE efficacy drive::

    dw = k_w * H_post * (act^2 * (w_hi - w) - lam * (w - w_lo))

  steady state at fixed act: ``w* = (act^2*w_hi + lam*w_lo)/(act^2 + lam)``,
  i.e. low rest efficacy, steep intermediate recruitment, bounded high state.

Reference geometry (NOT a fitted target): sealed reduced attractor
``(rE,rI)=(20.08,5.53)``, Hill ``Wb=20/th=5/n=4``, lower separatrix
``r_u in (2,4)``, upper ``(25,35)`` (results/attractor_solve.json).

Predeclared rule params (frozen before execution; no post-hoc tuning):
``tau_act_ms=20, lam=0.09, w_lo=0.002, w_hi=0.05, k_w=0.2, k_h=0.0,
gamma_h=0.02``; ``w_bounds=(1e-3, 0.1)``.
"""

from __future__ import annotations

from typing import Any

RULE_NAME = "jomission_sat_ee_v0"

RULE_PARAMS: dict[str, float] = {
    "k_h": 0.0,
    "gamma_h": 0.02,
    "tau_act_ms": 20.0,
    "k_w": 0.2,
    "w_lo": 0.002,
    "w_hi": 0.05,
    "lam": 0.09,
}

W_BOUNDS = (1e-3, 0.1)

# Predeclared probe rate bands (Hz driver rate; drive is only the knob).
PROBE_BANDS: dict[str, tuple[float, float]] = {
    "rest": (0.0, 1.0),
    "below": (1.0, 4.0),
    "transition": (4.0, 12.0),
    "active": (12.0, 25.0),
    "above": (25.0, 45.0),
}

PROBE_DRIVES: dict[str, float] = {
    "rest": 0.0,
    "below": 2.5,
    "transition": 4.5,
    "active": 8.0,
    "above": 11.0,
}

# Predeclared PASS bands for the measured transformation.
# Instrument correction (documented 2026-09-14): the target-Vm proxy sits at
# a ~1e-3 mV floor across all probes (single-edge currents real but tiny in
# Vm terms -- sealed current-ineffective lesson), so current is computed
# EXACTLY offline as w_e(t)*syn_e(t) from recorded spikes + recorded w, plus
# a causal on/off ratio at matched drive. Thresholds from K-map scale
# (I_active ~1e-3 estimated), set conservatively before seeing currents.
PASS_BANDS: dict[str, Any] = {
    "w_rest_max": 0.005,
    "w_trans_over_below_min": 3.0,
    "w_active_lo": 0.015,
    "w_active_hi": 0.05,
    "w_above_hi": 0.05,
    "w_above_over_active_max": 1.5,
    "I_active_min": 5e-4,
    "I_on_over_off_min": 3.0,
    "w_post_max": 0.008,
}


def _step_fn(ctx: Any) -> Any:
    import jax.numpy as jnp
    from jaxfne.hdp_rule import HDPRuleUpdate

    p = ctx.rule_params
    k_h = jnp.asarray(p.get("k_h", 0.0), dtype=ctx.H.dtype)
    gamma_h = jnp.asarray(p.get("gamma_h", 0.0), dtype=ctx.H.dtype)
    tau_act = jnp.asarray(p.get("tau_act_ms", 20.0), dtype=ctx.H.dtype)
    k_w = jnp.asarray(p.get("k_w", 0.0), dtype=ctx.H.dtype)
    w_lo = jnp.asarray(p.get("w_lo", 0.002), dtype=ctx.H.dtype)
    w_hi = jnp.asarray(p.get("w_hi", 0.05), dtype=ctx.H.dtype)
    lam = jnp.asarray(p.get("lam", 0.09), dtype=ctx.H.dtype)

    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    # NOTE (bug fix 2026-09-14): the kernel integrates aux + dt*d_aux with dt
    # in ms, so a bare +pre_sp term contributes dt (0.1) per spike, not 1.
    # Divide by dt so one spike adds 1 and E_ss = r*tau_act/1000 as designed.
    # Params unchanged; this restores the predeclared steady state.
    dt_safe = jnp.maximum(ctx.dt, jnp.asarray(1e-9, dtype=ctx.H.dtype))
    d_act = -ctx.aux / jnp.maximum(tau_act, jnp.asarray(1e-6, dtype=ctx.H.dtype))
    d_act = d_act + pre_sp / dt_safe
    dH = -gamma_h * (ctx.H - 1.0)
    # postsynaptic event sum via plain indexed-add (no private engine import);
    # predeclared k_h=0 keeps H idle, H_post~1 gates the drive in form.
    post_drive = jnp.zeros((ctx.n_neurons,), dtype=ctx.H.dtype).at[ctx.post].add(k_h * pre_sp)
    dH = dH + post_drive
    H_post = ctx.H[ctx.post]
    wmag = jnp.abs(ctx.w)
    act2 = ctx.aux * ctx.aux
    dw = k_w * H_post * (act2 * (w_hi - wmag) - lam * (wmag - w_lo))
    return HDPRuleUpdate(dH=dH, d_aux=d_act, d_theta={"edge_weight": dw})


def ensure_registered() -> str:
    """Register the rule once via the public ADVANCED surface; return its name."""
    from jaxfne.hdp_rule import (
        HDPRuleDescriptor,
        is_registered_hdp_rule,
        register_hdp_rule,
    )

    if not is_registered_hdp_rule(RULE_NAME):
        register_hdp_rule(
            HDPRuleDescriptor(
                name=RULE_NAME,
                h_coords=("H",),
                theta_targets=("edge_weight",),
                aux_coords=("act",),
                aux_layout="per_edge",
                scope="node",
                h_bounds=(0.1, 10.0),
                w_bounds=W_BOUNDS,
                default_params=dict(RULE_PARAMS),
            ),
            _step_fn,
        )
    return RULE_NAME

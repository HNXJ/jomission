"""S6 selective native rule: one pathway-selection bit on the qualified form.

Rule ``jomission_sat_eout_v0`` (public ADVANCED registration surface only).
Only scientific difference from ``jomission_sat_ee_v0`` (Phase 2): the
saturating efficacy drive actuates with E-pathway parameters on
presynaptic-E edges and with the original s=1 parameters on presynaptic
inhibitory edges, selected per-edge by the engine-provided ``exc_mask``
(``ri == 0`` in the registrable kernel: True iff presynaptic E)::

    w_hi_e = where(exc, s_E * w_hi, w_hi)      w_lo_e = where(exc, s_E * w_lo, w_lo)
    dw = k_w * H_post * (act^2 * (w_hi_e - wmag) - lam * (wmag - w_lo_e))

- Hidden-state dynamics, aux dynamics, shape params (tau_act, lam, k_w),
  H bounds, cells, tonic, homeostatic config: unchanged.
- Inhibitory-origin edges see arithmetically the original rule (same form,
  same s=1 params, untouched aux/H path) -> "existing measured efficacy".
- No target-class-specific gains: EE, E->PV, E->SST share one common
  E-out transformation by construction (selection is presynaptic only).

Predeclared bracket (from the inverse feasible region [54.1, 60.8];
interior + one flank each side to verify reduced-to-native transfer)::

    s_E in {50, 54, 57, 61, 65}

Engine bound: descriptor hi = 0.1 * s_E (max 6.5 at s_E=65) << kernel
ceiling 50.0. No geometry in S6 (explicit scope exclusion).
"""

from __future__ import annotations

from typing import Any

from jomission.qualification.hdp_rule_sat_ee import RULE_PARAMS as V0

RULE_NAME = "jomission_sat_eout_v0"

S_BRACKET: tuple[float, ...] = (50.0, 54.0, 57.0, 61.0, 65.0)

S_MAX_ENGINE = 500.0

assert max(S_BRACKET) <= S_MAX_ENGINE


def scale_tag(s: float) -> str:
    return ("%.1f" % s).replace(".", "p")


def scaled_rule_name(s: float) -> str:
    return f"{RULE_NAME}_s{scale_tag(s)}"


def scaled_params(s: float) -> dict[str, float]:
    p = dict(V0)
    p["w_lo_E"] = V0["w_lo"] * s
    p["w_hi_E"] = V0["w_hi"] * s
    p["w_lo_I"] = V0["w_lo"]
    p["w_hi_I"] = V0["w_hi"]
    return p


def scaled_w_bounds(s: float) -> tuple[float, float]:
    return (1e-3, 0.1 * s)


def scaled_w_base_E(s: float) -> float:
    return V0["w_lo"] * s


W_BASE_I = V0["w_lo"]


def _step_fn(ctx: Any) -> Any:
    import jax.numpy as jnp
    from jaxfne.hdp_rule import HDPRuleUpdate

    p = ctx.rule_params
    k_h = jnp.asarray(p.get("k_h", 0.0), dtype=ctx.H.dtype)
    gamma_h = jnp.asarray(p.get("gamma_h", 0.0), dtype=ctx.H.dtype)
    tau_act = jnp.asarray(p.get("tau_act_ms", 20.0), dtype=ctx.H.dtype)
    k_w = jnp.asarray(p.get("k_w", 0.0), dtype=ctx.H.dtype)
    lam = jnp.asarray(p.get("lam", 0.09), dtype=ctx.H.dtype)
    exc = ctx.exc_mask
    w_hi = jnp.where(exc, jnp.asarray(p.get("w_hi_E"), dtype=ctx.H.dtype),
                     jnp.asarray(p.get("w_hi_I"), dtype=ctx.H.dtype))
    w_lo = jnp.where(exc, jnp.asarray(p.get("w_lo_E"), dtype=ctx.H.dtype),
                     jnp.asarray(p.get("w_lo_I"), dtype=ctx.H.dtype))

    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    # Same dt-corrected aux integration as v0 (one spike adds 1).
    dt_safe = jnp.maximum(ctx.dt, jnp.asarray(1e-9, dtype=ctx.H.dtype))
    d_act = -ctx.aux / jnp.maximum(tau_act, jnp.asarray(1e-6, dtype=ctx.H.dtype))
    d_act = d_act + pre_sp / dt_safe
    dH = -gamma_h * (ctx.H - 1.0)
    post_drive = jnp.zeros((ctx.n_neurons,), dtype=ctx.H.dtype).at[ctx.post].add(k_h * pre_sp)
    dH = dH + post_drive
    H_post = ctx.H[ctx.post]
    wmag = jnp.abs(ctx.w)
    act2 = ctx.aux * ctx.aux
    dw = k_w * H_post * (act2 * (w_hi - wmag) - lam * (wmag - w_lo))
    return HDPRuleUpdate(dH=dH, d_aux=d_act, d_theta={"edge_weight": dw})


def ensure_registered_scale(s: float) -> str:
    """Register the selective rule instance once; return its name."""
    from jaxfne.hdp_rule import (
        HDPRuleDescriptor,
        is_registered_hdp_rule,
        register_hdp_rule,
    )

    name = scaled_rule_name(s)
    if not is_registered_hdp_rule(name):
        register_hdp_rule(
            HDPRuleDescriptor(
                name=name,
                h_coords=("H",),
                theta_targets=("edge_weight",),
                aux_coords=("act",),
                aux_layout="per_edge",
                scope="node",
                h_bounds=(0.1, 10.0),
                w_bounds=scaled_w_bounds(s),
                default_params=scaled_params(s),
            ),
            _step_fn,
        )
    return name

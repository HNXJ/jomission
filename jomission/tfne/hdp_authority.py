"""HDP rule `jomission_authority_v0`: outgoing efficacy scaled by the source's own H.

Written because the engine does not implement the intended semantics. The two shipped families
were both checked first:

    signed_linear     dm_E/dt = +K_HDP*(H_post - H_pre)*m_E
                      keys on the pre/post difference, so outward efficacy rises when the TARGET
                      is better resourced. That is not "activity in one ensemble increases its
                      authority over downstream ensembles".

    hebbian_product   dm_E/dt = +K_HDP*H_pre*H_post*m_E
                      includes the source's H but carries no negative term, so excitatory weights
                      grow exponentially to w_ceiling for any H > 0 and inhibitory weights decay
                      to w_floor. Measured: mean w 0.225 -> 10.83 against a ceiling of 10.0 in
                      2.4 s. It saturates rather than redistributes.

This rule instead relaxes each edge toward a target set by the SOURCE's state:

    w_target_e = w_base_e * m(H_pre(e)),     m(H) = 2H / (1 + H)
    dw_e/dt    = k_relax * (w_target_e - w_e)

m is neutral and equal to 1 at H = 1, rises toward a hard ceiling of 2 as H grows, and falls to 0
as H approaches 0. Efficacy is therefore bounded by construction: w_e cannot leave
[0, 2*w_base_e] however long the run, which is the property hebbian_product lacks.

H dynamics reproduce the engine kernel (jaxfne emitters.py:3201) term for term, so the arms differ
only in the weight rule:

    C(H)      = barrier_c/(H - H_min) + barrier_d/(H_max - H)
    tau*dH/dt = alpha*I_syn + beta - gamma*H*r - delta*W + rho_passive/H**2 - dC/dH

Spending stays in the kernel's currency, gamma*H*r with r the per-step spike indicator, because
that is the form the reviewer's gamma was derived against. It is not dt-independent, and that is
recorded rather than silently repaired.

w_base is latched per edge into aux on the first step from the constructed weights, so the target
is anchored to the realized architecture instead of to a constant assumed here.
"""

from __future__ import annotations

from typing import Any

RULE_NAME = "jomission_authority_v0"

H_MIN, H_MAX = 0.1, 10.0
W_BOUNDS = (0.0, 100.0)          # wide: m caps the target at 2*w_base, so this only guards runaway

#: m(H) = M_SAT*H/(1+H). M_SAT = 2.0 puts m(1) = 1 exactly, so H = 1 is efficacy-neutral.
M_SAT = 2.0

RULE_PARAMS: dict[str, float] = {
    "alpha": 0.05,
    "beta": 0.0,
    "gamma": 247.461,
    "delta": 0.0,
    "rho_passive": 0.02,
    "barrier_c": 0.01,
    "barrier_d": 1.0,
    "barrier_eps": 1e-3,
    "tau_0_ms": 5.0,
    "k_relax": 0.01,
    "m_sat": M_SAT,
}


def m_of_H(H, m_sat=M_SAT):
    """Efficacy multiplier. Pure function, so the shape can be checked without the engine."""
    return m_sat * H / (1.0 + H)


def _step_fn(ctx: Any) -> Any:
    import jax.numpy as jnp
    from jaxfne.hdp_rule import HDPRuleUpdate

    p = ctx.rule_params
    dt = ctx.H.dtype

    def par(name):
        return jnp.asarray(p.get(name, RULE_PARAMS.get(name, 0.0)), dtype=dt)

    alpha, beta, gamma, delta = par("alpha"), par("beta"), par("gamma"), par("delta")
    rho, bc, bd, beps = par("rho_passive"), par("barrier_c"), par("barrier_d"), par("barrier_eps")
    tau = jnp.maximum(par("tau_0_ms"), jnp.asarray(1e-6, dtype=dt))
    k_relax, m_sat = par("k_relax"), par("m_sat")
    h_min = jnp.asarray(H_MIN, dtype=dt)
    h_max = jnp.asarray(H_MAX, dtype=dt)

    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    wmag = jnp.abs(ctx.w)
    n = ctx.n_neurons

    # I_syn_i = sum_j w_ji x_j, incoming; W_i = sum_j |w_ij|, i's own outgoing burden.
    I_syn = jnp.zeros((n,), dtype=dt).at[ctx.post].add(ctx.w * pre_sp)
    W_out = jnp.zeros((n,), dtype=dt).at[ctx.pre].add(wmag)

    # -dC/dH for C(H) = bc/(H - H_min) + bd/(H_max - H), denominators floored by barrier_eps.
    lo = jnp.maximum(ctx.H - h_min, beps)
    hi = jnp.maximum(h_max - ctx.H, beps)
    minus_dCdH = bc / (lo * lo) - bd / (hi * hi)

    r = ctx.prev_spikes if ctx.prev_spikes is not None else ctx.spikes
    H_safe = jnp.maximum(ctx.H, beps)
    dH = (alpha * I_syn + beta - gamma * ctx.H * r - delta * W_out
          + rho / (H_safe * H_safe) + minus_dCdH) / tau

    # aux holds w_base per edge, latched once from the constructed weights.
    latched = ctx.aux > 0.0
    w_base = jnp.where(latched, ctx.aux, wmag)
    dt_safe = jnp.maximum(ctx.dt, jnp.asarray(1e-9, dtype=dt))
    d_aux = jnp.where(latched, 0.0, (wmag - ctx.aux) / dt_safe)

    target = w_base * m_of_H(ctx.H[ctx.pre], m_sat)
    dw = k_relax * (target - wmag) * jnp.sign(ctx.w)
    return HDPRuleUpdate(dH=dH, d_aux=d_aux, d_theta={"edge_weight": dw})


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
                aux_coords=("w_base",),
                aux_layout="per_edge",
                scope="node",
                h_bounds=(H_MIN, H_MAX),
                w_bounds=W_BOUNDS,
                default_params=dict(RULE_PARAMS),
            ),
            _step_fn,
        )
    return RULE_NAME

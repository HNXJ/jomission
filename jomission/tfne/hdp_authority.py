"""HDP rule `jomission_authority_v3`: outgoing efficacy scaled by the source's own H.

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

H dynamics follow the engine kernel (jaxfne emitters.py:3201) term for term, with two
corrections, each of which was measured before and after:

    C(H)      = barrier_c/(H - H_min) + barrier_d/(H_max - H)
    tau*dH/dt = alpha*I_net + beta - gamma*H*A - delta*W + rho_passive/H**2 - dC/dH

INCOME (v2). I_net is the signed native current that actually enters the Izhikevich update,
not a spike count. v1 used sum_j w_ji * spike_j, the raw per-step indicator; the engine's own
income is the decayed per-edge trace:

    edge_current = w * syn_state ; syn = segment_sum(edge_current, post)   (kernel :258-260)

Measured at the same operating point, v1's quantity was 28.5x too small for E and carried the
WRONG SIGN for PV and VIP, whose inhibition dominates. With the correct variable the income term
also stops being inert: it was 0.14% of E's budget under v1.

    I_net = I_tonic + I_syn      signed; inhibition carries w < 0 and subtracts

Scope boundary. A registered rule cannot see three of the kernel's four current terms. Tonic is
injected per neuron through hdp_rule_params (see tonic_params). The schedule term is exactly zero
for cortical neurons, because the diagnostic drives only retinal units. The noise term,
noise_coef * N(0,1) with noise_coef = 0.5, is NOT represented in H's income: it is zero-mean but
its per-step sd is 7.9x E's mean I_syn. That omission is declared, not approximated away.

SPENDING. A low-pass filtered activity coordinate, not the per-step spike indicator. The
indicator form was tried first and failed measurably: with gamma balanced on the time-average
(247.461), one spike removes gamma*H*r*dt/tau = 4.95 units of H from a nominal 1.0 while income
restores 0.005 per step, so a single spike takes 99 ms to undo at a 91.7 ms mean inter-spike
interval. H spanned its whole range 0.1 to 10.0 within 1 ms.

The state therefore carries a second coordinate A, a filtered firing rate normalised so that A = 1
at the HDP-off baseline rate, which makes gamma dimensionless. At the reference point H = A = 1
the barrier contributes zero (barrier_d/barrier_c = 100 centres it there), so

    gamma = alpha*I_net + rho_passive = 0.05*5.0637 + 0.02 = 0.27320

H = 1 is the reference the balance is struck at, NOT a forced setpoint: a neuron whose I_net or
activity differs from the reference settles wherever its own budget closes.

ACTIVITY FLOOR (v3). A's fixed point is A* = r_filtered/rate_ref, so a silent neuron must reach
A = 0. Under v2 the kernel's shared floor of 0.1 clipped it: PV firing at 0.27 Hz has A* = 0.025
but read A = 0.1000, so it was charged four times its own activity and settled at H = 4.40. The
floor is now 0.0 and H's floor moved into the rule, which is identically equivalent and leaves
alpha, gamma, the restore and barrier terms, m(H), W and the connectivity untouched.

w_base is latched per edge into aux on the first step from the constructed weights, so the target
is anchored to the realized architecture instead of to a constant assumed here.
"""

from __future__ import annotations

from typing import Any

RULE_NAME = "jomission_authority_v3"

H_MIN, H_MAX = 0.1, 10.0

#: A is a normalised rate and must reach 0 when the neuron is silent. The kernel applies ONE
#: h_bounds to the whole state vector, and register_hdp_rule validates each bound with float(),
#: so a per-coordinate (0.1, 0.0) cannot be declared. The floor is therefore 0.0 for both
#: coordinates and H's own floor is enforced inside the rule, on its derivative: requiring
#: dH >= (H_MIN - H)/dt makes H + dt*dH >= H_MIN identically, so the kernel's clip is a no-op
#: for H. A_MAX is the kernel's shared ceiling, kept only as numerical protection.
STATE_FLOOR = 0.0
A_MAX = H_MAX
W_BOUNDS = (0.0, 100.0)          # wide: m caps the target at 2*w_base, so this only guards runaway

#: m(H) = M_SAT*H/(1+H). M_SAT = 2.0 puts m(1) = 1 exactly, so H = 1 is efficacy-neutral.
M_SAT = 2.0

#: Filtered-rate reference: the HDP-off baseline E rate in spikes per ms, so A = 1 there.
RATE_REF_PER_MS = 0.0109089

RULE_PARAMS: dict[str, float] = {
    "alpha": 0.05,
    "beta": 0.0,
    "gamma": 0.27320,
    "delta": 0.0,
    "rho_passive": 0.02,
    "barrier_c": 0.01,
    "barrier_d": 1.0,
    "barrier_eps": 1e-3,
    "tau_0_ms": 5.0,
    "tau_act_ms": 500.0,
    "rate_ref_per_ms": RATE_REF_PER_MS,
    "k_relax": 0.01,
    "m_sat": M_SAT,
    "tonic_drive": 0.0,   # driver injects params.drive as a per-neuron array
}


def tonic_params(model):
    """Per-neuron tonic income for hdp_rule_params, read from the realized model.

    The kernel adds params.drive to current_native but does not pass it to a registered rule,
    so H would otherwise see only the synaptic part of its own income.
    """
    import numpy as np

    return {"tonic_drive": np.asarray(model.params["emitter"].drive)}


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
    tau_act = jnp.maximum(par("tau_act_ms"), jnp.asarray(1e-6, dtype=dt))
    rate_ref = jnp.maximum(par("rate_ref_per_ms"), jnp.asarray(1e-12, dtype=dt))
    rho, bc, bd, beps = par("rho_passive"), par("barrier_c"), par("barrier_d"), par("barrier_eps")
    tau = jnp.maximum(par("tau_0_ms"), jnp.asarray(1e-6, dtype=dt))
    k_relax, m_sat = par("k_relax"), par("m_sat")
    h_min = jnp.asarray(H_MIN, dtype=dt)
    h_max = jnp.asarray(H_MAX, dtype=dt)

    wmag = jnp.abs(ctx.w)
    n = ctx.n_neurons

    # The state vector is (H, A): resource and normalised filtered rate.
    H = ctx.H[:, 0]
    A = ctx.H[:, 1]

    # I_syn is the engine's own per-neuron signed synaptic current, reproduced term for term
    # from _hdp_registrable_kernel.py:258-260:
    #     edge_current = w * syn_state ; syn = segment_sum(edge_current, post)
    # syn_state is the decayed per-edge trace, NOT the raw spike indicator. Inhibition carries
    # w < 0 and therefore subtracts, so a neuron whose inhibition dominates has I_syn < 0.
    I_syn = jnp.zeros((n,), dtype=dt).at[ctx.post].add(ctx.w * ctx.syn_state)
    # Tonic is invisible to a registered rule, so the driver injects params.drive per neuron.
    I_tonic = jnp.asarray(p.get("tonic_drive", 0.0), dtype=dt)
    I_net = I_tonic + I_syn
    W_out = jnp.zeros((n,), dtype=dt).at[ctx.pre].add(wmag)

    # -dC/dH for C(H) = bc/(H - H_min) + bd/(H_max - H), denominators floored by barrier_eps.
    lo = jnp.maximum(H - h_min, beps)
    hi = jnp.maximum(h_max - H, beps)
    minus_dCdH = bc / (lo * lo) - bd / (hi * hi)

    r = ctx.prev_spikes if ctx.prev_spikes is not None else ctx.spikes
    dt_ms = jnp.maximum(ctx.dt, jnp.asarray(1e-9, dtype=dt))
    dA = (-A + (r / dt_ms) / rate_ref) / tau_act
    H_safe = jnp.maximum(H, beps)
    dH = (alpha * I_net + beta - gamma * H * A - delta * W_out
          + rho / (H_safe * H_safe) + minus_dCdH) / tau
    # H's lower bound, enforced here rather than by the kernel's shared clip (see STATE_FLOOR).
    dH = jnp.maximum(dH, (h_min - H) / dt_ms)
    dstate = jnp.stack([dH, dA], axis=1)

    # aux holds w_base per edge, latched once from the constructed weights.
    latched = ctx.aux > 0.0
    w_base = jnp.where(latched, ctx.aux, wmag)
    d_aux = jnp.where(latched, 0.0, (wmag - ctx.aux) / dt_ms)

    target = w_base * m_of_H(H[ctx.pre], m_sat)
    # The kernel integrates this on the MAGNITUDE (wmag + dt*dw) and restores the sign from
    # exc_mask, so dw must carry no sign of its own. Multiplying by sign(w) here negated the
    # relaxation on every inhibitory edge and made |w| diverge from its target.
    dw = k_relax * (target - wmag)
    return HDPRuleUpdate(dH=dstate, d_aux=d_aux, d_theta={"edge_weight": dw})


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
                h_coords=("H", "A"),
                h_shape=(2,),
                theta_targets=("edge_weight",),
                aux_coords=("w_base",),
                aux_layout="per_edge",
                scope="node",
                h_bounds=(STATE_FLOOR, H_MAX),
                w_bounds=W_BOUNDS,
                default_params=dict(RULE_PARAMS),
            ),
            _step_fn,
        )
    return RULE_NAME

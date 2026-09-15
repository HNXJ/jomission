"""HDP efficacy-scale lineage: one scalar amplitude delta on the qualified rule.

Scale definition (single delta; everything else frozen)::

    w_eff(H; s) = s * w_eff(H; 1)

Mechanically: rule ``jomission_sat_ee_v0`` form, hidden-state dynamics
(``k_h=0``, ``gamma_h``), shape params (``tau_act``, ``lam``, ``k_w``),
cell/inhibition/tonic/homeostatic configuration all unchanged. Only the
efficacy operating point (``w_lo``, ``w_hi``, descriptor ``w_bounds``,
probe-plant ``w_base``) is multiplied by ``s``. With ``u = w/s`` the
``u``-dynamics are then identical to the ``s=1`` qualification, so the
low-rest -> steep-mid -> bounded-high shape is preserved by construction
and verified by requalification (not assumed).

Per-scale registration names (``register_hdp_rule`` rejects re-registration;
the step function is shared unchanged)::

    jomission_sat_ee_s2 / s4p5 / s9 / s18 / s36

Predeclared bracket (frozen before execution; see results/hdp_scale_bracket.json
for the sealed derivation from measured K, rheobases, Phase-3A currents)::

    s in {2.0, 4.5, 9.0, 18.0, 36.0}   (below / near / above s~4.44)

Admissible upper bound (JaxFNE constraint, before outcomes): the registrable
kernel clips efficacy magnitude to the descriptor ``w_bounds`` hi, and the
engine absolute ceiling is ``w_ceiling = 50.0``. Scaled descriptor hi is
``0.1 * s``; ``0.1 * s <= 50`` gives ``s <= 500``. All bracket points lie
well inside. No Hill refit, no convergence repair, no tonic, no attractor
selection: the estimand is strictly whether absolute scaling creates viable
E/I geometry.

Stop rule (ascending; predeclared): stop at the first PASS (freeze the
minimum qualified scale, confirmed interior by exactly one above-bracket
run where available; PASS at the bracket maximum 36 with no above point is
a BOUNDARY, not a freeze), first PATHOLOGICAL, else exhaust the bracket
(SCALE_INSUFFICIENT).
"""

from __future__ import annotations

from jomission.qualification.hdp_rule_sat_ee import (
    RULE_PARAMS,
    W_BOUNDS,
    _step_fn,
)

SCALE_BRACKET: tuple[float, ...] = (2.0, 4.5, 9.0, 18.0, 36.0)

# Engine absolute efficacy ceiling (jaxfne registrable kernel / w_ceiling).
S_MAX_ENGINE = 500.0

assert max(SCALE_BRACKET) <= S_MAX_ENGINE


def scale_tag(s: float) -> str:
    return ("%.1f" % s).replace(".", "p")


def scaled_rule_name(s: float) -> str:
    return f"jomission_sat_ee_s{scale_tag(s)}"


def scaled_params(s: float) -> dict[str, float]:
    p = dict(RULE_PARAMS)
    p["w_lo"] = RULE_PARAMS["w_lo"] * s
    p["w_hi"] = RULE_PARAMS["w_hi"] * s
    return p


def scaled_w_bounds(s: float) -> tuple[float, float]:
    return (W_BOUNDS[0] * s, W_BOUNDS[1] * s)


def scaled_w_base(s: float) -> float:
    return RULE_PARAMS["w_lo"] * s


def ensure_registered_scale(s: float) -> str:
    """Register the scaled rule instance once; return its name."""
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

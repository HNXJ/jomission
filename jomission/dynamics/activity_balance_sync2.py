"""Balance + corrected synchrony term (spec: docs/designs/H_SYNC2_TEST_spec.md).

sigma = max(0, P_f - (1 + g) P_s) / r_ref with the constant r_ref = r_hi (owner, Q2): bounded at start-up
where the slow trace P_s is about 0. The step is the fixed-reference step of activity_balance_sync_fixref
(one home for the equations); only the default reference differs (parameter name `p_ref`, Hz).
"""
from __future__ import annotations

from jaxfne.hdp_rule import HDPRuleDescriptor, is_registered_hdp_rule, register_hdp_rule

from jomission.dynamics.activity_balance import DEFAULT_PARAMS as BALANCE_PARAMS
from jomission.dynamics.activity_balance_sync_fixref import DEFAULT_PARAMS as FIXREF_PARAMS
from jomission.dynamics.activity_balance_sync_fixref import _step

RULE_NAME = "activity_balance_sync2"
DEFAULT_PARAMS = {**FIXREF_PARAMS, "p_ref": BALANCE_PARAMS["r_hi"]}  # Hz


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

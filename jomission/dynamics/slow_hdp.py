"""Slow plasticity timescale intervention — tau_0 5→1000 (200×) rate-only, fixed points preserved."""

import jaxfne.hdp_network as hdp

def slow_hdp_params():
    hp = dict(hdp.v1_pfc_aaab_hdp_params())
    hp["tau_0_ms"] = 1000.0  # 5*200
    # K_HDP and K_w_ctrl unchanged to preserve ratio and fixed points
    return hp

SLOW_HP_HASH = "slow_200x"  # placeholder, actual hash computed via config

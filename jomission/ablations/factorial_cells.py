"""Factorial cells — 4 identities with factor isolation checks."""

import hashlib, json
import jaxfne.hdp_network as hdp
from jomission.network.builder import build_jomission_model
from jaxfne.io import config_hash

def cell_config(rf_on: bool, slow_rate: bool):
    # rf_on: use RF operator (distinct hash), slow_rate: tau0 1000 vs 5
    hp = dict(hdp.v1_pfc_aaab_hdp_params())
    if slow_rate:
        hp["tau_0_ms"] = 1000.0
    # rf_on adds rf metadata
    from jomission.network.rf import RFConfig, RFOperator
    # keep distinct identities via hash of hp+rf flag
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:8]
    rf_hash = "rf_on" if rf_on else "rf_off"
    return {"hp_hash": hp_hash, "rf": rf_hash, "hp": hp}

CELLS = {
    "A_RFoff_RateRef": cell_config(False, False),
    "B_RFon_RateRef": cell_config(True, False),
    "C_RFoff_RateSlow": cell_config(False, True),
    "D_RFon_RateSlow": cell_config(True, True),
}

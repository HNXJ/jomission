"""Recording — C_t -> q_t -> φ_t -> y_t via JaxFNE source/readout.

Exposes SPK, MUAe-like, LFP-like, CSD-like where JaxFNE supports it.
All field-derived modes are proxy readouts, not physical measurements.
"""

from __future__ import annotations

from typing import Any

import jaxfne as jtfne


PROBE_MODES: tuple[str, ...] = ("spikes", "V_m", "lfp_proxy", "csd_proxy", "phi_e_proxy", "source_proxy")

# Virtual laminar contacts — explicit geometry
N_CONTACTS_DEFAULT: int = 16
CONTACT_SPACING_PROXY: float = 0.06  # proxy depth units (0-1 range)


def probe_config(*, n_contacts: int = N_CONTACTS_DEFAULT) -> dict[str, Any]:
    return {"n_contacts": int(n_contacts), "modes": list(PROBE_MODES), "spacing_proxy": CONTACT_SPACING_PROXY, "claim_level": "proxy_readout"}


# Re-export motif helper for convenience (canonical impl in observables.py:partition_currents_by_motif)
def partition_currents_by_motif(edge_currents, edge_list, neuron_table, *, reduce: str = "mean"):
    """Thin re-export of jomission.recording.observables.partition_currents_by_motif.

    Splits I_e (sign +1) vs I_i (sign -1) per edge per step and aggregates per
    motif area×layer×class → area×layer×class. Uses presynaptic cell_type E vs
    PV/SST/VIP (emitters.py:2846 sign). Canonical seam: jaxfne.compile_step_fn
    (record_edge_current=True) -> edge_current_trace [n_steps,n_edges].
    """
    from jomission.recording.observables import partition_currents_by_motif as _f

    return _f(edge_currents, edge_list, neuron_table, reduce=reduce)


def validate_recording(model: jtfne.Model) -> dict[str, Any]:
    issues: list[str] = []
    # Check that model was built with probes
    meta = getattr(model, "manifest", {}) or {}
    # Check signals have field
    try:
        from jaxfne import Simulation
        sig = jtfne.simulate(model, Simulation(duration_ms=200.0, dt_ms=0.1, seed=0))
        if sig.field is None:
            issues.append("field is None (record_fields=False or not wired)")
        else:
            # Check proxy claim
            if sig.metadata.get("field_claim_level") not in ("proxy_readout", None):
                issues.append(f"field_claim_level {sig.metadata.get('field_claim_level')}")
    except Exception as e:
        issues.append(f"simulate failed: {e}")
    return {"valid": not issues, "issues": issues, "modes": list(PROBE_MODES), "claim": "proxy_readout"}

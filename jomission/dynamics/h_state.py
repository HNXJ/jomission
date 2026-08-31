"""H-state — multi-timescale finite-dimensional biophysical/history state.

GEN-2 IMPLEMENTATION NOTE (M-06): H_Gen2 is scalar h in R per neuron
(h_state_dim=1) with tau_i = tau_0_ms * size^3 (site-packages/jaxfne/emitters.py:3703).
The five HCoordinate entries below are conceptual vocabulary and future design
(H5_DESIGN = future/not implemented), not five implemented dimensions.
Do not claim 5 independent H timescales are implemented — conceptual vs implemented
scalar h, 1-D H truth, H_Gen2 scalar per neuron. See manifests/h_gen2_spec.json
spec_id H_Gen2_scalar_v1.

τ_H ∈ {0.1, 1, 10, 100, 1000} s is initial design, not biological constant.
Each H_k has explicit meaning/driver/domain/update/coupling — conceptual only.

H is NOT homeostasis; it is history state carried in ContinuationState.dynamic + HDP.
JaxFNE carries H via DynamicState and ContinuationState; this module declares the
semantic mapping. Implemented dimensionality is scalar (h_state_dim=1); the 5-D
HStateConfig below is declarative/future-design, never forwarded to builder/kernel
(build_jomission_network never reads HStateConfig; config_hash scalar by omission).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HCoordinate:
    name: str
    tau_s: float
    meaning: str
    driver: str
    domain: str
    coupling: str
    update: str = "exponential_decay"


# Initial design — 5 timescales, each with explicit semantics
H_COORDINATES: tuple[HCoordinate, ...] = (
    HCoordinate(
        name="H_fast",
        tau_s=0.1,
        meaning="fast adaptation / spike-frequency adaptation trace",
        driver="recent spiking (population rate error)",
        domain="per-neuron activity trace",
        coupling="scales intrinsic excitability / threshold",
    ),
    HCoordinate(
        name="H_medium",
        tau_s=1.0,
        meaning="synaptic/resource state (short depression/facilitation proxy)",
        driver="presynaptic spike history",
        domain="per-edge resource",
        coupling="scales synaptic efficacy",
    ),
    HCoordinate(
        name="H_slow",
        tau_s=10.0,
        meaning="excitability/gain state",
        driver="sustained input / firing rate",
        domain="per-neuron gain",
        coupling="multiplies drive/current",
    ),
    HCoordinate(
        name="H_very_slow",
        tau_s=100.0,
        meaning="contextual sequence state (block/pattern holding)",
        driver="sequence-level statistics (AAAB/BBBA counts)",
        domain="per-population context",
        coupling="modulates inter-area gain",
    ),
    HCoordinate(
        name="H_context",
        tau_s=1000.0,
        meaning="slow contextual prior (≥17 min exposure)",
        driver="long exposure history",
        domain="per-area prior",
        coupling="biases omission expectation",
    ),
)


@dataclass(frozen=True)
class HStateConfig:
    """Declarative H-state configuration. Passed to JaxFNE via hdp/homeostasis metadata.

    M-06 scalar qualifier: H_Gen2 implemented as scalar h (h_state_dim=1).
    This declarative 5-D config is conceptual/future-design (H5_DESIGN = future/not
    implemented). Do not instantiate as live kernel dims; builder never forwards it.
    Kept for vocabulary only; h_state_dim=5 here is declarative, not realized.
    """

    coordinates: tuple[HCoordinate, ...] = H_COORDINATES
    h_state_dim: int = 5  # declarative 5-D vocabulary (H5_DESIGN future); implemented scalar is 1 — see module docstring
    locality: str = "node"  # node-local vs population; initial = node

    def to_jaxfne_hdp_params(self) -> dict[str, Any]:
        # Map to JaxFNE's expected hdp_params keys where possible; keep semantic names.
        return {
            "h_state_dim": int(self.h_state_dim),
            "h_state_locality": str(self.locality),
            "h_taus_s": [float(c.tau_s) for c in self.coordinates],
            "h_meanings": [c.meaning for c in self.coordinates],
        }

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []
        if self.h_state_dim != len(self.coordinates):
            issues.append(f"h_state_dim {self.h_state_dim} != len(coords) {len(self.coordinates)}")
        taus = [c.tau_s for c in self.coordinates]
        expected = [0.1, 1.0, 10.0, 100.0, 1000.0]
        if taus != expected:
            issues.append(f"taus {taus} != {expected}")
        if self.locality not in ("node", "population"):
            issues.append(f"locality {self.locality} not in node/population")
        return {"valid": not issues, "issues": issues, "taus": taus}

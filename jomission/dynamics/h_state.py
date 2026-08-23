"""H-state — multi-timescale finite-dimensional biophysical/history state.

τ_H ∈ {0.1, 1, 10, 100, 1000} s is initial design, not biological constant.
Each H_k has explicit meaning/driver/domain/update/coupling.

H is NOT homeostasis; it is history state carried in ContinuationState.dynamic + HDP.
JaxFNE carries H via DynamicState and ContinuationState; this module declares the semantic mapping.
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
    """Declarative H-state configuration. Passed to JaxFNE via hdp/homeostasis metadata."""

    coordinates: tuple[HCoordinate, ...] = H_COORDINATES
    h_state_dim: int = 5  # one per coordinate
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

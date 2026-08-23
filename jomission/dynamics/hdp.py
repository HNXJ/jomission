"""HDP — adaptive parameters Θ, slower/bounded, switchable for ablation.

τ_Θ ≫ τ_X, bounded updates, separately disable-able.
JaxFNE's HDP machinery (hdp_network / PlasticParams) is the execution engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HDPConfig:
    enabled: bool = True
    tau_theta_s: float = 1000.0  # ≫ τ_X (ms scale)
    learning_rate: float = 1e-4
    bounds_lo: float = -0.1
    bounds_hi: float = 0.1
    # Which parameter channels are adaptive (JaxFNE ThetaChannelSpec)
    channels: tuple[str, ...] = ("edge_weight", "intrinsic_drive")
    locality: str = "population"

    def to_jaxfne_hdp_params(self) -> dict[str, Any]:
        if not self.enabled:
            return {"hdp_enabled": False}
        return {
            "hdp_enabled": True,
            "h_state_dim": 2,  # JaxFNE population restoring requires 2; our 5-coord design maps to 2-channel controller for now
            "h_state_locality": str(self.locality),
            "tau_theta_s": float(self.tau_theta_s),
            "learning_rate": float(self.learning_rate),
            "bounds_lo": float(self.bounds_lo),
            "bounds_hi": float(self.bounds_hi),
            "channels": list(self.channels),
        }

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []
        if self.tau_theta_s < 100:
            issues.append(f"tau_theta_s {self.tau_theta_s} not ≫ tau_X (expected ≥100 s)")
        if self.bounds_lo >= self.bounds_hi:
            issues.append("bounds_lo >= bounds_hi")
        if self.locality not in ("node", "population"):
            issues.append(f"locality {self.locality} invalid")
        return {"valid": not issues, "issues": issues, "enabled": self.enabled}

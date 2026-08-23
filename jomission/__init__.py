"""Jomission — dense laminar omission simulation.

Top-level package. All simulation dynamics delegate to JaxFNE; this package owns
experiment definition, parameterization, orchestration, analysis, and manifests.
"""

__version__ = "0.1.0"

from jomission.paradigm.spec import (
    CANONICAL_EPOCHS,
    CANONICAL_CONDITIONS,
    CONDITION_FAMILIES,
    OMISSION_POSITIONS,
    JOMISSION_PARADIGM,
    jomission_paradigm,
    paradigm_exact_gate,
)
from jomission.network.builder import build_jomission_network
from jomission.network.populations import JOMISSION_AREAS, JOMISSION_LAYERS, JOMISSION_CELL_TYPES
from jomission.dynamics.h_state import HStateConfig
from jomission.dynamics.hdp import HDPConfig

__all__ = [
    "__version__",
    "CANONICAL_EPOCHS",
    "CANONICAL_CONDITIONS",
    "CONDITION_FAMILIES",
    "OMISSION_POSITIONS",
    "JOMISSION_PARADIGM",
    "jomission_paradigm",
    "paradigm_exact_gate",
    "build_jomission_network",
    "JOMISSION_AREAS",
    "JOMISSION_LAYERS",
    "JOMISSION_CELL_TYPES",
    "HStateConfig",
    "HDPConfig",
]

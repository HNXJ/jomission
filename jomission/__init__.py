"""Jomission — dense laminar omission simulation.

Top-level package. All simulation dynamics delegate to JaxFNE; this package owns
experiment definition, parameterization, orchestration, analysis, and manifests.
"""

__version__ = "0.1.0"

# Persistent XLA compilation cache. Compilation is about two thirds of a canonical build
# (measured: warm build 1.0 s vs cold 3.7 s). Compiled code is unchanged; only its reuse is.
# Disable with JOMISSION_JAX_CACHE=0; the cache lives outside the repository.
import os as _os

if _os.environ.get("JOMISSION_JAX_CACHE", "1") != "0":
    try:
        import jax as _jax

        _cache_dir = _os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            _os.path.join(_os.path.expanduser("~"), ".cache", "jomission-jax"),
        )
        _jax.config.update("jax_compilation_cache_dir", _cache_dir)
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        _jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except Exception:  # cache is an optimization; never block import
        pass

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

"""Jomission — dense laminar omission simulation.

Top-level package. All simulation dynamics delegate to JaxFNE; this package owns
experiment definition, parameterization, orchestration, analysis, and manifests.
"""

__version__ = "0.1.0"

# Persistent XLA compilation cache (optional, performance only).
# Measured on one machine: canonical build cold ~3.7 s, warm ~1.1 s (single run each; not a
# multi-run benchmark). Compiled code is the same; only its reuse across processes changes.
# Rules: a setting the caller already made in the environment always wins; the cache
# directory must be trusted, because another writer to it can run arbitrary code when JAX loads
# entries. Disable with JOMISSION_JAX_CACHE=0. Default directory: ~/.cache/jomission-jax.
import os as _os

if _os.environ.get("JOMISSION_JAX_CACHE", "1") != "0":
    try:
        import jax as _jax

        if "JAX_COMPILATION_CACHE_DIR" not in _os.environ:
            _jax.config.update(
                "jax_compilation_cache_dir",
                _os.path.join(_os.path.expanduser("~"), ".cache", "jomission-jax"),
            )
        if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in _os.environ:
            _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        if "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES" not in _os.environ:
            _jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)  # -1: no size limit
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

"""Sampling + log scaling utilities — V0-P1 shared.

Read-only helpers for deterministic stratified sampling and log scaling.
Consumes same generated-owner arrays as analyses (builder.py:62 MOTIF_GAIN,
builder.py:395 spatial_sigma 0.08, jaxfne/emitters EdgeList via model.params).

No science mutation — statistics remain computed from complete EdgeList;
rendering may sample.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

import numpy as np


_SAMPLING_DISCLOSURE_TEMPLATE = (
    "Rendering {n_rendered:,} of {n_total:,} edges using deterministic stratified sampling. "
    "All reported degree, weight, probability and delay statistics use the complete EdgeList."
)


def config_hash_for_model(model: Any) -> str:
    try:
        from jaxfne.io import config_hash as _ch  # type: ignore

        cfg = getattr(model, "config", None)
        if cfg is None:
            cfg = getattr(model, "cfg", None)
        if cfg is not None:
            return str(_ch(cfg))
    except Exception:
        pass
    try:
        el = model.params.get("edge_list")  # type: ignore
        if el is not None:
            return hashlib.sha256(np.asarray(el.pre).tobytes()).hexdigest()[:16]
    except Exception:
        pass
    return "0000000000000000"


def deterministic_seed(config_hash: str, salt: str = "") -> int:
    try:
        s = f"{config_hash}:{salt}"
        return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
    except Exception:
        return 0


def sampling_disclosure(n_total: int, n_rendered: int) -> str:
    return _SAMPLING_DISCLOSURE_TEMPLATE.format(n_total=int(n_total), n_rendered=int(n_rendered))


def safe_log_abs_weight(w, eps: float = 1e-6):
    w_arr = np.asarray(w, dtype=float)
    return np.sign(w_arr) * np.log1p(np.abs(w_arr) / float(eps))


def safe_log_tau(tau, eps: float = 1e-9):
    tau_arr = np.asarray(tau, dtype=float)
    return np.log10(np.maximum(tau_arr, float(eps)))


def scaling_for_metric(metric: str) -> str:
    mk = str(metric).lower()
    if mk in ("weight", "w", "meanw", "realized_current", "weight_lognormal", "tau", "tau_h", "h_tau"):
        return "log"
    if mk in ("cv", "cv_isi", "fano", "rho"):
        return "linear"
    return "linear" if mk in ("probability", "p", "delay_ms", "rate") else "linear"

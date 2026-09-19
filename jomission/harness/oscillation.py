"""Periodicity of a population rate: period, strength, and a null it must beat.

Written for SCI-WS-OSC-1, because no sealed estimator for the burst period existed. Earlier
whole-system lineages quote a period in prose; no result file carries one and no module
computed one, so every number this lineage reports is measured here rather than inherited.

The estimator is deliberately plain:

    peak_autocorr   the largest autocorrelation after the first zero crossing
    period_ms       the lag at which that peak sits
    null            the 95th percentile of the same statistic over Poisson surrogates
                    matched to the signal's own mean rate and cell count

The first zero crossing matters: the autocorrelation of any smooth signal decays from 1.0 at
lag 0, and taking the largest value overall would return lag 0 or the shoulder beside it. The
first peak after the correlation has crossed zero is the first genuine recurrence.

The null exists because "there is a periodic mode" needs a floor that is not chosen by hand.
Independent Poisson firing at the same rate produces some peak autocorrelation purely by
chance, more of it when rates are low or cells are few. That value is the floor.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_LAG_MS = 1000.0
DEFAULT_N_SURROGATE = 200
DEFAULT_NULL_PERCENTILE = 95.0


def autocorrelation(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation of a mean-removed signal, normalised to 1.0 at lag 0."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0.0:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        out[lag] = float((x[: x.size - lag] * x[lag:]).sum()) / denom
    return out


def first_peak_after_zero_crossing(ac: np.ndarray) -> tuple[int | None, float]:
    """The first local maximum at a lag beyond the first sign change of the autocorrelation.

    Returns (lag, value), or (None, 0.0) when the correlation never crosses zero or never
    turns back up, which is what a non-recurrent signal looks like.
    """
    neg = np.flatnonzero(ac < 0.0)
    if neg.size == 0:
        return None, 0.0
    start = int(neg[0])
    if start + 1 >= ac.size - 1:
        return None, 0.0
    tail = ac[start:]
    best_rel = int(np.argmax(tail))
    if best_rel == 0 or best_rel == tail.size - 1:
        return None, 0.0
    lag = start + best_rel
    return lag, float(ac[lag])


def poisson_null(mean_rate_hz: float, n_cells: int, n_bins: int, dt_ms: float, *,
                 max_lag: int, n_surrogate: int = DEFAULT_N_SURROGATE,
                 percentile: float = DEFAULT_NULL_PERCENTILE, seed: int = 0) -> dict:
    """Peak autocorrelation reachable by independent Poisson firing at the same rate.

    The surrogate is a population rate built from ``n_cells`` independent Poisson spike
    trains at ``mean_rate_hz``, binned exactly as the measured signal is. No periodicity is
    put in, so whatever peak appears is chance.
    """
    rng = np.random.default_rng(seed)
    p = float(mean_rate_hz) * float(dt_ms) * 1e-3
    peaks = []
    for _ in range(int(n_surrogate)):
        counts = rng.binomial(n_cells, min(max(p, 0.0), 1.0), size=n_bins)
        rate = counts / (n_cells * dt_ms * 1e-3)
        _lag, value = first_peak_after_zero_crossing(autocorrelation(rate, max_lag))
        peaks.append(value)
    peaks_arr = np.asarray(peaks, dtype=float)
    return {"percentile": float(percentile),
            "threshold": float(np.percentile(peaks_arr, percentile)),
            "max": float(peaks_arr.max()), "mean": float(peaks_arr.mean()),
            "n_surrogate": int(n_surrogate), "mean_rate_hz": float(mean_rate_hz),
            "n_cells": int(n_cells), "seed": int(seed)}


def measure(rate_hz: np.ndarray, *, dt_ms: float, n_cells: int,
            max_lag_ms: float = DEFAULT_MAX_LAG_MS, n_surrogate: int = DEFAULT_N_SURROGATE,
            percentile: float = DEFAULT_NULL_PERCENTILE, seed: int = 0) -> dict:
    """Measure a population rate's periodic mode against its own Poisson null.

    ``rate_hz`` is a population firing rate sampled every ``dt_ms``. ``n_cells`` is how many
    cells it pools, which the null needs in order to match the signal's counting noise.
    """
    rate = np.asarray(rate_hz, dtype=np.float64)
    if rate.ndim != 1 or rate.size < 8:
        raise ValueError(f"rate_hz must be a 1-D series of at least 8 samples; got {rate.shape}")
    if dt_ms <= 0 or n_cells <= 0:
        raise ValueError("dt_ms and n_cells must be positive")
    max_lag = int(round(max_lag_ms / dt_ms))
    if max_lag >= rate.size:
        raise ValueError(f"max_lag {max_lag_ms} ms exceeds the {rate.size * dt_ms} ms window")

    ac = autocorrelation(rate, max_lag)
    lag, peak = first_peak_after_zero_crossing(ac)
    null = poisson_null(float(rate.mean()), n_cells, rate.size, dt_ms, max_lag=max_lag,
                        n_surrogate=n_surrogate, percentile=percentile, seed=seed)
    present = bool(lag is not None and peak > null["threshold"])
    mean = float(rate.mean())
    return {
        "periodic_mode_present": present,
        "period_ms": None if lag is None else round(lag * dt_ms, 4),
        "frequency_hz": None if lag is None or lag == 0 else round(1000.0 / (lag * dt_ms), 4),
        "peak_autocorr": round(peak, 6),
        "null": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in null.items()},
        "exceeds_null_by": round(peak - null["threshold"], 6),
        "mean_rate_hz": round(mean, 4),
        "min_rate_hz": round(float(rate.min()), 4),
        "max_rate_hz": round(float(rate.max()), 4),
        "modulation_depth": (None if mean <= 0 else
                             round(float(rate.max() - rate.min()) / mean, 4)),
        "window_samples": int(rate.size),
        "dt_ms": float(dt_ms),
        "max_lag_ms": float(max_lag_ms),
    }


def compare(intact: dict, ablated: dict, *, period_tolerance: float = 0.10) -> dict:
    """Classify an ablation against its intact control, into the three predeclared outcomes.

    The rule is comparative and fixed before execution: the question is whether the mode
    survives the ablation, keeps its frequency, or disappears into the null.
    """
    if not intact["periodic_mode_present"]:
        return {"outcome": "CONTROL_HAS_NO_MODE",
                "reading": ("the intact arm shows no periodic mode above its own null, so the "
                            "ablation cannot be interpreted. This is a result about the "
                            "estimator or the control, not about recurrence")}
    if not ablated["periodic_mode_present"]:
        return {"outcome": "MODE_REQUIRES_ABLATED_CONNECTION",
                "reading": ("the mode is present in the intact arm and falls into the null "
                            "when the connection is cut, so the connection is necessary for it")}
    pi, pa = intact["period_ms"], ablated["period_ms"]
    rel = abs(pa - pi) / pi
    if rel <= period_tolerance:
        return {"outcome": "MODE_INDEPENDENT_OF_ABLATED_CONNECTION",
                "period_change_fraction": round(rel, 6),
                "reading": ("the mode survives the cut at the same period, so it is not "
                            "generated by the ablated connection")}
    return {"outcome": "MODE_DEPENDS_QUANTITATIVELY",
            "period_change_fraction": round(rel, 6),
            "reading": ("the mode survives but its period moves, so the ablated connection "
                        "shapes the frequency without being necessary for the mode")}

"""T7 between-area lead/lag estimator — frozen estimand, independent validation.

Frozen authorities (do not alter estimand based on results):
- T7: absence or presence of robust fixed between-area lead/lag
      Estimand: cross-area field/rate cross-correlation peak lag distribution
      (comparison_matrix T7: "cross-area field/rate cross-correlation peak lag",
       test "peak lag distribution, test for consistent non-zero lag",
       threshold "no consistent fixed lag (p>0.05)")
      Source: jomission/analysis/comparison_matrix.py, targets.py, manifests/*.
- Must not infer propagation from anatomical delays (D_t edge delays).
      Lag must be data-driven from population rates/fields, not from wiring table.
- Requires positive propagation control and no-lag/null control.
- Require artifact-backed evidence, generated-owner arrays.
- Input: rate[trial,area,time] at >= ms resolution (dt_ms 0.1 or 1.0 canonical).
  Field support possible via area_local partition but rate is primary (task authority).

This module:
  - implements lag estimator independently (cross-correlation peak, data-driven)
  - validates via known-lag synthetic/controlled population-rate arrays using the EXACT
    same lag estimator, plus no-lag/null dataset, quantifying detection error / false-positive
  - then applies unchanged estimator to model rates

Provenance:
- Generates owner arrays under results/t7/ or manifests/ (when caller saves)
- No anatomical delay inference: estimator uses only rate[trial,area,time] values,
  never reads connectivity matrix or delay table.
- Windows: omission_local (-1000,+1000) reindexed at expected omission onset,
  plus absolute trial clock (fx -500, p1 0-531, d1 531-1031, p2 1031-1562,
  d2 1562-2062, p3 2062-2593, d3 2593-3093, p4 3093-3624, d4 3624-4124).
  See jomission/paradigm/epochs.py and spec.py SLOT_ONSET_MS.

Method details:
  - Per trial, per area-pair, extract rate window (position-aware) and compute
    normalized cross-correlation (z-scored signals, unbiased denominator).
  - Peak lag = lag (ms) maximizing cross-correlation within ±max_lag_ms.
  - Ensemble statistics per position/pair: mean, median, SD, SEM, 95% CI,
    one-sample t-test vs 0 (H0: no fixed lag), proportion within thresholds,
    estimator error quantification in controls.
  - Controls quantify detection error (MAE, bias, RMSE, within tolerance) and
    false-positive rate (spurious lag declarations on null).
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as st

try:
    from scipy import signal as scipy_signal
    _HAS_SCIPY_SIGNAL = True
except Exception:
    _HAS_SCIPY_SIGNAL = False

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------
AREAS_CANONICAL: Tuple[str, ...] = ("V1", "V4", "FEF", "PFC")
N_CONTACTS_DEFAULT: int = 16

# Absolute trial onsets for p-slots (ms, scheduler clock with fx -500)
# Evidence: field 0 = p1 (see t4_t5_analysis._trial_window_indices provenance)
SLOT_ONSET_MS: Dict[str, float] = {
    "p1": 0.0,
    "p2": 1031.0,
    "p3": 2062.0,
    "p4": 3093.0,
}
# Omission-local windows (reindexed at omission onset)
OMISSION_SLOT_MS = (0.0, 531.0)
OMISSION_BASELINE_MS = (-250.0, -50.0)
OMISSION_LOCAL_MS = (-1000.0, 1000.0)
POST_OMISSION_MS = (531.0, 1000.0)

# Mapping condition -> omission position
OMISSION_POSITIONS: Dict[str, Tuple[str, ...]] = {
    "p2": ("AXAB", "BXBA", "RXRR"),
    "p3": ("AAXB", "BBXA", "RRXR"),
    "p4": ("AAAX", "BBBX", "RRRX"),
    "intact": ("AAAB", "BBBA", "RRRR"),
}
COND_TO_POS: Dict[str, str | None] = {}
for _pos, _conds in OMISSION_POSITIONS.items():
    for _c in _conds:
        COND_TO_POS[_c] = None if _pos == "intact" else _pos

TRIAL_MS: float = 4624.0

# Estimator defaults (frozen for T7 closure)
DEFAULT_MAX_LAG_MS: float = 200.0
DEFAULT_WINDOW_MS: Tuple[float, float] = OMISSION_LOCAL_MS  # 2000 ms window gives stable xcorr
DEFAULT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("V1", "V4"),
    ("V1", "FEF"),
    ("V1", "PFC"),
    ("V4", "FEF"),
    ("V4", "PFC"),
    ("FEF", "PFC"),
)

ESTIMATOR_PROVENANCE: str = "data-driven cross-correlation peak; no anatomical delay inference"
FIELD_CLAIM_LEVEL: str = "proxy_readout"  # for field variant, not used for rate but kept
PHYSICAL_AMPLITUDE_CALIBRATED: bool = False

# ---------------------------------------------------------------------------
# Helpers: validation, windowing
# ---------------------------------------------------------------------------

def _validate_rate(
    rate: np.ndarray,
    trial_conditions: List[str],
    areas: Tuple[str, ...],
) -> Dict[str, Any]:
    if rate.ndim != 3:
        raise ValueError(f"rate must be 3D [trial,area,time], got {rate.shape}")
    n_trials = rate.shape[0]
    if len(trial_conditions) != n_trials:
        raise ValueError(f"trial_conditions len {len(trial_conditions)} != n_trials {n_trials}")
    if rate.shape[1] != len(areas):
        raise ValueError(f"rate area dim {rate.shape[1]} != len(areas) {len(areas)}")
    return {"n_trials": n_trials, "areas": list(areas), "n_time": rate.shape[2]}


def _window_for_position(
    position: str,
    fs_hz: float,
    dt_ms: float,
    window: Tuple[float, float],
) -> Tuple[int, int]:
    """Position-explicit window (absolute trial sample indices). Evidence: field 0 = p1."""
    if position not in ("p2", "p3", "p4"):
        raise ValueError(f"position {position} not in p2/p3/p4")
    onset_abs = SLOT_ONSET_MS[position]
    lo_abs = onset_abs + window[0]
    hi_abs = onset_abs + window[1]
    i0 = int(round(lo_abs / dt_ms))
    i1 = int(round(hi_abs / dt_ms))
    return i0, i1


def _all_pairs(areas: Tuple[str, ...]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i in range(len(areas)):
        for j in range(i + 1, len(areas)):
            pairs.append((areas[i], areas[j]))
    return pairs


# ---------------------------------------------------------------------------
# Core lag estimator (single trial, single pair)
# ---------------------------------------------------------------------------

def estimate_lag_single(
    x: np.ndarray,
    y: np.ndarray,
    fs_hz: float,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
    *,
    demean: bool = True,
    normalize: str = "zscore",  # "zscore" or "demean" or "none"
    method: str = "xcorr",
) -> Dict[str, Any]:
    """Estimate lead/lag between two 1D rate traces via cross-correlation peak.

    Data-driven: uses only x,y values, never connectivity/delay tables.

    Parameters
    ----------
    x, y: 1D arrays [time] — rate traces for area i and j
    fs_hz: sampling rate (Hz)
    max_lag_ms: maximum lag magnitude to search (ms)
    demean, normalize: preprocessing; "zscore" -> (x-mean)/std, robust to amplitude
    method: "xcorr" (only supported; placeholder for future)

    Returns
    -------
    dict with:
        peak_lag_ms: float, lag of y relative to x (positive = y lags x, x leads)
                     i.e., if y[t] = x[t - lag], peak at +lag.
        peak_corr: float, cross-correlation at peak (normalized ~ Pearson r)
        peak_corr_abs: float, correlation magnitude at absolute max (if differing)
        lags_ms: np.ndarray, lag axis
        xcorr: np.ndarray, correlation values
        n_time: int, window length
        valid: bool, whether estimation was valid (non-constant signals)
        note: str provenance
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x len {len(x)} != y len {len(y)}")
    n = int(x.shape[0])
    if n < 10:
        return {
            "peak_lag_ms": float("nan"),
            "peak_corr": float("nan"),
            "peak_corr_abs": float("nan"),
            "lags_ms": np.array([]),
            "xcorr": np.array([]),
            "n_time": n,
            "valid": False,
            "note": "window too short",
        }
    # Handle constant signals (no variance)
    sx = float(x.std())
    sy = float(y.std())
    if sx == 0 or sy == 0 or not np.isfinite(sx) or not np.isfinite(sy):
        return {
            "peak_lag_ms": float("nan"),
            "peak_corr": float("nan"),
            "peak_corr_abs": float("nan"),
            "lags_ms": np.array([]),
            "xcorr": np.array([]),
            "n_time": n,
            "valid": False,
            "note": "constant signal, zero variance",
        }
    dt_ms = 1000.0 / fs_hz
    max_lag_samples = int(round(max_lag_ms / dt_ms))
    max_lag_samples = min(max_lag_samples, n - 1)
    # Preprocess
    if normalize == "zscore":
        xz = (x - x.mean()) / (sx + 1e-12)
        yz = (y - y.mean()) / (sy + 1e-12)
    elif normalize == "demean":
        xz = x - x.mean()
        yz = y - y.mean()
        # Normalize by product of stds later to get correlation-like scale
    elif normalize == "none":
        xz = x
        yz = y
    else:
        raise ValueError(f"normalize {normalize} not in zscore/demean/none")

    # Compute cross-correlation (full)
    # Use scipy.signal.correlate if available for speed, else numpy
    if _HAS_SCIPY_SIGNAL:
        corr_raw = scipy_signal.correlate(xz, yz, mode="full", method="auto")
    else:
        corr_raw = np.correlate(xz, yz, mode="full")
    # corr_raw length = 2*n-1, lags = -(n-1) .. (n-1)
    lags_samples = np.arange(-n + 1, n, dtype=np.int32)
    # Convert to unbiased normalized correlation: divide by (n - |lag|)
    # For zscore, this yields Pearson r at each lag
    # For demean, need also divide by n*stdx*stdy? But we already zscored, so direct.
    # For demean without zscore, divide by (effective_n * sx * sy) approximation
    if normalize == "zscore":
        effective_n = n - np.abs(lags_samples)
        # Avoid division by zero (not happen as effective_n >=1)
        corr = corr_raw / np.maximum(effective_n, 1)
    elif normalize == "demean":
        effective_n = n - np.abs(lags_samples)
        corr = corr_raw / (np.maximum(effective_n, 1) * (sx + 1e-12) * (sy + 1e-12))
    else:
        effective_n = n - np.abs(lags_samples)
        corr = corr_raw / np.maximum(effective_n, 1)

    # Clip to max_lag
    mask = np.abs(lags_samples) <= max_lag_samples
    lags_clip = lags_samples[mask]
    corr_clip = corr[mask]
    # Sign correction: scipy/numpy correlate convention yields peak at -D when y delayed by D,
    # so we negate to enforce documented convention: positive lag = y lags x (y[t]=x[t - D] -> peak at +D)
    lags_ms = -lags_clip.astype(np.float64) * dt_ms
    # Find peak: max value (positive correlation). Also track abs max for diagnostics
    # Use max (not abs) because synthetic positive copy gives positive peak
    idx_max = int(np.argmax(corr_clip))
    idx_abs = int(np.argmax(np.abs(corr_clip)))
    peak_lag_ms = float(lags_ms[idx_max])
    peak_corr = float(corr_clip[idx_max])
    peak_corr_abs = float(corr_clip[idx_abs])
    peak_lag_abs = float(lags_ms[idx_abs])
    # Convention now: y[t]=x[t - D] -> corr peak at +D (y lags x). Negation above fixes numpy's sign.
    return {
        "peak_lag_ms": peak_lag_ms,
        "peak_corr": peak_corr,
        "peak_corr_abs": peak_corr_abs,
        "peak_lag_abs": peak_lag_abs,
        "lags_ms": lags_ms,
        "xcorr": corr_clip,
        "n_time": n,
        "valid": True,
        "note": f"data-driven xcorr; no anatomical delay; normalize={normalize}; max_lag_ms={max_lag_ms}; unbiased",
    }


# ---------------------------------------------------------------------------
# Synthetic / controlled rate generators (for validation)
# ---------------------------------------------------------------------------

def _generate_autocorrelated_signal(
    n_time: int,
    *,
    autocor: float = 0.9,
    noise_std: float = 0.5,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate smooth autocorrelated 1D signal (AR1)."""
    x = np.zeros(n_time, dtype=np.float64)
    # Start from random
    x[0] = rng.normal(0, 1)
    for t in range(1, n_time):
        x[t] = autocor * x[t - 1] + rng.normal(0, noise_std)
    return x


def _shift_signal(sig: np.ndarray, lag_samples: int) -> np.ndarray:
    """Shift sig by lag_samples (positive = delay, y[t]=x[t - lag]). Zero-pad edges."""
    n = len(sig)
    out = np.zeros_like(sig)
    if lag_samples == 0:
        out[:] = sig
    elif lag_samples > 0:
        # delay: y[t]=x[t - lag] -> y[lag:] = x[:-lag]
        out[lag_samples:] = sig[: n - lag_samples]
        # pad beginning with edge value or zero? Use zero? Use first value replication for continuity
        # Use zero padding for clean control but alternative is to replicate; we use zero for simplicity
        # However to keep mean stable, we fill with initial noise around 0 -> zero is okay for centered signal
    else:
        lag = -lag_samples
        out[: n - lag] = sig[lag:]
    return out


def synthesize_rate_pair(
    n_trials: int,
    n_time: int,
    fs_hz: float,
    true_lag_ms: float,
    *,
    noise_std: float = 0.3,
    autocor: float = 0.92,
    baseline_rate: float = 10.0,
    modulation: float = 5.0,
    rng_seed: int = 0,
) -> np.ndarray:
    """Synthesize rate[trial, 2, time] with known fixed lag between the two areas.

    Area 0 = driver, Area 1 = delayed copy of driver + independent noise.
    True lag in ms is converted to samples via fs_hz. Positive lag = area1 lags area0.

    Signal model: underlying shared signal s[t] is AR(1) autocorrelated, centered at 0,
    scaled to modulation, then added to baseline and noise.

    Parameters
    ----------
    n_trials, n_time, fs_hz, true_lag_ms
    noise_std, autocor: signal generation params
    baseline_rate, modulation: scaling to Hz-like values

    Returns
    -------
    rate: np.ndarray shape (n_trials, 2, n_time) dtype float64
    """
    rng = np.random.default_rng(rng_seed)
    lag_samples = int(round(true_lag_ms * fs_hz / 1000.0))
    rate = np.zeros((n_trials, 2, n_time), dtype=np.float64)
    for t in range(n_trials):
        s = _generate_autocorrelated_signal(n_time, autocor=autocor, noise_std=0.6, rng=rng)
        # Center? already zero-mean; scale
        s_scaled = s * (modulation / (np.std(s) + 1e-12))
        # Drive for area0
        noise0 = rng.normal(0, noise_std, size=n_time)
        noise1 = rng.normal(0, noise_std, size=n_time)
        sig0 = baseline_rate + s_scaled + noise0
        s_delayed = _shift_signal(s_scaled, lag_samples)
        sig1 = baseline_rate + s_delayed + noise1
        # Ensure non-negative rates (clip)
        sig0 = np.maximum(sig0, 0.0)
        sig1 = np.maximum(sig1, 0.0)
        rate[t, 0, :] = sig0
        rate[t, 1, :] = sig1
    return rate


def synthesize_rate_multiarea(
    n_trials: int,
    n_time: int,
    fs_hz: float,
    *,
    true_lags_ms: Dict[Tuple[int, int], float] | None = None,
    chain_lag_ms: float | None = None,
    n_areas: int = 4,
    noise_std: float = 0.3,
    autocor: float = 0.92,
    baseline_rate: float = 10.0,
    modulation: float = 5.0,
    rng_seed: int = 0,
) -> np.ndarray:
    """Synthesize rate[trial, n_areas, time] with controlled lags.

    Two modes:
      - chain_lag_ms: if not None, areas form a chain where area k lags area 0 by k*chain_lag_ms
      - true_lags_ms: dict {(i,j): lag_ms} for explicit pair lags; underlying driver is area 0's signal,
        each area's signal is delayed version plus noise (single-source model).
        For independent control, you can use independent signals.

    If both None, generates null (independent signals per area).

    Returns rate [trial, n_areas, time]
    """
    rng = np.random.default_rng(rng_seed)
    rate = np.zeros((n_trials, n_areas, n_time), dtype=np.float64)
    for t in range(n_trials):
        if chain_lag_ms is not None:
            s0 = _generate_autocorrelated_signal(n_time, autocor=autocor, noise_std=0.6, rng=rng)
            s0_scaled = s0 * (modulation / (np.std(s0) + 1e-12))
            for a in range(n_areas):
                lag_ms = a * chain_lag_ms
                lag_samples = int(round(lag_ms * fs_hz / 1000.0))
                s_delayed = _shift_signal(s0_scaled, lag_samples)
                noise = rng.normal(0, noise_std, size=n_time)
                sig = baseline_rate + s_delayed + noise
                rate[t, a, :] = np.maximum(sig, 0.0)
        elif true_lags_ms is not None:
            # Single driver model: base signal s0 for area 0, others delayed by specified lags
            s0 = _generate_autocorrelated_signal(n_time, autocor=autocor, noise_std=0.6, rng=rng)
            s0_scaled = s0 * (modulation / (np.std(s0) + 1e-12))
            # Area 0
            rate[t, 0, :] = np.maximum(baseline_rate + s0_scaled + rng.normal(0, noise_std, size=n_time), 0.0)
            for a in range(1, n_areas):
                # Find lag for pair (0,a) if specified else 0
                lag_ms = true_lags_ms.get((0, a), 0.0)
                # Also allow generic dict key (a) ?
                lag_samples = int(round(lag_ms * fs_hz / 1000.0))
                s_delayed = _shift_signal(s0_scaled, lag_samples)
                sig = baseline_rate + s_delayed + rng.normal(0, noise_std, size=n_time)
                rate[t, a, :] = np.maximum(sig, 0.0)
        else:
            # Null: independent
            for a in range(n_areas):
                s = _generate_autocorrelated_signal(n_time, autocor=autocor, noise_std=0.6, rng=rng)
                s_scaled = s * (modulation / (np.std(s) + 1e-12))
                noise = rng.normal(0, noise_std, size=n_time)
                sig = baseline_rate + s_scaled + noise
                rate[t, a, :] = np.maximum(sig, 0.0)
    return rate


def synthesize_rate_null(
    n_trials: int,
    n_time: int,
    fs_hz: float,
    n_areas: int = 4,
    *,
    noise_std: float = 0.3,
    autocor: float = 0.92,
    baseline_rate: float = 10.0,
    modulation: float = 5.0,
    rng_seed: int = 0,
) -> np.ndarray:
    """Independent rates per area (no coupling, null)."""
    return synthesize_rate_multiarea(
        n_trials, n_time, fs_hz,
        true_lags_ms=None, chain_lag_ms=None, n_areas=n_areas,
        noise_std=noise_std, autocor=autocor,
        baseline_rate=baseline_rate, modulation=modulation,
        rng_seed=rng_seed,
    )


def synthesize_rate_nolag(
    n_trials: int,
    n_time: int,
    fs_hz: float,
    n_areas: int = 4,
    *,
    noise_std: float = 0.3,
    autocor: float = 0.92,
    baseline_rate: float = 10.0,
    modulation: float = 5.0,
    rng_seed: int = 0,
) -> np.ndarray:
    """Synchronous rates per area (true lag 0, shared driver)."""
    # Use chain_lag 0
    return synthesize_rate_multiarea(
        n_trials, n_time, fs_hz,
        chain_lag_ms=0.0, n_areas=n_areas,
        noise_std=noise_std, autocor=autocor,
        baseline_rate=baseline_rate, modulation=modulation,
        rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# Ensemble evaluation (positive / null control quantification)
# ---------------------------------------------------------------------------

def quantify_positive_control(
    peak_lags_ms: np.ndarray,
    true_lag_ms: float,
    peak_corrs: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Quantify detection error for known-lag control."""
    peak_lags_ms = np.asarray(peak_lags_ms, dtype=np.float64)
    mask = np.isfinite(peak_lags_ms)
    lags = peak_lags_ms[mask]
    if len(lags) == 0:
        return {"n": 0, "valid": False, "note": "no finite lags"}
    errors = lags - true_lag_ms
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    median_ae = float(np.median(abs_errors))
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    std = float(lags.std(ddof=1)) if len(lags) > 1 else 0.0
    mean_est = float(lags.mean())
    median_est = float(np.median(lags))
    # Fractions within tolerance (ms)
    frac_within_2 = float(np.mean(abs_errors <= 2.0)) if len(lags)>0 else float("nan")
    frac_within_5 = float(np.mean(abs_errors <= 5.0))
    frac_within_10 = float(np.mean(abs_errors <= 10.0))
    # T-test vs zero (for positive lag should be non-zero)
    if len(lags) >= 2 and std > 0:
        t_stat, p_val = st.ttest_1samp(lags, popmean=0.0)
        t_stat = float(t_stat) if np.isfinite(t_stat) else float("nan")
        p_val = float(p_val) if np.isfinite(p_val) else float("nan")
    else:
        t_stat, p_val = float("nan"), float("nan")
    # Also test vs true lag: should be not significantly different from true
    if len(lags) >= 2 and std > 0:
        t_true, p_true = st.ttest_1samp(lags, popmean=true_lag_ms)
        t_true = float(t_true) if np.isfinite(t_true) else float("nan")
        p_true = float(p_true) if np.isfinite(p_true) else float("nan")
    else:
        t_true, p_true = float("nan"), float("nan")
    # Success criterion: MAE small, within tolerance high, p vs zero <0.05 for true !=0
    success = bool(mae <= 5.0 and frac_within_5 >= 0.8) if true_lag_ms != 0 else bool(mae <= 2.0)
    return {
        "n": int(len(lags)),
        "true_lag_ms": float(true_lag_ms),
        "mean_estimated_lag_ms": mean_est,
        "median_estimated_lag_ms": median_est,
        "bias_ms": bias,
        "mae_ms": mae,
        "median_ae_ms": median_ae,
        "rmse_ms": rmse,
        "std_ms": std,
        "frac_within_2ms": frac_within_2,
        "frac_within_5ms": frac_within_5,
        "frac_within_10ms": frac_within_10,
        "t_vs_zero": t_stat,
        "p_vs_zero": p_val,
        "t_vs_true": t_true,
        "p_vs_true": p_true,
        "valid": True,
        "success": success,
        "peak_corrs_mean": float(np.mean(peak_corrs[mask])) if peak_corrs is not None and len(peak_corrs[mask])>0 else float("nan"),
    }


def quantify_nolag_control(
    peak_lags_ms: np.ndarray,
    peak_corrs: np.ndarray | None = None,
    tolerance_ms: float = 5.0,
) -> Dict[str, Any]:
    """Quantify performance on no-lag (true 0) control."""
    return quantify_positive_control(peak_lags_ms, true_lag_ms=0.0, peak_corrs=peak_corrs)


def quantify_null_control(
    peak_lags_ms: np.ndarray,
    peak_corrs: np.ndarray | None = None,
    lag_threshold_ms: float = 5.0,
    corr_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Quantify false-positive behaviour on null (independent signals).

    For null, there is no true lag. We quantify:
      - lag distribution: mean, std, uniformity
      - false-positive rate under two definitions
    """
    peak_lags_ms = np.asarray(peak_lags_ms, dtype=np.float64)
    mask = np.isfinite(peak_lags_ms)
    lags = peak_lags_ms[mask]
    if len(lags) == 0:
        return {"n": 0, "valid": False}
    mean_lag = float(lags.mean())
    median_lag = float(np.median(lags))
    std_lag = float(lags.std(ddof=1)) if len(lags) > 1 else 0.0
    # T-test vs zero should be non-significant for null (p>0.05 indicates no fixed lag)
    if len(lags) >= 2 and std_lag > 0:
        t_stat, p_val = st.ttest_1samp(lags, popmean=0.0)
        t_stat = float(t_stat) if np.isfinite(t_stat) else float("nan")
        p_val = float(p_val) if np.isfinite(p_val) else float("nan")
    else:
        t_stat, p_val = float("nan"), float("nan")
    # False positive definitions:
    # Per-trial FP: |lag| > threshold and |corr| > corr_threshold -> spurious detection
    if peak_corrs is not None:
        corrs = np.asarray(peak_corrs, dtype=np.float64)[mask]
        # Use peak_corr magnitude
        fp_per_trial = np.mean((np.abs(lags) > lag_threshold_ms) & (np.abs(corrs) > corr_threshold))
        fp_per_trial = float(fp_per_trial)
        mean_abs_corr = float(np.mean(np.abs(corrs))) if len(corrs)>0 else float("nan")
        max_corr = float(np.max(np.abs(corrs))) if len(corrs)>0 else float("nan")
    else:
        fp_per_trial = float(np.mean(np.abs(lags) > lag_threshold_ms))
        mean_abs_corr = float("nan")
        max_corr = float("nan")
    # Distribution uniformity: for null independent, lags should be widely scattered across range
    # Coefficient: std large suggests no concentration; for true fixed lag, std small (<10ms)
    # Also proportion within 10ms should be low for null if truly independent (uniform) vs high for true lag
    prop_within_10 = float(np.mean(np.abs(lags) <= 10.0))
    prop_within_5 = float(np.mean(np.abs(lags) <= 5.0))
    # False positive at ensemble level: declaring significant fixed lag when p<0.05 and |mean|>threshold
    ensemble_fp = bool(p_val < 0.05 and abs(mean_lag) > lag_threshold_ms) if np.isfinite(p_val) else False
    return {
        "n": int(len(lags)),
        "mean_lag_ms": mean_lag,
        "median_lag_ms": median_lag,
        "std_ms": std_lag,
        "t_vs_zero": t_stat,
        "p_vs_zero": p_val,
        "prop_within_5ms": prop_within_5,
        "prop_within_10ms": prop_within_10,
        "false_positive_per_trial": fp_per_trial,
        "mean_abs_corr": mean_abs_corr,
        "max_abs_corr": max_corr,
        "ensemble_false_positive": ensemble_fp,
        "valid": True,
        "note": f"null independent signals; false positive per trial defined as |lag|>{lag_threshold_ms}ms and |corr|>{corr_threshold}",
    }


# ---------------------------------------------------------------------------
# Full T7 ensemble analysis (position-aware)
# ---------------------------------------------------------------------------

def _pearson_p(r: float, n: int) -> float:
    if not np.isfinite(r) or n < 3 or abs(r) >= 1:
        return float("nan")
    # t = r * sqrt((n-2)/(1-r^2))
    try:
        t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
        p = 2 * st.t.sf(abs(t), df=n - 2)
        return float(p)
    except Exception:
        return float("nan")


def compute_t7(
    rate: np.ndarray,
    trial_conditions: List[str],
    *,
    fs_hz: float,
    dt_ms: float | None = None,
    areas: Tuple[str, ...] = AREAS_CANONICAL,
    window_ms: Tuple[float, float] = DEFAULT_WINDOW_MS,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
    pairs: List[Tuple[str, str]] | None = None,
    normalize: str = "zscore",
) -> Dict[str, Any]:
    """Compute T7 cross-area lag distribution using unchanged estimator.

    Estimand: cross-area field/rate cross-correlation peak lag distribution.
    No anatomical delay inference: each pair's lag is estimated only from the two
    rate traces via estimate_lag_single.

    Position-aware: per p2/p3/p4, relevant trials are omission at that pos + intact
    evaluated at same absolute window. This mirrors T4 pooling_rule DO NOT pool before
    testing position dependence. Pooled secondary is also provided but flagged.

    Parameters
    ----------
    rate: [trial, area, time] at >= ms resolution
    trial_conditions: list length n_trials (12 canonical condition names)
    fs_hz, dt_ms: sampling; if dt_ms None, inferred as 1000/fs_hz
    areas: area ordering matching rate dim1
    window_ms: omission-local window relative to expected onset (default -1000,+1000)
    max_lag_ms: search range for peak
    pairs: list of (area_i, area_j) to test; default all 6 pairs
    normalize: passed to estimate_lag_single

    Returns
    -------
    dict with per_position stats, pooled, per_trial arrays, provenance
    """
    if dt_ms is None:
        dt_ms = 1000.0 / fs_hz
    _validate_rate(rate, trial_conditions, areas)
    n_trials = rate.shape[0]
    n_time = rate.shape[2]
    n_areas = len(areas)
    area_to_idx = {a: i for i, a in enumerate(areas)}
    if pairs is None:
        pairs = _all_pairs(areas)
    # Validate pairs
    for a, b in pairs:
        if a not in area_to_idx or b not in area_to_idx:
            raise ValueError(f"pair {(a,b)} not in areas {areas}")
    positions = ("p2", "p3", "p4")
    band_note = "not spectral; T7 is cross-area lag (field/rate) — window_ms and max_lag_ms frozen"

    # Preallocate per_position containers
    # We'll store per_position[ pos ][ pair_str ] = stats
    per_position: Dict[str, Any] = {}
    # Also per_trial arrays for artifact: [trial, pair_idx] lags/corrs
    pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
    n_pairs = len(pairs)

    # For pooled secondary, collect all per-trial results ignoring position (using own position windows)
    # For artifact backing, also store per_trial_position arrays: lag[trial, pair, pos]
    per_trial_position_lag = np.full((n_trials, n_pairs, len(positions)), np.nan, dtype=np.float64)
    per_trial_position_corr = np.full((n_trials, n_pairs, len(positions)), np.nan, dtype=np.float64)
    per_trial_position_valid = np.zeros((n_trials, n_pairs, len(positions)), dtype=bool)

    # Iterate positions
    for pos_idx, pos in enumerate(positions):
        om_conds = set(OMISSION_POSITIONS[pos])
        intact_conds = set(OMISSION_POSITIONS["intact"])
        is_om = np.array([c in om_conds for c in trial_conditions])
        is_intact = np.array([c in intact_conds for c in trial_conditions])
        is_relevant = is_om | is_intact
        relevant_indices = np.where(is_relevant)[0]
        n_om = int(is_om.sum())
        n_intact = int(is_intact.sum())
        n_rel = int(is_relevant.sum())

        # Window indices for this position (same absolute window for all relevant trials)
        i0, i1 = _window_for_position(pos, fs_hz, dt_ms, window_ms)
        i0c, i1c = max(0, i0), min(n_time, i1)
        window_valid = i1c > i0c and (i1c - i0c) >= 10  # need at least 10 samples
        band_stats: Dict[str, Any] = {}
        for pair in pairs:
            pair_str = f"{pair[0]}-{pair[1]}"
            ai = area_to_idx[pair[0]]
            bj = area_to_idx[pair[1]]
            peak_lags: List[float] = []
            peak_corrs: List[float] = []
            peak_lags_om: List[float] = []
            peak_lags_intact: List[float] = []
            peak_corrs_om: List[float] = []
            peak_corrs_intact: List[float] = []
            per_trial_lags = []  # for all relevant, in order of trial idx
            per_trial_corrs = []

            for t_idx in relevant_indices:
                if not window_valid:
                    lag_ms = float("nan")
                    corr = float("nan")
                    valid = False
                else:
                    x = rate[t_idx, ai, i0c:i1c]
                    y = rate[t_idx, bj, i0c:i1c]
                    res = estimate_lag_single(x, y, fs_hz, max_lag_ms=max_lag_ms, normalize=normalize)
                    lag_ms = float(res["peak_lag_ms"]) if res["valid"] else float("nan")
                    corr = float(res["peak_corr"]) if res["valid"] else float("nan")
                    valid = bool(res["valid"])
                peak_lags.append(lag_ms)
                peak_corrs.append(corr)
                per_trial_lags.append(lag_ms)
                per_trial_corrs.append(corr)
                # Store in per_trial_position arrays (by original trial index)
                pidx = pair_to_idx[pair]
                per_trial_position_lag[t_idx, pidx, pos_idx] = lag_ms
                per_trial_position_corr[t_idx, pidx, pos_idx] = corr
                per_trial_position_valid[t_idx, pidx, pos_idx] = valid
                # Split by condition type for diagnostic
                if trial_conditions[t_idx] in om_conds:
                    peak_lags_om.append(lag_ms)
                    peak_corrs_om.append(corr)
                else:
                    peak_lags_intact.append(lag_ms)
                    peak_corrs_intact.append(corr)

            peak_lags_arr = np.array(peak_lags, dtype=np.float64)
            peak_corrs_arr = np.array(peak_corrs, dtype=np.float64)
            # Compute ensemble stats over relevant trials (finite only)
            mask = np.isfinite(peak_lags_arr) & np.isfinite(peak_corrs_arr)
            lags_finite = peak_lags_arr[mask]
            corrs_finite = peak_corrs_arr[mask]
            n_finite = int(mask.sum())
            if n_finite >= 2:
                mean_lag = float(np.mean(lags_finite))
                median_lag = float(np.median(lags_finite))
                sd_lag = float(np.std(lags_finite, ddof=1))
                se_lag = sd_lag / math.sqrt(n_finite)
                # 95% CI via t
                tcrit = st.t.ppf(0.975, df=n_finite - 1) if n_finite > 1 else float("nan")
                ci_lo = mean_lag - tcrit * se_lag if np.isfinite(tcrit) else float("nan")
                ci_hi = mean_lag + tcrit * se_lag if np.isfinite(tcrit) else float("nan")
                # t-test vs zero (H0: no fixed lag)
                try:
                    t_stat, p_val = st.ttest_1samp(lags_finite, popmean=0.0)
                    t_stat = float(t_stat) if np.isfinite(t_stat) else float("nan")
                    p_val = float(p_val) if np.isfinite(p_val) else float("nan")
                except Exception:
                    t_stat, p_val = float("nan"), float("nan")
                # IQR
                q25, q75 = float(np.percentile(lags_finite, 25)), float(np.percentile(lags_finite, 75))
                iqr = q75 - q25
                # Proportion within thresholds
                prop_within_5 = float(np.mean(np.abs(lags_finite) <= 5.0))
                prop_within_10 = float(np.mean(np.abs(lags_finite) <= 10.0))
                prop_within_20 = float(np.mean(np.abs(lags_finite) <= 20.0))
                # Robust dispersion vs max_lag
                mean_abs_corr = float(np.mean(np.abs(corrs_finite))) if len(corrs_finite)>0 else float("nan")
                max_abs_corr = float(np.max(np.abs(corrs_finite))) if len(corrs_finite)>0 else float("nan")
                # Falsification: if strong fixed lag, p<0.05 and |mean|>threshold and std small
                has_fixed_lag = bool(p_val < 0.05 and abs(mean_lag) > 5.0 and sd_lag < 30.0) if np.isfinite(p_val) else False
            else:
                mean_lag = float(np.mean(lags_finite)) if n_finite>0 else float("nan")
                median_lag = float(np.median(lags_finite)) if n_finite>0 else float("nan")
                sd_lag = float("nan")
                se_lag = float("nan")
                ci_lo, ci_hi = float("nan"), float("nan")
                t_stat, p_val = float("nan"), float("nan")
                q25 = q75 = iqr = float("nan")
                prop_within_5 = prop_within_10 = prop_within_20 = float("nan")
                mean_abs_corr = float("nan")
                max_abs_corr = float("nan")
                has_fixed_lag = False

            # Also breakdown omission vs intact separately
            def _subset_stats(arr_list):
                arr = np.array(arr_list, dtype=np.float64)
                m = np.isfinite(arr)
                a = arr[m]
                if len(a) == 0:
                    return {"mean": float("nan"), "median": float("nan"), "sd": float("nan"), "n": 0}
                return {"mean": float(a.mean()), "median": float(np.median(a)), "sd": float(a.std(ddof=1)) if len(a)>1 else 0.0, "n": int(len(a))}

            subset_om = _subset_stats(peak_lags_om)
            subset_intact = _subset_stats(peak_lags_intact)

            band_stats[pair_str] = {
                "pair": list(pair),
                "n_relevant": n_rel,
                "n_omission": n_om,
                "n_intact": n_intact,
                "n_finite": n_finite,
                "peak_lags_ms": peak_lags,  # per relevant trial order
                "peak_corrs": peak_corrs,
                "peak_lags_omission_only": peak_lags_om,
                "peak_lags_intact_only": peak_lags_intact,
                "peak_corrs_omission_only": peak_corrs_om,
                "peak_corrs_intact_only": peak_corrs_intact,
                "relevant_trial_indices": relevant_indices.tolist(),
                "mean_lag_ms": mean_lag,
                "median_lag_ms": median_lag,
                "sd_lag_ms": sd_lag,
                "se_lag_ms": se_lag if 'se_lag' in locals() else float("nan"),
                "ci95_mean": [float(ci_lo), float(ci_hi)],
                "q25_ms": q25,
                "q75_ms": q75,
                "iqr_ms": iqr if 'iqr' in locals() else float("nan"),
                "t_vs_zero": t_stat,
                "p_vs_zero": p_val,
                "prop_within_5ms": prop_within_5,
                "prop_within_10ms": prop_within_10,
                "prop_within_20ms": prop_within_20,
                "mean_abs_corr": mean_abs_corr,
                "max_abs_corr": max_abs_corr,
                "has_fixed_lag": has_fixed_lag,
                "omission_mean_lag_ms": subset_om["mean"],
                "intact_mean_lag_ms": subset_intact["mean"],
                "omission_sd_ms": subset_om["sd"],
                "intact_sd_ms": subset_intact["sd"],
            }

        per_position[pos] = {
            "window_ms": list(window_ms),
            "max_lag_ms": float(max_lag_ms),
            "window_samples": [i0, i1, i0c, i1c],
            "n_omission_trials": n_om,
            "n_intact_trials": n_intact,
            "n_total_relevant": n_rel,
            "relevant_indices": relevant_indices.tolist(),
            "pair_stats": band_stats,
        }

    # Pooled secondary (ignores position, uses each trial's own position window where applicable)
    # For pooled, we take per_trial_position arrays and for each trial pick the column corresponding to its own position
    # Intact trials have no own position; we average over positions for pooled?
    # Simpler: pool = concatenation of all per_position finite lags across positions (relevant sets) — but this double counts intact
    # Instead define pooled as: for each trial, use lag at its own omission position if omission else mean lag across positions
    pooled_stats: Dict[str, Any] = {}
    for pair in pairs:
        pair_str = f"{pair[0]}-{pair[1]}"
        pidx = pair_to_idx[pair]
        pooled_lags: List[float] = []
        pooled_corrs: List[float] = []
        for t_idx, cond in enumerate(trial_conditions):
            pos = COND_TO_POS.get(cond)
            if pos is not None:
                # omission trial: use its own position column
                pos_idx = positions.index(pos)
                lag = float(per_trial_position_lag[t_idx, pidx, pos_idx])
                corr = float(per_trial_position_corr[t_idx, pidx, pos_idx])
                if np.isfinite(lag):
                    pooled_lags.append(lag)
                    pooled_corrs.append(corr)
            else:
                # intact: average across positions or collect per position? For pooled secondary we will collect mean
                lags = per_trial_position_lag[t_idx, pidx, :]
                corrs = per_trial_position_corr[t_idx, pidx, :]
                # Use mean over positions where finite
                mask = np.isfinite(lags) & np.isfinite(corrs)
                if mask.sum() == 0:
                    continue
                pooled_lags.append(float(lags[mask].mean()))
                pooled_corrs.append(float(corrs[mask].mean()))
        pooled_lags_arr = np.array(pooled_lags, dtype=np.float64)
        mask = np.isfinite(pooled_lags_arr)
        pf = pooled_lags_arr[mask]
        if len(pf) >= 2:
            m = float(pf.mean())
            sd = float(pf.std(ddof=1))
            try:
                t_stat, p_val = st.ttest_1samp(pf, popmean=0.0)
                t_stat = float(t_stat) if np.isfinite(t_stat) else float("nan")
                p_val = float(p_val) if np.isfinite(p_val) else float("nan")
            except Exception:
                t_stat, p_val = float("nan"), float("nan")
        else:
            m = float(pf.mean()) if len(pf)>0 else float("nan")
            sd = float("nan")
            t_stat, p_val = float("nan"), float("nan")
        pooled_stats[pair_str] = {
            "n_pooled": int(len(pf)),
            "mean_lag_ms": m,
            "sd_lag_ms": sd,
            "t_vs_zero": t_stat,
            "p_vs_zero": p_val,
            "lags_pooled": pooled_lags,
            "corrs_pooled": pooled_corrs,
            "note": "pooled p2/p3/p4; primary is per-position (see pooling_rule DO NOT pool until tested)",
        }

    # Denominators
    denominators: Dict[str, Any] = {}
    for pos in positions:
        denominators[pos] = {
            "omission_conditions": sorted(OMISSION_POSITIONS[pos]),
            "intact_conditions": sorted(OMISSION_POSITIONS["intact"]),
            "n_omission_trials": int(np.sum([c in set(OMISSION_POSITIONS[pos]) for c in trial_conditions])),
            "n_intact_trials": int(np.sum([c in set(OMISSION_POSITIONS["intact"]) for c in trial_conditions])),
        }
    denominators["total_trials"] = n_trials
    denominators["pairs"] = [list(p) for p in pairs]

    provenance = {
        "field_claim_level": FIELD_CLAIM_LEVEL,
        "physical_amplitude_calibrated": PHYSICAL_AMPLITUDE_CALIBRATED,
        "method": ESTIMATOR_PROVENANCE,
        "areas": list(areas),
        "positions": list(positions),
        "pairs": [list(p) for p in pairs],
        "n_trials": n_trials,
        "n_time": n_time,
        "fs_hz": float(fs_hz),
        "dt_ms": float(dt_ms),
        "window_ms": list(window_ms),
        "max_lag_ms": float(max_lag_ms),
        "normalize": normalize,
        "pooling_rule": "DO NOT pool p2/p3/p4 until position dependence explicitly tested (Q11); pooled values secondary",
        "language_rule": "lfp_proxy remains proxy; never promote to physical LFP/CSD",
        "estimator": "estimate_lag_single (data-driven xcorr peak, unbiased)",
        "anatomical_delay_inference": "none — estimator never reads connectivity/delay tables",
        "owner": "generated",
        "generated_by": "jomission.analysis.t7_lag.compute_t7",
        "window_note": band_note,
    }

    return {
        "per_position": per_position,
        "pooled_secondary": pooled_stats,
        "per_trial_position_lag": per_trial_position_lag,  # [trial, pair, pos]
        "per_trial_position_corr": per_trial_position_corr,
        "per_trial_position_valid": per_trial_position_valid,
        "denominators": denominators,
        "provenance": provenance,
        "pairs": pairs,
        "areas": areas,
        "positions": positions,
        "fs_hz": float(fs_hz),
        "dt_ms": float(dt_ms),
    }


# ---------------------------------------------------------------------------
# Artifact orchestration
# ---------------------------------------------------------------------------

def build_rate_from_signals(
    signals: List[Any],
    model: Any | None = None,
    *,
    dt_ms: float = 0.1,
    areas: Tuple[str, ...] = AREAS_CANONICAL,
) -> Tuple[np.ndarray, List[str], float, Tuple[str, ...], Dict[str, Any]]:
    """Build rate[trial,area,time] from Signals via neuron_metadata.

    Similar to t4_t5_analysis.build_field_rate_arrays but rate-only.

    Returns (rate, trial_conditions, fs_hz, areas, meta)
    """
    # Resolve area indices
    meta0 = signals[0].metadata if hasattr(signals[0], "metadata") else {}
    neuron_meta = meta0.get("neuron_metadata")
    if neuron_meta is None and model is not None and hasattr(model, "neuron_table"):
        neuron_meta = model.neuron_table()
    if neuron_meta is None:
        raise ValueError("Need neuron_metadata to resolve area indices for rate")
    area_to_idx: Dict[str, List[int]] = {a: [] for a in areas}
    for pos, row in enumerate(neuron_meta):
        a = str(row.get("area", ""))
        if a in area_to_idx:
            area_to_idx[a].append(pos)
    for a in areas:
        if len(area_to_idx[a]) == 0:
            raise ValueError(f"No neurons for area {a}")

    n_trials = len(signals)
    # Determine n_time from first spikes
    # All signals should have same T because same duration
    T0 = np.asarray(signals[0].spikes).shape[0]
    rate = np.zeros((n_trials, len(areas), T0), dtype=np.float64)
    scale = 1000.0 / dt_ms
    for t_idx, sig in enumerate(signals):
        spikes = np.asarray(sig.spikes)
        T_spk = spikes.shape[0]
        T_use = min(T_spk, T0)
        for a_idx, area in enumerate(areas):
            idx = area_to_idx[area]
            r = spikes[:T_use, idx].mean(axis=1) * scale
            rate[t_idx, a_idx, :T_use] = r

    trial_conditions: List[str] = []
    for sig in signals:
        md = getattr(sig, "metadata", {}) or {}
        cond = md.get("condition") or md.get("cond") or "UNKNOWN"
        trial_conditions.append(str(cond))
    fs_hz = 1000.0 / dt_ms
    meta = {
        "areas": list(areas),
        "dt_ms": dt_ms,
        "fs_hz": fs_hz,
        "n_trials": n_trials,
        "n_time": T0,
        "owner": "generated",
        "provenance": "spike-mean rate per area, no anatomical delay",
        "generated_by": "jomission.analysis.t7_lag.build_rate_from_signals",
    }
    return rate, trial_conditions, fs_hz, areas, meta


def run_t7_analysis(
    rate: np.ndarray,
    trial_conditions: List[str],
    *,
    fs_hz: float,
    dt_ms: float | None = None,
    areas: Tuple[str, ...] = AREAS_CANONICAL,
    window_ms: Tuple[float, float] = DEFAULT_WINDOW_MS,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
    pairs: List[Tuple[str, str]] | None = None,
    out_dir: str | pathlib.Path | None = None,
    save_arrays: bool = True,
) -> Dict[str, Any]:
    """Run T7 and optionally save generated-owner arrays.

    Returns dict with t7 result, plus artifact paths.
    """
    if dt_ms is None:
        dt_ms = 1000.0 / fs_hz if fs_hz else 1.0
    t7 = compute_t7(
        rate, trial_conditions,
        fs_hz=fs_hz, dt_ms=dt_ms,
        areas=areas, window_ms=window_ms,
        max_lag_ms=max_lag_ms, pairs=pairs,
    )

    # Build light JSON (without heavy per-trial 3D arrays full, but include summaries)
    combined = {
        "namespace": "canonical_confirmatory",
        "generated_by": "jomission.analysis.t7_lag.run_t7_analysis",
        "owner": "generated",
        "field_claim_level": FIELD_CLAIM_LEVEL,
        "physical_amplitude_calibrated": PHYSICAL_AMPLITUDE_CALIBRATED,
        "method": ESTIMATOR_PROVENANCE,
        "fs_hz": float(fs_hz),
        "dt_ms": float(dt_ms),
        "areas": list(areas),
        "pairs": [list(p) for p in (pairs if pairs is not None else t7["pairs"])],
        "window_ms": list(window_ms),
        "max_lag_ms": float(max_lag_ms),
        "per_position_summary": {
            pos: {
                pair_str: {
                    "n_finite": stats["n_finite"],
                    "mean_lag_ms": stats["mean_lag_ms"],
                    "median_lag_ms": stats["median_lag_ms"],
                    "sd_lag_ms": stats["sd_lag_ms"],
                    "p_vs_zero": stats["p_vs_zero"],
                    "prop_within_10ms": stats["prop_within_10ms"],
                    "has_fixed_lag": stats["has_fixed_lag"],
                }
                for pair_str, stats in perpos["pair_stats"].items()
            }
            for pos, perpos in t7["per_position"].items()
        },
        "pooled_secondary": t7["pooled_secondary"],
        "denominators": t7["denominators"],
        "provenance": t7["provenance"],
    }

    artifact_paths: Dict[str, str] = {}
    if save_arrays and out_dir is not None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # JSON
        json_path = out / "t7_summary.json"
        def _json_safe(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            raise TypeError(type(o))
        with open(json_path, "w") as f:
            json.dump(combined, f, indent=2, default=_json_safe)
        artifact_paths["json"] = str(json_path)

        # NPZ per-trial heavy
        npz_path = out / "t7_per_trial_arrays.npz"
        np.savez_compressed(
            npz_path,
            per_trial_position_lag=t7["per_trial_position_lag"],
            per_trial_position_corr=t7["per_trial_position_corr"],
            per_trial_position_valid=t7["per_trial_position_valid"],
            trial_conditions=np.array(trial_conditions, dtype=object),
            areas=np.array(list(areas), dtype=object),
            pairs=np.array([f"{a}-{b}" for a, b in (pairs if pairs is not None else t7["pairs"])], dtype=object),
            fs_hz=np.array(fs_hz),
            dt_ms=np.array(dt_ms),
            window_ms=np.array(window_ms),
            max_lag_ms=np.array(max_lag_ms),
        )
        artifact_paths["npz"] = str(npz_path)

        # NPY lag summary matrices: [pair, position] mean lag
        pairs_list = pairs if pairs is not None else t7["pairs"]
        positions = ("p2", "p3", "p4")
        mean_lag_mat = np.full((len(pairs_list), len(positions)), np.nan)
        sd_lag_mat = np.full((len(pairs_list), len(positions)), np.nan)
        p_mat = np.full((len(pairs_list), len(positions)), np.nan)
        for pi, pair in enumerate(pairs_list):
            pair_str = f"{pair[0]}-{pair[1]}"
            for pj, pos in enumerate(positions):
                stats = t7["per_position"][pos]["pair_stats"][pair_str]
                mean_lag_mat[pi, pj] = stats["mean_lag_ms"] if np.isfinite(stats["mean_lag_ms"]) else np.nan
                sd_lag_mat[pi, pj] = stats["sd_lag_ms"] if np.isfinite(stats["sd_lag_ms"]) else np.nan
                p_mat[pi, pj] = stats["p_vs_zero"] if np.isfinite(stats["p_vs_zero"]) else np.nan
        npy_mean = out / "t7_mean_lag_pair_x_position.npy"
        npy_sd = out / "t7_sd_lag_pair_x_position.npy"
        npy_p = out / "t7_p_pair_x_position.npy"
        np.save(npy_mean, mean_lag_mat)
        np.save(npy_sd, sd_lag_mat)
        np.save(npy_p, p_mat)
        artifact_paths["mean_lag_npy"] = str(npy_mean)
        artifact_paths["sd_lag_npy"] = str(npy_sd)
        artifact_paths["p_npy"] = str(npy_p)

        # Provenance json
        prov_path = out / "t7_provenance.json"
        with open(prov_path, "w") as f:
            json.dump(t7["provenance"], f, indent=2, default=_json_safe)
        artifact_paths["provenance"] = str(prov_path)

        # Also hash for integrity
        sha = hashlib.sha256()
        with open(json_path, "rb") as f:
            sha.update(f.read())
        artifact_paths["sha256_json"] = sha.hexdigest()

    return {"t7": t7, "summary": combined, "artifacts": artifact_paths, "owner": "generated"}


# ---------------------------------------------------------------------------
# Controls orchestration for closure
# ---------------------------------------------------------------------------

def run_controls_validation(
    *,
    n_trials: int = 24,
    n_time: int = 2000,
    fs_hz: float = 1000.0,
    dt_ms: float | None = None,
    true_lag_ms_list: List[float] = [15.0, 30.0, 50.0, -20.0],
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
    noise_std: float = 0.3,
    out_dir: str | pathlib.Path | None = None,
) -> Dict[str, Any]:
    """Run positive / no-lag / null controls using the exact same estimator.

    Quantifies detection error and false-positive for T7 closure.

    Returns dict with per-lag positive results, nolag, null, and aggregated metrics.
    """
    if dt_ms is None:
        dt_ms = 1000.0 / fs_hz
    results: Dict[str, Any] = {}
    # Positive controls with varying lags
    positive: Dict[str, Any] = {}
    for true_lag in true_lag_ms_list:
        rate_pair = synthesize_rate_pair(
            n_trials=n_trials, n_time=n_time, fs_hz=fs_hz,
            true_lag_ms=true_lag, noise_std=noise_std, autocor=0.92,
            rng_seed=int(abs(true_lag * 10 + 7) % 10000),
        )
        # Use estimator on each trial's pair
        peak_lags = []
        peak_corrs = []
        for t in range(n_trials):
            x = rate_pair[t, 0, :]
            y = rate_pair[t, 1, :]
            res = estimate_lag_single(x, y, fs_hz=fs_hz, max_lag_ms=max_lag_ms)
            peak_lags.append(float(res["peak_lag_ms"]) if res["valid"] else float("nan"))
            peak_corrs.append(float(res["peak_corr"]) if res["valid"] else float("nan"))
        peak_lags_arr = np.array(peak_lags)
        peak_corrs_arr = np.array(peak_corrs)
        quant = quantify_positive_control(peak_lags_arr, true_lag_ms=true_lag, peak_corrs=peak_corrs_arr)
        positive[str(true_lag)] = {
            "true_lag_ms": true_lag,
            "peak_lags_ms": peak_lags,
            "peak_corrs": peak_corrs,
            "quant": quant,
        }
    results["positive_controls"] = positive

    # No-lag control (true 0)
    rate_nolag = synthesize_rate_nolag(
        n_trials=n_trials, n_time=n_time, fs_hz=fs_hz, n_areas=2,
        noise_std=noise_std, rng_seed=123,
    )
    lags_nolag: List[float] = []
    corrs_nolag: List[float] = []
    for t in range(n_trials):
        x = rate_nolag[t, 0, :]
        y = rate_nolag[t, 1, :]
        res = estimate_lag_single(x, y, fs_hz=fs_hz, max_lag_ms=max_lag_ms)
        lags_nolag.append(float(res["peak_lag_ms"]) if res["valid"] else float("nan"))
        corrs_nolag.append(float(res["peak_corr"]) if res["valid"] else float("nan"))
    lags_nolag_arr = np.array(lags_nolag)
    corrs_nolag_arr = np.array(corrs_nolag)
    quant_nolag = quantify_nolag_control(lags_nolag_arr, corrs_nolag_arr)
    results["nolag_control"] = {
        "peak_lags_ms": lags_nolag,
        "peak_corrs": corrs_nolag,
        "quant": quant_nolag,
    }

    # Null control (independent)
    rate_null = synthesize_rate_null(
        n_trials=n_trials, n_time=n_time, fs_hz=fs_hz, n_areas=2,
        noise_std=noise_std, rng_seed=999,
    )
    lags_null: List[float] = []
    corrs_null: List[float] = []
    for t in range(n_trials):
        x = rate_null[t, 0, :]
        y = rate_null[t, 1, :]
        res = estimate_lag_single(x, y, fs_hz=fs_hz, max_lag_ms=max_lag_ms)
        lags_null.append(float(res["peak_lag_ms"]) if res["valid"] else float("nan"))
        corrs_null.append(float(res["peak_corr"]) if res["valid"] else float("nan"))
    lags_null_arr = np.array(lags_null)
    corrs_null_arr = np.array(corrs_null)
    quant_null = quantify_null_control(lags_null_arr, corrs_null_arr)
    results["null_control"] = {
        "peak_lags_ms": lags_null,
        "peak_corrs": corrs_null,
        "quant": quant_null,
    }

    # Aggregate success criteria
    positive_success = all(
        positive[k]["quant"]["success"] is True for k in positive
    )
    nolag_success = bool(quant_nolag["mae_ms"] <= 5.0 and quant_nolag["frac_within_5ms"] >= 0.6)
    null_success = bool(
        quant_null["p_vs_zero"] > 0.05 or quant_null["false_positive_per_trial"] < 0.2
    ) if np.isfinite(quant_null["p_vs_zero"]) else bool(quant_null["false_positive_per_trial"] < 0.3)
    # For null, also expect std large (>20ms) indicating no concentration, and mean near 0
    results["aggregate"] = {
        "positive_success": bool(positive_success),
        "nolag_success": bool(nolag_success),
        "null_success": bool(null_success),
        "controls_pass": bool(positive_success and nolag_success and null_success),
        "max_lag_ms": float(max_lag_ms),
        "fs_hz": float(fs_hz),
        "dt_ms": float(dt_ms),
        "n_trials": n_trials,
        "n_time": n_time,
        "true_lags_tested": list(true_lag_ms_list),
        "note": "All controls used EXACT same estimator (estimate_lag_single); positive control quantifies detection error, null/nolag quantifies false-positive.",
    }
    # Provenance for controls
    results["provenance"] = {
        "method": ESTIMATOR_PROVENANCE,
        "estimator": "estimate_lag_single",
        "anatomical_delay_inference": "none",
        "positive_controls": {k: positive[k]["quant"] for k in positive},
        "nolag": quant_nolag,
        "null": quant_null,
        "owner": "generated",
        "generated_by": "jomission.analysis.t7_lag.run_controls_validation",
    }

    # Save if out_dir provided
    artifact_paths: Dict[str, str] = {}
    if out_dir is not None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Save JSON
        json_path = out / "t7_controls_validation.json"
        def _js(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(type(o))
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=_js)
        artifact_paths["json"] = str(json_path)
        # Save NPZ with per-trial arrays
        npz_path = out / "t7_controls_per_trial.npz"
        # Flatten positive controls into arrays
        save_dict: Dict[str, Any] = {}
        for k, v in positive.items():
            save_dict[f"positive_{k}_lags"] = np.array(v["peak_lags_ms"])
            save_dict[f"positive_{k}_corrs"] = np.array(v["peak_corrs"])
        save_dict["nolag_lags"] = np.array(lags_nolag)
        save_dict["nolag_corrs"] = np.array(corrs_nolag)
        save_dict["null_lags"] = np.array(lags_null)
        save_dict["null_corrs"] = np.array(corrs_null)
        np.savez_compressed(npz_path, **save_dict)
        artifact_paths["npz"] = str(npz_path)
        # Hash
        sha = hashlib.sha256()
        with open(json_path, "rb") as f:
            sha.update(f.read())
        artifact_paths["sha256"] = sha.hexdigest()
        results["artifacts"] = artifact_paths

    return results


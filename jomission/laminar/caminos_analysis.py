"""E0 estimators: peak frequency, PV-to-SST lag, population rates (docs/laminar_motif/E0_SEAL.md).

Subset of the JaxFNE etude estimators, verbatim except one call written in the 0.5.9 millisecond form
(``population_rates``; see the comment at the call) (artifacts/etudes/papers/caminos/caminos_analysis.py,
spec PREREG_P3): the constants and the functions ``kept_slice``, ``e_welch``, ``spectral_peak``,
``_gaussian_kernel``, ``pv_som_lag`` and ``analyze_run``. The R1 to R4 scorer is not ported (E0 scores
Q1 to Q3 only). The JaxFNE ``SOM`` label is the SST class. Input is per-step population counts
[steps, 3] in E, PV, SOM order with step ``dt`` in seconds.
"""

from __future__ import annotations

import numpy as np
from jaxfne.analysis import (
    aperiodic_spectral_peak,
    oscillation_call,
    population_rates,
    rate_welch_psd,
    xcorr_peak_lag,
)

DISCARD_S = 2.0  # first 2 s discarded (4 x tau_w)
WIN_S = 1.0  # Welch 1 s Hann windows
OVERLAP_FRAC = 0.5  # 50 % overlap
FIT_LO_HZ = 10.0  # aperiodic fit band
FIT_HI_HZ = 150.0
BAND_LO_HZ = 20.0  # peak search band
BAND_HI_HZ = 80.0
EXCL_HALF_HZ = 8.0  # refit excludes f_pk +- 8 Hz
PE_THRESHOLD = 4.0  # oscillates when peak excess >= 4
EDGE_HZ = 2.0  # f_pk not within 2 Hz of 20 or 80 Hz
LAG_WINDOW_MS = 14.0  # R2: search -14..+14 ms
SMOOTH_MS = 2.0  # R2: 2 ms Gaussian smoothing
POPS = ("E", "PV", "SOM")


def _as_float_counts(counts):
    arr = np.asarray(counts)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("counts must be [steps, 3], got shape %r" % (arr.shape,))
    return arr


def kept_slice(n_steps, dt):
    """Row slice after discarding the first 2 s."""
    cut = int(round(DISCARD_S / dt))
    if cut >= n_steps:
        raise ValueError("run shorter than the 2 s discard: %d steps at dt=%g" % (n_steps, dt))
    return slice(cut, None)


def e_welch(e_rate, dt):
    """Welch PSD of the E rate: mean removed, 1 s Hann, 50 % overlap."""
    return rate_welch_psd(e_rate, dt, win_s=WIN_S, overlap_frac=OVERLAP_FRAC)


def spectral_peak(e_rate, dt, pe_threshold=PE_THRESHOLD):
    """Peak excess and oscillation call for the E rate (R1, R4; v4)."""
    freqs, pxx = e_welch(e_rate, dt)
    pk = aperiodic_spectral_peak(
        freqs, pxx,
        fit_lo_hz=FIT_LO_HZ, fit_hi_hz=FIT_HI_HZ,
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ,
        excl_half_hz=EXCL_HALF_HZ, edge_hz=EDGE_HZ,
    )
    return {
        "peak_freq_Hz": pk["peak_freq_Hz"],
        "peak_excess": pk["peak_excess"],
        "background_Hz": pk["background_Hz"],
        "local_max": pk["local_max"],
        "edge_ok": pk["edge_ok"],
        "oscillates": oscillation_call(pk["peak_excess"], pk["edge_ok"], pe_threshold),
        "fit": pk["fit"],
        "fit_pass1": pk["fit_pass1"],
    }


def _gaussian_kernel(sigma_steps):
    radius = int(3 * sigma_steps)
    g = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (g / sigma_steps) ** 2)
    return k / k.sum()


def pv_som_lag(counts, dt, sizes):
    """PV->SOM rate cross-correlation peak lag in ms (R2). Positive means SOM after PV."""
    arr = _as_float_counts(counts)
    keep = arr[kept_slice(arr.shape[0], dt)]
    dt_ms = dt * 1000.0
    pv = keep[:, 1] / dt / sizes["PV"]
    som = keep[:, 2] / dt / sizes["SOM"]
    kernel = _gaussian_kernel(SMOOTH_MS / dt_ms)
    pv_s = np.convolve(pv, kernel, mode="same")
    som_s = np.convolve(som, kernel, mode="same")
    lag_steps = xcorr_peak_lag(som_s, pv_s, dt_ms, window_ms=LAG_WINDOW_MS)
    return float(lag_steps * dt_ms)


def analyze_run(counts, dt, sizes, pe_threshold=PE_THRESHOLD):
    """Estimator set for one run; a population with zero spikes leaves its estimands undefined."""
    arr = _as_float_counts(counts)
    keep = arr[kept_slice(arr.shape[0], dt)]
    spikes = {p: int(keep[:, i].sum()) for i, p in enumerate(POPS)}
    # Seconds form: accepted by JaxFNE 0.5.8 (the pinned authority) and by 0.5.9,
    # where it is deprecated. Same quantity as the ms form, equal up to rounding.
    rates = population_rates(arr, dt=dt, sizes=sizes, pops=POPS, discard_s=DISCARD_S)
    out = {
        "rates_Hz": rates,
        "spikes_kept": spikes,
        "peak_freq_E_Hz": None,
        "peak_excess_E": None,
        "oscillates": False,
        "lag_PV_SOM_ms": None,
        "undefined": [],
    }
    if spikes["E"] == 0:
        out["undefined"].append("peak")
    else:
        pk = spectral_peak(keep[:, 0] / dt / sizes["E"], dt, pe_threshold)
        out["peak_freq_E_Hz"] = pk["peak_freq_Hz"]
        out["peak_excess_E"] = pk["peak_excess"]
        out["oscillates"] = pk["oscillates"]
    if spikes["PV"] == 0 or spikes["SOM"] == 0:
        out["undefined"].append("lag")
    else:
        out["lag_PV_SOM_ms"] = pv_som_lag(arr, dt, sizes)
    return out

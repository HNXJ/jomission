"""T4/T5 area-resolved field analysis — frozen five-band x four-area x p2/p3/p4.

Frozen authorities (do not alter estimands based on results):
- T4: area/frontal omission-related LFP-like spectral changes across declared bands
      (theta 4-8, alpha 8-14, beta 14-30, low_gamma 30-50, high_gamma 50-80),
      preserving area and omission position. Source: jomission/analysis/comparison_matrix.py
      Comparison_matrix T4 estimand: "LFP-like band power (20-50 Hz) omission vs intact"
      contrast FEF/PFC vs V1, corrected. Task closure expands to five-band x 4-area x 3-position
      while preserving frontal vs V1 contrast (not opportunistic).
      Pooling rule: DO NOT pool p2/p3/p4 until position dependence explicitly tested (Q11).
- T5: gamma/rate coupling versus lower-frequency coupling, band-resolved, with frozen
      gamma-vs-low contrast. Comparison_matrix T5: "trial gamma power vs spike rate
      correlation" per unit across trials. Here area-level band-resolved (proxy of per-unit).
- Windows (reindexed t=0 at expected omission onset + absolute trial clock):
      omission_local (-1000,+1000), omission_baseline (-250,-50),
      omission_slot (0,531), post_omission (531,1000). Trial clock: fx -500, p1 0-531,
      d1 531-1031, p2 1031-1562, d2 1562-2062, p3 2062-2593, d3 2593-3093, p4 3093-3624, d4 3624-4124.
      Full trial = 4624 ms including fx. See jomission/paradigm/epochs.py and spec.py.
- Field claim: proxy_readout, physical_amplitude_calibrated=False, linear_solver
      (area_local partition). See jomission/recording/area_local.py provenance.
- No causal field->spike claim: coupling is correlational (Pearson r), not causal.

Consumes:
    field[trial,area,contact,time] — from Subagent A area_local path
          shape (n_trials, n_areas, n_contacts, n_time) or (n_trials, n_areas, time, contact)
          via jomission.recording.area_local.field_by_area_4d  (trial_A_C_T or trial_A_T_C)
    rate[trial,area,time] — area-mean instantaneous rate or spike count per bin, aligned in time
    event metadata: trial_conditions list[str] length n_trials, trial_phases optional,
          omission_positions dict, fs_hz / dt_ms.

If field data unavailable, caller should build via area_local path from Signals
(build_field_rate_arrays helper) — do not fabricate contact-averaged copies.

Generated-owner arrays: this module owns results under results/t4_t5/ or manifests/
with provenance json + npz. Claim stays proxy_readout.

Invariants:
- No pooling of p2/p3/p4 before per-position test.
- All 5 bands x 4 areas x 3 positions reported even if null.
- Denominators (n_trials per condition/phase/position) explicit.
- Uncertainty per-trial (mean, SD, SEM, 95% CI, Cohen d) and for coupling (Fisher z CI).
- Truth gates preserved in output metadata.
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

# Frozen five-band definition (task authority)
BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 30.0),
    "low_gamma": (30.0, 50.0),
    "high_gamma": (50.0, 80.0),
}
BAND_ORDER: Tuple[str, ...] = ("theta", "alpha", "beta", "low_gamma", "high_gamma")
LOWER_BANDS: Tuple[str, ...] = ("theta", "alpha", "beta")
GAMMA_BANDS: Tuple[str, ...] = ("low_gamma", "high_gamma")

AREAS_CANONICAL: Tuple[str, ...] = ("V1", "V4", "FEF", "PFC")
N_CONTACTS_DEFAULT: int = 16

# Absolute trial onsets for p-slots (ms, scheduler clock with fx -500)
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
# intact stays None mapping for convenience; explicit check via OMISSION_POSITIONS["intact"]

TRIAL_MS: float = 4624.0

PHYSICAL_AMPLITUDE_CALIBRATED: bool = False
FIELD_CLAIM_LEVEL: str = "proxy_readout"
FIELD_SOLVER_STATUS: str = "linear_solver"


# ---------------------------------------------------------------------------
# Helpers: spectral, windowing, validation
# ---------------------------------------------------------------------------

def _validate_field_rate(
    field: np.ndarray,
    rate: np.ndarray | None,
    trial_conditions: List[str],
    areas: Tuple[str, ...],
    n_contacts: int | None = None,
) -> Dict[str, Any]:
    if field.ndim != 4:
        raise ValueError(f"field must be 4D [trial,area,contact,time] or [trial,area,time,contact], got {field.shape}")
    n_trials = field.shape[0]
    if len(trial_conditions) != n_trials:
        raise ValueError(f"trial_conditions len {len(trial_conditions)} != n_trials {n_trials}")
    if field.shape[1] != len(areas):
        raise ValueError(f"field area dim {field.shape[1]} != len(areas) {len(areas)}")
    if rate is not None:
        if rate.shape[0] != n_trials or rate.shape[1] != len(areas):
            raise ValueError(f"rate shape {rate.shape} incompatible with field {field.shape}")
        if rate.shape[2] != field.shape[-1] and rate.shape[2] != field.shape[2]:
            # time axis may be swapped; check both layouts
            # field could be trial_A_C_T (time last) or trial_A_T_C (time third)
            # rate is trial_A_T
            pass
    return {"n_trials": n_trials, "areas": list(areas)}


def _infer_layout(field: np.ndarray, n_contacts: int = N_CONTACTS_DEFAULT) -> str:
    """Infer whether field is trial_A_C_T or trial_A_T_C."""
    # field shape (T,A,X,Y) with n_trials,T first, A second
    # One of X,Y is n_contacts (16), other is n_time (~46240 at dt 0.1)
    # n_contacts is small (16) vs n_time large
    dim2, dim3 = field.shape[2], field.shape[3]
    if dim2 == n_contacts and dim3 != n_contacts:
        return "trial_A_C_T"  # time last
    elif dim3 == n_contacts and dim2 != n_contacts:
        return "trial_A_T_C"  # time third, contact last
    else:
        # fallback: assume trial_A_C_T if ambiguous (both 16? not plausible for time)
        return "trial_A_C_T"


def _field_slice(
    field: np.ndarray,
    trial_idx: int,
    area_idx: int,
    t0: int,
    t1: int,
    layout: str,
) -> np.ndarray:
    """Extract [n_contacts, window_time] for one trial/area."""
    if layout == "trial_A_C_T":
        # field[trial, area, contact, time]
        return field[trial_idx, area_idx, :, t0:t1]  # [C, Tw]
    else:
        # trial_A_T_C: field[trial, area, time, contact] -> transpose
        return field[trial_idx, area_idx, t0:t1, :].T  # [C, Tw]


def _rate_slice(
    rate: np.ndarray,
    trial_idx: int,
    area_idx: int,
    t0: int,
    t1: int,
) -> np.ndarray:
    """Extract rate window [window_time] for one trial/area."""
    return rate[trial_idx, area_idx, t0:t1]


def _bandpower_periodogram(
    sig: np.ndarray,
    fs_hz: float,
    band: Tuple[float, float],
) -> float:
    """Periodogram band power for 1D signal sig [T]."""
    n = sig.shape[0]
    if n < 2:
        return 0.0
    x = sig - sig.mean()
    # rfft
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    psd = (np.abs(np.fft.rfft(x)) ** 2) / n  # not density-normalized; sum equals variance
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    # band power = sum(psd[mask]); optionally scale by 2/n? consistent across conditions, ratio unbiased
    # For real signal, rfft omits negative freqs except DC/Nyquist; periodogram doubling not needed for contrast
    return float(psd[mask].sum())


def _bandpower_multicontact(
    window_ct: np.ndarray,
    fs_hz: float,
    band: Tuple[float, float],
    average_contacts: bool = True,
) -> float | np.ndarray:
    """Window [C, Tw] -> band power per contact or mean over contacts."""
    per_contact = np.array([_bandpower_periodogram(window_ct[c], fs_hz, band) for c in range(window_ct.shape[0])])
    if average_contacts:
        return float(per_contact.mean())
    return per_contact


def _trial_window_indices(
    condition: str,
    fs_hz: float,
    dt_ms: float,
    window: Tuple[float, float],
    reference: str = "slot",
) -> Tuple[int, int]:
    """Map omission-local window to absolute trial sample indices.

    Evidence update (2026-08-24): empirical verification shows field time 0
    corresponds to p1 onset (schedule time 0), not fx onset. StimulusSchedule
    maps fx -500 to negative index clamped to 0, so p1 at 0, p2 at 1031, etc.
    Previous analysis (field_analysis.py) used 1031 directly and tests with
    rate confirm offset 0 gives omission decrement -20 Hz at p2, while offset
    500 gives no effect. Therefore we do NOT add 500 ms fx offset.
    See test_rate_alignment in tests for provenance.

    Parameters
    ----------
    condition: str — e.g. AXAB (p2 omission) or AAAB (intact)
    window: omission-local (lo,hi) ms relative to expected onset for that position.
            For intact trials, caller should use _window_for_position.
    reference: not used; kept for API symmetry.
    """
    pos = COND_TO_POS.get(condition)
    if pos is None:
        pos = "p2"
    onset_abs = SLOT_ONSET_MS[pos]
    lo_abs = onset_abs + window[0]
    hi_abs = onset_abs + window[1]
    i0 = int(round(lo_abs / dt_ms))
    i1 = int(round(hi_abs / dt_ms))
    return i0, i1


def _window_for_position(
    position: str,
    fs_hz: float,
    dt_ms: float,
    window: Tuple[float, float],
) -> Tuple[int, int]:
    """Position-explicit window (for intact trials). Evidence: field 0 = p1."""
    if position not in ("p2", "p3", "p4"):
        raise ValueError(f"position {position} not in p2/p3/p4")
    onset_abs = SLOT_ONSET_MS[position]
    lo_abs = onset_abs + window[0]
    hi_abs = onset_abs + window[1]
    i0 = int(round(lo_abs / dt_ms))
    i1 = int(round(hi_abs / dt_ms))
    return i0, i1


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = float(a.mean()), float(b.mean())
    sa, sb = float(a.std(ddof=1)), float(b.std(ddof=1))
    pooled = math.sqrt(((len(a)-1)*sa*sa + (len(b)-1)*sb*sb) / (len(a)+len(b)-2))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def _mean_ci(a: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    if len(a) < 2:
        return (float("nan"), float("nan"))
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a)))
    # t critical
    tcrit = st.t.ppf(1 - alpha/2, df=len(a)-1)
    return (m - tcrit*se, m + tcrit*se)


# ---------------------------------------------------------------------------
# Build field/rate arrays from Signals via area_local (fallback when field unavailable)
# ---------------------------------------------------------------------------

def build_field_rate_arrays(
    signals: List[Any],
    model: Any | None = None,
    *,
    dt_ms: float = 0.1,
    areas: Tuple[str, ...] = AREAS_CANONICAL,
    layout: str = "trial_A_C_T",
) -> Tuple[np.ndarray, np.ndarray, List[str], float, Tuple[str, ...], Dict[str, Any]]:
    """Build field[trial,area,contact,time] and rate[trial,area,time] from Signals via area_local.
    
    Parameters
    ----------
    signals: list of jaxfne Signals (one per trial)
    model: jomission Model for neuron_metadata (optional if metadata in signals)
    dt_ms: sampling interval (0.1 canonical, 1.0 pilot)
    areas: area ordering
    layout: "trial_A_C_T" (default, contact second) or "trial_A_T_C"
    
    Returns
    -------
    field: np.ndarray shape (n_trials, n_areas, n_contacts, n_time) if trial_A_C_T else (n_trials,n_areas,n_time,n_contacts)
    rate: np.ndarray shape (n_trials, n_areas, n_time) — area-mean spike rate (Hz) derived from spikes
    trial_conditions: list[str] — caller must set externally; here we return placeholder conditions from metadata if present else "UNKNOWN"
    fs_hz: float
    areas: tuple
    meta: dict provenance (proxy_readout etc)
    """
    from jomission.recording.area_local import field_by_area_4d, field_by_area_array

    # Determine layout flag for area_local
    time_major = layout == "trial_A_T_C"
    field, ret_areas, field_meta = field_by_area_4d(signals, model, time_major=time_major)
    # field_meta already carries proxy_readout etc

    # Build rate[trial, area, time] from spikes
    # Need area indices
    # Resolve from model or first signal metadata
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
    # Validate
    for a in areas:
        if len(area_to_idx[a]) == 0:
            raise ValueError(f"No neurons for area {a}")

    n_trials = len(signals)
    # time length from field
    if layout == "trial_A_C_T":
        n_time = field.shape[3]
    else:
        n_time = field.shape[2]
    rate = np.zeros((n_trials, len(areas), n_time), dtype=np.float32)
    # spikes are [T, N] per trial, at dt_ms resolution, binary 0/1 per bin?
    # Convert to Hz: spike * (1000/dt_ms) if binary, or already rate? Use mean per bin scaled.
    scale = 1000.0 / dt_ms
    for t_idx, sig in enumerate(signals):
        spikes = np.asarray(sig.spikes)  # [T, N]
        # spikes may have different T than field if record_fields differently? Assume same T
        # Align lengths: if spikes longer than n_time, crop; if shorter, pad? We'll crop/pad.
        T_spk = spikes.shape[0]
        T_use = min(T_spk, n_time)
        for a_idx, area in enumerate(areas):
            idx = area_to_idx[area]
            # mean over neurons -> [T]
            r = spikes[:T_use, idx].mean(axis=1) * scale  # Hz
            rate[t_idx, a_idx, :T_use] = r
            # if n_time longer, remaining stays 0 (should not happen)
    # Trial conditions placeholder: try to extract from signal metadata if available
    trial_conditions = []
    for sig in signals:
        md = getattr(sig, "metadata", {}) or {}
        cond = md.get("condition") or md.get("cond") or "UNKNOWN"
        trial_conditions.append(str(cond))
    fs_hz = 1000.0 / dt_ms
    meta = {
        "field_meta": field_meta,
        "areas": list(areas),
        "layout": layout,
        "dt_ms": dt_ms,
        "fs_hz": fs_hz,
        "n_trials": n_trials,
        "n_contacts": field.shape[2] if layout == "trial_A_C_T" else field.shape[3],
        "field_claim_level": FIELD_CLAIM_LEVEL,
        "physical_amplitude_calibrated": PHYSICAL_AMPLITUDE_CALIBRATED,
        "field_solver_status": FIELD_SOLVER_STATUS,
        "provenance": "area_local linear partition (proxy) + spike-mean rate; no causal field->spike",
        "generated_by": "t4_t5_analysis.build_field_rate_arrays",
        "owner": "generated",
    }
    return field, rate, trial_conditions, fs_hz, ret_areas, meta


# ---------------------------------------------------------------------------
# T4: five-band x four-area x p2/p3/p4 omission-related LFP-like band power
# ---------------------------------------------------------------------------

def compute_t4(
    field: np.ndarray,
    trial_conditions: List[str],
    *,
    fs_hz: float,
    dt_ms: float | None = None,
    areas: Tuple[str, ...] = AREAS_CANONICAL,
    bands: Dict[str, Tuple[float, float]] = BANDS,
    window_slot: Tuple[float, float] = OMISSION_SLOT_MS,
    window_baseline: Tuple[float, float] = OMISSION_BASELINE_MS,
    average_contacts: bool = True,
    baseline_normalization: str = "none",  # "none" or "ratio" or "difference"
) -> Dict[str, Any]:
    """T4: area x band x position omission vs intact band power.
    
    Frozen estimand: LFP-like band power omission vs intact (proxy_readout).
    Expanded to 5 bands x 4 areas x 3 positions, preserving area and position.
    
    Contrast per area/band/position: 
        omission power (slot window) vs intact power (same absolute slot position)
        Statistic: difference (omission - intact), ratio, Cohen d, t-test.
    
    Also computes slot_vs_baseline per trial (diagnostic) but primary contrast is
    omission vs intact at same position.
    
    Parameters
    ----------
    field: [trial, area, contact, time] or [trial, area, time, contact]
    trial_conditions: list of condition names (12 canonical) length n_trials
    fs_hz: sampling rate
    dt_ms: if None, inferred as 1000/fs_hz
    areas: area order matching field dim1
    bands: band dict
    window_slot: omission slot (0,531)
    window_baseline: (-250,-50) for diagnostic ratio
    average_contacts: mean power over contacts before statistics (True) else keep per-contact
    baseline_normalization: if "ratio" compute slot/baseline per trial then compare
    
    Returns
    -------
    dict with keys:
        per_trial: dict band -> array [n_trials, n_areas] (power per trial per area)
        per_position: dict position -> band -> area -> stats dict
        denominators: n per condition/position
        provenance
    """
    if dt_ms is None:
        dt_ms = 1000.0 / fs_hz
    _validate_field_rate(field, None, trial_conditions, areas)
    layout = _infer_layout(field)
    n_trials = field.shape[0]
    n_areas = len(areas)
    # Determine n_time for clipping
    n_time = field.shape[3] if layout == "trial_A_C_T" else field.shape[2]

    # Precompute per-trial, per-area, per-band power for BOTH slot and baseline windows
    # For omission trials, slot is at their own position; for intact, we will evaluate at each position separately
    # To allow omission vs intact contrast per position, we need power evaluated at each position's window
    # So we compute a 4D array: power[trial, area, band, position] for position in p2/p3/p4
    positions = ("p2", "p3", "p4")
    band_names = list(bands.keys())
    n_bands = len(band_names)

    # per_position_power[trial, area, band, pos_idx] 
    per_trial_position_power = np.full((n_trials, n_areas, n_bands, len(positions)), np.nan, dtype=np.float64)
    per_trial_baseline_power = np.full((n_trials, n_areas, n_bands, len(positions)), np.nan, dtype=np.float64)

    for t_idx, cond in enumerate(trial_conditions):
        for a_idx, area in enumerate(areas):
            for pos_idx, pos in enumerate(positions):
                i0, i1 = _window_for_position(pos, fs_hz, dt_ms, window_slot)
                b0, b1 = _window_for_position(pos, fs_hz, dt_ms, window_baseline)
                # Clip to valid range
                i0c, i1c = max(0, i0), min(n_time, i1)
                b0c, b1c = max(0, b0), min(n_time, b1)
                if i1c <= i0c or b1c <= b0c:
                    continue
                w_slot = _field_slice(field, t_idx, a_idx, i0c, i1c, layout)  # [C, Tw]
                w_base = _field_slice(field, t_idx, a_idx, b0c, b1c, layout)
                for b_idx, bname in enumerate(band_names):
                    band = bands[bname]
                    p_slot = _bandpower_multicontact(w_slot, fs_hz, band, average_contacts=average_contacts)
                    p_base = _bandpower_multicontact(w_base, fs_hz, band, average_contacts=average_contacts)
                    per_trial_position_power[t_idx, a_idx, b_idx, pos_idx] = float(p_slot) if np.ndim(p_slot)==0 else float(np.asarray(p_slot).mean())
                    per_trial_baseline_power[t_idx, a_idx, b_idx, pos_idx] = float(p_base) if np.ndim(p_base)==0 else float(np.asarray(p_base).mean())

    # Also compute per-trial power at own position's slot vs baseline ratio diagnostic
    per_trial_slot_power_own = np.full((n_trials, n_areas, n_bands), np.nan)
    per_trial_baseline_power_own = np.full((n_trials, n_areas, n_bands), np.nan)
    for t_idx, cond in enumerate(trial_conditions):
        pos = COND_TO_POS.get(cond)
        # For intact, no own position — leave nan; for omission, use its position
        if pos is None:
            continue
        pos_idx = positions.index(pos)
        for a_idx in range(n_areas):
            for b_idx in range(n_bands):
                per_trial_slot_power_own[t_idx, a_idx, b_idx] = per_trial_position_power[t_idx, a_idx, b_idx, pos_idx]
                per_trial_baseline_power_own[t_idx, a_idx, b_idx] = per_trial_baseline_power[t_idx, a_idx, b_idx, pos_idx]

    # Build denominators and per-position contrasts omission vs intact
    # For each position, define omission conditions = OMISSION_POSITIONS[pos], intact = OMISSION_POSITIONS["intact"]
    denominators: Dict[str, Any] = {}
    per_position_stats: Dict[str, Any] = {}

    for pos_idx, pos in enumerate(positions):
        om_conds = set(OMISSION_POSITIONS[pos])
        intact_conds = set(OMISSION_POSITIONS["intact"])
        # Masks
        is_om = np.array([c in om_conds for c in trial_conditions])
        is_intact = np.array([c in intact_conds for c in trial_conditions])
        n_om = int(is_om.sum())
        n_intact = int(is_intact.sum())
        denominators[pos] = {
            "omission_conditions": sorted(om_conds),
            "intact_conditions": sorted(intact_conds),
            "n_omission_trials": n_om,
            "n_intact_trials": n_intact,
            "n_total_position_relevant": n_om + n_intact,
            "omission_trial_indices": np.where(is_om)[0].tolist(),
            "intact_trial_indices": np.where(is_intact)[0].tolist(),
        }
        # For each band and area, compute stats
        band_stats: Dict[str, Any] = {}
        for b_idx, bname in enumerate(band_names):
            band_stats[bname] = {}
            for a_idx, area in enumerate(areas):
                om_vals = per_trial_position_power[is_om, a_idx, b_idx, pos_idx]
                intact_vals = per_trial_position_power[is_intact, a_idx, b_idx, pos_idx]
                # Remove nans
                om_vals = om_vals[np.isfinite(om_vals)]
                intact_vals = intact_vals[np.isfinite(intact_vals)]
                # Also compute baseline-normalized if requested
                if baseline_normalization == "ratio":
                    om_base = per_trial_baseline_power[is_om, a_idx, b_idx, pos_idx]
                    int_base = per_trial_baseline_power[is_intact, a_idx, b_idx, pos_idx]
                    om_base = om_base[np.isfinite(om_base)]
                    int_base = int_base[np.isfinite(int_base)]
                    # ratio slot/baseline per trial
                    om_vals_ratio = om_vals / np.maximum(om_base, 1e-12) if len(om_base)==len(om_vals) else om_vals
                    intact_vals_ratio = intact_vals / np.maximum(int_base, 1e-12) if len(int_base)==len(intact_vals) else intact_vals
                    # use ratio values for contrast
                    om_use = om_vals_ratio
                    intact_use = intact_vals_ratio
                else:
                    om_use = om_vals
                    intact_use = intact_vals

                # Difference and ratio
                mean_om = float(np.mean(om_use)) if len(om_use)>0 else float("nan")
                mean_intact = float(np.mean(intact_use)) if len(intact_use)>0 else float("nan")
                diff = mean_om - mean_intact if np.isfinite(mean_om) and np.isfinite(mean_intact) else float("nan")
                ratio = mean_om / max(mean_intact, 1e-12) if np.isfinite(mean_om) and np.isfinite(mean_intact) and mean_intact!=0 else float("nan")
                log_ratio = math.log(ratio) if np.isfinite(ratio) and ratio>0 else float("nan")
                # Variability
                sd_om = float(np.std(om_use, ddof=1)) if len(om_use)>1 else 0.0
                sd_intact = float(np.std(intact_use, ddof=1)) if len(intact_use)>1 else 0.0
                se_om = sd_om / math.sqrt(len(om_use)) if len(om_use)>0 else float("nan")
                se_intact = sd_intact / math.sqrt(len(intact_use)) if len(intact_use)>0 else float("nan")
                sem_diff = math.sqrt(se_om**2 + se_intact**2) if np.isfinite(se_om) and np.isfinite(se_intact) else float("nan")
                ci_om = _mean_ci(om_use)
                ci_intact = _mean_ci(intact_use)
                # Cohen d
                d = _cohens_d(om_use, intact_use)
                # t-test (two-sided, unequal var)
                if len(om_use)>=2 and len(intact_use)>=2:
                    tstat, pval = st.ttest_ind(om_use, intact_use, equal_var=False, nan_policy='omit')
                    tstat = float(tstat) if np.isfinite(tstat) else float("nan")
                    pval = float(pval) if np.isfinite(pval) else 1.0
                    # also permutation? For closure we provide t; permutation can be added later
                else:
                    tstat, pval = float("nan"), float("nan")
                # 95% CI for diff via t
                if np.isfinite(diff) and np.isfinite(sem_diff) and len(om_use)+len(intact_use)>2:
                    # approximate using normal
                    ci_diff_lo = diff - 1.96*sem_diff
                    ci_diff_hi = diff + 1.96*sem_diff
                else:
                    ci_diff_lo, ci_diff_hi = float("nan"), float("nan")

                band_stats[bname][area] = {
                    "n_omission": len(om_use),
                    "n_intact": len(intact_use),
                    "mean_omission": mean_om,
                    "mean_intact": mean_intact,
                    "diff_om_minus_intact": diff,
                    "ratio_om_over_intact": ratio,
                    "log_ratio": log_ratio,
                    "sd_omission": sd_om,
                    "sd_intact": sd_intact,
                    "se_omission": se_om,
                    "se_intact": se_intact,
                    "ci95_omission": [float(ci_om[0]), float(ci_om[1])],
                    "ci95_intact": [float(ci_intact[0]), float(ci_intact[1])],
                    "ci95_diff": [float(ci_diff_lo), float(ci_diff_hi)],
                    "cohens_d": float(d),
                    "t_stat": tstat,
                    "p_value_two_sided": pval,
                    # per-trial values for artifact
                    "per_trial_omission_values": om_use.tolist() if len(om_use)<1000 else om_use[:1000].tolist(),
                    "per_trial_intact_values": intact_use.tolist() if len(intact_use)<1000 else intact_use[:1000].tolist(),
                    # also report frontal vs V1 contrast later aggregated
                }
            # Add frontal vs V1 summary for this band/position
            # Frontal = FEF,PFC average; V1 single
        per_position_stats[pos] = band_stats

    # Also compute frontal vs V1 contrast aggregate per band/position
    frontal_vs_v1: Dict[str, Any] = {}
    for pos in positions:
        frontal_vs_v1[pos] = {}
        for bname in band_names:
            # diff frontal minus V1 for omission-vs-intact effect?
            # For each area, we have diff_om_minus_intact; frontal contrast = mean(FEF,PFC diff) - V1 diff
            diffs = {}
            for area in areas:
                diffs[area] = per_position_stats[pos][bname][area]["diff_om_minus_intact"]
            frontal_mean = float(np.mean([diffs[a] for a in ("FEF","PFC") if np.isfinite(diffs[a])]))
            v1_diff = diffs["V1"]
            contrast = frontal_mean - v1_diff if np.isfinite(frontal_mean) and np.isfinite(v1_diff) else float("nan")
            frontal_vs_v1[pos][bname] = {
                "frontal_mean_diff": frontal_mean,
                "v1_diff": float(v1_diff) if np.isfinite(v1_diff) else float("nan"),
                "frontal_minus_v1": float(contrast) if np.isfinite(contrast) else float("nan"),
            }

    # Pooled across positions (secondary, flagged as pooled; primary is per-position)
    pooled: Dict[str, Any] = {}
    for b_idx, bname in enumerate(band_names):
        pooled[bname] = {}
        for a_idx, area in enumerate(areas):
            # Collect all omission trials across p2/p3/p4 and all intact
            # For omission pooled, average over positions' powers? Instead collect per trial's own position power where available
            # Use per_trial_slot_power_own for omissions, and average over positions for intact? Simpler: collect over all position evaluations
            # We will pool by taking all omission trials' values at their own position vs all intact trials averaged over positions
            om_vals = per_trial_slot_power_own[:, a_idx, b_idx]
            om_vals = om_vals[np.isfinite(om_vals)]
            # For intact, we need power averaged over positions? Compute mean over positions for each intact trial
            # For each intact trial, per_trial_position_power gives 3 positions; average them
            intact_indices = np.where(np.array([c in set(OMISSION_POSITIONS["intact"]) for c in trial_conditions]))[0]
            intact_vals = []
            for t_idx in intact_indices:
                vals = per_trial_position_power[t_idx, a_idx, b_idx, :]
                vals = vals[np.isfinite(vals)]
                if len(vals)>0:
                    intact_vals.append(float(vals.mean()))
            intact_vals = np.array(intact_vals)
            mean_om = float(om_vals.mean()) if len(om_vals)>0 else float("nan")
            mean_int = float(intact_vals.mean()) if len(intact_vals)>0 else float("nan")
            diff = mean_om - mean_int if np.isfinite(mean_om) and np.isfinite(mean_int) else float("nan")
            pooled[bname][area] = {
                "n_omission_pooled": len(om_vals),
                "n_intact_pooled": len(intact_vals),
                "mean_omission_pooled": mean_om,
                "mean_intact_pooled": mean_int,
                "diff_pooled": diff,
                "note": "pooled p2/p3/p4; primary is per-position (see pooling_rule DO NOT pool until tested)",
            }

    provenance = {
        "field_claim_level": FIELD_CLAIM_LEVEL,
        "physical_amplitude_calibrated": PHYSICAL_AMPLITUDE_CALIBRATED,
        "field_solver_status": FIELD_SOLVER_STATUS,
        "bands": {k: list(v) for k, v in bands.items()},
        "band_order": list(band_names),
        "areas": list(areas),
        "positions": list(positions),
        "n_trials": n_trials,
        "n_contacts": field.shape[2] if layout=="trial_A_C_T" else field.shape[3],
        "layout": layout,
        "fs_hz": float(fs_hz),
        "dt_ms": float(dt_ms),
        "window_slot_ms": list(window_slot),
        "window_baseline_ms": list(window_baseline),
        "baseline_normalization": baseline_normalization,
        "average_contacts": bool(average_contacts),
        "pooling_rule": "DO NOT pool p2/p3/p4 until position dependence explicitly tested (Q11); pooled values secondary",
        "language_rule": "lfp_proxy remains proxy; never promote to physical LFP/CSD",
        "owner": "generated",
        "generated_by": "jomission.analysis.t4_t5_analysis.compute_t4",
        "method": "periodogram (rfft) bandpower, per-contact mean, omission vs intact same position, t-test per band/area/position",
    }

    return {
        "per_trial_position_power": per_trial_position_power,  # [trial, area, band, pos]
        "per_trial_baseline_power": per_trial_baseline_power,
        "per_trial_slot_power_own": per_trial_slot_power_own,
        "per_position": per_position_stats,
        "frontal_vs_v1": frontal_vs_v1,
        "pooled_secondary": pooled,
        "denominators": denominators,
        "provenance": provenance,
        "band_order": band_names,
        "positions": positions,
        "areas": areas,
    }


# ---------------------------------------------------------------------------
# T5: band-resolved gamma vs lower coupling
# ---------------------------------------------------------------------------

def _pearson_r(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson r and two-sided p."""
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3 or np.std(x)==0 or np.std(y)==0:
        return 0.0, 1.0
    r, p = st.pearsonr(x, y)
    return float(r), float(p)


def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if not np.isfinite(r) or n < 4:
        return (float("nan"), float("nan"))
    # Fisher z
    z = 0.5 * math.log((1 + r) / (1 - r)) if abs(r) < 1 else math.copysign(10, r)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = st.norm.ppf(1 - alpha/2)
    lo_z = z - zcrit*se
    hi_z = z + zcrit*se
    # inverse
    lo = (math.exp(2*lo_z) - 1) / (math.exp(2*lo_z) + 1)
    hi = (math.exp(2*hi_z) - 1) / (math.exp(2*hi_z) + 1)
    return (float(lo), float(hi))


def compute_t5(
    field: np.ndarray,
    rate: np.ndarray,
    trial_conditions: List[str],
    *,
    fs_hz: float,
    dt_ms: float | None = None,
    areas: Tuple[str, ...] = AREAS_CANONICAL,
    bands: Dict[str, Tuple[float, float]] = BANDS,
    window_slot: Tuple[float, float] = OMISSION_SLOT_MS,
    average_contacts: bool = True,
) -> Dict[str, Any]:
    """T5: band-resolved gamma/rate coupling versus lower-frequency coupling.

    For each area and band, correlate across trials:
        X = band power in omission slot (position-specific or pooled)
        Y = mean spike rate in same slot
    Primary contrast (frozen gamma-vs-low): mean gamma (low+high) vs mean lower (theta/alpha/beta).
    Also report per-band r and low_gamma vs lower, high_gamma vs lower.

    Field is proxy_readout; coupling is correlational, no causal field->spike claim.

    Returns dict with per trial arrays, per area band r, gamma-vs-low contrast, denominators.
    """
    if dt_ms is None:
        dt_ms = 1000.0 / fs_hz
    _validate_field_rate(field, rate, trial_conditions, areas)
    layout = _infer_layout(field)
    n_trials = field.shape[0]
    n_areas = len(areas)
    n_time = field.shape[3] if layout == "trial_A_C_T" else field.shape[2]
    band_names = list(bands.keys())
    n_bands = len(band_names)
    positions = ("p2", "p3", "p4")

    # Compute per trial, per area, per band: band power in slot (position-specific? need to choose)
    # For T5 overall coupling, we likely pool across positions but use each trial's own slot position's power vs rate
    # For intact trials, average over positions or use mean? Instead compute power at each position and also pooled.
    # We'll compute both per-position and pooled.

    # per_trial_bandpower_own: [trial, area, band] using trial's own omission position if omission else mean over positions
    per_trial_bandpower = np.full((n_trials, n_areas, n_bands), np.nan, dtype=np.float64)
    per_trial_rate = np.full((n_trials, n_areas), np.nan, dtype=np.float64)

    for t_idx, cond in enumerate(trial_conditions):
        pos = COND_TO_POS.get(cond)
        # For rate, window is same as field slot
        if pos is not None:
            i0, i1 = _window_for_position(pos, fs_hz, dt_ms, window_slot)
        else:
            # intact: average over positions for both field and rate? Let's compute mean over positions' windows
            # We'll later average; for now compute mean across positions for this trial
            # Instead compute power as mean over positions for intact
            pass
        # For intact handling below, we branch

    # Actually loop with explicit handling
    for t_idx, cond in enumerate(trial_conditions):
        for a_idx in range(n_areas):
            # Rate window: if omission, use its position slot; if intact, average over 3 positions
            pos = COND_TO_POS.get(cond)
            if pos is not None:
                i0, i1 = _window_for_position(pos, fs_hz, dt_ms, window_slot)
                i0c, i1c = max(0, i0), min(n_time, i1)
                if i1c > i0c:
                    w_rate = _rate_slice(rate, t_idx, a_idx, i0c, i1c)
                    per_trial_rate[t_idx, a_idx] = float(w_rate.mean()) if len(w_rate)>0 else float("nan")
                    w_field = _field_slice(field, t_idx, a_idx, i0c, i1c, layout)
                    for b_idx, bname in enumerate(band_names):
                        p = _bandpower_multicontact(w_field, fs_hz, bands[bname], average_contacts=average_contacts)
                        per_trial_bandpower[t_idx, a_idx, b_idx] = float(p) if np.ndim(p)==0 else float(np.asarray(p).mean())
                else:
                    per_trial_rate[t_idx, a_idx] = float("nan")
            else:
                # intact: compute mean over p2/p3/p4 windows
                rates = []
                powers = {b: [] for b in band_names}
                for pos in positions:
                    i0, i1 = _window_for_position(pos, fs_hz, dt_ms, window_slot)
                    i0c, i1c = max(0, i0), min(n_time, i1)
                    if i1c <= i0c:
                        continue
                    w_rate = _rate_slice(rate, t_idx, a_idx, i0c, i1c)
                    rates.append(float(w_rate.mean()))
                    w_field = _field_slice(field, t_idx, a_idx, i0c, i1c, layout)
                    for b_idx, bname in enumerate(band_names):
                        p = _bandpower_multicontact(w_field, fs_hz, bands[bname], average_contacts=average_contacts)
                        powers[bname].append(float(p) if np.ndim(p)==0 else float(np.asarray(p).mean()))
                per_trial_rate[t_idx, a_idx] = float(np.mean(rates)) if len(rates)>0 else float("nan")
                for b_idx, bname in enumerate(band_names):
                    per_trial_bandpower[t_idx, a_idx, b_idx] = float(np.mean(powers[bname])) if len(powers[bname])>0 else float("nan")

    # Now per area per band correlation across trials
    per_area_band: Dict[str, Any] = {}
    for a_idx, area in enumerate(areas):
        per_area_band[area] = {}
        for b_idx, bname in enumerate(band_names):
            x = per_trial_bandpower[:, a_idx, b_idx]
            y = per_trial_rate[:, a_idx]
            # mask finite
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n >= 3:
                r, p = _pearson_r(x[mask], y[mask])
                ci = _fisher_ci(r, n)
                # also Spearman for robustness
                try:
                    rs, ps = st.spearmanr(x[mask], y[mask])
                    rs = float(rs) if np.isfinite(rs) else float("nan")
                    ps = float(ps) if np.isfinite(ps) else float("nan")
                except Exception:
                    rs, ps = float("nan"), float("nan")
            else:
                r, p = float("nan"), float("nan")
                ci = (float("nan"), float("nan"))
                rs, ps = float("nan"), float("nan")
            per_area_band[area][bname] = {
                "n": n,
                "pearson_r": float(r) if np.isfinite(r) else float("nan"),
                "p_value": float(p) if np.isfinite(p) else float("nan"),
                "ci95": [float(ci[0]), float(ci[1])],
                "spearman_r": rs,
                "spearman_p": ps,
                # per-trial values for artifact (capped)
                "per_trial_bandpower": x.tolist() if n<1000 else x[:1000].tolist(),
                "per_trial_rate": y.tolist() if len(y)<1000 else y[:1000].tolist(),
            }

    # Gamma vs low contrast per area
    gamma_vs_low: Dict[str, Any] = {}
    for area in areas:
        lower_rs = [per_area_band[area][b]["pearson_r"] for b in LOWER_BANDS if np.isfinite(per_area_band[area][b]["pearson_r"])]
        gamma_rs = [per_area_band[area][b]["pearson_r"] for b in GAMMA_BANDS if np.isfinite(per_area_band[area][b]["pearson_r"])]
        mean_lower = float(np.mean(lower_rs)) if len(lower_rs)>0 else float("nan")
        mean_gamma = float(np.mean(gamma_rs)) if len(gamma_rs)>0 else float("nan")
        diff_gamma_minus_lower = mean_gamma - mean_lower if np.isfinite(mean_gamma) and np.isfinite(mean_lower) else float("nan")
        # Also low_gamma alone vs lower, high_gamma alone vs lower
        low_g = per_area_band[area]["low_gamma"]["pearson_r"]
        high_g = per_area_band[area]["high_gamma"]["pearson_r"]
        low_vs_lower = low_g - mean_lower if np.isfinite(low_g) and np.isfinite(mean_lower) else float("nan")
        high_vs_lower = high_g - mean_lower if np.isfinite(high_g) and np.isfinite(mean_lower) else float("nan")
        # Statistical test: are gamma rs > lower? Use permutation or t? Simple: compare means via Fisher? Provide descriptive.
        # We can also do Steiger test for dependent correlations? But for now provide diff and note.

        gamma_vs_low[area] = {
            "mean_lower_r": float(mean_lower) if np.isfinite(mean_lower) else float("nan"),
            "mean_gamma_r": float(mean_gamma) if np.isfinite(mean_gamma) else float("nan"),
            "gamma_minus_lower": float(diff_gamma_minus_lower) if np.isfinite(diff_gamma_minus_lower) else float("nan"),
            "low_gamma_r": float(low_g) if np.isfinite(low_g) else float("nan"),
            "high_gamma_r": float(high_g) if np.isfinite(high_g) else float("nan"),
            "low_gamma_minus_lower": float(low_vs_lower) if np.isfinite(low_vs_lower) else float("nan"),
            "high_gamma_minus_lower": float(high_vs_lower) if np.isfinite(high_vs_lower) else float("nan"),
            "n_trials": int(np.mean([per_area_band[area][b]["n"] for b in band_names if np.isfinite(per_area_band[area][b]["n"])] ) ) if band_names else 0,
            "interpretation": "positive if gamma coupling > lower; correlational, not causal field->spike",
        }

    # Also per-position T5 coupling (secondary): compute r per position
    per_position: Dict[str, Any] = {}
    for pos in positions:
        per_position[pos] = {}
        om_conds = set(OMISSION_POSITIONS[pos])
        intact_conds = set(OMISSION_POSITIONS["intact"])
        # For per-position, include both omission and intact trials that are relevant to that position
        # Actually for correlation we need trials at that position: omission pos trials + intact trials evaluated at that position
        # So we need to recompute per-trial values at that specific position (not pooled mean)
        # Recompute quickly: per_trial_bandpower_pos and per_trial_rate_pos
        # Build arrays for this position only
        pos_bandpower = np.full((n_trials, n_areas, n_bands), np.nan)
        pos_rate = np.full((n_trials, n_areas), np.nan)
        for t_idx, cond in enumerate(trial_conditions):
            # Only include trials that are either omission at pos or intact
            if cond not in om_conds and cond not in intact_conds:
                continue
            for a_idx in range(n_areas):
                i0, i1 = _window_for_position(pos, fs_hz, dt_ms, window_slot)
                i0c, i1c = max(0, i0), min(n_time, i1)
                if i1c <= i0c:
                    continue
                w_rate = _rate_slice(rate, t_idx, a_idx, i0c, i1c)
                pos_rate[t_idx, a_idx] = float(w_rate.mean())
                w_field = _field_slice(field, t_idx, a_idx, i0c, i1c, layout)
                for b_idx, bname in enumerate(band_names):
                    p = _bandpower_multicontact(w_field, fs_hz, bands[bname], average_contacts=average_contacts)
                    pos_bandpower[t_idx, a_idx, b_idx] = float(p) if np.ndim(p)==0 else float(np.asarray(p).mean())
        # Now compute per area per band correlation for this position's trial subset
        for a_idx, area in enumerate(areas):
            per_position[pos][area] = {}
            # Find relevant indices
            mask_trials = np.array([c in om_conds or c in intact_conds for c in trial_conditions])
            for b_idx, bname in enumerate(band_names):
                x = pos_bandpower[mask_trials, a_idx, b_idx]
                y = pos_rate[mask_trials, a_idx]
                m = np.isfinite(x) & np.isfinite(y)
                n = int(m.sum())
                if n >= 3:
                    r, p = _pearson_r(x[m], y[m])
                    ci = _fisher_ci(r, n)
                else:
                    r, p = float("nan"), float("nan")
                    ci = (float("nan"), float("nan"))
                per_position[pos][area][bname] = {"n": n, "pearson_r": float(r) if np.isfinite(r) else float("nan"), "p": float(p) if np.isfinite(p) else float("nan"), "ci95": [float(ci[0]), float(ci[1])]}

    denominators = {
        "n_trials_total": n_trials,
        "n_per_condition": {c: int(np.sum(np.array(trial_conditions)==c)) for c in sorted(set(trial_conditions))},
        "n_per_position": {pos: int(np.sum([c in set(OMISSION_POSITIONS[pos]) for c in trial_conditions])) for pos in positions},
        "n_intact": int(np.sum([c in set(OMISSION_POSITIONS["intact"]) for c in trial_conditions])),
        "areas": list(areas),
        "bands": list(band_names),
    }

    provenance = {
        "field_claim_level": FIELD_CLAIM_LEVEL,
        "physical_amplitude_calibrated": PHYSICAL_AMPLITUDE_CALIBRATED,
        "field_solver_status": FIELD_SOLVER_STATUS,
        "bands": {k: list(v) for k, v in bands.items()},
        "band_order": list(band_names),
        "lower_bands": list(LOWER_BANDS),
        "gamma_bands": list(GAMMA_BANDS),
        "areas": list(areas),
        "positions": list(positions),
        "n_trials": n_trials,
        "fs_hz": float(fs_hz),
        "dt_ms": float(dt_ms),
        "window_slot_ms": list(window_slot),
        "average_contacts": bool(average_contacts),
        "method": "bandpower (periodogram) vs rate Pearson r across trials, Fisher z CI; gamma-vs-low contrast frozen",
        "causal_claim": "correlational only; no causal field->spike interpretation (see 00_RULES_INVARIANTS)",
        "owner": "generated",
        "generated_by": "jomission.analysis.t4_t5_analysis.compute_t5",
    }

    return {
        "per_trial_bandpower": per_trial_bandpower,  # [trial, area, band]
        "per_trial_rate": per_trial_rate,  # [trial, area]
        "per_area_band": per_area_band,
        "gamma_vs_low": gamma_vs_low,
        "per_position": per_position,
        "denominators": denominators,
        "provenance": provenance,
        "band_order": band_names,
        "areas": areas,
    }


# ---------------------------------------------------------------------------
# Orchestration + artifact saving
# ---------------------------------------------------------------------------

def run_t4_t5_analysis(
    field: np.ndarray,
    rate: np.ndarray,
    trial_conditions: List[str],
    *,
    fs_hz: float,
    dt_ms: float | None = None,
    areas: Tuple[str, ...] = AREAS_CANONICAL,
    out_dir: str | pathlib.Path | None = None,
    save_arrays: bool = True,
) -> Dict[str, Any]:
    """Run both T4 and T5 and optionally save generated-owner arrays.
    
    Returns dict with t4, t5, combined provenance and artifact paths.
    """
    if dt_ms is None:
        dt_ms = 1000.0 / fs_hz
    t4 = compute_t4(field, trial_conditions, fs_hz=fs_hz, dt_ms=dt_ms, areas=areas)
    t5 = compute_t5(field, rate, trial_conditions, fs_hz=fs_hz, dt_ms=dt_ms, areas=areas)

    # Build combined result for JSON (arrays converted to lists or omitted for JSON)
    # Keep heavy arrays separate as npz
    combined = {
        "namespace": "canonical_confirmatory",
        "generated_by": "jomission.analysis.t4_t5_analysis.run_t4_t5_analysis",
        "owner": "generated",
        "field_claim_level": FIELD_CLAIM_LEVEL,
        "physical_amplitude_calibrated": PHYSICAL_AMPLITUDE_CALIBRATED,
        "field_solver_status": FIELD_SOLVER_STATUS,
        "causal_interpretation": "T5 coupling is correlational; no field->spike causal claim",
        "language_rule": "lfp_proxy remains proxy; never promote to physical LFP/CSD",
        "fs_hz": float(fs_hz),
        "dt_ms": float(dt_ms),
        "areas": list(areas),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "t4_summary": {
            "per_position_frontal_vs_v1": t4["frontal_vs_v1"],
            "denominators": t4["denominators"],
            "provenance": t4["provenance"],
        },
        "t5_summary": {
            "gamma_vs_low": t5["gamma_vs_low"],
            "per_area_band_r": {area: {b: t5["per_area_band"][area][b]["pearson_r"] for b in t5["band_order"]} for area in areas},
            "denominators": t5["denominators"],
            "provenance": t5["provenance"],
        },
        # Include full per-position diff tables (lightweight JSON)
        "t4_per_position": t4["per_position"],
        "t5_per_area_band_full": t5["per_area_band"],
    }

    artifact_paths: Dict[str, str] = {}
    if save_arrays and out_dir is not None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # JSON summary
        json_path = out / "t4_t5_summary.json"
        # Convert numpy types for JSON
        def _json_safe(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, (np.bool_)):
                return bool(o)
            raise TypeError(f"not json serializable {type(o)}")
        # Need to ensure combined is json serializable (per_position contains numpy)
        # Use json.dumps with default
        with open(json_path, "w") as f:
            json.dump(combined, f, indent=2, default=_json_safe)
        artifact_paths["json"] = str(json_path)

        # NPZ with heavy per-trial arrays
        npz_path = out / "t4_t5_per_trial_arrays.npz"
        np.savez_compressed(
            npz_path,
            t4_per_trial_position_power=t4["per_trial_position_power"],
            t4_per_trial_baseline_power=t4["per_trial_baseline_power"],
            t4_per_trial_slot_power_own=t4["per_trial_slot_power_own"],
            t5_per_trial_bandpower=t5["per_trial_bandpower"],
            t5_per_trial_rate=t5["per_trial_rate"],
            # also save trial conditions as bytes
            trial_conditions=np.array(trial_conditions, dtype=object),
            areas=np.array(list(areas), dtype=object),
            bands=np.array(list(BANDS.keys()), dtype=object),
            fs_hz=np.array(fs_hz),
            dt_ms=np.array(dt_ms),
        )
        artifact_paths["npz"] = str(npz_path)

        # Also save per-area band matrices as npy for generated-owner
        # Provide explicit generated-owner arrays: t4_diff_om_minus_intact [area,band,pos]
        positions = ("p2", "p3", "p4")
        diff_arr = np.full((len(areas), len(BANDS), len(positions)), np.nan)
        for pi, pos in enumerate(positions):
            for bi, bname in enumerate(BANDS):
                for ai, area in enumerate(areas):
                    diff_arr[ai, bi, pi] = t4["per_position"][pos][bname][area]["diff_om_minus_intact"]
        np.save(out / "t4_diff_om_minus_intact__area_band_pos.npy", diff_arr)
        artifact_paths["t4_diff_npy"] = str(out / "t4_diff_om_minus_intact__area_band_pos.npy")
        # t5 r matrix [area, band]
        r_arr = np.full((len(areas), len(BANDS)), np.nan)
        for ai, area in enumerate(areas):
            for bi, bname in enumerate(BANDS):
                r_arr[ai, bi] = t5["per_area_band"][area][bname]["pearson_r"]
        np.save(out / "t5_pearson_r__area_band.npy", r_arr)
        artifact_paths["t5_r_npy"] = str(out / "t5_pearson_r__area_band.npy")
        # gamma vs low contrast [area]
        gamma_low_arr = np.array([t5["gamma_vs_low"][area]["gamma_minus_lower"] for area in areas])
        np.save(out / "t5_gamma_minus_lower__area.npy", gamma_low_arr)
        artifact_paths["t5_gamma_low_npy"] = str(out / "t5_gamma_minus_lower__area.npy")

        # Provenance file
        prov_path = out / "provenance.json"
        prov = {
            "generated_by": "t4_t5_analysis.run_t4_t5_analysis",
            "owner": "generated",
            "field_claim_level": FIELD_CLAIM_LEVEL,
            "physical_amplitude_calibrated": PHYSICAL_AMPLITUDE_CALIBRATED,
            "field_solver_status": FIELD_SOLVER_STATUS,
            "no_causal_field_to_spike": True,
            "bands": BANDS,
            "areas": list(areas),
            "fs_hz": fs_hz,
            "dt_ms": dt_ms,
            "hash": hashlib.sha256(json.dumps(combined["t4_summary"], sort_keys=True, default=str).encode()).hexdigest()[:12],
        }
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2)
        artifact_paths["provenance"] = str(prov_path)

    return {"t4": t4, "t5": t5, "combined": combined, "artifacts": artifact_paths}


# Backwards-compat: expose build helper
__all__ = [
    "BANDS", "BAND_ORDER", "LOWER_BANDS", "GAMMA_BANDS",
    "AREAS_CANONICAL", "SLOT_ONSET_MS",
    "OMISSION_POSITIONS", "COND_TO_POS",
    "FIELD_CLAIM_LEVEL", "PHYSICAL_AMPLITUDE_CALIBRATED", "FIELD_SOLVER_STATUS",
    "build_field_rate_arrays", "compute_t4", "compute_t5", "run_t4_t5_analysis",
]

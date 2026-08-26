"""Semantic factor-isolation validator — Jomission Factorial v0.2.

Fixes the v0.1 defect where the syntactic validator (config_hash/hp_hash diff)
PASSED even though:
  (i)   cell A ran with the canonical RF-ON hdp params (hp f327f9d2ad64cc88,
        K_HDP=0.003) instead of the frozen RF-OFF params (bb8277e7a8e0bca2,
        K_HDP=0.0) — A failed frozen predicate C1;
  (ii)  RF drive total input energy differed 185× between RFoff (uniform
        all-400 drive, Σ≈42,480,000 per AAAB trial) and RFon (V1 L4 E/PV
        graded retinotopic drive, Σ≈230,181 per AAAB trial) — the "RF"
        intervention was a compound of config + graded drive + K_HDP, not an
        isolated retinotopy toggle;
  (iii) the rate (timescale) effect was measured from run-means over settled
        (ref) vs mid-relaxation (LONG) regimes — a transient-sampling artifact
        (β_tau), never a measured τ_eff from dense Θ(t).

This validator tests BOTH:
  (a) Δconfiguration  — config_hash / hp_hash / hdp-params diff as intended,
      but checked against the frozen PER-CELL identity (not the canonical
      label), and
  (b) Δrealized inputs/dynamics — the actual generated intervention, measured
      from the generated drive array (StimulusSchedule.to_array) and dense Θ(t)
      trace, NOT read from config.

Frozen reference numbers below are MEASURED from the real v0.1 cell runs
(results/rf_rate_factorial/{A,B}_RF*_RateRef/recording/external_drive_examples.npz
and H_t_dense.npz), so the validator evaluates realized facts, not nominal ones.

No frozen scientific config (jomission/ablations/rf_rate_factorial.py,
manifests/) is altered. Frozen artifacts are read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from jomission.ablations.rf_rate_factorial import (
    CELL_FACTORS,
    CELL_HP_HASHES,
    CELL_ORDER,
    FROZEN_CONFIG_HASH,
    hp_for_cell,
)
from jomission.paradigm.spec import SLOT_DURATION_MS, SLOT_ONSET_MS

SEMANTIC_VERSION = "factor_isolation.v0.2.0"

# ---------------------------------------------------------------------------
# Frozen tolerances (v0.2)
# ---------------------------------------------------------------------------
ENERGY_TOL_REL = 0.05          # total input energy must match within ±5%
SLOT_TOL_REL = 0.05            # per-slot envelope within ±5% of expected
STIM_IDENTITY_TOL_REL = 0.05   # A vs B per-presentation energy within ±5%
OMISSION_TOL_ABS = 1e-6        # omission slot energy must be ~exactly 0
MOMENT_TOL_REL = 0.10          # drive mean/std vs reference within ±10%
MOMENT_STD_ABS_RF_OFF = 1e-6   # uniform (RFoff) drive std must be ~0
HP_ABS_TOL = 1e-9              # hdp-param equality (fixed points / bounds)
TAU_RATIO_MIN = 50.0           # LONG must show measured τ_eff ≥ 50× ref (target 203×)
TAU_REF_RANGE_S = (2.0, 8.0)   # measured ref τ_eff ~4.1 s
TAU_LONG_MIN_S = 400.0         # measured LONG τ_eff target 833 s
TAU_LONG_MAX_S = 1500.0
TAU0_RATE_RATIO = 200.0        # frozen: tau_0 slow / standard = 1000/5

# ---------------------------------------------------------------------------
# Measured reference numbers from real v0.1 drives/schedules (not config)
# ---------------------------------------------------------------------------
# RFoff (cells A, C): uniform drive to all 400 neurons, amplitude 5.0,
# 4 intact p-slots × 531 ms @ dt 0.1 → 5310 steps/slot.
RF_OFF_REFERENCE_ENERGY_AAAB = 42480000.0
RF_OFF_REFERENCE_PER_SLOT = 10620000.0
RF_OFF_REFERENCE_ACTIVE_UNITS = 400
RF_OFF_REFERENCE_DRIVE_MEAN = 2.2967        # time-averaged drive, mean across units
RF_OFF_REFERENCE_DRIVE_STD = 0.0            # uniform → zero dispersion

# RFon (cells B, D): V1 L4 E/PV graded retinotopic drive (12 target units,
# sparsity 0.25 → 6 units active at 0.2·max over trial).
RF_ON_REFERENCE_ENERGY_AAAB = 230180.8125
RF_ON_REFERENCE_PER_SLOT = 57528.3          # p1/p2/p3; p4 ≈ 57595.9 (<0.2% spread)
RF_ON_REFERENCE_ACTIVE_UNITS = 6
RF_ON_REFERENCE_N_TARGET = 12               # V1 L4 E/PV resolved units
RF_ON_REFERENCE_SPARSITY = 0.25
RF_ON_REFERENCE_DRIVE_MEAN = 0.0124
RF_ON_REFERENCE_DRIVE_STD = 0.1170
# Graded-retinotopy dispersion, scale-invariant (v0.1 measured CV ≈ 9.4):
# survives energy normalization, unlike the absolute mean/std above.
RF_ON_REFERENCE_CV_RANGE = (4.0, 15.0)

# The v0.1 defect, frozen as the ratio the semantic invariant must reject.
RF_ENERGY_IMBALANCE_V01 = RF_OFF_REFERENCE_ENERGY_AAAB / RF_ON_REFERENCE_ENERGY_AAAB  # ≈ 184.6×

# Frozen timescale targets (design doc, PLASTICITY_RATE_INTERVENTION_DESIGN.md):
#   Rate standard tau_0 5.0 → τ_eff,E ≈ 4.1 s (saturates in ~12 s)
#   Rate slow     tau_0 1000.0 → τ_eff,E ≈ 833 s (76% of asymptote at 1200 s)
TAU_REF_NOMINAL_S = 4.1
TAU_LONG_NOMINAL_S = 833.0

# ---------------------------------------------------------------------------
# Realized data structures — MEASURED, not nominal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizedInputs:
    """Measured statistics of the actual generated external drive for one cell.

    Every field must be measured from the generated schedule/drive array
    (e.g. ``StimulusSchedule.to_array``) and the runtime, not read from config.
    """

    cell: str
    total_input_energy: float  # Σ_{t,i} I_i(t) over one representative intact trial
    target_indices: Tuple[int, ...] = ()
    v1_indices: Tuple[int, ...] = ()
    target_area: str = ""
    target_layers: Tuple[str, ...] = ()
    target_cell_types: Tuple[str, ...] = ()
    per_slot_energy: Dict[str, float] = field(default_factory=dict)
    slot_onsets_ms: Dict[str, float] = field(default_factory=dict)
    slot_durations_ms: Dict[str, float] = field(default_factory=dict)
    omission_energy: float = 0.0  # Σ drive in the omitted slot (must be 0)
    stimulus_identity_energy: Dict[str, float] = field(default_factory=dict)  # per-presentation A/B
    active_unit_count: Dict[str, int] = field(default_factory=dict)  # per stimulus at 0.2·max
    drive_mean: float = 0.0  # time-averaged drive, mean across units
    drive_std: float = 0.0  # time-averaged drive, std across units
    n_neurons: int = 400
    n_steps_per_trial: int = 46240
    # Δconfiguration facts — realized at runtime, not frozen labels
    config_hash: str = ""
    hp_hash: str = ""
    hdp_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealizedDynamics:
    """Measured dynamics from the dense Θ(t) trace (not nominal values)."""

    cell: str
    theta_t: np.ndarray  # dense Theta(t), shape (n,), over the measurement window
    theta_t_ms: np.ndarray  # time axis in ms, shape (n,)
    window_start_s: float = 0.0
    window_end_s: Optional[float] = None
    tau_effective_s: Optional[float] = None  # if None, fitted from theta_t
    theta_final: Optional[float] = None
    theta_initial: Optional[float] = None
    saturated: Optional[bool] = None  # Θ reached its asymptote inside the window
    h_bounds: Tuple[float, float] = (0.1, 10.0)
    w_bounds: Tuple[float, float] = (0.01, 10.0)


@dataclass(frozen=True)
class FactorPair:
    """A factor-isolation comparison between two frozen cells.

    intervention: "rf" (RFoff→RFon) or "timescale" (ref→LONG).
    """

    a: str
    b: str
    intervention: str

    @classmethod
    def of(cls, a: str, b: str) -> "FactorPair":
        fa, fb = CELL_FACTORS[a], CELL_FACTORS[b]
        if fa[0] != fb[0]:
            return cls(a=a, b=b, intervention="rf")
        if fa[1] != fb[1]:
            return cls(a=a, b=b, intervention="timescale")
        raise ValueError(f"{a} and {b} differ in neither RF nor Rate factor")

    @property
    def cells(self) -> Tuple[str, str]:
        return (self.a, self.b)


def _rel_error(x: float, ref: float) -> float:
    if ref == 0:
        return float("inf") if abs(x) > 0 else 0.0
    return float(abs(x - ref) / abs(ref))


# ---------------------------------------------------------------------------
# Δconfiguration (a)
# ---------------------------------------------------------------------------


def _check_configuration(cell: str, realized: RealizedInputs) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    rf, rate = CELL_FACTORS[cell]
    frozen_hp = hp_for_cell(cell)
    exp_hash = CELL_HP_HASHES[cell]

    # config_hash: RFoff cells must realize the canonical hash; RFon cells must
    # realize a distinct RF-metadata hash (presence of retinotopic config).
    cfg_expected = FROZEN_CONFIG_HASH if rf == "off" else f"!={FROZEN_CONFIG_HASH}"
    cfg_ok = bool(realized.config_hash == FROZEN_CONFIG_HASH) if rf == "off" else bool(
        realized.config_hash and realized.config_hash != FROZEN_CONFIG_HASH
    )
    checks["config_hash_match"] = {
        "pass": cfg_ok,
        "measured": realized.config_hash or "(not provided)",
        "expected": cfg_expected,
        "detail": f"{cell} ({rf}) config_hash must {'equal' if rf=='off' else 'differ from'} {FROZEN_CONFIG_HASH}",
    }

    # hp_hash must equal the frozen PER-CELL hash (this catches the A misconfig:
    # realized f327f9d2ad64cc88 ≠ frozen bb8277e7a8e0bca2 for A).
    hp_ok = bool(realized.hp_hash) and realized.hp_hash == exp_hash
    checks["hp_hash_match"] = {
        "pass": hp_ok,
        "measured": realized.hp_hash or "(not provided)",
        "expected": exp_hash,
        "detail": f"{cell} realized hp_hash must equal frozen CELL_HP_HASHES ({exp_hash})",
    }

    # hdp params: the realized runtime params must match the frozen per-cell
    # params — K_HDP by RF level, tau_0_ms by Rate level, and the fixed-point /
    # bound params (K_ctrl, K_w_ctrl, alpha, gamma, H_min/max, w_floor/ceiling)
    # unchanged across all cells.
    p = realized.hdp_params
    if not p:
        checks["hp_params_match"] = {
            "pass": False,
            "measured": "(not provided)",
            "expected": dict(frozen_hp),
            "detail": "realized hdp_params must be measured from the runtime, not omitted",
        }
    else:
        issues: List[str] = []
        for key, tol in [
            ("K_HDP", HP_ABS_TOL),
            ("tau_0_ms", HP_ABS_TOL),
            ("K_ctrl", HP_ABS_TOL),
            ("K_w_ctrl", HP_ABS_TOL),
            ("alpha", HP_ABS_TOL),
            ("gamma", HP_ABS_TOL),
            ("H_min", HP_ABS_TOL),
            ("H_max", HP_ABS_TOL),
            ("w_floor", HP_ABS_TOL),
            ("w_ceiling", HP_ABS_TOL),
        ]:
            got = p.get(key)
            exp = frozen_hp.get(key)
            if got is None or abs(float(got) - float(exp)) > tol:
                issues.append(f"{key} realized {got!r} != frozen {exp!r}")
        if rf == "off" and abs(float(p.get("K_HDP", -1)) - 0.0) > HP_ABS_TOL:
            issues.append("RFoff cell must realize K_HDP=0.0 (w frozen); got %r" % p.get("K_HDP"))
        if rate == "slow" and abs(float(p.get("tau_0_ms", -1)) - 1000.0) > HP_ABS_TOL:
            issues.append("Rate slow cell must realize tau_0_ms=1000.0; got %r" % p.get("tau_0_ms"))
        checks["hp_params_match"] = {
            "pass": not issues,
            "measured": dict(p),
            "expected": dict(frozen_hp),
            "detail": "; ".join(issues) if issues else f"{cell} realized hdp params match frozen per-cell params",
        }
    return checks


# ---------------------------------------------------------------------------
# RF semantics (b) — Δrealized inputs for an RFoff→RFon intervention
# ---------------------------------------------------------------------------


def _check_rf_semantics(pair: FactorPair, ri: Dict[str, RealizedInputs]) -> Dict[str, Any]:
    a, b = pair.a, pair.b
    ra, rb = ri[a], ri[b]
    checks: Dict[str, Any] = {}
    rf_levels = {"off": ra, "on": rb} if CELL_FACTORS[a][0] == "off" else {"off": rb, "on": ra}
    roff, ron = rf_levels["off"], rf_levels["on"]

    # --- target populations (V1 L4 E/PV vs all-400) -------------------------
    off_ok = (
        len(roff.target_indices) == roff.n_neurons
        and roff.target_area in ("all", "")
        and off_target_all_units(roff)
    )
    on_ok = (
        ron.target_area == "V1"
        and bool(set(ron.target_indices))
        and set(ron.target_indices) <= set(ron.v1_indices)
        and set(ron.target_layers) <= {"L4"}
        and set(ron.target_cell_types) <= {"E", "PV"}
        and RF_ON_REFERENCE_N_TARGET - 2 <= len(ron.target_indices) <= RF_ON_REFERENCE_N_TARGET + 2
    )
    checks["target_population"] = {
        "pass": bool(off_ok and on_ok),
        "measured": {
            "off": {"area": roff.target_area, "n_target": len(roff.target_indices)},
            "on": {"area": ron.target_area, "layers": list(ron.target_layers), "types": list(ron.target_cell_types), "n_target": len(ron.target_indices), "within_v1": bool(set(ron.target_indices) <= set(ron.v1_indices))},
        },
        "expected": {
            "off": {"area": "all", "n_target": 400},
            "on": {"area": "V1", "layers": ["L4"], "types": ["E", "PV"], "n_target": 12},
        },
        "detail": "RFoff must drive all-400; RFon must drive V1 L4 E/PV only (retinotopic targeting)",
    }

    # --- total input energy parity (isolates retinotopy, not energy) --------
    e_off, e_on = roff.total_input_energy, ron.total_input_energy
    e_hi = max(e_off, e_on)
    parity_ok = bool(e_hi > 0 and _rel_error(e_off, e_on) <= ENERGY_TOL_REL)
    checks["total_input_energy_parity"] = {
        "pass": parity_ok,
        "measured": {"RFoff": e_off, "RFon": e_on, "ratio": (e_off / e_on) if e_on else float("inf")},
        "expected": f"|E_off − E_on|/max ≤ {ENERGY_TOL_REL}",
        "detail": (
            f"RF intervention must preserve total input energy (v0.1 defect: "
            f"{RF_ENERGY_IMBALANCE_V01:.1f}× imbalance, {e_off:.3g} vs {e_on:.3g})"
        ),
    }

    # --- temporal envelope (per-slot drive sums, onset/duration) ------------
    env_issues: List[str] = []
    # slot geometry preserved
    for slot in SLOT_ONSET_MS:
        got_on = ra.slot_onsets_ms.get(slot)
        exp_on = SLOT_ONSET_MS[slot]
        if got_on is not None and abs(float(got_on) - exp_on) > 1e-6:
            env_issues.append(f"onset {slot}: {got_on} != {exp_on}")
        got_dur = ra.slot_durations_ms.get(slot)
        exp_dur = SLOT_DURATION_MS[slot]
        if got_dur is not None and abs(float(got_dur) - exp_dur) > 1e-6:
            env_issues.append(f"duration {slot}: {got_dur} != {exp_dur}")
    # per-slot energy: intact slots positive & ~equal share; delay/fx slots zero
    for cell, r in (("off", roff), ("on", ron)):
        per_slot = r.per_slot_energy
        intact = [s for s in ("p1", "p2", "p3", "p4") if per_slot.get(s, 0) > 0]
        if not intact:
            env_issues.append(f"{cell}: no intact p-slot energy measured")
            continue
        tot = sum(per_slot.get(s, 0) for s in intact)
        for s in intact:
            if _rel_error(per_slot[s], tot / len(intact)) > SLOT_TOL_REL:
                env_issues.append(f"{cell} {s} energy {per_slot[s]:.3g} deviates from share {tot/len(intact):.3g}")
        for s in ("d1", "d2", "d3", "d4", "fx"):
            if abs(per_slot.get(s, 0)) > OMISSION_TOL_ABS:
                env_issues.append(f"{cell} non-drive slot {s} energy {per_slot.get(s,0):.3g} != 0")
    checks["temporal_envelope"] = {
        "pass": not env_issues,
        "measured": {"per_slot_energy": {c: dict(r.per_slot_energy) for c, r in (("off", roff), ("on", ron))}},
        "expected": "intact slots ≈ equal share (±5%); delay/fx slots exactly 0; onsets/durations frozen",
        "detail": "; ".join(env_issues) if env_issues else "temporal envelope consistent with frozen paradigm",
    }

    # --- stimulus identity energy (A vs B per-presentation) -----------------
    sa = ra.stimulus_identity_energy.get("A", 0.0)
    sb = ra.stimulus_identity_energy.get("B", 0.0)
    stim_ok = bool(sa > 0 and sb > 0 and _rel_error(sa, sb) <= STIM_IDENTITY_TOL_REL)
    checks["stimulus_identity_energy"] = {
        "pass": stim_ok,
        "measured": {"A": sa, "B": sb},
        "expected": f"|E_A − E_B|/max ≤ {STIM_IDENTITY_TOL_REL} (symmetric blobs)",
        "detail": "A and B stimuli must inject equal energy per presentation (symmetric blobs at (8,8)/(24,24))",
    }

    # --- omission energy (must be exactly 0 in omitted slot) ----------------
    om_ok = bool(abs(ra.omission_energy) <= OMISSION_TOL_ABS and abs(rb.omission_energy) <= OMISSION_TOL_ABS)
    checks["omission_energy"] = {
        "pass": om_ok,
        "measured": {pair.a: ra.omission_energy, pair.b: rb.omission_energy},
        "expected": f"≤ {OMISSION_TOL_ABS} (exactly 0)",
        "detail": "omission must zero the drive in the omitted slot, preserving timing (no energy leak)",
    }

    # --- active-unit count per stimulus -------------------------------------
    act_off = ra.active_unit_count
    act_on = rb.active_unit_count
    off_cnt = [v for v in act_off.values() if v is not None] or []
    on_cnt = [v for v in act_on.values() if v is not None] or []
    off_ok = bool(off_cnt) and all(c == RF_OFF_REFERENCE_ACTIVE_UNITS for c in off_cnt)
    on_ok = bool(on_cnt) and all(RF_ON_REFERENCE_ACTIVE_UNITS - 3 <= c <= RF_ON_REFERENCE_ACTIVE_UNITS + 6 for c in on_cnt)
    checks["active_unit_count"] = {
        "pass": bool(off_ok and on_ok),
        "measured": {"RFoff": act_off, "RFon": act_on},
        "expected": {
            "RFoff": RF_OFF_REFERENCE_ACTIVE_UNITS,
            "RFon": f"{RF_ON_REFERENCE_ACTIVE_UNITS} ± (sparsity 0.18–0.30 of 12 targets)",
        },
        "detail": "RFoff drives all 400 units; RFon drives a sparse retinotopic subset",
    }

    # --- first/second moments of external drive across units ----------------
    # Gates are scale-invariant so a correctly energy-normalized RFon schedule
    # (which moves absolute mean up to the RFoff level) still passes:
    #   - internal consistency: mean drive == energy/(n_neurons × n_steps);
    #   - RFoff uniform → zero dispersion (std ≈ 0);
    #   - RFon graded → CV = std/mean in the frozen retinotopic range
    #     (v0.1 measured CV ≈ 9.4; absolute refs RF_ON_REFERENCE_DRIVE_* are
    #     reported for documentation of the defective v0.1 schedule).
    mom_issues: List[str] = []
    for r in (ra, rb):
        exp_mean = r.total_input_energy / (r.n_neurons * r.n_steps_per_trial) if r.n_neurons * r.n_steps_per_trial else 0.0
        if exp_mean > 0 and _rel_error(r.drive_mean, exp_mean) > MOMENT_TOL_REL:
            mom_issues.append(f"{r.cell} drive_mean {r.drive_mean:.4g} inconsistent with energy (expected {exp_mean:.4g})")
    if _rel_error(roff.drive_mean, RF_OFF_REFERENCE_DRIVE_MEAN) > MOMENT_TOL_REL:
        mom_issues.append(f"RFoff drive_mean {roff.drive_mean:.4g} != ref {RF_OFF_REFERENCE_DRIVE_MEAN}")
    if roff.drive_std > MOMENT_STD_ABS_RF_OFF:
        mom_issues.append(f"RFoff drive_std {roff.drive_std:.4g} != 0 (uniform drive must have zero dispersion)")
    cv_on = (ron.drive_std / ron.drive_mean) if ron.drive_mean > 0 else float("nan")
    if not (RF_ON_REFERENCE_CV_RANGE[0] <= cv_on <= RF_ON_REFERENCE_CV_RANGE[1]):
        mom_issues.append(f"RFon drive CV {cv_on:.3f} outside {RF_ON_REFERENCE_CV_RANGE} (v0.1 graded CV ≈ 9.4)")
    checks["drive_moments"] = {
        "pass": not mom_issues,
        "measured": {"RFoff": {"mean": roff.drive_mean, "std": roff.drive_std}, "RFon": {"mean": ron.drive_mean, "std": ron.drive_std, "cv": cv_on}},
        "expected": {"RFoff": {"mean": RF_OFF_REFERENCE_DRIVE_MEAN, "std": 0.0}, "RFon": {"cv": RF_ON_REFERENCE_CV_RANGE, "v0.1_abs_ref": {"mean": RF_ON_REFERENCE_DRIVE_MEAN, "std": RF_ON_REFERENCE_DRIVE_STD}}},
        "detail": "; ".join(mom_issues) if mom_issues else "drive first/second moments consistent with energy and frozen dispersion structure",
    }
    return checks


def off_target_all_units(r: RealizedInputs) -> bool:
    """RFoff drive must reach all n_neurons (uniform), not a targeted subset."""
    if not r.target_indices:
        return True  # no target_indices recorded → full-field drive (as in v0.1 A/C)
    return len(r.target_indices) == r.n_neurons


# ---------------------------------------------------------------------------
# Timescale semantics (b) — Δrealized inputs/dynamics for a ref→LONG intervention
# ---------------------------------------------------------------------------


def measure_tau_effective(theta_t: np.ndarray, theta_t_ms: np.ndarray) -> Dict[str, Any]:
    """Measure τ_eff from a dense Θ(t) trace (not nominal).

    Fits the first-order relaxation
        Θ(t) = Θ∞ − (Θ∞ − Θ0)·exp(−t/τ_eff)
    by nonlinear least squares (scipy.optimize.curve_fit) with Θ0 fixed at the
    measured initial value and (Θ∞, τ_eff) free. This is unbiased even for a
    partial relaxation (e.g. LONG at 76% of asymptote), unlike a log-linear
    fit which compresses when Θ∞ is underestimated from a truncated tail.

    Also reports how much of the relaxation was covered (tail_fraction = y at
    window end vs the fitted asymptote) so a measurement captured mid-relaxation
    can be flagged by the relaxation-window check.
    """
    from scipy.optimize import curve_fit

    theta = np.asarray(theta_t, dtype=float)
    t = np.asarray(theta_t_ms, dtype=float) / 1000.0  # ms → s
    if theta.ndim != 1 or t.ndim != 1 or theta.shape != t.shape or theta.shape[0] < 4:
        return {"tau_effective_s": float("nan"), "method": "invalid", "tail_fraction": 0.0, "span_s": 0.0}
    theta0 = float(theta[0])
    theta_hi = float(theta[-1])
    span_s = float(t[-1] - t[0])
    if span_s <= 0 or theta_hi <= theta0 + 1e-12:
        return {
            "tau_effective_s": float("inf") if span_s > 0 else float("nan"),
            "theta_initial": theta0,
            "theta_final": float(np.mean(theta[-max(1, int(0.2 * len(theta))):])),
            "tail_fraction": 1.0 if theta_hi > theta0 else 0.0,
            "span_s": span_s,
            "method": "no-dynamics",
        }

    def model(tt: np.ndarray, theta_inf: float, tau: float) -> np.ndarray:
        return theta_inf - (theta_inf - theta0) * np.exp(-tt / tau)

    # Initial guess: τ from the observed 63%-crossing (observed max as proxy).
    obs_amp = theta_hi - theta0
    thr = theta0 + 0.632 * obs_amp
    above = np.where(theta >= thr)[0]
    guess_tau = float(t[above[0]] - t[0]) if len(above) else span_s / 2.0
    guess_tau = max(guess_tau, span_s / len(theta))
    guess_inf = theta_hi + 0.25 * obs_amp
    tau, theta_inf, residual = float("nan"), float("nan"), float("nan")
    method = "unresolved"
    try:
        lo = [theta0 + 1e-9, 1e-6]
        hi = [theta0 + 10.0 * obs_amp + 1e-3, 1e7]
        popt, _ = curve_fit(model, t, theta, p0=[guess_inf, guess_tau], bounds=(lo, hi), maxfev=20000)
        theta_inf, tau = float(popt[0]), float(popt[1])
        residual = float(np.sqrt(np.mean((model(t, theta_inf, tau) - theta) ** 2)))
        method = "curve_fit"
    except Exception:
        # fallback: time-to-63% crossing with observed max as asymptote proxy
        if len(above):
            tau = float(t[above[0]] - t[0])
            theta_inf = guess_inf
            method = "crossing"
    y_end = float(np.clip((theta[-1] - theta0) / max(theta_inf - theta0, 1e-12), 0.0, 1.0)) if np.isfinite(theta_inf) else 0.0
    return {
        "tau_effective_s": float(tau),
        "theta_initial": theta0,
        "theta_final": float(np.mean(theta[-max(1, int(0.2 * len(theta))):])),
        "theta_infinity_fit": float(theta_inf),
        "tail_fraction": y_end,
        "span_s": span_s,
        "method": method,
        "fit_residual": float(residual) if np.isfinite(residual) else float("nan"),
    }


def _check_timescale_semantics(
    pair: FactorPair, ri: Dict[str, RealizedInputs], rd: Dict[str, RealizedDynamics]
) -> Dict[str, Any]:
    a, b = pair.a, pair.b
    checks: Dict[str, Any] = {}
    # Timescale pair: same RF level, ref → LONG (rate standard → slow)
    ref_cell, long_cell = (a, b) if CELL_FACTORS[a][1] == "standard" else (b, a)
    pa = ri[ref_cell].hdp_params
    pb = ri[long_cell].hdp_params

    # --- fixed-point equivalence (Θ∞ and H∞ invariant under Rate) -----------
    fp_issues: List[str] = []
    if not pa or not pb:
        fp_issues.append("realized hdp_params missing for one/both cells")
    else:
        for key in ("K_ctrl", "K_w_ctrl", "alpha", "gamma", "H_min", "H_max", "w_floor", "w_ceiling"):
            va, vb = pa.get(key), pb.get(key)
            if va is None or vb is None or abs(float(va) - float(vb)) > HP_ABS_TOL:
                fp_issues.append(f"{key} ref={va!r} != long={vb!r}")
        # K_HDP/K_w_ctrl ratio must be unchanged (w* invariant for given ΔH)
        def _ratio(p: Dict[str, Any]) -> float:
            kh = p.get("K_HDP", 0.0)
            kw = p.get("K_w_ctrl", 0.0)
            return float(kh / kw) if kw else float("nan")
        ra_, rb_ = _ratio(pa), _ratio(pb)
        if not (np.isfinite(ra_) and np.isfinite(rb_) and abs(ra_ - rb_) <= HP_ABS_TOL):
            fp_issues.append(f"K_HDP/K_w_ctrl ratio ref={ra_:.6g} != long={rb_:.6g}")
        # tau_0 must be the only rate knob, scaling by frozen 200×
        ta_, tb_ = pa.get("tau_0_ms"), pb.get("tau_0_ms")
        if ta_ is None or tb_ is None or abs(float(tb_) / float(ta_) - TAU0_RATE_RATIO) > HP_ABS_TOL:
            fp_issues.append(f"tau_0_ms ratio long/ref={float(tb_)/float(ta_) if ta_ else 'nan'} != {TAU0_RATE_RATIO}")
    checks["fixed_point_equivalence"] = {
        "pass": not fp_issues,
        "measured": {
            "ref_hp": dict(pa) if pa else None,
            "long_hp": dict(pb) if pb else None,
        },
        "expected": "K_HDP/K_w_ctrl, K_ctrl, alpha, gamma, bounds unchanged; tau_0 scales rate 200× only",
        "detail": "; ".join(fp_issues) if fp_issues else "fixed points (Θ∞, H∞) preserved — tau_0 scales rate only",
    }

    # --- measured τ_eff from dense Θ(t), not nominal ------------------------
    tau_issues: List[str] = []
    tau_meas: Dict[str, Dict[str, Any]] = {}
    if rd is None or ref_cell not in rd or long_cell not in rd:
        checks["measured_tau_effective"] = {
            "pass": False,
            "measured": {},
            "expected": "τ_eff measured from dense Θ(t) for both cells",
            "detail": "realized_dynamics (dense Θ(t)) required for timescale intervention — nominal config is not sufficient",
        }
    else:
        for cell in (ref_cell, long_cell):
            dyn = rd[cell]
            if dyn.tau_effective_s is not None:
                fit = {"tau_effective_s": float(dyn.tau_effective_s), "method": "reported"}
                if dyn.theta_t is not None and dyn.theta_t_ms is not None and len(np.asarray(dyn.theta_t)) > 1:
                    fit = measure_tau_effective(dyn.theta_t, dyn.theta_t_ms)
            elif dyn.theta_t is not None and dyn.theta_t_ms is not None and len(np.asarray(dyn.theta_t)) > 1:
                fit = measure_tau_effective(dyn.theta_t, dyn.theta_t_ms)
            else:
                fit = {"tau_effective_s": float("nan"), "method": "unresolved"}
            tau_meas[cell] = fit
        t_ref = tau_meas[ref_cell]["tau_effective_s"]
        t_long = tau_meas[long_cell]["tau_effective_s"]
        if not (np.isfinite(t_ref) and t_ref > 0):
            tau_issues.append(f"ref τ_eff not measurable: {t_ref!r}")
        elif not (TAU_REF_RANGE_S[0] <= t_ref <= TAU_REF_RANGE_S[1]):
            tau_issues.append(f"ref τ_eff {t_ref:.2f}s outside frozen {TAU_REF_RANGE_S} (target {TAU_REF_NOMINAL_S}s)")
        if not (np.isfinite(t_long) and t_long > 0):
            tau_issues.append(f"LONG τ_eff not measurable: {t_long!r}")
        elif not (TAU_LONG_MIN_S <= t_long <= TAU_LONG_MAX_S):
            tau_issues.append(f"LONG τ_eff {t_long:.1f}s outside frozen [{TAU_LONG_MIN_S},{TAU_LONG_MAX_S}] (target {TAU_LONG_NOMINAL_S}s)")
        if np.isfinite(t_ref) and np.isfinite(t_long) and t_ref > 0 and t_long / t_ref < TAU_RATIO_MIN:
            tau_issues.append(f"LONG τ_eff/ref = {t_long:.1f}/{t_ref:.2f} = {t_long/t_ref:.1f} < {TAU_RATIO_MIN} (target 203×, 833s vs 4.1s) — LONG NOT measurably slower")
    checks["measured_tau_effective"] = {
        "pass": not tau_issues,
        "measured": {k: {"tau_effective_s": v["tau_effective_s"], "method": v["method"]} for k, v in tau_meas.items()},
        "expected": {
            "ref": f"τ_eff ∈ {TAU_REF_RANGE_S}s (target {TAU_REF_NOMINAL_S}s)",
            "LONG": f"τ_eff ∈ [{TAU_LONG_MIN_S},{TAU_LONG_MAX_S}]s (target {TAU_LONG_NOMINAL_S}s)",
            "ratio": f"LONG/ref ≥ {TAU_RATIO_MIN}",
        },
        "detail": "; ".join(tau_issues) if tau_issues else "LONG measured τ_eff ≫ ref (833s vs 4.1s) confirmed from dense Θ(t)",
    }

    # --- relaxation window (catches "measured during relaxation") -----------
    win_issues: List[str] = []
    for cell in (ref_cell, long_cell):
        if cell not in rd:
            continue
        dyn = rd[cell]
        fit = tau_meas.get(cell, {})
        tau = fit.get("tau_effective_s", float("nan"))
        span = fit.get("span_s", 0.0)
        if np.isfinite(tau) and tau > 0 and span > 0 and span < tau:
            win_issues.append(
                f"{cell}: measurement window spans {span:.0f}s < τ_eff {tau:.0f}s — τ_eff estimated during relaxation"
            )
    checks["relaxation_window"] = {
        "pass": not win_issues,
        "measured": {k: {"span_s": v.get("span_s", float("nan")), "tail_fraction": v.get("tail_fraction", float("nan"))} for k, v in tau_meas.items()},
        "expected": "measurement window must span ≥ 1× τ_eff (fixed point reached, or ≥ full timescale covered)",
        "detail": "; ".join(win_issues) if win_issues else "dense windows span ≥ τ_eff — τ_eff measured at/after sufficient relaxation",
    }

    # --- bounds / saturation unchanged --------------------------------------
    b_issues: List[str] = []
    for cell in (ref_cell, long_cell):
        dyn = rd.get(cell)
        if dyn is None:
            continue
        if abs(dyn.h_bounds[0] - 0.1) > HP_ABS_TOL or abs(dyn.h_bounds[1] - 10.0) > HP_ABS_TOL:
            b_issues.append(f"{cell} observed H bounds {dyn.h_bounds} != frozen [0.1,10]")
        if abs(dyn.w_bounds[0] - 0.01) > HP_ABS_TOL or abs(dyn.w_bounds[1] - 10.0) > HP_ABS_TOL:
            b_issues.append(f"{cell} observed w bounds {dyn.w_bounds} != frozen [0.01,10]")
    checks["bounds_saturation"] = {
        "pass": not b_issues,
        "measured": {k: {"h_bounds": list(rd[k].h_bounds), "w_bounds": list(rd[k].w_bounds)} for k in rd if k in pair.cells},
        "expected": "H∈[0.1,10], w∈[0.01,10] unchanged across ref/LONG",
        "detail": "; ".join(b_issues) if b_issues else "observed bounds unchanged — saturation envelope preserved",
    }
    return checks


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------


def assert_factor_isolation(
    cell_pair: Tuple[str, str] | FactorPair,
    realized_inputs: Dict[str, RealizedInputs],
    realized_dynamics: Optional[Dict[str, RealizedDynamics]] = None,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Semantic factor-isolation validator for a frozen cell pair.

    Checks BOTH (a) Δconfiguration and (b) Δrealized inputs/dynamics.

    Parameters
    ----------
    cell_pair : tuple[str, str] or FactorPair
        Two frozen cell names (same order arbitrary). Intervention is inferred:
        differing RF factor → "rf"; differing Rate factor → "timescale".
    realized_inputs : dict[str, RealizedInputs]
        Measured drive/config facts per cell (see RealizedInputs).
    realized_dynamics : dict[str, RealizedDynamics], optional
        Measured dense Θ(t) facts per cell. Required for a timescale
        intervention; optional for an RF intervention.
    strict : bool
        If True, raise AssertionError listing failures. Default False —
        returns a pass/fail report per semantic check.

    Returns
    -------
    dict with "valid", "version", "intervention", "pair", "checks"
    (each check: pass / measured / expected / detail) and "issues".
    """
    if not isinstance(cell_pair, FactorPair):
        pair = FactorPair.of(*cell_pair)
    else:
        pair = cell_pair
    if pair.a not in CELL_ORDER or pair.b not in CELL_ORDER:
        raise KeyError(f"cell pair {pair.a},{pair.b} not in frozen {CELL_ORDER}")
    for c in pair.cells:
        if c not in realized_inputs:
            raise KeyError(f"realized_inputs missing for {c}")

    checks: Dict[str, Any] = {}
    for c in pair.cells:
        for name, check in _check_configuration(c, realized_inputs[c]).items():
            checks[f"config.{c}.{name}"] = check

    if pair.intervention == "rf":
        checks.update(_check_rf_semantics(pair, realized_inputs))
    else:  # timescale
        checks.update(_check_timescale_semantics(pair, realized_inputs, realized_dynamics or {}))

    issues = [name for name, c in checks.items() if not c["pass"]]
    report = {
        "valid": not issues,
        "version": SEMANTIC_VERSION,
        "intervention": pair.intervention,
        "pair": [pair.a, pair.b],
        "checks": checks,
        "issues": issues,
    }
    if strict and issues:
        details = "\n".join(f"  - {name}: {checks[name]['detail']}" for name in issues)
        raise AssertionError(f"factor-isolation FAILED ({pair.intervention}, {pair.a}→{pair.b}):\n{details}")
    return report
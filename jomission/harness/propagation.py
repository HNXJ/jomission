"""Causal propagation estimator for multi-area architectures.

The estimator this replaces compared two windows of one run, per area:

    delta_A = r_A(stimulus window) - r_A(baseline window)

WS-AUTH-2 showed that quantity cannot identify transmission here. With every long-range weight
set to exactly zero, so that no current can reach FEF or PFC from V1 at all, the old gate still
reported FEF +0.7971 Hz and PFC +0.8406 Hz, as much as V1's +0.7536. Anything common to the two
windows -- intrinsic phase drift, burst-count sampling in a 190 ms periodic signal, a slow
network-wide trend -- enters delta_A unopposed and is indistinguishable from a response.

A causal estimate needs two contrasts, not one. Stimulus ON against OFF removes what is common
to the two windows; coupled against disconnected removes what survives that and is still common
to areas the stimulus cannot reach:

    R_A(g) = [r_A_on(g) - r_A_off(g)] - [r_A_on(0) - r_A_off(0)]

All four runs are matched in every respect but the two contrasted factors, and the ON and OFF
runs share an initial state and RNG stream so their difference is not a seed difference.

One caution, stated because it is easy to misread. Evaluating R on the same g = 0 pair that
supplies the control gives identically zero by construction: it is X - X. That is a plumbing
check on the wiring, not evidence the estimator rejects non-transmission. The adversarial test
is `noise_floor`, which evaluates R on a HELD-OUT g = 0 replicate at a different seed. Its
residual is what a null area can produce by chance, and it is the only defensible source for the
tolerance R must exceed.
"""

from __future__ import annotations

import math

#: Areas the stimulus cannot reach when long-range coupling is zero. At g = 0 these are the
#: adversarial null: any non-zero R here is estimator error, not transmission.
DISCONNECTED_AT_ZERO_G = ("V4_1", "V4_2", "FEF", "PFC")


def _areas(*maps):
    """The area set common to every argument, refusing to guess when they disagree."""
    keys = [set(m) for m in maps]
    common = set.intersection(*keys)
    extra = set.union(*keys) - common
    if extra:
        raise ValueError(f"rate maps disagree on areas; not shared by all: {sorted(extra)}")
    return sorted(common)


def paired_difference(on, off):
    """r_on - r_off per area. Both runs must be matched in seed and initial state."""
    return {a: float(on[a]) - float(off[a]) for a in _areas(on, off)}


def causal_response(on, off, control_on, control_off):
    """R_A: the stimulus effect at this coupling, minus the stimulus effect with none.

    Each argument maps area name to that area's mean E rate in Hz over the same window. The
    result is in Hz and is zero for an area the stimulus reaches only through coupling that is
    absent from both conditions.
    """
    test = paired_difference(on, off)
    control = paired_difference(control_on, control_off)
    return {a: test[a] - control[a] for a in _areas(test, control)}


def noise_floor(replicate_on, replicate_off, control_on, control_off, *, areas=None):
    """R evaluated on a held-out null replicate: what a non-transmitting area yields by chance.

    The replicate must be another g = 0 pair at a DIFFERENT seed from the control pair. Passing
    the control pair back in gives zero by construction and measures nothing.
    """
    r = causal_response(replicate_on, replicate_off, control_on, control_off)
    sel = sorted(r) if areas is None else sorted(areas)
    missing = [a for a in sel if a not in r]
    if missing:
        raise ValueError(f"replicate is missing areas {missing}")
    mags = [abs(r[a]) for a in sel]
    return {"by_area": {a: r[a] for a in sel},
            "max_abs": max(mags), "mean_abs": sum(mags) / len(mags),
            "rms": math.sqrt(sum(x * x for x in mags) / len(mags)),
            "n_areas": len(sel),
            "is_degenerate": all(m == 0.0 for m in mags)}


def evaluate(on, off, control_on, control_off, *, tolerance_hz,
             levels=None, disconnected=DISCONNECTED_AT_ZERO_G):
    """Full causal propagation gate: R per area, the null check, and the level verdicts.

    `tolerance_hz` must come from a measured noise floor, not from a convention. `levels` maps a
    level name to the areas any one of which responding satisfies it; the default is the V1 /
    V4 / frontal hierarchy the whole-system diagnostics use.
    """
    if tolerance_hz <= 0:
        raise ValueError("tolerance_hz must be positive and derived from a measured null")
    r = causal_response(on, off, control_on, control_off)
    levels = levels or {"V1_driven": ("V1_1",), "V4": ("V4_1", "V4_2"),
                        "frontal": ("FEF", "PFC")}
    responded = {a: bool(abs(v) > tolerance_hz) for a, v in r.items()}
    by_level = {name: bool(any(responded.get(a, False) for a in members))
                for name, members in levels.items()}
    return {
     "estimator": "R_A = (on - off) - (control_on - control_off), per area, Hz",
     "R_hz": {a: round(v, 6) for a, v in r.items()},
     "tolerance_hz": tolerance_hz,
     "tolerance_source": "measured on a held-out null replicate; see noise_floor",
     "responded": responded,
     "levels": by_level,
     "pass": bool(all(by_level.values())),
     "null_areas_at_zero_coupling": list(disconnected),
     "supersedes": ("the within-run window difference delta_A, which WS-AUTH-2 showed reports a "
                    "response at FEF and PFC when no current can reach them"),
    }

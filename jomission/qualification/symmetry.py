"""H1-H5 symmetry-breaking analysis: single-cell assays, no plant change.

H1: per class (E/PV/SST/VIP, exact plant params), perturb one intrinsic
parameter (a/b/c/d) x{0.8,0.9,1.1,1.2} at the HIGH-regime tonic
(results/v21_vectors.json V_HIGH). Single-cell Euler (sealed protocol):
rate + ISI-CV over 10 s; rheobase via 0..40 step-1 scan (2 s each).
Metrics per (class,param): rate gain |d ln r/d ln p| (== detuning gain
under common drive: phase-drift rate between detuned cells is Delta-f,
so no separate coupled assay is needed for dispersion ranking),
rheobase shift (mean-movement risk), single-cell ISI-CV max (bursting
onset = deterministic ISI mechanism if >=0.5).
H2 memo (in test): constant-drive cell -> limit cycle -> CV->0; H1
decides whether class-spanning (RS->burster) gives deterministic ISI.
H3 history: sealed doc evidence (no new sims).
H4 selection rule (predeclared): ISI-viable param (isi>=0.5,
rheo-shift<=0.25, rate within [0.5x,2x]) -> INTRINSIC + bracket
eps in {0.05,0.10,0.15,0.20} on the winning (class,param); elif best
detuning gain <0.1 -> STRUCTURAL (plant insensitive to intrinsics);
elif no ISI mechanism -> PRIVATE_STOCHASTIC (calibrated private shot
noise per H3 lessons, not the failed 2kHz weak regime); else UNRESOLVED.
H5 verdict = H4 output. Only the verdict authorizes re-execution.
"""

from __future__ import annotations

import numpy as np

CLASSES = ("E", "PV", "SST", "VIP")
PARAMS = ("a", "b", "c", "d")
LEVELS = (0.8, 0.9, 1.1, 1.2)

DT_MS = 0.1
ASSAY_S = 10.0
RHEO_GRID = tuple(float(x) for x in np.arange(0.0, 41.0, 1.0))

BRACKET = (0.05, 0.10, 0.15, 0.20)


def run_cell(p, I, dur_s=ASSAY_S):
    """Single-cell Euler (sealed protocol). Returns (rate_Hz, isi_cv)."""
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    v, u = float(c), float(b) * float(c)
    n_steps = int(round(dur_s * 1000.0 / DT_MS))
    skip = n_steps // 2
    spikes = []
    for t in range(n_steps):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + float(I)
        du = a * (b * v - u)
        v += DT_MS * dv
        u += DT_MS * du
        if v >= 30.0:
            if t >= skip:
                spikes.append(t)
            v = float(c)
            u += float(d)
    spikes = np.array(spikes)
    rate = len(spikes) / (dur_s / 2.0)
    if len(spikes) >= 4:
        isi = np.diff(spikes).astype(float) * DT_MS
        cv = float(isi.std() / max(isi.mean(), 1e-9))
    else:
        cv = 0.0
    return float(rate), float(cv)


def rheobase(p):
    for I in RHEO_GRID:
        r, _ = run_cell(p, I, dur_s=2.0)
        if r > 0.5:
            return float(I)
    return None


def assay_class(c, base, tonic):
    """Full perturbation assay for one class at its HIGH tonic."""
    rows = {}
    r0, cv0 = run_cell(base, tonic)
    rh0 = rheobase(base)
    rows["base"] = {"rate": r0, "cv": cv0, "rheobase": rh0}
    for par in PARAMS:
        for lv in LEVELS:
            p = dict(base)
            p[par] = base[par] * lv
            r, cv = run_cell(p, tonic)
            rows[f"{par}x{lv}"] = {"rate": r, "cv": cv,
                                   "rheobase": rheobase(p)}
    return rows

"""N0-N1 native trajectory geometry: frozen plant, boundary trajectories,
full-state observables with spike/reset structure preserved.

N0 freeze: s_E in {50,54,57,61} selective plants unchanged (rule, gains,
cells, inhibition, convergence, tonic=0, HDP detached-history state).
Only intervention: uniform E-drive prep amplitude (already-qualified
steering) to traverse silence<->sync.

N1 trajectories per candidate (predeclared): the adjacent flip pair from
sealed b_bisect flip fields (lo=max silent amp, hi=min sync amp above lo;
s61 non-monotonic noted: pair (3.5,4.0)) plus mid=(lo+hi)/2. 8 s prep,
2 s-chunk execution with full spikes+V retention, online reduction.

Sealed reduced observables per run (raw discarded after deterministic
reduction; reduction code is the provenance):
  rates_2ms   per-class Hz in 20-step bins (4000 pts, spike structure kept)
  sync_2ms    E-population co-spike fraction per 2 ms bin
  means_10ms  per-class mean V in 100-step bins (fast voltage tracking)
  snaps_500ms per-pathway aux/w means + H + per-class u/Vbar (16 snaps;
              resolves aux/w tau 20-55 ms)
  V_win       full-resolution V of first E/PV/SST indices over
              [t_commit-1s, t_commit+1s] (or [5s,7s] if no commit event)
  Ixy_10ms    constructed pathway currents K*wbar*synbar, synbar from the
              exact kernel driven by binned class rates (constructed
              observable, labeled as such)
  t_commit    first 2 ms bin with E-bin-rate > COMMIT_R (500 Hz), else null
  outcome     settled endpoint class (SILENT/SYNC/OTHER by S9 bands)
COMMIT_R=500 Hz predeclared (10% of the sync level; S9 active band max 40).
"""

from __future__ import annotations

import numpy as np

CANDIDATES: tuple[float, ...] = (50.0, 54.0, 57.0, 61.0)

PREP_S = 8.0
DT_MS = 0.1
SEED = 11
BIN_2MS = 20
BIN_10MS = 100
COMMIT_R = 500.0
WIN_S = 1.0


def scale_tag(s: float) -> str:
    return ("%.1f" % s).replace(".", "p")


def flip_amps(s):
    """Adjacent flip pair + midpoint from sealed bisection (deterministic)."""
    import json

    b = json.load(open(f"results/b_bisect_s{scale_tag(s)}.json"))
    grid = {float(a): float(r) for a, r in b["grid"].items()}
    silent = sorted(a for a, r in grid.items() if r < 1.0)
    sync_above = {}
    for a in silent:
        above = [x for x, r in grid.items() if x > a and r >= 1.0]
        if above:
            sync_above[a] = min(above)
    assert sync_above, "no adjacent flip pair in sealed grid"
    lo = max(sync_above)
    hi = sync_above[lo]
    return lo, hi, (lo + hi) / 2.0

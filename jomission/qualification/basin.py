"""S9 native nonlinear basin: predeclared init battery + release protocol.

Parent: 23c56d5 (S7) + ac4422e (S8). Candidates: s_E in {50,54,57,61}
(frozen; s61 boundary-marginal included, no pre-selection). No tuning of
any parameter: pure adjudication of the frozen candidates.

Plant: canonical C-min column (400 neurons), zero tonic everywhere,
selective rule jomission_sat_eout_v0 at the candidate s_E. No schedule
drive during release: finite initialization + free evolution.

Initializations (predeclared; REVISED 2026-09-15: explicit V construction
abandoned for active inits -- a constructed (V,u) pair is unphysical
(demonstrated: V=c+FP-weights guarantees silence, V-spread+FP-weights
artificially ignites via inconsistent V-u pairing into a sync 2-cycle).
Active inits are driven-preparation snapshots with self-consistent
V/u/syn/aux/w/H phases, released to zero drive):
  P0 REST     fresh rest (control; expected collapse)
  P1 PREP-AT  E-drive prep, settled E rate nearest rE* (|d|<=30% else unusable)
  P3 PREP-LOW E-drive prep nearest 0.5*rE*
  P4 PREP-HIGH E-drive prep nearest 1.5*rE*
  P5 PREP-IBIAS E-drive + fixed PV-drive prep nearest rE* (I-dominance recovery)
  P7 HDPCONSTRUCT neural rest (consistent V=c,u=b*c pair) + FP aux/w + H=1
Prep: uniform schedule drive on all E cells (P1/P3/P4) or E cells + PV
cells at fixed PV amp (P5); 8 s settle; usability = nearest settled E rate
within 30% of target (else init UNUSABLE for that candidate, documented).
Prep amp grid {2,4,...,16}, one bounded extension {18,20,24} if unusable.
FP E-out w homogenized per pathway (documented finite perturbation of the
heterogeneous base). E-pre w selected by presynaptic class only, mirroring
the rule's own selectivity (no target-class gains).

Release: 20 s zero-drive in 4 x 5 s chunks (memory-bounded; no weight
trace -- HDP state read from chunk-end dynamic carry). Per-chunk class
rates; w/aux/H/u means at chunk ends; last-chunk spikes for synchrony.

Capture (predeclared): rE in [3,40] over last 5 s; rate drift <=20% vs
previous 5 s; w drift <=5%; rPV>=1, rSST>=0.5; max class rate <80;
synchrony guard (steps with >50% E co-spiking) <0.01. Same regime:
captured rE within +/-30%. Collapse: rE<1. Else OTHER (saturated /
oscillatory / unresolved-zone). P0 is a control (expected collapse;
spontaneous capture recorded, not gated). VIP reported; >1 Hz noted as
scope expansion, not failure.

Basin PASS per candidate: >=2 genuinely distinct finite inits (P1/P3/P4/P5/P7)
captured to the same regime. Near-FP prep-release collapse (P1) =
closure-transfer failure (reported, no tuning). Silent co-attractor may
remain stable. Superseded 2026-09-15 construction-protocol runs
(results/s9_init_s50p0_*) are preserved but excluded from the verdict
(P2/P5 artifactual ignition; P1/P3/P4/P6 unphysical neural silence).
"""

from __future__ import annotations

import numpy as np

CANDIDATES: tuple[float, ...] = (50.0, 54.0, 57.0, 61.0)

TAU_ACT = 20.0
RELEASE_S = 20.0
CHUNK_S = 5.0
DT_MS = 0.1
SEED = 11

INITS = ("P0", "P1", "P3", "P4", "P5", "P7")

PREP_AMPS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)
PREP_AMPS_EXT = (18.0, 20.0, 24.0)
PREP_IBIAS_PV_AMP = 2.0
PREP_S = 8.0
PREP_USABLE_TOL = 0.30
PREP_SETTLE_TOL = 0.15

CAPTURE = {"rE_lo": 3.0, "rE_hi": 40.0, "drift_max": 0.20, "w_drift_max": 0.05,
           "rPV_min": 1.0, "rSST_min": 0.5, "sat_max": 80.0,
           "sync_max": 0.01, "regime_tol": 0.30}


def scale_tag(s: float) -> str:
    return ("%.1f" % s).replace(".", "p")


def fp_tables(s):
    """Fixed-point construction tables for candidate s (sealed JSONs)."""
    import json

    tag = scale_tag(s)
    b1 = json.load(open(f"results/s7_battery_b1_s{tag}.json"))
    b2 = json.load(open(f"results/s7_battery_b2_s{tag}.json"))
    b3 = json.load(open(f"results/s7_battery_b3_s{tag}.json"))
    geo = json.load(open(f"results/s7_geometry_s{tag}.json"))
    assert geo["verdict"] == "S7_PASS", (s, geo["verdict"])
    root = geo["cortical"][0]
    FV = json.load(open("results/s8_assay.json"))
    rE, rPV, rSST = root["rE"], root["rPV"], root["rSST"]

    def curve(rows, key, field="I"):
        xs = np.array([r["driver_rate"] for r in rows])
        ys = np.array([r["classes"][key][field] for r in rows])
        o = np.argsort(xs)
        return xs[o], ys[o]

    x_ee, i_ee = curve(b1["rows"], "E")
    x_epv, i_epv = curve(b1["rows"], "PV")
    x_esst, i_esst = curve(b1["rows"], "SST")
    I = {"E": root["I_E"],
         "PV": 300.0 * float(np.interp(rE, x_epv, i_epv)),
         "SST": 300.0 * float(np.interp(rE, x_esst, i_esst))}
    r = {"E": rE, "PV": rPV, "SST": rSST}
    xw_ee, w_ee = curve(b1["rows"], "E", "w")
    xw_epv, w_epv = curve(b1["rows"], "PV", "w")
    xw_esst, w_esst = curve(b1["rows"], "SST", "w")
    xw_pve, w_pve = curve(b2["rows"], "E", "w")
    xw_sste, w_sste = curve(b3["rows"], "E", "w")
    xw_ev, w_ev = curve(b1["rows"], "VIP", "w")
    wpath = {"EE": float(np.interp(rE, xw_ee, w_ee)),
             "EPV": float(np.interp(rE, xw_epv, w_epv)),
             "ESST": float(np.interp(rE, xw_esst, w_esst)),
             "EVIP": float(np.interp(rE, xw_ev, w_ev)),
             "PVE": -float(np.interp(rPV, xw_pve, w_pve)),
             "SSTE": -float(np.interp(rSST, xw_sste, w_sste))}
    Vstar = {c: float(np.interp(I[c], FV[c]["I"], FV[c]["V"]))
             for c in ("E", "PV", "SST")}
    # VIP out of S7 scope (no K_?VIP convergence measured): constructed at
    # rest (V at zero current). VIP recruitment during release is reported.
    Vstar["VIP"] = float(np.interp(0.0, FV["VIP"]["I"], FV["VIP"]["V"]))
    return {"r": r, "I": I, "wpath": wpath, "Vstar": Vstar}

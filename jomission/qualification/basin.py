"""S9 native nonlinear basin: predeclared init battery + release protocol.

Parent: 23c56d5 (S7) + ac4422e (S8). Candidates: s_E in {50,54,57,61}
(frozen; s61 boundary-marginal included, no pre-selection). No tuning of
any parameter: pure adjudication of the frozen candidates.

Plant: canonical C-min column (400 neurons), zero tonic everywhere,
selective rule jomission_sat_eout_v0 at the candidate s_E. No schedule
drive during release: finite initialization + free evolution.

Initializations (8, exact construction predeclared; FP = fixed-point
tables from S7 geometry/batteries + S8 assay/u*):
  P0 REST     fresh rest (V=c, u=b*c, H=1, aux=0, w=model base, syn=0)
  P1 FPSYNC   FP w/aux/H/u + V=c everywhere + syn=0
  P2 FPSPREAD FP + V ramped c->0 per neuron (desync)
  P3 EWEAK    FP with E-pre aux x0.5, E-pre w x0.7
  P4 ESTRONG  FP with E-pre aux x1.5, E-pre w x1.2 (capped 0.9*hi)
  P5 EONLY    E subpopulation at FP-spread, I cells + I-pre aux/w at rest
  P6 IBIAS    FP with I-pre aux x1.5, I-pre w x1.2 (capped 0.9*hi_I)
  P7 HDPONLY  neural rest everywhere + aux/w at FP + H=1
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

Basin PASS per candidate: >=2 genuinely distinct finite inits (P1-P7)
captured to the same regime. Near-FP collapse (P1/P2) = closure-transfer
failure (reported, no tuning). Silent co-attractor may remain stable.
"""

from __future__ import annotations

import numpy as np

CANDIDATES: tuple[float, ...] = (50.0, 54.0, 57.0, 61.0)

TAU_ACT = 20.0
RELEASE_S = 20.0
CHUNK_S = 5.0
DT_MS = 0.1
SEED = 11

INITS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")

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

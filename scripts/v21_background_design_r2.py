"""SCI-V21-BACKGROUND design regeneration after PV_CALIBRATION_FAIL (BKG1-BKG3, analytic only).

    python scripts/v21_background_design_r2.py         # derived JSON on stdout

No simulation. Reads sealed results; computes rest-point bifurcations of the frozen Izhikevich cells.
Model (engine order, per step): v' = 0.04 v^2 + 5 v + 140 - u + I, u' = a (b v - u), forward Euler dt 0.1 ms.
Rest points solve 0.04 v^2 + (5 - b) v + 140 + I = 0 (lower root is the candidate stable rest).
Currents in JaxFNE native drive units; rates in Hz.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DT_MS = 0.1
# Cell parameters as read from build_cmin() emitter (homogeneous per class; PV confirmed in
# results/v21_bkg_pv_calibration.json cell_params_min_max). Rounded float32 values.
CELLS = {"E": (0.02, 0.2, -65.0, 8.0), "PV": (0.1, 0.2, -65.0, 2.0),
         "SST": (0.05, 0.25, -65.0, 2.0), "VIP": (0.02, -0.1, -55.0, 6.0)}
CLASSES = ("E", "PV", "SST", "VIP")


def load(rel):
    raw = (ROOT / rel).read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))


def lower_rest(b, current):
    disc = (5.0 - b) ** 2 - 0.16 * (140.0 + current)
    if disc < 0:
        return None
    return (-(5.0 - b) - np.sqrt(disc)) / 0.08


def jacobian(a, b, v):
    return np.array([[0.08 * v + 5.0, -1.0], [a * b, -a]])


def rest_unstable(a, b, current, discrete):
    v = lower_rest(b, current)
    if v is None:
        return True
    J = jacobian(a, b, v)
    if discrete:
        return float(np.max(np.abs(np.linalg.eigvals(np.eye(2) + DT_MS * J)))) > 1.0
    return float(np.max(np.linalg.eigvals(J).real)) > 0.0


def onset(a, b, discrete, lo=-50.0, tol=1e-6):
    """Smallest I at which the lower rest point is lost or loses linear stability (bisection)."""
    hi = (5.0 - b) ** 2 / 0.16 - 140.0 + 1e-9
    if rest_unstable(a, b, lo, discrete):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if rest_unstable(a, b, mid, discrete):
            hi = mid
        else:
            lo = mid
    return round(hi, 5)


def onsets():
    out = {}
    for c, (a, b, _, _) in CELLS.items():
        sn = (5.0 - b) ** 2 / 0.16 - 140.0
        cont = onset(a, b, discrete=False)
        disc = onset(a, b, discrete=True)
        v_sn = -(5.0 - b) / 0.08
        det_at_trace0 = a * (b - 0.08 * ((a - 5.0) / 0.08) - 5.0)
        out[c] = {"saddle_node_I": round(sn, 5), "continuous_instability_I": cont, "euler_map_instability_I": disc,
                  "instability_kind": "Hopf" if (cont is not None and cont < sn - 1e-4 and det_at_trace0 > 0) else "saddle-node",
                  "v_at_saddle_node": round(v_sn, 4)}
    return out


def derive():
    ons = onsets()
    mid = load("results/v21_mid.json")
    high = load("results/v21_high.json")
    cal = load("results/v21_bkg_pv_calibration.json")
    support = {}
    for c in CLASSES:
        first = min(x for x in (ons[c]["continuous_instability_I"], ons[c]["euler_map_instability_I"]) if x is not None)
        support[c] = {
            "I0_V_MID": mid["tonic"][c], "I0_V_HIGH": high["tonic"][c], "rest_instability_I_min": first,
            "V_MID_above_onset": bool(mid["tonic"][c] > first), "V_HIGH_above_onset": bool(high["tonic"][c] > first),
            "network_rate_V_MID_hz": [w[c] for w in mid["rates"]], "network_rate_V_HIGH_hz": [w[c] for w in high["rates"]]}
    return {"cell_onsets": ons, "tonic_support": support,
            "pv_calibration": {"label": cal["label"],
                               "isolated_rates_hz": {str(e["I"]): e["rate_hz"] for e in cal["evaluations"]}},
            "tonic_only_supported_at_V_MID": [c for c in CLASSES if support[c]["V_MID_above_onset"]],
            "below_onset_at_V_MID_but_firing_in_network": [
                c for c in CLASSES if not support[c]["V_MID_above_onset"] and min(support[c]["network_rate_V_MID_hz"]) > 1.0],
            "below_onset_at_V_HIGH_but_firing_in_network": [
                c for c in CLASSES if not support[c]["V_HIGH_above_onset"] and min(support[c]["network_rate_V_HIGH_hz"]) > 1.0]}


if __name__ == "__main__":
    print(json.dumps(derive(), indent=1))

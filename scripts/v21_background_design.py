"""SCI-V21-BACKGROUND design derivation (BKG0-BKG3). Reads sealed results only; no simulation.

    python scripts/v21_background_design.py            # derived JSON on stdout

Every number in results/v21_background_design.json "derived" comes from derive();
tests/test_v21_background_design.py checks the record against it.

Evidence classes: MEASURED (executed cell value), INTERPOLATED (between two executed values of the
same class at the same tonic along the V2.1b sigma/mu chain), BOUNDED (between two executed values,
assuming rate is monotone in fluctuation amplitude), UNSUPPORTED (would need extrapolation).
Rates in Hz; currents in JaxFNE native current units; sigma/mu dimensionless.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jomission.harness import gates  # noqa: E402

CLASSES = ("E", "PV", "SST", "VIP")
REGIMES = {"low": "V_LOW", "mid": "V_MID", "high": "V_HIGH"}
SM = {"sm0p5": 0.5, "sm1p0": 1.0, "sm2p0": 2.0}
TARGET_HZ = 10.0


def load(rel):
    raw = (ROOT / rel).read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))


def bands():
    checks = gates.load_gate()["profiles"]["prospective_v2"]["checks"]
    spec = next(c for c in checks.values() if c["kind"] == "class_rate_band_by_class")
    return spec["rate_hz"]


def fi_table():
    """battery_b1 F-I: grid bin holding TARGET_HZ, whether that bin starts at zero rate (onset bin)."""
    F = load("results/battery_b1.json")["F"]
    out = {}
    for c in CLASSES:
        cur, r = np.asarray(F[c]["I"], float), np.asarray(F[c]["r"], float)
        k = int(np.flatnonzero(r >= TARGET_HZ)[0])
        lo, hi = float(cur[k - 1]), float(cur[k])
        out[c] = {"bin_I": [lo, hi], "bin_r": [float(r[k - 1]), float(r[k])],
                  "I_interp": round(float(lo + (TARGET_HZ - r[k - 1]) * (hi - lo) / (r[k] - r[k - 1])), 4),
                  "onset_bin": bool(r[k - 1] == 0.0),
                  "suprathreshold_slope_hz_per_unit": round(float((r[k + 1] - r[k]) / (cur[k + 1] - cur[k])), 3)}
    return out


def native_noise_rates():
    """V2.1: tonic vectors with engine native noise only (no shot). Window min/max per class."""
    out = {}
    for reg in REGIMES:
        res = load(f"results/v21_{reg}.json")
        out[reg] = {"tonic": res["tonic"],
                    "rate_min": {c: round(min(w[c] for w in res["rates"]), 3) for c in CLASSES},
                    "rate_max": {c: round(max(w[c] for w in res["rates"]), 3) for c in CLASSES}}
    return out


def shot_response():
    """V2.1b: per regime, sigma/mu, lambda, per-class sigma_abs = sigma/mu * mu_c, window-max rate, band flag."""
    b = bands()
    out = {}
    for reg in REGIMES:
        for cell, sm in SM.items():
            res = load(f"results/v21b_{reg}_{cell}.json")
            rmax = {c: round(max(w[c] for w in res["rates"]), 3) for c in CLASSES}
            rmin = {c: round(min(w[c] for w in res["rates"]), 3) for c in CLASSES}
            out[f"{reg}_{cell}"] = {
                "sigma_over_mu": sm, "lambda_hz": res["shot"]["lambda_hz"],
                "sigma_abs": {c: round(sm * res["tonic"][c], 4) for c in CLASSES},
                "rate_min": rmin, "rate_max": rmax,
                "in_band": {c: bool(b[c][0] <= rmin[c] and rmax[c] <= b[c][1]) for c in CLASSES},
                "pooled_isi_frac_in_min": round(min(x["frac_in"] for x in res["cv"]), 3)}
    return out


def gains(native):
    """Network rate gain between V_MID and V_HIGH with native noise only (Hz per native current unit)."""
    lo, hi = native["mid"], native["high"]
    g = {}
    for c in CLASSES:
        dI = hi["tonic"][c] - lo["tonic"][c]
        g[c] = {"dI": round(dI, 4), "r_mid": lo["rate_min"][c], "r_high": hi["rate_min"][c],
                "gain": round((hi["rate_min"][c] - lo["rate_min"][c]) / dI, 3),
                "mid_silent": bool(lo["rate_max"][c] == 0.0)}
    return g


def interp_or_bound(points, x):
    """points: sorted [(amp, rate_max)] measured at one tonic. Rate at amp x and its evidence class."""
    amps = [p[0] for p in points]
    for a, r in points:
        if abs(a - x) < 1e-9:
            return {"rate_max": r, "evidence": "MEASURED"}
    if x > amps[-1]:
        return {"rate_max": None, "evidence": "UNSUPPORTED"}
    j = next(i for i, a in enumerate(amps) if a > x)
    (a0, r0), (a1, r1) = points[j - 1], points[j]
    kind = "BOUNDED" if a0 == 0.0 else "INTERPOLATED"
    if kind == "BOUNDED":
        return {"rate_max": round(r1, 3), "rate_lower_bound": round(min(r0, r1), 3), "evidence": kind}
    return {"rate_max": round(r0 + (r1 - r0) * (x - a0) / (a1 - a0), 3), "evidence": kind}


def normalization_analysis(native, shot, g):
    """Predicted window-max rates at tonic V_MID for sigma/mu_E in {0.5, 1.0, 1.25, 1.5, 2.0}.

    A: s_c = I0_c (sigma/mu common; the V2.1b family). B: s_c = I0_E * gain_E / gain_c (rate-equivalent
    amplitude). Class chains: native-noise V2.1 point at amplitude 0, then the V2.1b mid cells.
    BOUNDED values assume monotone rate in amplitude and ignore the lambda difference between chains.
    """
    b = bands()
    I0 = native["mid"]["tonic"]
    chain = {c: [(0.0, native["mid"]["rate_max"][c])]
             + sorted((shot[f"mid_{cell}"]["sigma_abs"][c], shot[f"mid_{cell}"]["rate_max"][c]) for cell in SM)
             for c in CLASSES}
    out = {}
    for name in ("A_tonic_relative", "B_gain_normalized"):
        rows = {}
        for s in (0.5, 1.0, 1.25, 1.5, 2.0):
            row = {}
            for c in CLASSES:
                if name == "A_tonic_relative":
                    amp = s * I0[c]
                elif c == "PV" and g[c]["mid_silent"]:
                    row[c] = {"sigma_abs": None, "rate_max": None, "evidence": "UNSUPPORTED",
                              "why": "PV gain undefined: PV silent at V_MID tonic"}
                    continue
                else:
                    amp = s * I0["E"] * g["E"]["gain"] / g[c]["gain"]
                pred = interp_or_bound(chain[c], round(amp, 4))
                pred["sigma_abs"] = round(amp, 4)
                if pred["rate_max"] is not None:
                    pred["in_band_upper"] = bool(pred["rate_max"] <= b[c][1])
                    pred["in_band_lower"] = bool(pred.get("rate_lower_bound", pred["rate_max"]) >= b[c][0])
                row[c] = pred
            rows[str(s)] = row
        out[name] = rows
    return out


def pv_at_high_support(native, shot, g, fi):
    """PV option: tonic from V_HIGH (executed 20 Hz), s_PV from the battery suprathreshold F-I slope.

    Rate-level weak recurrence assumed: the PV chain comes from V_HIGH cells, where the other classes ran near 20 Hz.
    """
    b = bands()
    chain = [(0.0, native["high"]["rate_max"]["PV"])] + sorted(
        (shot[f"high_{cell}"]["sigma_abs"]["PV"], shot[f"high_{cell}"]["rate_max"]["PV"]) for cell in SM)
    slope = fi["PV"]["suprathreshold_slope_hz_per_unit"]
    rows = {}
    for s in (0.5, 1.0, 1.25, 1.5, 2.0):
        amp = s * native["mid"]["tonic"]["E"] * g["E"]["gain"] / slope
        pred = interp_or_bound(chain, round(amp, 4))
        pred["sigma_abs"] = round(amp, 4)
        pred["in_band_upper"] = bool(pred["rate_max"] <= b["PV"][1])
        pred["in_band_lower"] = bool(pred.get("rate_lower_bound", pred["rate_max"]) >= b["PV"][0])
        rows[str(s)] = pred
    return {"I0_PV": native["high"]["tonic"]["PV"], "s_gain_hz_per_unit": slope, "rows": rows}


def e_band_sigma_cap(shot):
    """sigma/mu_E where the interpolated V_MID E window-max rate reaches the E band upper edge."""
    hi = bands()["E"][1]
    pts = [(SM[cell], shot[f"mid_{cell}"]["rate_max"]["E"]) for cell in ("sm0p5", "sm1p0", "sm2p0")]
    for (s0, r0), (s1, r1) in zip(pts, pts[1:]):
        if r0 <= hi < r1:
            return round(s0 + (hi - r0) * (s1 - s0) / (r1 - r0), 4)
    return None


def vip_band_sigma_cap_tonic_relative(shot):
    """Same for VIP under s_c = I0_c (normalization A)."""
    hi = bands()["VIP"][1]
    pts = [(SM[cell], shot[f"mid_{cell}"]["rate_max"]["VIP"]) for cell in ("sm0p5", "sm1p0", "sm2p0")]
    for (s0, r0), (s1, r1) in zip(pts, pts[1:]):
        if r0 <= hi < r1:
            return round(s0 + (hi - r0) * (s1 - s0) / (r1 - r0), 4)
    return None


def derive():
    native = native_noise_rates()
    shot = shot_response()
    g = gains(native)
    fi = fi_table()
    return {"bands_hz": bands(), "fi_battery_b1": fi, "native_noise_v21": native, "shot_v21b": shot,
            "gain_mid_to_high": g, "normalization": normalization_analysis(native, shot, g),
            "pv_option_high_support": pv_at_high_support(native, shot, g, fi),
            "sigma_cap_E_band": e_band_sigma_cap(shot),
            "sigma_cap_VIP_band_tonic_relative": vip_band_sigma_cap_tonic_relative(shot)}


if __name__ == "__main__":
    print(json.dumps(derive(), indent=1))

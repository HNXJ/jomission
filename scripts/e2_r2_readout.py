"""Phase 1 readout conditions G0, G1, G2, G3, G1' and control K1 on fixed R2 unit spectra.

Protocol: docs/laminar_motif/GEOMETRY_PHASE1_PROTOCOL_2026-11-25.md sections 4-6. Estimand per held-out area a:
Pearson over the 21 contacts between the model's band-mean profile and area a's own band-mean profile (Y[bin, freq, a]).
Conditions have no selection step, so the LOAO fold is the area itself. K2 (depth-shuffled null under the LOAO
selection) is not implemented: the protocol does not define the selection rule.

crossing_2se (protocol section 4) is a property of the model profile, not of the held-out area: it is computed once per
condition with the scorer's own function (e2_motif_figure.score) on the same spectra. It is not defined for K1, which
has no window variance.

Cross-checks (both asserted): G0 against the area mean reproduces the sealed scorer's r_gm and r_ab, and G0's
alpha-beta and gamma profiles, diff_se, contrast and crossing reproduce the saved motif json (sum_over_cells, image).

Sensitivity (sealed spec section 7): pass/fail of the median r_gm and r_ab at 0.5, 0.6 and 0.7, descriptive only.
Only 0.6 carries the claim.

Usage: python scripts/e2_r2_readout.py TAG [TAG ...]   (cwd = tree under test; reads results/laminar_e2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import e2_motif_figure as mf  # noqa: E402
import e2_motif_similarity as sim  # noqa: E402
import e2_scorer as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "docs" / "laminar_motif" / "data" / "lansa_lfp_Y.json"
RD = ROOT / "results" / "laminar_e2"
SIG, SIG_ALT = 0.15, 0.25
D_SEALED = np.array([-0.40, 0.00, 0.30])  # S, M, D (sealed R2 layout)
D_ANAT = np.array([-0.216, 0.000, 0.198])  # L2/3 mid, L4 mid, L5 mid (primary)
D_ANAT_L6 = np.array([-0.216, 0.000, 0.406])  # G1' sensitivity: D at L6 mid
CONDS = {"G0": (D_SEALED, SIG), "G1": (D_ANAT, SIG), "G2": (D_SEALED, SIG_ALT),
         "G3": (D_ANAT, SIG_ALT), "G1p": (D_ANAT_L6, SIG)}
BANDS = mf.BANDS[mf.FIG_BANDS]  # ((10, 30), (50, 150)), the verdict bands
F = np.arange(1, 151)
AB = (F >= BANDS[0][0]) & (F <= BANDS[0][1])
GM = (F >= BANDS[1][0]) & (F <= BANDS[1][1])
THRESHOLDS = (0.5, 0.6, 0.7)


def kernel(depths: np.ndarray, sigma: float) -> np.ndarray:
    """Same form as sc.unit_spectra: K[c, u] = exp(-(z_c - z_u)^2 / (2 sigma^2)) on the 21 LANSA contacts."""
    return np.exp(-((mf.CHANNELS[:, None] - depths[None, :]) ** 2) / (2 * sigma ** 2))


def readout(S_mean: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-frequency relative power over contacts, 21 x 150 (rp_map convention)."""
    P = mf.probe_power(S_mean, K)[:, F]
    return P / P.max(axis=0, keepdims=True)


def band_r(R: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(r_ab, r_gm) between R and target band-mean profiles, as the scorer's band_r."""
    return (sim.pearson(R[:, AB].mean(axis=1), target[:, AB].mean(axis=1)),
            sim.pearson(R[:, GM].mean(axis=1), target[:, GM].mean(axis=1)))


def per_area(R: np.ndarray, Y: np.ndarray, areas: list[str]) -> dict:
    out = {}
    for a, name in enumerate(areas):
        rab, rgm = band_r(R, Y[:, :, a])
        out[name] = {"r_ab": rab, "r_gm": rgm}
    return out


def pass_at(x: float) -> dict:
    return {f"{t:g}": bool(x >= t) for t in THRESHOLDS}


def summary(pa: dict) -> dict:
    ab = np.array([v["r_ab"] for v in pa.values()])
    gm = np.array([v["r_gm"] for v in pa.values()])
    mab, mgm = float(np.median(ab)), float(np.median(gm))
    return {"median_r_ab": mab, "median_r_gm": mgm, "n_r_gm_ge_0.6": int((gm >= 0.6).sum()), "n": int(gm.size),
            "median_r_gm_ge": pass_at(mgm), "median_r_ab_ge": pass_at(mab)}


def crossing(f: np.ndarray, S: np.ndarray, K: np.ndarray) -> dict:
    """The scorer's crossing test (e2_motif_figure.score, image bands) on these spectra."""
    s = mf.score(f, S, K, mf.CHANNELS, BANDS_IMAGE)
    return {"crossing_2se": s["crossing_2se"], "contrast_M": s["contrast_M"], "z": s["z"],
            "alpha_beta_profile": s["alpha_beta_profile"], "gamma_profile": s["gamma_profile"],
            "diff_se": s["diff_se"]}


BANDS_IMAGE = mf.BANDS[mf.FIG_BANDS]


def main(tags: list[str]) -> None:
    obs = json.loads(OBS.read_text(encoding="utf-8"))
    areas = obs["axes"]["areas"]
    Y = np.array(obs["Y"], dtype=np.float64)  # [bin 21, freq 150, area 11]
    assert Y.shape == (21, 150, 11), Y.shape
    PROTO = Y.mean(axis=2)
    assert np.allclose(PROTO, sim.PROTO), "observed area mean differs from e2_motif_similarity.PROTO"
    for tag in tags:
        motif = json.loads((RD / f"{tag}_motif.json").read_text(encoding="utf-8"))
        f, S, _ = sc.unit_spectra(tag, float(motif["discard_s"]), RD)
        S_mean = S.mean(axis=0)
        ver = json.loads((RD / f"{tag}_verdict.json").read_text(encoding="utf-8"))
        # cross-check 1: sealed G0 against the area mean reproduces the scorer's r_gm and r_ab
        R0 = readout(S_mean, kernel(D_SEALED, SIG))
        rab0, rgm0 = band_r(R0, PROTO)
        assert abs(rgm0 - ver["r_gm"]) < 1e-9 and abs(rab0 - ver["r_ab"]) < 1e-9, (rgm0, ver["r_gm"], rab0, ver["r_ab"])
        # cross-check 2: sealed G0 crossing and profiles reproduce the saved motif json
        g0x = crossing(f, S, kernel(D_SEALED, SIG))
        saved = motif["sum_over_cells"]["scores"][mf.FIG_BANDS]
        assert g0x["crossing_2se"] == saved["crossing_2se"], (g0x["crossing_2se"], saved["crossing_2se"])
        for key in ("alpha_beta_profile", "gamma_profile", "diff_se"):
            assert np.allclose(np.asarray(g0x[key]), np.asarray(saved[key]), atol=1e-9, rtol=0), key
        assert abs(g0x["contrast_M"] - saved["contrast_M"]) < 1e-9
        res = {"tag": tag, "units": ["S", "M", "D"], "discard_s": float(motif["discard_s"]),
               "crosscheck_G0_areamean": {"r_gm": rgm0, "r_ab": rab0, "scorer_r_gm": ver["r_gm"],
                                          "scorer_r_ab": ver["r_ab"], "pass": True},
               "crosscheck_G0_motif": {"crossing_2se": g0x["crossing_2se"], "contrast_M": g0x["contrast_M"],
                                       "profiles_match_saved_motif": True, "pass": True},
               "conditions": {}}
        for cid, (dep, sig) in CONDS.items():
            K = kernel(dep, sig)
            R = readout(S_mean, K)
            pa = per_area(R, Y, areas)
            res["conditions"][cid] = {"depths": dep.tolist(), "sigma": sig, "per_area": pa, **summary(pa),
                                      **crossing(f, S, K)}
        # K1: identical flat unit spectra (unit power 1 at every frequency, no cross terms), G1 geometry, sigma 0.15
        S_flat = np.zeros_like(S_mean)
        for u in range(3):
            S_flat[:, u, u] = 1.0
        RK1 = readout(S_flat, kernel(D_ANAT, SIG))
        pa = per_area(RK1, Y, areas)
        res["conditions"]["K1"] = {"depths": D_ANAT.tolist(), "sigma": SIG, "per_area": pa, **summary(pa),
                                   "crossing_2se": None,
                                   "crossing_note": "not defined: K1 has no window variance, so no jackknife SE"}
        c = res["conditions"]
        res["effects_median_r_gm"] = {
            "placement_G1_minus_G0": c["G1"]["median_r_gm"] - c["G0"]["median_r_gm"],
            "width_G2_minus_G0": c["G2"]["median_r_gm"] - c["G0"]["median_r_gm"],
            "width_G3_minus_G1": c["G3"]["median_r_gm"] - c["G1"]["median_r_gm"],
            "interaction": (c["G3"]["median_r_gm"] - c["G1"]["median_r_gm"]) - (c["G2"]["median_r_gm"] - c["G0"]["median_r_gm"]),
            "G1_minus_K1": c["G1"]["median_r_gm"] - c["K1"]["median_r_gm"],
        }
        g1, k1 = c["G1"], c["K1"]
        reasons = []
        if g1["median_r_gm"] < 0.6:
            reasons.append("G1 median r_gm < 0.6")
        if g1["median_r_ab"] < 0.6:
            reasons.append("G1 median r_ab < 0.6")
        if not g1["median_r_gm"] > k1["median_r_gm"]:
            reasons.append("G1 does not exceed K1")
        res["sealed_rule"] = {
            "support": bool(g1["median_r_gm"] >= 0.6 and g1["median_r_gm"] > k1["median_r_gm"] and g1["median_r_ab"] >= 0.6),
            "falsified": bool(reasons), "falsification_reasons": reasons,
            "text": "spec section 7: support = G1 median r_gm >= 0.6, G1 > K1, median r_ab >= 0.6; "
                    "falsification = G1 median r_gm < 0.6, or median r_ab < 0.6, or G1 does not exceed K1",
        }
        out = RD / f"r2_readout_{tag}.json"
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(tag, "crosscheck ok", f"G0 r_gm {rgm0:.6f}", "| G1 median r_gm", round(g1["median_r_gm"], 4),
              "r_ab", round(g1["median_r_ab"], 4), "| K1 median r_gm", round(k1["median_r_gm"], 4),
              "| crossing G0", g0x["crossing_2se"], "G1", c["G1"]["crossing_2se"],
              "| falsified", bool(reasons))


if __name__ == "__main__":
    main(sys.argv[1:])

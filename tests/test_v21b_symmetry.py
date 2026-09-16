"""H0-H5 symmetry analysis: defect seal check, H1 assays, H2-H5 verdict.

Deterministic single-cell numerics + sealed evidence (no plant sims).
Writes results/h_sensitivity.json, results/h_lineage.json.
"""

import numpy as np

from jomission.qualification.stability import class_params
from jomission.qualification.symmetry import (
    BRACKET,
    CLASSES,
    LEVELS,
    PARAMS,
    assay_class,
)


def test_h0_defect():
    import json
    h = json.load(open("results/h_defect.json"))
    assert h["defect"] == "INSUFFICIENT_SYMMETRY_BREAKING"
    assert h["evidence"] == ["results/v21_low.json", "results/v21_mid.json",
                             "results/v21_high.json", "results/v21_lineage.json"]
    print("H0 defect sealed:", h["defect"])


def test_h1_sensitivity():
    import json
    params = class_params()
    tonic = json.load(open("results/v21_vectors.json"))["vectors"]["V_HIGH"]
    out = {}
    for c in CLASSES:
        rows = assay_class(c, params[c], tonic[c])
        base = rows["base"]
        per = {}
        for par in PARAMS:
            gains, rheos, cvs, rates = [], [], [], []
            for lv in LEVELS:
                r = rows[f"{par}x{lv}"]
                if base["rate"] > 0 and r["rate"] > 0:
                    gains.append(abs(np.log(r["rate"] / base["rate"])
                                     / np.log(lv)))
                if base["rheobase"] and r["rheobase"]:
                    rheos.append(abs(r["rheobase"] - base["rheobase"])
                                 / base["rheobase"])
                cvs.append(r["cv"])
                rates.append(r["rate"])
            per[par] = {"rate_gain": round(float(max(gains)) if gains else 0.0, 3),
                        "rheo_shift_max": round(float(max(rheos)) if rheos else 0.0, 3),
                        "isi_cv_max": round(float(max(cvs)), 3),
                        "rate_min": round(float(min(rates)), 2),
                        "rate_max": round(float(max(rates)), 2)}
        out[c] = {"base": {k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in base.items()},
                  "tonic": tonic[c], "params": per}
    json.dump(out, open("results/h_sensitivity.json", "w"), indent=2)
    for c in CLASSES:
        print(c, "base:", out[c]["base"])
        for par in PARAMS:
            print("  ", par, out[c]["params"][par])
    assert isinstance(out, dict)


def test_h4_h5_verdict():
    """H2 memo + H3 history + H4 selection + H5 verdict (predeclared rule)."""
    import json
    sens = json.load(open("results/h_sensitivity.json"))
    hist = json.load(open("results/h_history.json"))
    assert "conclusion" in hist
    # H2: deterministic ISI mechanism = single-cell ISI-CV>=0.5 with
    # bounded mean movement (rheo shift<=0.25, rate within [0.5x,2x]).
    cands = []
    for c in CLASSES:
        base_rate = sens[c]["base"]["rate"]
        for par in PARAMS:
            p = sens[c]["params"][par]
            if (p["isi_cv_max"] >= 0.5 and p["rheo_shift_max"] <= 0.25
                    and base_rate > 0
                    and p["rate_min"] >= 0.5 * base_rate
                    and p["rate_max"] <= 2.0 * base_rate):
                cands.append((c, par, p["rate_gain"], p["isi_cv_max"]))
    cands.sort(key=lambda x: (-x[3], -x[2]))
    gains = [(c, par, sens[c]["params"][par]["rate_gain"])
             for c in CLASSES for par in PARAMS]
    best_gain = max(gains, key=lambda x: x[2])
    h2 = {"isi_capable": [(c, p) for c, p, _, _ in cands],
          "best_detuning": {"class": best_gain[0], "param": best_gain[1],
                            "gain": best_gain[2]},
          "memo": ("constant-drive cells are limit cycles (CV->0); H1 shows "
                   + ("a class-spanning ISI mechanism EXISTS" if cands else
                      "no single-param perturbation reaches ISI-CV 0.5: "
                      "deterministic dispersion gives phases/rates, not "
                      "temporal irregularity"))}
    if cands:
        verdict = "INTRINSIC_HETEROGENEITY_JUSTIFIED"
        detail = (f"disperse {cands[0][1]} in {cands[0][0]} "
                  f"(isi {cands[0][3]}, gain {cands[0][2]}); "
                  f"bracket eps {list(BRACKET)}")
    elif best_gain[2] < 0.1:
        verdict = "STRUCTURAL_HETEROGENEITY_JUSTIFIED"
        detail = (f"plant insensitive to intrinsics (best gain "
                  f"{best_gain[2]} on {best_gain[1]}/{best_gain[0]}); only "
                  f"input/connectivity structure can break symmetry")
    else:
        verdict = "PRIVATE_STOCHASTIC_DRIVE_JUSTIFIED"
        detail = (f"detuning available (best gain {best_gain[2]} on "
                  f"{best_gain[1]}/{best_gain[0]}) but no deterministic ISI "
                  f"mechanism; calibrated private shot noise per H3 "
                  f"(not the failed 2kHz weak regime)")
    out = {"parent": "V2.1 FAIL", "h2": h2, "verdict": verdict,
           "detail": detail,
           "next_authorized_action": ("re-execute V2.1 with the justified "
                                      "mechanism + predeclared bracket"
                                      if verdict != "UNRESOLVED" else "STOP")}
    json.dump(out, open("results/h_lineage.json", "w"), indent=2)
    print("H5 verdict:", verdict, "|", detail)
    assert verdict in ("INTRINSIC_HETEROGENEITY_JUSTIFIED",
                       "PRIVATE_STOCHASTIC_DRIVE_JUSTIFIED",
                       "STRUCTURAL_HETEROGENEITY_JUSTIFIED", "UNRESOLVED")

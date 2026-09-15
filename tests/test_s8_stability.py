"""S8 full-state slow-manifold stability: assays, Jacobians, verdict.

Deterministic numerics + fresh single-cell assays (no network sims, no
tuning). Writes results/s8_assay.json, results/s8_candidate_s{TAG}.json,
results/s8_lineage.json. Stops before nonlinear basin execution.
"""

import numpy as np

from jomission.qualification.stability import (
    CANDIDATES,
    MARGIN,
    assay_FV,
    build_jacobian,
    class_params,
    scale_tag,
)

GROUPS = {"neural-U": (0, 3), "HDP-aux": (3, 8), "HDP-w": (8, 13), "H": (13, 14)}


def analyze(s, FV, params):
    import json
    b = build_jacobian(s, FV, params)
    J = b["J"]
    vals, vecs = np.linalg.eig(J)
    assert np.all(np.isfinite(vals))
    assert J.shape == (14, 14)  # all active coordinates represented
    i = int(np.argmax(vals.real))
    slowest = {"lambda": complex(vals[i]),
               "maxRe": float(vals.real.max())}
    v = np.abs(vecs[:, i]) ** 2
    v = v / v.sum()
    part = {g: float(v[a:b].sum()) for g, (a, b) in GROUPS.items()}
    slowest["participation"] = part
    slowest["dominant"] = max(part, key=part.get)
    slowest["boundary_distance"] = float(MARGIN - slowest["maxRe"])
    out = {"scale": s, "margin": MARGIN,
           "maxRe": slowest["maxRe"],
           "slowest": {"re": slowest["lambda"].real,
                       "im": slowest["lambda"].imag,
                       "participation": part, "dominant": slowest["dominant"]},
           "boundary_distance": slowest["boundary_distance"],
           "fast_rate_maxeig": b["fast_max"],
           "slaving_sigma_min": b["slaving_sigma_min"],
           "gains": {"Fprime": b["Fp"], "Vprime": b["Vp"], "G": b["G"],
                     "G_R2": b["R2"]},
           "pass": bool(slowest["maxRe"] <= MARGIN)}
    json.dump(out, open(f"results/s8_candidate_s{scale_tag(s)}.json", "w"),
              indent=2)
    print(f"s={s} maxRe={out['maxRe']:.6f} dominant={slowest['dominant']} "
          f"pass={out['pass']}")
    return out


def test_s8_assay():
    """Fresh joint (F,Vbar) assay; sealed-F consistency gate."""
    import json
    params = class_params()
    FV = assay_FV(params)
    sealed = json.load(open("results/battery_b1.json"))["F"]
    for c in ("E", "PV", "SST", "VIP"):
        Ii = np.array(sealed[c]["I"])
        rr = np.array(sealed[c]["r"])
        for i0, r0 in zip(Ii, rr):
            r1 = float(np.interp(i0, FV[c]["I"], FV[c]["r"]))
            if r0 > 1.0:
                assert abs(r1 - r0) / r0 <= 0.05, (c, i0, r0, r1)
    json.dump({c: {"I": FV[c]["I"].tolist(), "r": FV[c]["r"].tolist(),
                   "V": FV[c]["V"].tolist()} for c in FV},
              open("results/s8_assay.json", "w"))
    print("S8 assay OK; fresh F matches sealed within 5%")


def test_s8_candidates():
    import json
    FV_raw = json.load(open("results/s8_assay.json"))
    FV = {c: {"I": np.array(v["I"]), "r": np.array(v["r"]),
              "V": np.array(v["V"])} for c, v in FV_raw.items()}
    params = class_params()
    res = [analyze(s, FV, params) for s in CANDIDATES]
    passing = [r["scale"] for r in res if r["pass"]]
    lin = {"parent": "23c56d5", "margin": MARGIN,
           "candidates": [{"scale": r["scale"], "maxRe": r["maxRe"],
                            "dominant": r["slowest"]["dominant"],
                            "participation": r["slowest"]["participation"],
                            "boundary_distance": r["boundary_distance"],
                            "pass": r["pass"]} for r in res],
           "passing_subset": passing,
           "selection": "none performed",
           "verdict": ("FULL_STATE_STABILITY_PASS" if passing
                       else "FULL_STATE_STABILITY_FAIL"),
           "next_authorized_action": ("STOP before nonlinear basin execution"
                                      if passing else
                                      "STOP; mode participation diagnoses the failed direction")}
    json.dump(lin, open("results/s8_lineage.json", "w"), indent=2)
    print("S8 verdict:", lin["verdict"], passing)
    assert lin["verdict"] in ("FULL_STATE_STABILITY_PASS",
                              "FULL_STATE_STABILITY_FAIL",
                              "FULL_STATE_STABILITY_UNRESOLVED")

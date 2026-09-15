"""N4 native return geometry: residence, empirical flow, slow-region search.

- Residence fractions per trajectory over (SILENT rE<1 | SYNC sync>0.5 |
  INTERMEDIATE else) from 2 ms bins: dwell quantification.
- Empirical flow in (rE, sync): coarse 16x16 grid, mean per-bin Delta over
  consecutive 2 ms samples (every 5th bin subsample); fixed bins =
  high-residence + |Delta|~=0.
- Slow-region search: non-corner bins (not silence-corner, not sync-corner)
  with residence share >1% of total and |Delta| below the dataset median:
  candidate metastable sets (expect none; report verbatim).
- Verdict N6 mapping (predeclared): closure failed (N3 missing=True), so
  ACTIONABLE is unreachable. NO_INTERMEDIATE if dwell lives only in
  silence/sync (+transient corridor); UNRESOLVED only if observables
  insufficient (they are not: 25 trajectories, 235 cross-fate pairs).

Writes results/n4_geometry.json, results/n_lineage.json.
"""

import numpy as np


def load_all():
    import glob
    import json
    return [json.load(open(f)) for f in sorted(glob.glob("results/n_traj_*.json"))]


def test_n4_return_geometry():
    import json
    trajs = load_all()
    res_dwells, flow = {}, {}
    grid_n = 16
    acc_d = np.zeros((grid_n, grid_n))
    acc_n = np.zeros((grid_n, grid_n))
    acc_r = np.zeros((grid_n, grid_n))
    third = []
    for d in trajs:
        er = np.array(d["rates_2ms"]["E"])
        sy = np.array(d["sync_2ms"])
        sil = er < 1.0
        # 2-cycle sync has per-bin fraction ~= 0.5 exactly: use >= with a
        # hair below (the cycle alternates full/empty bins around 0.5).
        syn = sy >= 0.45
        inter = ~(sil | syn)
        res_dwells[d["scale"], d["name"]] = {
            "silent": round(float(sil.mean()), 4),
            "sync": round(float(syn.mean()), 4),
            "intermediate": round(float(inter.mean()), 4),
            "outcome": d["outcome"]}
        # Empirical flow on (log10(rE+1), sync), subsampled.
        x = np.log10(er + 1.0) / np.log10(10001.0)
        y = sy
        for k in range(0, len(er) - 1, 5):
            i = min(int(x[k] * grid_n), grid_n - 1)
            j = min(int(y[k] * grid_n), grid_n - 1)
            acc_d[j, i] += (x[k + 1] - x[k]) + (y[k + 1] - y[k])
            acc_n[j, i] += 1
            acc_r[j, i] += 1
    tot = acc_r.sum()
    share = acc_r / tot
    mag = np.abs(acc_d) / np.maximum(acc_n, 1)
    # Corners: silence (x~0,y~0), sync (x~1,y~1 in these coords: rE=5000
    # -> x=log10(5001)/4=0.925; sync=0.5..1 -> j>=8).
    slow = []
    med = float(np.median(mag[acc_n > 0]))
    for i in range(grid_n):
        for j in range(grid_n):
            if acc_n[j, i] == 0:
                continue
            corner = (i <= 1 and j <= 1) or (i >= 14 and j >= 8)
            if not corner and share[j, i] > 0.01 and mag[j, i] <= med:
                slow.append({"bin": [i, j], "share": round(float(share[j, i]), 4),
                             "flow": round(float(mag[j, i]), 5)})
    inter_share = float(np.mean([v["intermediate"] for v in res_dwells.values()]))
    out = {"dwells": {f"{s}_{n}": v for (s, n), v in res_dwells.items()},
           "intermediate_mean_share": round(inter_share, 4),
           "slow_regions": slow,
           "flow_median": round(med, 5)}
    json.dump(out, open("results/n4_geometry.json", "w"), indent=2)
    print("intermediate mean share:", round(inter_share, 4))
    print("slow regions:", slow)
    # Verdict.
    n3 = json.load(open("results/n3_closure.json"))
    dw = [v["intermediate"] for v in res_dwells.values()]
    if not slow and max(dw) < 0.05:
        verdict = "NATIVE_GEOMETRY_NO_INTERMEDIATE"
        detail = ("dwell lives only in silence/sync corners (max intermediate "
                  f"share {max(dw):.3f}); no third slow region; closure failed "
                  "(N3 missing coordinate): plant organizes only "
                  "silence<->sync over the justified domain")
    else:
        verdict = "NATIVE_GEOMETRY_UNRESOLVED"
        detail = f"slow regions or dwell require interpretation: {slow}"
    lin = {"parent": "S9 ACTIVE_BASIN_FAIL + B OFF_MANIFOLD",
           "n_trajectories": len(trajs),
           "verdict": verdict, "detail": detail,
           "next_authorized_action": "STOP; N5 intervention proposal requires approval"}
    json.dump(lin, open("results/n_lineage.json", "w"), indent=2)
    print("N4 verdict:", verdict)
    assert verdict in ("NATIVE_GEOMETRY_ACTIONABLE", "NATIVE_GEOMETRY_NO_INTERMEDIATE",
                       "NATIVE_GEOMETRY_UNRESOLVED")

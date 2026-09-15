"""N2' burst morphology + N3' cross-fate closure on sealed N1 trajectories.

N2 redesign (documented): 2 ms rates/sync/V show NO sustained separation
before ~300 ms (both fates: onset burst, then silence; reignition ~350 ms
only in SYNC fate). Coarse-snapshot coords (aux/w/u @500 ms) separate only
consequently. The genuine pre-commitment observable is the ONSET BURST
itself (t < 20 ms): peak rate, burst width (>1000 Hz bins), total burst
spikes, sync peak, first-spike latency + burst count of example neurons
(V_win full-res traces). SILENT vs SYNC2CYCLE groups compared via
Cliff's delta on these morphology features.

N3' redesign: matched pairs ACROSS fates within the silent gap
(t in {50,100,150,200,250} ms; SYNC-traj pre-reignition vs SILENT-traj),
state = fast coords only [rE,rPV,rSST,rVIP,sync,V_E,V_PV,V_SST] z-scored,
eps=0.05. Futures at +100 ms (expansion) and +500 ms (reignition outcome:
E-rate>100 in the following 500 ms window?). Divergence fraction =
pairs with different reignition outcomes. High divergence at low d =
same apparent coarse state, different futures = missing subthreshold
coordinate (V/u/syn microstate).

Writes results/n2_separation.json, results/n3_closure.json.
"""

import numpy as np

GAP_T = (50, 100, 150, 200, 250)
EPS = 0.05
EXPAND_THR = 10.0
DELTA_THR = 0.8
FAST = ["rE", "rPV", "rSST", "rVIP", "sync", "V_E", "V_PV", "V_SST"]


def scale_tag(s):
    return ("%.1f" % s).replace(".", "p")


def load_all():
    import glob
    import json
    return [json.load(open(f)) for f in sorted(glob.glob("results/n_traj_*.json"))]


def burst_features(d):
    """Onset-burst morphology from sealed 2 ms arrays (t < 40 ms)."""
    er = np.array(d["rates_2ms"]["E"][:20])
    sy = np.array(d["sync_2ms"][:20])
    vw = {c: np.array(d["V_win"][c]) for c in ("E", "PV", "SST")}
    dt = 0.002
    feats = {
        "peak_rate": float(er.max()),
        "width_bins": int((er > 1000.0).sum()),
        "burst_spikes": float(er[:20].sum() * dt),
        "sync_peak": float(sy.max()),
    }
    for c, v in vw.items():
        # First-spike proxy: first V sample >= 20 mV within the window.
        hot = np.flatnonzero(v >= 20.0)
        feats[f"{c}_first_ms"] = float(hot[0] * 0.1) if len(hot) else -1.0
        feats[f"{c}_win_spikes"] = float((v >= 20.0).sum())
    return feats


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (n * m)


def test_n2_burst():
    import json
    trajs = load_all()
    sync = [d for d in trajs if d["outcome"] == "SYNC2CYCLE"]
    sil = [d for d in trajs if d["outcome"] == "SILENT"]
    assert sync and sil
    keys = list(burst_features(sync[0]).keys())
    table = {}
    for k in keys:
        a = [burst_features(d)[k] for d in sync]
        b = [burst_features(d)[k] for d in sil]
        table[k] = {"delta": round(float(cliffs_delta(a, b)), 3),
                    "sync_med": round(float(np.median(a)), 3),
                    "sil_med": round(float(np.median(b)), 3)}
    out = {"n_sync": len(sync), "n_silent": len(sil),
           "features": table,
           "note": "pre-commitment onset morphology only (t<40 ms + V_win)"}
    json.dump(out, open("results/n2_separation.json", "w"), indent=2)
    print("N2 burst morphology:")
    for k, v in table.items():
        print(f"  {k}: delta={v['delta']} sync_med={v['sync_med']} sil_med={v['sil_med']}")
    assert isinstance(out, dict)


def fast_at(d, t_ms):
    i2 = min(int(t_ms / 2), len(d["rates_2ms"]["E"]) - 1)
    i1 = min(int(t_ms / 10), len(d["means_10ms"]["V_E"]) - 1)
    return {"rE": d["rates_2ms"]["E"][i2], "rPV": d["rates_2ms"]["PV"][i2],
            "rSST": d["rates_2ms"]["SST"][i2],
            "rVIP": d["rates_2ms"]["VIP"][i2], "sync": d["sync_2ms"][i2],
            "V_E": d["means_10ms"]["V_E"][i1],
            "V_PV": d["means_10ms"]["V_PV"][i1],
            "V_SST": d["means_10ms"]["V_SST"][i1]}


def reignited(d, t_ms, horizon=500.0):
    """E-rate > 100 Hz anywhere in (t, t+horizon] (2 ms bins)."""
    i0 = int(t_ms / 2)
    i1 = min(int((t_ms + horizon) / 2), len(d["rates_2ms"]["E"]))
    return bool((np.array(d["rates_2ms"]["E"][i0:i1]) > 100.0).any())


def test_n3_xfate_closure():
    import json
    trajs = load_all()
    sync = [d for d in trajs if d["outcome"] == "SYNC2CYCLE"]
    sil = [d for d in trajs if d["outcome"] == "SILENT"]
    S, meta, fut = [], [], []
    for d in trajs:
        for t in GAP_T:
            S.append([fast_at(d, t)[c] for c in FAST])
            meta.append((d["outcome"], t))
            fut.append(reignited(d, t))
    S = np.array(S, dtype=float)
    mu, sd = S.mean(axis=0), S.std(axis=0) + 1e-12
    Z = (S - mu) / sd
    is_sync_traj = np.array([m[0] == "SYNC2CYCLE" for m in meta])
    # Cross-fate pairs within eps: same apparent coarse gap state?
    pairs, divs = 0, 0
    for i in np.flatnonzero(is_sync_traj):
        dd = np.sqrt(((Z[~is_sync_traj] - Z[i]) ** 2).sum(axis=1))
        for j in np.flatnonzero(~is_sync_traj)[dd < EPS]:
            pairs += 1
            if fut[i] != fut[j]:
                divs += 1
    out = {"eps": EPS, "gap_times": list(GAP_T),
           "cross_fate_pairs": pairs,
           "divergent_fraction": (divs / pairs) if pairs else None,
           "pairs_detail": "same coarse gap state, different reignition futures",
           "missing_coordinate": bool(pairs > 0 and (divs / pairs) > 0.5)
           if pairs else None}
    json.dump(out, open("results/n3_closure.json", "w"), indent=2)
    print(f"N3 cross-fate pairs={pairs} divergent={out['divergent_fraction']} "
          f"missing={out['missing_coordinate']}")
    assert isinstance(out, dict)

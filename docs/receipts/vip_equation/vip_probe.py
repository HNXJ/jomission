"""VIP equation probe: which v-equation each construction path yields for VIP cells.

Read-only with respect to the repo. Constructs models (no simulation).
Usage: python vip_probe.py <out.json>   (run with PYTHONPATH = engine tree + jomission checkout)
"""
import dataclasses
import json
import sys

import numpy as np

import jaxfne as jtfne
from jomission.network.builder import build_jomission_model
from jomission.qualification import cmin


def emitter_fields(e):
    if dataclasses.is_dataclass(e):
        return [f.name for f in dataclasses.fields(e)]
    return sorted(vars(e))


def per_type_eq(model):
    tbl = model.neuron_table()
    e = model.params["emitter"]
    ct = np.array([str(r["cell_type"]) for r in tbl])
    vl = getattr(e, "v_lin", None)
    vc = getattr(e, "v_const", None)
    out = {}
    for t in sorted(set(ct.tolist())):
        idx = np.where(ct == t)[0]
        if vl is None:
            out[t] = {"n": int(idx.size), "v_eq": "no v_lin field (engine standard 5/140)"}
            continue
        pairs = sorted(set(zip(np.round(np.asarray(vl)[idx], 6).tolist(),
                               np.round(np.asarray(vc)[idx], 6).tolist())))
        out[t] = {"n": int(idx.size), "v_lin_const_pairs": pairs}
    return out


def vip_summary(model, label):
    tbl = model.neuron_table()
    e = model.params["emitter"]
    ct = np.array([str(r["cell_type"]) for r in tbl])
    idx = np.where(ct == "VIP")[0]
    rec = {"path": label, "n_cells": int(ct.size), "n_VIP": int(idx.size),
           "emitter_fields": emitter_fields(e), "per_type": per_type_eq(model)}
    if idx.size:
        b = np.asarray(e.b, dtype=float)
        rec["VIP_b_min_max"] = [round(float(b[idx].min()), 6), round(float(b[idx].max()), 6)]
        for name in ("a", "c", "d"):
            v = np.asarray(getattr(e, name), dtype=float)
            rec[f"VIP_{name}_min_max"] = [round(float(v[idx].min()), 6), round(float(v[idx].max()), 6)]
    return rec


def main(out_path):
    results = {"engine": jtfne.__file__, "paths": []}
    jomission_model = build_jomission_model(n_per_area=100, seed=0)
    results["paths"].append(vip_summary(jomission_model, "builder: build_jomission_model(n_per_area=100, seed=0)"))

    raw = cmin.build_cmin(seed=0)
    results["paths"].append(vip_summary(raw, "cmin: build_cmin(seed=0), raw"))

    repaired = cmin.repair_jitter(cmin.repair_tonic(cmin.build_cmin(seed=0)), seed=0)
    results["paths"].append(vip_summary(repaired, "cmin: repair_jitter(repair_tonic(build_cmin(seed=0)))"))

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, sort_keys=True, default=str)
    print("engine:", results["engine"])
    for p in results["paths"]:
        print(p["path"], "| VIP cells:", p["n_VIP"], "| VIP b:", p.get("VIP_b_min_max"))
    print("wrote:", out_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])

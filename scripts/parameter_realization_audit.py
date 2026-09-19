"""Read-only parameter-realization audit of the whole-system architecture.

    python scripts/parameter_realization_audit.py [--out results/parameter_realization_audit.json]

WS-REALIZE-R4 proved graph identity: objects, populations, projection identities and the index
map all reconciled exactly, and that PASS stands. It did not compare parameters. The string
"weight" does not appear anywhere in results/whole_system_realization_r2.json, so the spec's
frozen `weight: 0.5` was never checked against what the engine realized. This audit supplies the
missing half of the contract, per projection class:

    identity, N_edges, realized p, w, tau, delay, mechanism

and then does the inverse calibration: for each projection family it reports the afferent
authority a postsynaptic cell actually receives,

    K_p = N_afferents,p * mean|w_p|     (equivalently, the summed afferent weight per target cell)

and the long-range to local ratio

    g_p = K_long,p / K_local,reference

separately for feedforward, feedback and lateral. It measures; it chooses nothing. The intended
value of g is a model-definition decision and is not inferred here.

No simulation is run and nothing about the model is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jomission.tfne import architecture, nf, realize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "parameter_realization_audit.json"

#: The relation each TFNE cross-area projection realizes, keyed by (pre layer, post layer).
#: ff and fb come from the O[fffb] clauses; lat from X[lat].
RELATION = {("L2", "L4"): "ff", ("L3", "L4"): "ff", ("L6", "L1"): "fb", ("L3", "L3"): "lat"}


def build():
    chain, rules, external, decl = architecture.build()
    n = nf.normalize(chain, rules, external)
    cfg = realize.declare_projections(realize.declare_mechanisms(realize.scaffold(n)[0], n), n)
    return realize.construct(cfg)


def audit(model) -> dict:
    idx, (area, lay, cls) = realize.index_map(model)
    area, lay, cls = np.asarray(area), np.asarray(lay), np.asarray(cls)
    el = model.params["edge_list"]
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    w = np.asarray(el.weight)
    # tau_ms and delay_steps are stored uniformly, not per edge (tau_storage says which). The
    # kernel resolves them through these helpers, so the audit reads what the kernel would use
    # rather than the empty arrays the edge list carries.
    import jax.numpy as jnp
    from jaxfne.emitters import resolve_edge_delay_steps, resolve_edge_tau_ms

    tau = np.asarray(resolve_edge_tau_ms(el, jnp.float32))
    delay = np.asarray(resolve_edge_delay_steps(el))
    if tau.size != w.size:
        tau = np.full(w.shape, float(tau.reshape(-1)[0]) if tau.size else float("nan"))
    if delay.size != w.size:
        delay = np.full(w.shape, int(delay.reshape(-1)[0]) if delay.size else 0)
    recep = np.asarray(el.receptor_index)
    group = np.array([f"{a}.{layer}.{c}" for a, layer, c in zip(area, lay, cls)])
    sizes = {g: int(v[2]) for g, v in idx.items()}
    same_area = area[pre] == area[post]

    def stats(mask, name, relation):
        ww = w[mask]
        aw = np.abs(ww)
        if aw.size == 0:
            return None
        g_pre = sorted(set(group[pre[mask]].tolist()))
        g_post = sorted(set(group[post[mask]].tolist()))
        n_pre = sum(sizes.get(g, 0) for g in g_pre)
        n_post = sum(sizes.get(g, 0) for g in g_post)
        # Afferents of this family per postsynaptic cell that actually receives any.
        per_target = np.bincount(post[mask], minlength=group.size)
        recv = per_target[per_target > 0]
        # K = summed afferent |w| per receiving cell.
        k_per = np.zeros(group.size)
        np.add.at(k_per, post[mask], aw)
        k_recv = k_per[per_target > 0]
        taus = sorted({round(float(t), 6) for t in tau[mask]})
        return {
            "name": name,
            "relation": relation,
            "pre_groups": g_pre,
            "post_groups": g_post,
            "n_edges": int(aw.size),
            "n_pre": n_pre,
            "n_post": n_post,
            "p_realized": float(aw.size / max(n_pre * n_post, 1)),
            "w_mean": float(ww.mean()), "w_abs_mean": float(aw.mean()),
            "w_abs_min": float(aw.min()), "w_abs_max": float(aw.max()),
            "w_uniform": bool(aw.min() == aw.max()),
            "sign": ("excitatory" if (ww > 0).all() else
                     "inhibitory" if (ww < 0).all() else "mixed"),
            "tau_ms": taus if len(taus) <= 4 else [taus[0], "...", taus[-1]],
            "delay_steps": sorted({int(d) for d in delay[mask]})[:4],
            "receptor_index": sorted({int(r) for r in recep[mask]}),
            "afferents_per_target_mean": float(recv.mean()),
            "K_per_target_mean": float(k_recv.mean()),
            "K_per_target_min": float(k_recv.min()),
            "K_per_target_max": float(k_recv.max()),
        }

    # --- cross-area: one entry per declared TFNE projection identity ---
    long_range = []
    seen = set()
    for e in np.flatnonzero(~same_area):
        key = (group[pre[e]], group[post[e]])
        if key in seen:
            continue
        seen.add(key)
        mask = (~same_area) & (group[pre] == key[0]) & (group[post] == key[1])
        rel = RELATION.get((key[0].split(".")[1], key[1].split(".")[1]), "unclassified")
        long_range.append(stats(mask, f"{key[0]} -> {key[1]}", rel))
    long_range.sort(key=lambda d: (d["relation"], d["name"]))

    # --- intra-area: one entry per (pre class -> post class) family, pooled over areas ---
    local = []
    for cp in ["E", "PV", "SST", "VIP"]:
        for cq in ["E", "PV", "SST", "VIP"]:
            mask = same_area & (cls[pre] == cp) & (cls[post] == cq)
            s = stats(mask, f"{cp} -> {cq} (intra-area)", "local")
            if s:
                s["pre_groups"] = [f"<all {cp}>"]
                s["post_groups"] = [f"<all {cq}>"]
                local.append(s)

    # --- authority per postsynaptic class: local excitatory reference vs long-range ---
    exc = w > 0
    authority = {}
    for c in ["E", "PV", "SST", "VIP"]:
        k = np.flatnonzero(cls == c)
        row = {}
        for nm, mask in [
            ("local_exc", same_area & exc),
            ("local_inh", same_area & ~exc),
            ("long_ff", (~same_area) & np.isin(
                [RELATION.get((a.split(".")[1], b.split(".")[1])) for a, b in
                 zip(group[pre], group[post])], ["ff"])),
            ("long_fb", (~same_area) & np.isin(
                [RELATION.get((a.split(".")[1], b.split(".")[1])) for a, b in
                 zip(group[pre], group[post])], ["fb"])),
            ("long_lat", (~same_area) & np.isin(
                [RELATION.get((a.split(".")[1], b.split(".")[1])) for a, b in
                 zip(group[pre], group[post])], ["lat"])),
        ]:
            t = np.zeros(group.size)
            np.add.at(t, post[mask], np.abs(w[mask]))
            row[nm] = float(t[k].mean())
        ref = row["local_exc"] or float("nan")
        row["g_ff"] = row["long_ff"] / ref
        row["g_fb"] = row["long_fb"] / ref
        row["g_lat"] = row["long_lat"] / ref
        row["g_long_total"] = (row["long_ff"] + row["long_fb"] + row["long_lat"]) / ref
        row["n"] = int(k.size)
        authority[c] = row

    aw_all = np.abs(w)
    return {
        "schema": "jomission.parameter_realization_audit.v1",
        "scope": ("parameters only. WS-REALIZE-R4 established graph identity and that PASS is not "
                  "retracted; this audit covers what R4 did not compare"),
        "read_only": "no simulation; the model is constructed and measured, never modified",
        "spec_frozen_weight": 0.5,
        "spec_frozen_probability": 1.0,
        "spec_source": "results/whole_system_diagnostic_3_spec.json frozen block",
        "tau_delay_storage": {"tau_storage": str(el.tau_storage),
                              "uniform_delay_steps": int(el.uniform_delay_steps),
                              "note": ("tau and delay are stored uniformly, not per edge; these are "
                                       "resolved through the engine helpers the kernel itself uses")},
        "totals": {
            "n_edges": int(w.size),
            "n_intra_area": int(same_area.sum()),
            "n_cross_area": int((~same_area).sum()),
            "w_abs_mean_intra": float(aw_all[same_area].mean()),
            "w_abs_mean_cross": float(aw_all[~same_area].mean()),
            "disparity_cross_over_intra": float(aw_all[~same_area].mean() / aw_all[same_area].mean()),
            "cross_edges_at_frozen_weight": float((aw_all[~same_area] == 0.5).mean()),
            "intra_edges_at_frozen_weight": float((aw_all[same_area] == 0.5).mean()),
        },
        "long_range_projections": long_range,
        "local_families": local,
        "authority_per_postsynaptic_class": authority,
        "qualification": "PARAMETER_REALIZATION_UNQUALIFIED",
        "qualification_reason": (
            "the spec freezes weight 0.5 and probability 1.0 without scoping them. Realization "
            "applied 0.5 to every cross-area edge and none of the intra-area edges, which carry "
            "engine-assigned weights. No gate in the chain compared them, so the parameter half of "
            "the realization contract is unestablished, not passed"),
        "decision_not_taken": (
            "the intended long-range to local authority ratio g is a model-definition decision. "
            "This audit reports the realized g and does not propose a value"),
    }


def main(out: Path | None = None) -> None:
    rec = audit(build())
    t = rec["totals"]
    print("PARAMETER_REALIZATION_UNQUALIFIED")
    print("  edges %d = intra %d + cross %d" % (t["n_edges"], t["n_intra_area"], t["n_cross_area"]))
    print("  mean |w|: intra %.6f  cross %.6f  disparity %.1fx"
          % (t["w_abs_mean_intra"], t["w_abs_mean_cross"], t["disparity_cross_over_intra"]))
    print("  at the frozen 0.5: cross %.4f of edges, intra %.4f"
          % (t["cross_edges_at_frozen_weight"], t["intra_edges_at_frozen_weight"]))
    print()
    print("  %-34s %-4s %7s %9s %10s %9s" % ("projection", "rel", "edges", "p", "|w|", "K/target"))
    for p in rec["long_range_projections"]:
        print("  %-34s %-4s %7d %9.4f %10.4f %9.4f"
              % (p["name"], p["relation"], p["n_edges"], p["p_realized"], p["w_abs_mean"],
                 p["K_per_target_mean"]))
    print()
    print("  %-22s %-4s %7s %9s %10s %9s" % ("local family", "rel", "edges", "p", "|w|", "K/target"))
    for p in rec["local_families"]:
        print("  %-22s %-4s %7d %9.4f %10.6f %9.4f"
              % (p["name"], p["relation"], p["n_edges"], p["p_realized"], p["w_abs_mean"],
                 p["K_per_target_mean"]))
    print()
    print("  authority per postsynaptic class (K = summed afferent |w| per cell):")
    print("  %-5s %6s %10s %10s %9s %9s %9s %9s" %
          ("class", "n", "local_exc", "local_inh", "long_ff", "long_fb", "long_lat", "g_long"))
    for c, r in rec["authority_per_postsynaptic_class"].items():
        print("  %-5s %6d %10.4f %10.4f %9.4f %9.4f %9.4f %9.2f"
              % (c, r["n"], r["local_exc"], r["local_inh"], r["long_ff"], r["long_fb"],
                 r["long_lat"], r["g_long_total"]))
    dest = (ROOT / out) if out and not Path(out).is_absolute() else (out or OUT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("\nwrote", dest.relative_to(ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(Path(a.out) if a.out else None)

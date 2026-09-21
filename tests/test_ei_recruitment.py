"""SCI-EI-RECRUITMENT-1: native-resolution current-budget measurement, read-only.

No construction is simulated. Static weights, counts, tonic and parameters come
from a read-only rebuild of the sealed construction; currents, rates, v and u
come from the retained EI 1 ms series; F-I boundaries come from sealed curves.
The E/I split is mean-field reconstructed and gated by its residual against
the retained engine totals before it may classify. Writes
results/whole_system_ei_recruitment.json plus the current-budget atlas.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "whole_system_ei_recruitment.json"
ATLAS = ROOT / "results" / "viz" / "SCI-EI-RECRUITMENT-1" / "current_budget_atlas.png"

CLASSES = ("E", "PV", "SST", "VIP")
SILENT_HZ = 1.0
EXC_FRACTION = 0.2
RESIDUAL_TOL = 0.2


def test_ei_recruitment():
    import scripts.whole_system_ei_regime_arm as fork

    model, _, _, _ = fork.build()
    index, (area, layer, cls) = fork.realize.index_map(model)
    model, _ = fork.enforce(model, index)
    model, authority = fork.o_authority.realize_equal_g(model, index, area, layer, cls, 0.0)
    assert authority["acceptance"]["pass"]
    members = {g: np.arange(index[g][0], index[g][1]) for g in fork.groups_of(index)}
    cls_arr = np.array([str(r.get("cell_type")) for r in model.neuron_table()])
    n = len(cls_arr)
    em = model.params["emitter"]
    tonic = np.asarray(em.drive, dtype=float)
    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)

    z = np.load(ROOT / "results" / "whole_system_ei_regime_diagnosis_y.npz")
    dt_s = float(z["dt_ms"]) / 1000.0
    groups = [str(g) for g in z["series_groups"]]
    grate = {g: np.asarray(z["series_rate_hz"])[i] for i, g in enumerate(groups)}
    gcurr = {g: np.asarray(z["series_current_mean"])[i] for i, g in enumerate(groups)}
    gvm = {g: np.asarray(z["series_v_mean"])[i] for i, g in enumerate(groups)}
    guu = {g: np.asarray(z["series_u_mean"])[i] for i, g in enumerate(groups)}

    def cortical(c):
        return [g for g in groups if g.split(".")[-1] == c and not g.startswith("Retina")]

    fbi = json.load(open(ROOT / "results" / "battery_b1.json"))["F"]

    out = {
        "spec": "results/whole_system_ei_recruitment_spec.json",
        "lineage": "SCI-EI-RECRUITMENT-1",
        "classes": {},
    }
    for c in CLASSES:
        is_c = cls_arr == c
        n_c = int(is_c.sum())
        tonic_c = float(tonic[is_c].mean())
        e_groups = cortical("E")
        i_e = 0.0
        for g in e_groups:
            gi = members[g]
            m = np.isin(pre, gi) & is_c[post]
            i_e += float(w[m].sum() * grate[g].mean() * dt_s)
        i_e /= max(n_c, 1)
        parts = {}
        for src in ("PV", "SST", "VIP"):
            tot = 0.0
            for g in cortical(src):
                m = np.isin(pre, members[g]) & is_c[post]
                tot += float(w[m].sum() * grate[g].mean() * dt_s)
            parts[src] = tot / max(n_c, 1)
        i_i = float(sum(parts.values()))
        i_syn_ret = float(np.mean([gcurr[g] for g in cortical(c)], axis=0).mean())
        i_net = float(tonic_c + i_syn_ret)
        i_net_t = tonic_c + np.mean([gcurr[g] for g in cortical(c)], axis=0)
        r = float(np.mean([grate[g] for g in cortical(c)], axis=0).mean())
        k_e = int(((cls_arr[pre] == "E") & is_c[post]).sum())
        w_e = w[(cls_arr[pre] == "E") & is_c[post]]
        I = np.asarray(fbi[c]["I"], dtype=float)
        rr = np.asarray(fbi[c]["r"], dtype=float)
        hit = np.flatnonzero(rr > 1.0)
        rheo = float(I[hit[0]]) if hit.size else None
        exp_at_net = float(np.interp(i_net, I, rr, left=0.0, right=float(rr[-1])))
        out["classes"][c] = {
            "n_cells": n_c,
            "tonic": round(tonic_c, 4),
            "k_E_in": k_e,
            "w_E_sum_per_cell": round(float(w_e.sum() / max(n_c, 1)), 6),
            "I_E_mean": round(i_e, 4),
            "I_I_split": {k: round(v, 4) for k, v in parts.items()},
            "I_I_mean": round(i_i, 4),
            "I_syn_retained": round(i_syn_ret, 4),
            "reconstruction_residual": round(float(i_e + i_i - i_syn_ret), 4),
            "residual_ok": bool(
                abs(float(i_e + i_i - i_syn_ret)) <= RESIDUAL_TOL * max(abs(i_syn_ret), 1e-9)
            ),
            "I_net": round(i_net, 4),
            "I_net_max": round(float(i_net_t.max()), 4),
            "I_net_frac_above_rheobase": round(float((i_net_t >= rheo).mean()), 4)
            if rheo is not None
            else None,
            "rate_hz": round(r, 4),
            "v_mean": round(float(np.mean([gvm[g] for g in cortical(c)])), 3),
            "u_mean": round(float(np.mean([guu[g] for g in cortical(c)])), 3),
            "rheobase": rheo,
            "F_expected_at_I_net": round(exp_at_net, 3),
        }

    def cause(c):
        d = out["classes"][c]
        if not d["rate_hz"] < SILENT_HZ:
            return "NOT_SILENT"
        if d["residual_ok"] and d["I_E_mean"] < EXC_FRACTION * d["tonic"]:
            return "E_I_RECRUITMENT_DEFICIT"
        if d["residual_ok"] and d["I_I_mean"] < 0 and abs(d["I_I_mean"]) > d["I_E_mean"]:
            return "INHIBITORY_VETO"
        if d["rheobase"] is not None and d["I_net"] >= d["rheobase"]:
            return "INTRINSIC_EXCITABILITY_LIMIT"
        if (
            d["rheobase"] is not None
            and d["I_net_max"] is not None
            and d["I_net_max"] >= d["rheobase"] > d["I_net"]
        ):
            return "STATE_DEPENDENT_RECRUITMENT_FAIL"
        return "EI_RECRUITMENT_UNRESOLVED"

    causes = {c: cause(c) for c in ("PV", "VIP")}
    if causes["PV"] == causes["VIP"] and causes["PV"] != "NOT_SILENT":
        verdict = causes["PV"]
    elif causes["PV"] != causes["VIP"] and "NOT_SILENT" not in causes.values():
        verdict = "MIXED_RECRUITMENT_CAUSES"
    else:
        verdict = "EI_RECRUITMENT_UNRESOLVED"
    out["causes"] = causes
    out["verdict"] = verdict
    out["fail_boundary"] = f"PV: {causes['PV']}; VIP: {causes['VIP']}"

    ATLAS.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    t = np.arange(len(next(iter(grate.values()))), dtype=float)
    for i, c in enumerate(CLASSES):
        ax = axes[i, 0]
        morate = np.mean([grate[g] for g in cortical(c)], axis=0)
        ax.plot(t, morate, label="r(t)")
        ax.set_title(f"{c} rate (Hz), mean {morate.mean():.3f}")
        ax = axes[i, 1]
        mocurr = np.mean([gcurr[g] for g in cortical(c)], axis=0)
        ax.plot(t, mocurr, label="I_syn(t)")
        ax.axhline(out["classes"][c]["tonic"], linestyle="--", label="tonic")
        rheo = out["classes"][c]["rheobase"]
        if rheo is not None:
            ax.axhline(rheo, linestyle=":", label="rheobase")
        e_t = np.zeros_like(t)
        for g in [x for x in groups if x.split(".")[-1] == "E" and not x.startswith("Retina")]:
            m = np.isin(pre, members[g]) & (cls_arr[post] == c)
            e_t = e_t + float(w[m].sum()) * np.asarray(grate[g]) * dt_s
        n_c = out["classes"][c]["n_cells"]
        ax.plot(t, e_t / max(n_c, 1), label="I_E(t)")
        ax.plot(t, mocurr - e_t / max(n_c, 1), label="I_I(t)")
        ax.set_title(f"{c} currents (cause: {causes.get(c, '-')})")
        ax.legend(fontsize=8)
        ax = axes[i, 2]
        ax.plot(t, np.mean([gvm[g] for g in cortical(c)], axis=0), label="v(t)")
        ax.plot(t, np.mean([guu[g] for g in cortical(c)], axis=0), label="u(t)")
        ax.set_title(f"{c} state")
        ax.legend(fontsize=8)
    fig.suptitle("SCI-EI-RECRUITMENT-1 current-budget atlas (1 ms bins, full run)")
    fig.tight_layout()
    fig.savefig(ATLAS, dpi=80)
    plt.close(fig)
    out["atlas"] = str(ATLAS.relative_to(ROOT)).replace("\\", "/")

    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    assert OUT.is_file()

"""OPEN-38 construction invariance across composition arms (SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md §5a).

Usage: python scripts/open38_construction_check.py OUT_JSON REF_LAYOUT ARM_LAYOUT [ARM_LAYOUT ...] --seeds 1 2 3 4 5

Builds each layout per seed (construct only, no simulation) and compares each arm with ref, same seed:
  sizes        E per unit equal; PV + SOM per unit equal (total inhibition); realized SOM/PV within 1 cell of rho
  e_to_e       every E->E rule (local e_e_U, inter-unit x_A_B_e_e): identical pre, post, weight, delay, receptor
  e_params     every AdEx parameter of every E cell identical
  inh_rules    rules into or out of PV/SOM: weight, delay, receptor sets equal to ref's; edge count inside the
               two-sided binomial interval of p * N_pre * N_post (N_pre * (N_pre - 1) for pv_pv: no self-edges
               assumed; the larger interval of the two is used, as self-exclusion is not read from the engine here).
               Level (owner ruling 2026-11-17, option B; was 99.9 % per test): 5 % family-wise over every PV/SOM
               test of this invocation (Bonferroni): per-test level 1 - 0.05 / N_tests, N_tests = PV/SOM rules per
               pair x arms x seeds. Recorded in the report as "family".
  drive        Poisson drive spec identical (a function of seed only); driven cells per unit (E + PV) reported
  geometry     unit depth, readout columns (unit, receptor), receptor tau and reversal identical
Writes OUT_JSON; exit 0 only if every gate passes for every arm and seed. A failure is kept, not retried.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402

from jomission.laminar import caminos_e0 as c0  # noqa: E402
from jomission.laminar import caminos_stack as cs  # noqa: E402

POP_OF = {name: (pre, post) for name, pre, post, _, _ in c0.RULES}
FAMILY_ALPHA = 0.05  # owner ruling 2026-11-17 (option B)


def family_level(n_tests: int, alpha: float = FAMILY_ALPHA) -> float:
    """Per-test two-sided interval level for a family-wise error alpha over n_tests tests (Bonferroni)."""
    if n_tests < 1:
        raise ValueError("no PV/SOM tests: the family is empty")
    return 1.0 - alpha / n_tests


def rule_interval(pairs: list[int], p: float, level: float) -> tuple[float, float]:
    """Union of the central binomial intervals over the candidate pair counts."""
    return (min(binom.interval(level, n, p)[0] for n in pairs), max(binom.interval(level, n, p)[1] for n in pairs))


def is_inh(rule: str) -> bool:
    pu, pp, qu, qp = pops_of(rule)
    return not (pp == "E" and qp == "E")


def build(lay: str, seed: int):
    sizes, rho, p_inter, depth = cs.layout(lay)
    mods = cs.variant(lay)[1]
    p_local = cs.local_p(lay, sizes)
    model, rows = cs.construct(seed, sizes, p_inter, p_local, tau_sst_e_ms=cs.sst_taus(lay, sizes),
                               k_scale=mods.get("k", 1.0), deep_size=mods.get("dsz", 1.0),
                               gaba_scale=mods.get("gi", 1.0), triple_drive=cs.drive_entries(lay, sizes),
                               w_scales=mods.get("w"))
    st = model.static["adex"]
    names = [r[0] for r in st["rules"]]
    rid = np.asarray(st["rule_id"])
    edges = {}
    for i, nm in enumerate(names):
        m = rid == i
        edges[nm] = {k: np.asarray(st[k])[m] for k in ("pre", "post", "weight_nS", "delay_ms", "receptor")}
    em = model.params["emitter"]
    labels = np.array(em.labels)
    e_idx = np.nonzero(labels == "E")[0]
    e_par = {f: np.asarray(getattr(em, f))[e_idx] for f in ("C_pF", "Delta_mV", "EL_mV", "Tref_ms", "VT_mV",
                                                             "Vpeak_mV", "Vreset_mV", "a_nS", "b_pA", "gL_nS", "tauw_ms")}
    _, gcols, gtau, grev = cs.lfp_weights(model, rows, list(sizes))
    p_rule = {}
    names_u = list(sizes)
    n_nb = {m: (len(names_u) > 1) * (2 - (m in (names_u[0], names_u[-1]))) for m in names_u}
    for nm in names:
        mm = re.fullmatch(r"x_(\w)_(\w)_e_(e|pv|som)", nm)
        if mm:
            a, b = mm.group(1), mm.group(2)
            p_rule[nm] = p_inter if p_inter is not None else 2 * cs.K_INTER * mods.get("k", 1.0) / n_nb[b] / sizes[a]["E"]
        else:
            u = nm.rsplit("_", 1)[1]
            p_rule[nm] = (p_local or {}).get(u, cs.K_E / sizes[u]["E"])
    return {"sizes": sizes, "rho": rho, "depth": depth, "edges": edges, "e_par": e_par, "p": p_rule,
            "gcols": [list(c) for c in gcols], "gtau": np.asarray(gtau), "grev": np.asarray(grev),
            "drive": c0.drive_spec(0.07, seed)}


def pops_of(rule: str) -> tuple[str, str, str, str]:
    """(pre_unit, pre_pop, post_unit, post_pop) of a rule name."""
    mm = re.fullmatch(r"x_(\w)_(\w)_e_(e|pv|som)", rule)
    if mm:
        return mm.group(1), "E", mm.group(2), {"e": "E", "pv": "PV", "som": "SOM"}[mm.group(3)]
    base, u = rule.rsplit("_", 1)
    pre, post = POP_OF[base]
    return u, pre, u, post


def compare(ref: dict, arm: dict, level: float) -> dict:
    g = {}
    s0, s1 = ref["sizes"], arm["sizes"]
    ratio_err = {u: abs(s1[u]["SOM"] - (s1[u]["PV"] + s1[u]["SOM"]) * arm["rho"][u] / (1 + arm["rho"][u])) for u in s1}
    g["sizes"] = {"ok": all(s0[u]["E"] == s1[u]["E"] and s0[u]["PV"] + s0[u]["SOM"] == s1[u]["PV"] + s1[u]["SOM"]
                            and s1[u]["PV"] > 0 and s1[u]["SOM"] > 0 and ratio_err[u] <= 1.0 for u in s0),
                  "arm": s1, "ratio_err_cells": ratio_err}
    ee = [r for r in ref["edges"] if pops_of(r)[1] == "E" and pops_of(r)[3] == "E"]
    diff = [r for r in ee if r not in arm["edges"] or any(
        not np.array_equal(ref["edges"][r][k], arm["edges"][r][k]) for k in ref["edges"][r])]
    g["e_to_e"] = {"ok": not diff and bool(ee), "rules": len(ee), "differing": diff,
                   "edges": int(sum(ref["edges"][r]["pre"].size for r in ee))}
    # Vpeak_mV is NaN for every cell in this engine (unset); NaN in the same place counts as equal
    pdiff = [f for f in ref["e_par"] if not np.array_equal(ref["e_par"][f], arm["e_par"][f], equal_nan=True)]
    g["e_params"] = {"ok": not pdiff and ref["e_par"]["C_pF"].size > 0, "differing": pdiff}
    inh = {}
    for r in arm["edges"]:
        pu, pp, qu, qp = pops_of(r)
        if pp == "E" and qp == "E":
            continue
        n_pre, n_post = s1[pu][pp], s1[qu][qp]
        pairs = [n_pre * n_post] + ([n_pre * (n_pre - 1)] if (pu, pp) == (qu, qp) else [])
        lo, hi = rule_interval(pairs, arm["p"][r], level)
        cnt = int(arm["edges"][r]["pre"].size)
        same_w = all(np.array_equal(np.unique(ref["edges"][r][k]), np.unique(arm["edges"][r][k]))
                     for k in ("weight_nS", "delay_ms", "receptor")) if r in ref["edges"] else False
        inh[r] = {"count": cnt, "interval": [float(lo), float(hi)], "params_equal": same_w,
                  "ok": lo <= cnt <= hi and same_w}
    g["inh_rules"] = {"ok": bool(inh) and all(v["ok"] for v in inh.values()) and set(arm["edges"]) == set(ref["edges"]),
                      "rules": inh}
    g["drive"] = {"ok": ref["drive"] == arm["drive"], "spec": arm["drive"],
                  "driven_cells_per_unit": {u: s1[u]["E"] + s1[u]["PV"] for u in s1}}
    g["geometry"] = {"ok": ref["depth"] == arm["depth"] and ref["gcols"] == arm["gcols"]
                     and np.array_equal(ref["gtau"], arm["gtau"]) and np.array_equal(ref["grev"], arm["grev"])}
    return g


def main(out: str, ref_lay: str, arms: list[str], seeds: list[int]) -> int:
    report, ok = {"ref": ref_lay, "arms": arms, "seeds": seeds, "results": {}}, True
    level = None
    for s in seeds:
        ref = build(ref_lay, s)
        if level is None:  # PV/SOM rules per pair are fixed by the layout family (set equality is gated below)
            n_tests = sum(is_inh(r) for r in ref["edges"]) * len(arms) * len(seeds)
            level = family_level(n_tests)
            report["family"] = {"alpha": FAMILY_ALPHA, "n_tests": n_tests, "per_test_level": level,
                                "ruling": "owner 2026-11-17, option B (Bonferroni); was 0.999 per test"}
        for lay in arms:
            g = compare(ref, build(lay, s), level)
            report["results"][f"{lay}|s{s}"] = g
            failed = [k for k, v in g.items() if not v["ok"]]
            ok &= not failed
            print(f"seed {s} {lay}: {'PASS' if not failed else 'FAIL ' + ','.join(failed)}", flush=True)
    report["pass"] = ok
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("ref")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    sys.exit(main(a.out, a.ref, a.arms, a.seeds))

"""OPEN-38 spec §6 analysis of the accepted runs (stage 1: ref, flat, rev; stage 2: v1, mt, mst, teo, chc, shape;
seeds 1-5). Implements the frozen decision rule only; no selection, threshold or outcome is added.

    python -B scripts/open38_section6.py OUT_JSON [--dir results/laminar_e2]
        [--mid results/laminar_e2/o38s_s6_mid_gamma.json] [--t8 paper/jomission_lansa/data/t8_summary.json]

Spec §6 / §5(c) / §4 items, each computed here:
- pairing: paired by seed when every construction receipt shows identical E->E edge lists (§5a, §6), else Welch;
- per arm and outcome: d_s = X(ref, s) - X(arm, s), mean, SD (ddof 1), 95 % t-interval (t_0.975,4);
- primary family: 8 arms on r_ab, two-sided paired t-test, Holm at 0.05; difference verdict "ref higher" / "arm
  higher" only if Holm-significant and every d_s has that sign; equivalence by TOST against +-0.1 (90 % interval of d
  inside (-0.1, 0.1)), unadjusted across the 8; INCONCLUSIVE when neither;
- MDD: 80 % power, n 5, two-sided 0.05/8, sigma = SD of r_ab over ref seeds 1-5 (the spec's wording); the T8
  rflat/rrev paired SD (n 3) beside it with its chi-square 95 % interval;
- pairing benefit: paired SD of d beside the unpaired SD sqrt(var_ref + var_arm), and the Welch interval;
- secondary outcomes (r_gm, r_proto, mid-depth relative gamma, unit E rates): intervals only, no tests;
  crossing_2se and scorer pass: counts (binary, no d);
- noise floor (§5c): SD of r_ab over ref seeds 1-5; T8 flat/rev effect quoted as the scale of a known effect;
- §4 illustrative: r_ab of the 4 area arms beside Delta = log2(R_D / R_S); no estimand, no test.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stx
import sys
from pathlib import Path

from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open38_stage1_check as s1  # noqa: E402

ARMS = ("v1", "mt", "mst", "teo", "chc", "shape", "flat", "rev")  # §6 primary family, spec order
AREA_ARMS = ("v1", "mt", "mst", "teo")  # §4 illustrative
SEEDS = (1, 2, 3, 4, 5)
ALPHA, MARGIN, N_FAMILY, POWER = 0.05, 0.1, 8, 0.80
CONTINUOUS = ("r_ab", "r_gm", "r_proto", "mid_gamma", "rate_S", "rate_M", "rate_D")
MID_KEY = "mid [-0.2,0]"


def tag(arm: str, seed: int) -> str:
    return f"o38s_pc_{arm}_n4000_s{seed}_T8" if arm in ("ref", "flat", "rev") else f"o38s_{arm}_n4000_s{seed}_T8"


def t_interval(x: list[float], level: float) -> list[float]:
    n, m, sd = len(x), stx.mean(x), stx.stdev(x)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n)
    return [m - h, m + h]


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    m, order, adj, run = len(p), sorted(range(len(p)), key=lambda i: p[i]), [0.0] * len(p), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def verdicts(d: list[float], p_holm: float) -> dict:
    lo90, hi90 = t_interval(d, 0.90)
    diff = None
    if p_holm <= ALPHA and all(x > 0 for x in d):
        diff = "ref higher"
    elif p_holm <= ALPHA and all(x < 0 for x in d):
        diff = "arm higher"
    equiv = bool(-MARGIN < lo90 and hi90 < MARGIN)
    return {"difference": diff, "equivalent": equiv, "ci90": [float(lo90), float(hi90)],
            "verdict": "INCONCLUSIVE" if diff is None and not equiv else
            " and ".join(v for v in (diff, "equivalent within ±0.1" if equiv else None) if v)}


def mdd(sigma: float, n: int = 5, alpha: float = ALPHA / N_FAMILY, power: float = POWER) -> float:
    """Smallest |d| a two-sided paired t-test (df n-1) detects with the given power, sigma the SD of d."""
    df, tc = n - 1, stats.t.ppf(1 - alpha / 2, n - 1)

    def pw(delta: float) -> float:
        nc = delta * math.sqrt(n) / sigma
        # lower tail as P(T > tc | -nc): equal by symmetry; scipy 1.18 nct.cdf returns NaN in that far tail
        return stats.nct.sf(tc, df, nc) + stats.nct.sf(tc, df, -nc) - power
    return float(optimize.brentq(pw, 1e-9, 40 * sigma / math.sqrt(n)))  # noncentrality 40: power ~1, nct stays finite


def sd_ci(sd: float, n: int, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [sd * math.sqrt((n - 1) / stats.chi2.ppf(1 - a, n - 1)), sd * math.sqrt((n - 1) / stats.chi2.ppf(a, n - 1))]


def load(d: Path, mid: dict) -> dict:
    runs = {}
    for arm in ("ref", *ARMS):
        for s in SEEDS:
            t = tag(arm, s)
            r = s1.run_checks(d, t)
            rec = json.loads((d / f"{t}.json").read_text(encoding="utf-8"))
            rt = r["e_rates_hz"]
            runs[(arm, s)] = {"tag": t, "ok": r["ok"], "stop": r["stop"], "r_ab": r["r_ab"], "r_gm": r["r_gm"],
                              "r_proto": r["r_proto"], "mid_gamma": mid["runs"][t][MID_KEY],
                              "rate_S": rt["S"], "rate_M": rt["M"], "rate_D": rt["D"],
                              "crossing_2se": r["crossing_2se"], "pass": r["pass"], "rho": rec["rho"]}
    return runs


def paired_ok(d: Path) -> dict:
    out = {}
    for f in ("o38s_pc_construction.json", "o38s_s2_construction.json"):
        c = json.loads((d / f).read_text(encoding="utf-8"))
        out[f] = all(r["e_to_e"]["ok"] and not r["e_to_e"]["differing"] for r in c["results"].values())
    return out


def main(out: str, d: Path, mid_p: Path, t8_p: Path) -> int:
    mid, t8 = json.loads(mid_p.read_text(encoding="utf-8")), json.loads(t8_p.read_text(encoding="utf-8"))
    runs = load(d, mid)
    bad = sorted(v["tag"] for v in runs.values() if not v["ok"] or v["stop"])
    pairing = paired_ok(d)
    paired = all(pairing.values())
    ref_ab = [runs[("ref", s)]["r_ab"] for s in SEEDS]
    sd_ref = stx.stdev(ref_ab)
    res = {"schema": "o38s_s6/1", "spec": "SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md §4-§6",
           "runs_not_evaluable": bad, "pairing": {"e_to_e_identical": pairing, "paired": paired},
           "noise_floor": {"ref_r_ab": ref_ab, "sd": sd_ref, "mean": stx.mean(ref_ab)},
           "mdd": {"sigma": sd_ref, "sigma_source": "SD of r_ab over ref seeds 1-5 (spec §6 wording)",
                   "alpha_two_sided": ALPHA / N_FAMILY, "power": POWER, "n": 5, "mdd": mdd(sd_ref)},
           "t8_prior": {}, "arms": {}}
    for arm in ("rflat", "rrev"):
        dd = [a["r_ab"] - b["r_ab"] for a, b in zip(t8["runs"]["ref"], t8["runs"][arm])]
        sd = stx.stdev(dd)
        res["t8_prior"][arm] = {"d_s": dd, "mean": stx.mean(dd), "sd": sd, "sd_ci95": sd_ci(sd, len(dd)),
                                "mdd_at_this_sd": mdd(sd)}
    pvals = []
    for arm in ARMS:
        a = {"outcomes": {}, "binary": {}}
        for o in CONTINUOUS:
            ds = [runs[("ref", s)][o] - runs[(arm, s)][o] for s in SEEDS]
            xr, xa = [runs[("ref", s)][o] for s in SEEDS], [runs[(arm, s)][o] for s in SEEDS]
            e = {"d_s": ds, "mean": stx.mean(ds), "sd": stx.stdev(ds), "ci95": t_interval(ds, 0.95)}
            if o == "r_ab":
                w = stats.ttest_ind(xr, xa, equal_var=False)
                ci = w.confidence_interval(0.95)
                e.update(unpaired_sd=math.sqrt(stx.variance(xr) + stx.variance(xa)),
                         welch_ci95=[float(ci.low), float(ci.high)],
                         p_paired=float(stats.ttest_rel(xr, xa).pvalue))
                pvals.append(e["p_paired"])
            a["outcomes"][o] = e
        for b in ("crossing_2se", "pass"):
            a["binary"][b] = {"ref_true": sum(runs[("ref", s)][b] is True for s in SEEDS),
                              "arm_true": sum(runs[(arm, s)][b] is True for s in SEEDS)}
        res["arms"][arm] = a
    for arm, p in zip(ARMS, holm(pvals)):
        e = res["arms"][arm]["outcomes"]["r_ab"]
        e["p_holm"] = p
        e.update(verdicts(e["d_s"], p))
    res["illustrative_area_arms"] = {
        arm: {"r_ab_mean": stx.mean(runs[(arm, s)]["r_ab"] for s in SEEDS),
              "delta_log2_RD_RS": math.log2(runs[(arm, 1)]["rho"]["D"] / runs[(arm, 1)]["rho"]["S"])}
        for arm in AREA_ARMS}
    Path(out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"not evaluable {bad}; paired {paired}; ref r_ab SD {sd_ref:.4f}; MDD {res['mdd']['mdd']:.4f}")
    for arm in ARMS:
        e = res["arms"][arm]["outcomes"]["r_ab"]
        print(f"{arm:6s} d {e['mean']:+.4f} CI95 [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] p {e['p_paired']:.2e} "
              f"holm {e['p_holm']:.2e} -> {e['verdict']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dir", default="results/laminar_e2")
    ap.add_argument("--mid", default="results/laminar_e2/o38s_s6_mid_gamma.json")
    ap.add_argument("--t8", default="paper/jomission_lansa/data/t8_summary.json")
    a = ap.parse_args()
    sys.exit(main(a.out, Path(a.dir), Path(a.mid), Path(a.t8)))

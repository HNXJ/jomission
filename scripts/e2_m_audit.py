"""Read-only audit of stored motif receipts: per-arm S/D band powers and the signed contrast M.

Reads existing `*_T8_motif.json` receipts (no simulation). For each file:
- S = depths < -0.175 and D = depths > +0.175 (the LANSA module rule, `docs/laminar_motif/MINIMAL_MODEL_SPEC.md`).
- Proposed M (2026-11-25 thesis): log(P_S,gamma / P_S,ab) - log(P_D,gamma / P_D,ab), P = mean over the module's bins.
- Pipeline contrast_M (`scripts/e2_motif_figure.py`): slope over depth of (ab - gamma). Checked for equality per file.

Usage:
  python3 scripts/e2_m_audit.py --dir results/laminar_e2 --glob "o38s_*_T8_motif.json"
  python3 scripts/e2_m_audit.py --dir DIR_WITH_T8_RECEIPTS --glob "t8_*_T8_motif.json"

Output: one TSV row per arm (mean over seeds, with min and max), plus a check count on the slope equality.
Arm = file name without the `_s<seed>_T8_motif.json` suffix.
"""
import argparse
import glob
import json
import math
import os
import re


def mean(xs):
    return sum(xs) / len(xs)


def slope(xs, ys):
    mx = mean(xs)
    my = mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def summarise(path):
    with open(path) as fh:
        obj = json.load(fh)
    img = obj["per_cell_mean"]["scores"]["image"]
    ys = obj["probe"]["channels_rel_L4"]
    ab = img["alpha_beta_profile"]
    gm = img["gamma_profile"]
    if not (len(ys) == len(ab) == len(gm)):
        raise ValueError(f"profile length mismatch in {path}")
    s = [i for i, y in enumerate(ys) if y < -0.175 - 1e-9]
    d = [i for i, y in enumerate(ys) if y > 0.175 + 1e-9]
    if not s or not d:
        raise ValueError(f"empty module in {path}")
    p_sg, p_sa = mean([gm[i] for i in s]), mean([ab[i] for i in s])
    p_dg, p_da = mean([gm[i] for i in d]), mean([ab[i] for i in d])
    m_user = math.log(p_sg / p_sa) - math.log(p_dg / p_da)
    m_slope = slope(ys, [a - g for a, g in zip(ab, gm)])
    return {
        "S_gm": p_sg,
        "S_ab": p_sa,
        "D_gm": p_dg,
        "D_ab": p_da,
        "M_user": m_user,
        "M_code": img["contrast_M"],
        "slope_matches_code": abs(m_slope - img["contrast_M"]) < 1e-9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--glob", required=True)
    a = ap.parse_args()
    arms = {}
    checks = 0
    for path in sorted(glob.glob(os.path.join(a.dir, a.glob))):
        name = os.path.basename(path)
        arm = re.sub(r"_s[0-9]+_T8_motif\.json$", "", name)
        row = summarise(path)
        checks += row["slope_matches_code"]
        arms.setdefault(arm, []).append(row)
    cols = ["S_gm", "S_ab", "D_gm", "D_ab", "M_user", "M_code"]
    print("arm\tn\t" + "\t".join(f"{c}_mean" for c in cols) + "\tM_user_min\tM_user_max\tM_code_min\tM_code_max")
    for arm in sorted(arms):
        rs = arms[arm]
        means = [mean([r[c] for r in rs]) for c in cols]
        mu = [r["M_user"] for r in rs]
        mc = [r["M_code"] for r in rs]
        vals = "\t".join(f"{v:+.4f}" if c.startswith("M") else f"{v:.4f}" for c, v in zip(cols, means))
        print(f"{arm}\t{len(rs)}\t{vals}\t{min(mu):+.4f}\t{max(mu):+.4f}\t{min(mc):+.4f}\t{max(mc):+.4f}")
    total = sum(len(v) for v in arms.values())
    print(f"# slope check: {checks} of {total} files reproduce contrast_M")


if __name__ == "__main__":
    main()

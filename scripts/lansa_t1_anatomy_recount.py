"""Read-only receipt: T1 qualitative motif flags under frozen anatomical depth rules.

Usage: python scripts/lansa_t1_anatomy_recount.py [--dir DIR] [--y LANSA_Y_JSON]
DIR defaults to the cowork laminar_e2 results, where the T1 motif receipts live (not in this repo).
Writes a TSV to stdout and a summary after '#'. Reads only: no simulation, no verdict, no criterion.

Depth convention: LANSA bins are cortical-depth fractions about mid-L4, negative superficial
(docs/laminar_motif/data/lansa_density_X.json, units).
Layer depths (um): L4 357-505, mid-L4 431, L2/3 mid 263, L5 mid 585, L6 mid 747; thickness 779 um
(scripts/allen_v1_figure.py THICK_UM). Source: the layer table of docs/laminar_motif/ALLEN_V1_REPLICA_SPEC.md
section 1. That spec is RETIRED (owner, 2026-11-17), and its source is a mouse V1 model (AllenInstitute
oscillations-biorealistic-mouse-v1-model, per docs/laminar_motif/OPEN38_ALT_ALLEN_INVENTORY.md). The depths are
not re-verified against that source in this repo, and their validity for the macaque LANSA areas is UNKNOWN.

Rules were written on 2026-11-25 and not committed before the run. Gamma superficial: peak z < sup. Alpha-beta
deep: peak z > deep. Rules A, B and C give identical counts on this data: no alpha-beta peak bin lies between
+0.095 and +0.25, and the peak bins lie on a 0.05 grid. They are threshold checks, not a sensitivity analysis.
  A  primary, anatomical: sup = z(L4 top), deep = z(L4 bottom).
  B  sensitivity: the LANSA module boundary, sup = -0.175, deep = +0.175
     (docs/laminar_motif/SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md, Laminar modules row).
  C  sensitivity, stricter: sup = z(L4 top), deep = z(L5 mid).
  O  reproduction of the 2026-11-25 chat count: sup = 0, deep = 0. Must give 21 of 57, or the script stops.
Every rule also needs the endpoint sign crossing (ab - gm < 0 at the first bin and > 0 at the last).
The scorer's crossing_2se is reported beside it and is not a flag here.
"""
import argparse
import glob
import json
import os
import sys

MID_UM, THICK_UM = 431.0, 779.0
L4_TOP_UM, L4_BOT_UM, L5_MID_UM = 357.0, 505.0, 585.0
N_RUNS = 57


def z_of(um):
    return (um - MID_UM) / THICK_UM


Z_L4_TOP = z_of(L4_TOP_UM)
Z_L4_BOT = z_of(L4_BOT_UM)
Z_L5_MID = z_of(L5_MID_UM)
RULES = {
    "A": (Z_L4_TOP, Z_L4_BOT),
    "B": (-0.175, 0.175),
    "C": (Z_L4_TOP, Z_L5_MID),
    "O": (0.0, 0.0),
}


def peak_index(xs):
    """First maximum: a tie goes to the more superficial bin."""
    best = 0
    for i, x in enumerate(xs):
        if x > xs[best]:
            best = i
    return best


def flags(z, gm, ab, sup, deep):
    ig, ia = peak_index(gm), peak_index(ab)
    d = [a - g for a, g in zip(ab, gm)]
    qg = z[ig] < sup
    qa = z[ia] > deep
    qx = d[0] < 0 and d[-1] > 0
    return z[ig], z[ia], qg, qa, qx


def load_run(path):
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    img = obj["sum_over_cells"]["scores"]["image"]
    z = obj["probe"]["channels_rel_L4"]
    return z, img["gamma_profile"], img["alpha_beta_profile"], bool(img["crossing_2se"])


def observed(y_path):
    """Scorer PROTO band means: area mean of Y, then 10-30 Hz and 50-150 Hz band means per bin."""
    with open(y_path, encoding="utf-8") as fh:
        obj = json.load(fh)
    z = obj["axes"]["bin_center_rel_L4"]
    freq = obj["axes"]["freq_hz"]
    Y = obj["Y"]
    ab_k = [k for k, f in enumerate(freq) if 10 <= f <= 30]
    gm_k = [k for k, f in enumerate(freq) if 50 <= f <= 150]
    ab, gm = [], []
    for i in range(len(z)):
        P = [sum(Y[i][k][a] for a in range(len(Y[i][k]))) / len(Y[i][k]) for k in range(len(freq))]
        ab.append(sum(P[k] for k in ab_k) / len(ab_k))
        gm.append(sum(P[k] for k in gm_k) / len(gm_k))
    return z, gm, ab


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    default_y = os.path.join(os.path.dirname(here), "docs", "laminar_motif", "data", "lansa_lfp_Y.json")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/d/cowork/jomission/results/laminar_e2")
    ap.add_argument("--y", default=default_y)
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(a.dir, "t1_*_motif.json")))
    if len(paths) != N_RUNS:
        raise SystemExit(f"fixture check failed: expected {N_RUNS} T1 motif receipts, found {len(paths)}")

    zo, gmo, abo = observed(a.y)
    obs = {r: flags(zo, gmo, abo, *RULES[r]) for r in RULES}
    if not all(obs["A"][2:4]):
        raise SystemExit("known-positive check failed: the observed map does not satisfy rule A")

    header = ["run", "peak_gm_z", "peak_ab_z", "crossing_2se"] + [f"{r}_{k}" for r in RULES for k in ("qg", "qa", "all3")]
    print("\t".join(header))
    counts = {r: {"qg": 0, "qa": 0, "all3": 0, "all3_cross": 0} for r in RULES}
    ab_peaks = {}
    for p in paths:
        z, gm, ab, cross = load_run(p)
        if len(z) != 21:
            raise SystemExit(f"{p}: expected 21 bins, found {len(z)}")
        name = os.path.basename(p).replace("_motif.json", "")
        row = [name]
        pg, pa, _, _, _ = flags(z, gm, ab, 0.0, 0.0)
        row += [f"{pg:+.3f}", f"{pa:+.3f}", str(cross).lower()]
        ab_peaks[f"{pa:+.2f}"] = ab_peaks.get(f"{pa:+.2f}", 0) + 1
        for r in RULES:
            _, _, qg, qa, qx = flags(z, gm, ab, *RULES[r])
            all3 = qg and qa and qx
            counts[r]["qg"] += qg
            counts[r]["qa"] += qa
            counts[r]["all3"] += all3
            counts[r]["all3_cross"] += all3 and cross
            row += [str(qg).lower(), str(qa).lower(), str(all3).lower()]
        print("\t".join(row))

    print(f"# runs {len(paths)}; observed (Y): peak_gm_z={obs['A'][0]:+.3f} peak_ab_z={obs['A'][1]:+.3f}")
    print(f"# thresholds: z(L4 top)={Z_L4_TOP:+.4f} z(L4 bottom)={Z_L4_BOT:+.4f} z(L5 mid)={Z_L5_MID:+.4f}")
    for r in RULES:
        c = counts[r]
        print(f"# rule {r}: gamma superficial {c['qg']}; alpha-beta deep {c['qa']}; all three {c['all3']}; "
              f"all three and crossing_2se {c['all3_cross']}")
    if counts["O"]["all3"] != 21:
        raise SystemExit(f"reproduction check failed: rule O gives {counts['O']['all3']}, expected 21")
    print("# reproduction check: rule O gives 21 (matches the 2026-11-25 chat count)")
    print("# alpha-beta peak bins (count): " + ", ".join(f"{k}:{v}" for k, v in sorted(ab_peaks.items())))


if __name__ == "__main__":
    main(sys.argv[1:])

"""rhou layout tokens for the spectrum-sensitivity arms from the composition table. Usage: COMPOSITION_CSV"""
import csv
import math
import sys

rows = {(r["area"], r["module"]): r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8"))}
AREAS = ("V1", "MT", "MST", "TEO")


def tok(R):
    return "_".join(f"rhou{m}{R[m]:.3f}".replace(".", "p") for m in "SMD")


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


for a in AREAS:
    R = {m: int(rows[(a, m)]["SST"]) / int(rows[(a, m)]["PVALB"]) for m in "SMD"}
    print(a, tok(R), "gm", f"{gm(R.values()):.3f}")
chc = {m: gm([int(rows[(a, m)]["SST"]) / (int(rows[(a, m)]["PVALB"]) + int(rows[(a, m)]["PV_CHC"])) for a in AREAS])
       for m in "SMD"}
pv = {m: gm([int(rows[(a, m)]["SST"]) / int(rows[(a, m)]["PVALB"]) for a in AREAS]) for m in "SMD"}
print("pooled-gm PVALB", tok(pv), "gm", f"{gm(pv.values()):.3f}")
print("pooled-gm +CHC", tok(chc), "gm", f"{gm(chc.values()):.3f}")
g0 = gm([0.25, 1.0, 2.0])
print("prescribed gm", f"{g0:.3f}")
shape = {m: pv[m] * g0 / gm(pv.values()) for m in "SMD"}
print("shape (pooled-gm PVALB rescaled to prescribed gm)", tok(shape), "gm", f"{gm(shape.values()):.3f}")

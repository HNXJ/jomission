"""Laminar SST/PVALB (and VIP) composition per area from Chen et al. 2023 portal Cell Summation exports (OPEN-38).

Usage: python scripts/open38_laminar_composition.py CLUSTER_DIR OUT_CSV

CLUSTER_DIR holds one portal export per cluster, `cell_summation_monkey_f001_<SUBCLASS>.<n>.csv` (columns
`region`, `Cell Count`; one sample, all 119 sections summed). Subclasses are summed over their clusters
(prereg A2): SST, PVALB, PV_CHC, VIP, VIP_RELN. Modules (prereg §3): S = l2 + l3, M = l4, D = l5 + l6; l1 is
reported separately and enters no ratio. A module with no layer row (agranular l4) is missing, never zero.

Output: one row per (area, module) with counts, R = SST/PVALB, and a 95% interval for R that covers counting
(Poisson) noise only, from the Clopper-Pearson interval of SST/(SST+PVALB). One animal (monkey #1): no
between-animal uncertainty exists in these data, and the interval must not be read as biological uncertainty.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta

SUBCLASSES = ("SST", "PVALB", "PV_CHC", "VIP", "VIP_RELN")
MODULES = {"L1": ("l1",), "S": ("l2", "l3"), "M": ("l4",), "D": ("l5", "l6")}
FILE_RE = re.compile(r"cell_summation_monkey_f001_(SST|PVALB|PV_CHC|VIP|VIP_RELN)\.(\d+)\.csv$")


def load(cluster_dir: Path) -> tuple[dict, dict]:
    """counts[subclass][region] summed over clusters; n_clusters[subclass]."""
    counts: dict = {s: defaultdict(int) for s in SUBCLASSES}
    n_clusters = defaultdict(int)
    for f in sorted(cluster_dir.iterdir()):
        m = FILE_RE.search(f.name)
        if not m:
            continue
        sub = m.group(1)
        n_clusters[sub] += 1
        with open(f, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh))
        if rows[0] != ["region", "Cell Count"]:
            raise ValueError(f"{f.name}: unexpected header {rows[0]}")
        seen = set()
        for region, value in rows[1:]:
            if region in seen:
                raise ValueError(f"{f.name}: duplicate region {region!r}")
            seen.add(region)
            counts[sub][region] += int(value)
    return counts, dict(n_clusters)


def ratio_ci(a: int, b: int, level: float = 0.95) -> tuple[float, float]:
    """Interval for a/b from the exact (Clopper-Pearson) interval of a/(a+b); counting noise only."""
    n, q = a + b, (1 - level) / 2
    lo = beta.ppf(q, a, n - a + 1) if a > 0 else 0.0
    hi = beta.ppf(1 - q, a + 1, n - a) if b > 0 else float("inf")
    return lo / (1 - lo), (hi / (1 - hi) if hi < 1 else float("inf"))


def module_rows(counts: dict, area: str) -> list[dict]:
    """One row per module of `area`; layers absent from every subclass make the module missing."""
    out = []
    for mod, layers in MODULES.items():
        present = [l for l in layers if any(f"{area}-{l}" in counts[s] for s in SUBCLASSES)]
        row = {"area": area, "module": mod, "layers": "+".join(present) if present else "", "missing": not present}
        for s in SUBCLASSES:
            row[s] = sum(counts[s].get(f"{area}-{l}", 0) for l in present) if present else ""
        if present and mod != "L1" and row["PVALB"] > 0:
            row["R_sst_pvalb"] = row["SST"] / row["PVALB"]
            row["R_lo"], row["R_hi"] = ratio_ci(row["SST"], row["PVALB"])
        else:
            row["R_sst_pvalb"] = row["R_lo"] = row["R_hi"] = ""
        out.append(row)
    return out


def main(cluster_dir: str, out_csv: str) -> None:
    counts, n_clusters = load(Path(cluster_dir))
    areas = sorted({re.sub(r"-l\d$", "", r) for r in counts["SST"] if re.search(r"-l\d$", r)})
    rows = [r for a in areas for r in module_rows(counts, a)]
    cols = ["area", "module", "layers", "missing", *SUBCLASSES, "R_sst_pvalb", "R_lo", "R_hi"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"clusters per subclass {n_clusters}; areas {len(areas)}; rows {len(rows)} -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

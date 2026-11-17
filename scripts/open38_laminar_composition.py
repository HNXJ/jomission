"""Laminar SST/PVALB (and VIP) composition per area from Chen et al. 2023 portal Cell Summation exports (OPEN-38).

Usage: python scripts/open38_laminar_composition.py CLUSTER_DIR ALL_CELLS_CSV OUT_CSV

ALL_CELLS_CSV is the same export with no cell-type filter (all cells); it gives the denominator `n_all` and the
fraction of all cells in each subclass. Every export must list the same regions; the cluster count per subclass
must equal EXPECTED (Chen 2023 Table S2), else the build stops.

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
EXPECTED = {"SST": 32, "PVALB": 32, "PV_CHC": 5, "VIP": 9, "VIP_RELN": 9}
MODULES = {"L1": ("l1",), "S": ("l2", "l3"), "M": ("l4",), "D": ("l5", "l6")}
FILE_RE = re.compile(r"cell_summation_monkey_f001_(SST|PVALB|PV_CHC|VIP|VIP_RELN)\.(\d+)\.csv$")


def read_export(f: Path) -> dict:
    """{region: count} from one Cell Summation export; refuses a wrong header or a duplicate region."""
    with open(f, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if rows[0] != ["region", "Cell Count"]:
        raise ValueError(f"{f.name}: unexpected header {rows[0]}")
    out: dict = {}
    for region, value in rows[1:]:
        if region in out:
            raise ValueError(f"{f.name}: duplicate region {region!r}")
        out[region] = int(value)
    return out


def load(cluster_dir: Path) -> tuple[dict, dict]:
    """counts[subclass][region] summed over clusters; n_clusters[subclass]. All exports must share one region set."""
    counts: dict = {s: defaultdict(int) for s in SUBCLASSES}
    n_clusters = defaultdict(int)
    regions = None
    for f in sorted(cluster_dir.iterdir()):
        m = FILE_RE.search(f.name)
        if not m:
            continue
        sub = m.group(1)
        n_clusters[sub] += 1
        data = read_export(f)
        if regions is None:
            regions = set(data)
        elif set(data) != regions:
            raise ValueError(f"{f.name}: region set differs from the other exports")
        for region, value in data.items():
            counts[sub][region] += value
    return counts, dict(n_clusters)


def ratio_ci(a: int, b: int, level: float = 0.95) -> tuple[float, float]:
    """Interval for a/b from the exact (Clopper-Pearson) interval of a/(a+b); counting noise only."""
    n, q = a + b, (1 - level) / 2
    lo = beta.ppf(q, a, n - a + 1) if a > 0 else 0.0
    hi = beta.ppf(1 - q, a + 1, n - a) if b > 0 else float("inf")
    return lo / (1 - lo), (hi / (1 - hi) if hi < 1 else float("inf"))


def module_rows(counts: dict, area: str, all_cells: dict | None = None) -> list[dict]:
    """One row per module of `area`; layers absent from every subclass make the module missing. With
    `all_cells`, adds the denominator n_all and frac_<subclass> = count / n_all."""
    out = []
    for mod, layers in MODULES.items():
        present = [l for l in layers if any(f"{area}-{l}" in counts[s] for s in SUBCLASSES)]
        row = {"area": area, "module": mod, "layers": "+".join(present) if present else "", "missing": not present}
        for s in SUBCLASSES:
            row[s] = sum(counts[s].get(f"{area}-{l}", 0) for l in present) if present else ""
        n_all = sum(all_cells.get(f"{area}-{l}", 0) for l in present) if (present and all_cells) else ""
        row["n_all"] = n_all
        for s in SUBCLASSES:
            row[f"frac_{s}"] = row[s] / n_all if n_all else ""
        if present and mod != "L1" and row["PVALB"] > 0:
            row["R_sst_pvalb"] = row["SST"] / row["PVALB"]
            row["R_lo"], row["R_hi"] = ratio_ci(row["SST"], row["PVALB"])
        else:
            row["R_sst_pvalb"] = row["R_lo"] = row["R_hi"] = ""
        out.append(row)
    return out


def main(cluster_dir: str, all_cells_csv: str, out_csv: str) -> None:
    counts, n_clusters = load(Path(cluster_dir))
    if n_clusters != EXPECTED:
        raise ValueError(f"cluster counts {n_clusters} != expected {EXPECTED}")
    all_cells = read_export(Path(all_cells_csv))
    if set(all_cells) != set(counts["SST"]):
        raise ValueError("all-cells export lists a different region set from the cluster exports")
    areas = sorted({re.sub(r"-l\d$", "", r) for r in counts["SST"] if re.search(r"-l\d$", r)})
    rows = [r for a in areas for r in module_rows(counts, a, all_cells)]
    cols = ["area", "module", "layers", "missing", *SUBCLASSES, "n_all", *(f"frac_{s}" for s in SUBCLASSES),
            "R_sst_pvalb", "R_lo", "R_hi"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"clusters per subclass {n_clusters}; areas {len(areas)}; rows {len(rows)} -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

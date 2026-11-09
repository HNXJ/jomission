"""Import the Allen V1 core tables into docs/laminar_motif/data/allen_v1_blocks.json (ALLEN_V1_REPLICA_SPEC.md 1-2).

Usage:
  python scripts/allen_v1_import.py SRC_DIR

SRC_DIR holds the files of AllenInstitute/oscillations-biorealistic-mouse-v1-model @ SOURCE_SHA (BSD-3-Clause):
pmax_matrix_v1dd.csv, sigma.csv, V1model_seed_file.xlsx, lgn_params.csv. Writes the 19 populations (core counts,
depth bounds), pmax, sigma, the space-averaging factor g per (pre ei, post ei) and the source hashes.
Exit 0 when the JSON is written.

g = E[N2(dxz; mu, C) / N2(mu; mu, C)] over pairs uniform in the 400 um core disk, target tuning uniform, the
Rossi form of edge_funcs.connect_cells (E source: C = R diag((s/1.5)^2, (s*1.5)^2) R^T, mu = -50 um along theta;
I source: isotropic, mu = +50 um). Orientation and target-size factors average 1 and are dropped.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "laminar_motif" / "data" / "allen_v1_blocks.json"
SOURCE = "AllenInstitute/oscillations-biorealistic-mouse-v1-model"
SOURCE_SHA = "bd25da66bfa92d097692d1c1c5d3621bcd62bcfe"
CORE_RADIUS_UM = 400.0
ROSSI_DISP_UM, ROSSI_SCALE = 50.0, 1.5
MC_PAIRS, MC_SEED = 400_000, 0
# seed-file pop_name -> pmax population (L4 E subtypes merge, as pmax_matrix_v1dd.csv)
POP_OF = {
    "i1Htr3a": "L1_Inh", "e23Cux2": "L2/3_Exc", "i23Pvalb": "L2/3_PV", "i23Sst": "L2/3_SST", "i23Vip": "L2/3_VIP",
    "e4Nr5a1": "L4_Exc", "e4Rorb": "L4_Exc", "e4Scnn1a": "L4_Exc", "e4other": "L4_Exc",
    "i4Pvalb": "L4_PV", "i4Sst": "L4_SST", "i4Vip": "L4_VIP",
    "e5ET": "L5_ET", "e5IT": "L5_IT", "e5NP": "L5_NP", "i5Pvalb": "L5_PV", "i5Sst": "L5_SST", "i5Vip": "L5_VIP",
    "e6Ntsr1": "L6_Exc", "i6Pvalb": "L6_PV", "i6Sst": "L6_SST", "i6Vip": "L6_VIP",
}


def table(path: Path) -> list[list[str]]:
    return [r.split() for r in path.read_text(encoding="utf-8").strip().splitlines()]


def xlsx_rows(path: Path, sheet_index: int = 0) -> list[dict[str, str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(path)
    shared = [
        "".join(t.text or "" for t in si.iter("{%s}t" % ns["m"]))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)
    ]
    root = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet_index + 1}.xml"))
    out = []
    for row in root.iter("{%s}row" % ns["m"]):
        d = {}
        for c in row.findall("m:c", ns):
            v = c.find("m:v", ns)
            if v is not None:
                d[re.match(r"[A-Z]+", c.get("r")).group(0)] = shared[int(v.text)] if c.get("t") == "s" else v.text
        out.append(d)
    return out


def g_factor(sigma: float, src_ei: str, rng: np.random.Generator) -> float:
    def disk(n):
        r = CORE_RADIUS_UM * np.sqrt(rng.random(n))
        a = 2 * np.pi * rng.random(n)
        return np.c_[r * np.cos(a), r * np.sin(a)]

    s, t = disk(MC_PAIRS), disk(MC_PAIRS)
    th = 2 * np.pi * rng.random(MC_PAIRS)
    c, sn = np.cos(th), np.sin(th)
    sign = 1.0 if src_ei == "i" else -1.0
    u = (s - t) - sign * ROSSI_DISP_UM * np.c_[c, sn]
    if src_ei == "i":
        q = (u ** 2).sum(axis=1) / sigma ** 2
    else:
        a = u[:, 0] * c + u[:, 1] * sn
        b = -u[:, 0] * sn + u[:, 1] * c
        q = (a / (sigma / ROSSI_SCALE)) ** 2 + (b / (sigma * ROSSI_SCALE)) ** 2
    return float(np.exp(-0.5 * q).mean())


def main(src: Path) -> None:
    files = ("pmax_matrix_v1dd.csv", "sigma.csv", "V1model_seed_file.xlsx", "lgn_params.csv")
    hashes = {f: hashlib.sha256((src / f).read_bytes()).hexdigest() for f in files}
    rows = xlsx_rows(src / "V1model_seed_file.xlsx")
    head = rows[0]
    col = {v: k for k, v in head.items()}
    pops: dict[str, dict] = {}
    for r in rows[1:]:
        name = r.get(col["pop_name"])
        if name not in POP_OF:
            continue
        key = POP_OF[name]
        p = pops.setdefault(key, {"N": 0, "subtypes": {}, "ei": r[col["ei"]],
                                  "depth_um": [float(r[col["upper_bound"]]), float(r[col["lower_bound"]])]})
        n = int(r[col["pop_core_count"]])
        p["N"] += n
        p["subtypes"][name] = n
    pm = table(src / "pmax_matrix_v1dd.csv")
    names = pm[0][1:]
    if sorted(names) != sorted(pops) or [r[0] for r in pm[1:]] != names:
        raise RuntimeError("pmax populations do not match the seed-file populations")
    pmax = {r[0]: {post: float(v) for post, v in zip(names, r[1:])} for r in pm[1:]}
    sg = table(src / "sigma.csv")
    sigma = {f"{a[0]}{k}": float(v) for a in sg[1:] for k, v in zip(sg[0][1:], a[1:])}
    rng = np.random.default_rng(MC_SEED)
    g = {k: g_factor(v, k[0], rng) for k, v in sigma.items()}
    lg = table(src / "lgn_params.csv")
    lgn = {r[0]: dict(zip(lg[0][1:], map(float, r[1:]))) for r in lg[1:]}
    out = {
        "source": SOURCE, "source_sha": SOURCE_SHA, "sha256": hashes,
        "core_radius_um": CORE_RADIUS_UM, "mc": {"pairs": MC_PAIRS, "seed": MC_SEED},
        "populations": names, "pops": {k: pops[k] for k in names},
        "N_total": int(sum(pops[k]["N"] for k in names)),
        "pmax": pmax, "sigma_um": sigma, "g": g, "lgn_params": lgn,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"N_total": out["N_total"], "g": g, "out": str(OUT.relative_to(ROOT))}))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))

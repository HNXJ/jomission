"""E0 scored runner (docs/laminar_motif/E0_SEAL.md). Run only under SCI-LAMINAR-MOTIF-1, tier 3.

Usage:
  python scripts/run_e0_scored.py seed N      run one scored seed (N in 1..5, 12 s, p = 0.07); writes the seed record
  python scripts/run_e0_scored.py score       apply the verdict rule to the seed records; writes the result file
Seed records and the result go to results/laminar_e0/. Counts arrays are written beside them (.npy, untracked).
Exit 0 whenever the file was written; the status is the file's `verdict`, never the exit code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "laminar_e0"
P = 0.07
T_S = 12.0


def _seed(n: int) -> None:
    import jaxfne
    import numpy as np

    from jomission.laminar import caminos_analysis as an
    from jomission.laminar import caminos_e0 as c

    if n not in (1, 2, 3, 4, 5):
        raise SystemExit(f"seed must be 1..5, got {n}")
    path = OUT / f"e0_seed{n}.json"
    if path.exists():
        raise SystemExit(f"{path.name} exists; a scored seed is run once")
    OUT.mkdir(parents=True, exist_ok=True)
    r = c.run(P, n, T_S)
    est = an.analyze_run(r["counts"], 1e-4, r["sizes"])
    est = {k: (v.item() if hasattr(v, "item") else v) for k, v in est.items()}
    rec = {
        "schema": "jomission.e0_seed.v1", "seed": n, "p": P, "T_s": T_S, "steps": r["steps"],
        "wall_s": r["wall_s"], "jax_devices": r["jax_devices"], "jaxfne_version": jaxfne.__version__,
        "jaxfne_file": "side-install" if "jaxfne-0.5.9-site" in str(jaxfne.__file__) else "env",
        "analysis": est,
    }
    np.save(OUT / f"e0_seed{n}_counts.npy", r["counts"])
    path.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"wrote {path.name} steps={r['steps']} wall_s={r['wall_s']:.0f}")


def _score() -> None:
    from jomission.laminar.e0_verdict import SEEDS, verdict

    seeds = {}
    versions = set()
    for n in SEEDS:
        p = OUT / f"e0_seed{n}.json"
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            seeds[n] = rec
            versions.add(rec["jaxfne_version"])
    v = verdict(seeds)
    res = {
        "schema": "jomission.e0_result.v1", "lineage": "SCI-LAMINAR-MOTIF-1", "seal": "docs/laminar_motif/E0_SEAL.md",
        "jaxfne_versions": sorted(versions), **v,
    }
    (OUT / "e0_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("verdict", v["verdict"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "seed":
        _seed(int(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "score":
        _score()
    else:
        raise SystemExit(__doc__)

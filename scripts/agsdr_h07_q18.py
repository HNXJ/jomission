"""Recompute H07 under ruling Q18 (term 4: locality_index >= 0.9 x canonical) (docs/designs/SCI-AGSDR-H07H11_ruling_Q18.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402

D = REPO / "docs/receipts/agsdr_h07h11"
RULING = REPO / "docs/designs/SCI-AGSDR-H07H11_ruling_Q18.md"
FACTOR = 0.9
GLOBALIZED_LOC = 1.0047210504246324  # sealed control 2 value (globalized model), recomputed below


def main() -> int:
    import numpy as np
    import agsdr_h07h11 as H

    s = json.loads((D / "summary.json").read_text(encoding="utf-8"))
    ref = s["H07_reference_terms"]
    thr = FACTOR * ref["t4_locality_index"]
    # controls: canonical passes; globalized (same construction as sealed control 2) fails
    m = H.canonical()
    g = H.graph(m)
    w = g["area"][g["pre"]] == g["area"][g["post"]]
    rng = np.random.default_rng(20261105)
    n, k = len(g["area"]), int(w.sum())
    pre, post = g["pre"].copy(), g["post"].copy()
    rp = rng.integers(0, n, k)
    rq = rng.integers(0, n - 1, k)
    rq = rq + (rq >= rp)
    pre[w], post[w] = rp, rq
    glob = H.locality_index(pre, post, g["area"], g["xyz"])
    controls = {"canonical_pass": ref["t4_locality_index"] >= thr, "globalized_locality": glob,
                "globalized_matches_sealed_control": abs(glob - GLOBALIZED_LOC) < 1e-12, "globalized_fail": glob < thr}
    per = {}
    for f in sorted(D.glob("AGS_S0_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        h = r["H07"]
        t4 = h["t4_locality_index"] >= thr
        per[r["candidate_id"]] = {"t4_locality_index": h["t4_locality_index"], "t4_pass_Q18": t4,
                                  "H07_pass_Q18": bool(h["t1_pass"] and h["t2_pass"] and h["t3_pass"] and t4 and h["t5_pass"]),
                                  "H11_pass": r["H11"]["H11_pass"]}
    ok = lambda c: per[c]["H07_pass_Q18"] and per[c]["H11_pass"]  # noqa: E731
    a4 = json.loads((REPO / "docs/receipts/agsdr_stage0_a4/summary.json").read_text(encoding="utf-8"))
    out = {"ruling": "docs/designs/SCI-AGSDR-H07H11_ruling_Q18.md", "ruling_sha256": A.sha256_of(RULING),
           "script_sha256": A.sha256_of(Path(__file__).resolve()), "threshold": thr, "factor": FACTOR,
           "controls": controls, "controls_pass": bool(controls["canonical_pass"] and controls["globalized_fail"]
                                                       and controls["globalized_matches_sealed_control"]),
           "per_candidate": per, "H07_pass_count": sum(v["H07_pass_Q18"] for v in per.values()),
           "final_hard_feasible_unamended": sorted(c for c in s["prior_feasible_H01_H10d_H05allE"] if ok(c)),
           "final_hard_feasible_amendment_4": sorted(c for c in a4["feasible_H01_H10d_tau_H05allE"] if ok(c)),
           "stage1_slots": 8}
    (D / "h07_q18_summary.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("threshold", "controls", "controls_pass", "H07_pass_count",
                                          "final_hard_feasible_unamended", "final_hard_feasible_amendment_4")}, indent=1, default=float))
    return 0 if out["controls_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

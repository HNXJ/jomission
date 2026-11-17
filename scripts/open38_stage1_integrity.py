"""OPEN-38 stage-1 batch integrity (owner 2026-11-16), run beside the frozen open38_stage1_check.py: PASS is impossible
when a required run is absent, incomplete, carries the wrong seed or arm, or is one execution reported twice.

Usage: python scripts/open38_stage1_integrity.py OUT_JSON --head SHA [--results-dir D]

Per expected run (arm x seed, tag o38s_pc_{arm}_n4000_s{seed}_T8):
  complete   .json, .npz, _similarity.json, _verdict.json present and parseable; npz has counts, block_keys, lfp_pA
  identity   recorded seed == tag seed; recorded layout == the arm's frozen layout; T_s == 8; verdict r_ab finite
  provenance recorded git_sha == HEAD given; git_dirty false; jaxfne_file under the verified export
Across runs:
  provenance_unique  no two runs share an execution record: wall_s (construct, simulate, total). A shared record is
             evidence of SUSPECTED duplication (one execution under two tags), not proof: timings can collide or be
             copied. It fails this check, and per owner ruling 2026-11-16 no scientific result is rejected on it
             without independent review. Later stages should record a unique execution ID in the run manifest.
  identical_outputs  runs sharing npz bytes or the counts array are listed; with distinct execution records this is
             a WARNING, not a failure (identical outputs can occur legitimately; owner 2026-11-16)
  no_extra   no o38s_pc_* run outside the expected 15
Exit 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open38_stage1_check as s1  # noqa: E402
import open38_smoke_check as sc  # noqa: E402

BASE = "e2_3unit_n4000_dens_k1_kd1_sst15"
LAYOUT = {"ref": BASE, "flat": BASE + "_rhof0p794", "rev": BASE + "_rhor"}
T_S = 8.0


def check_run(d: Path, arm: str, seed: int, head: str) -> dict:
    t = s1.tag(arm, seed)
    out = {"tag": t, "problems": []}
    try:
        rec = json.loads((d / f"{t}.json").read_text(encoding="utf-8"))
        ver = json.loads((d / f"{t}_verdict.json").read_text(encoding="utf-8"))
        json.loads((d / f"{t}_similarity.json").read_text(encoding="utf-8"))
        raw = (d / f"{t}.npz").read_bytes()
        with np.load(d / f"{t}.npz") as z:
            missing = [k for k in ("counts", "block_keys", "lfp_pA") if k not in z.files]
            counts = z["counts"] if "counts" in z.files else None
    except Exception as e:  # absent or unreadable file = incomplete run
        out["problems"].append(f"incomplete: {type(e).__name__}: {e}")
        out["ok"] = False
        return out
    if missing:
        out["problems"].append(f"incomplete: npz lacks {missing}")
    if rec.get("seed") != seed:
        out["problems"].append(f"wrong seed: recorded {rec.get('seed')}, tag {seed}")
    if rec.get("layout") != LAYOUT[arm]:
        out["problems"].append(f"wrong arm: recorded layout {rec.get('layout')!r}, expected {LAYOUT[arm]!r}")
    if rec.get("T_s") != T_S:
        out["problems"].append(f"T_s {rec.get('T_s')} != {T_S}")
    r = ver.get("r_ab")
    if not (isinstance(r, (int, float)) and math.isfinite(r)):
        out["problems"].append(f"r_ab not finite: {r!r}")
    if rec.get("git_sha") != head or rec.get("git_dirty") is not False:
        out["problems"].append(f"provenance: git_sha {rec.get('git_sha')}, dirty {rec.get('git_dirty')}")
    jf = (rec.get("jaxfne_file") or "").replace("\\", "/").lower()
    if not jf.startswith(sc.ENGINE_EXPORT.lower() + "/"):
        out["problems"].append(f"engine file outside export: {rec.get('jaxfne_file')}")
    w = rec.get("wall_s")
    if not (isinstance(w, dict) and w):
        out["problems"].append("provenance: no wall_s execution record")
    out["exec_record"] = json.dumps(w, sort_keys=True)
    out["npz_sha256"] = hashlib.sha256(raw).hexdigest()
    out["counts_sha256"] = hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest() if counts is not None else None
    out["ok"] = not out["problems"]
    return out


def main(out: str, d: Path, head: str) -> int:
    runs = [check_run(d, a, s, head) for a in s1.ARMS for s in s1.SEEDS]
    dup = {}
    for key in ("exec_record", "npz_sha256", "counts_sha256"):
        seen: dict = {}
        for r in runs:
            h = r.get(key)
            if h:
                seen.setdefault(h, []).append(r["tag"])
        dup[key] = [v for v in seen.values() if len(v) > 1]
    expected = {s1.tag(a, s) for a in s1.ARMS for s in s1.SEEDS}
    found = {re.sub(r"\.json$", "", p.name) for p in d.glob("o38s_pc_*_T8.json")}
    extra = sorted(found - expected)
    checks = {"runs": {"ok": all(r["ok"] for r in runs), "failed": {r["tag"]: r["problems"] for r in runs if not r["ok"]}},
              "provenance_unique": {"ok": not dup["exec_record"], "shared_execution_record": dup["exec_record"],
                                    "meaning": "suspected duplication; independent review required before rejection"},
              "no_extra": {"ok": not extra, "extra": extra}}
    warnings = {"identical_outputs": {"npz_sha256": dup["npz_sha256"], "counts_sha256": dup["counts_sha256"]}}
    ok = all(c["ok"] for c in checks.values())
    Path(out).write_text(json.dumps({"pass": ok, "head": head, "checks": checks, "warnings": warnings, "runs": runs},
                                    indent=1), encoding="utf-8")
    print(json.dumps({"pass": ok, "failed": [k for k, c in checks.items() if not c["ok"]],
                      "identical_output_groups": len(dup["npz_sha256"]) + len(dup["counts_sha256"])}))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--head", required=True)
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "laminar_e2"))
    a = ap.parse_args()
    sys.exit(main(a.out, Path(a.results_dir), a.head))

"""OPEN-38 batch integrity (owner 2026-11-16; stage 2 added 2026-11-17), run beside the frozen open38_stage1_check.py:
PASS is impossible when a required run is absent, incomplete, carries the wrong seed or arm, or is one execution
reported twice.

Usage: python scripts/open38_stage1_integrity.py OUT_JSON --head SHA [--stage 1|2] [--runs ARM:SEED ...]
                                                 [--results-dir D] [--stage-log F]

Per expected run (stage 1: ref/flat/rev x seeds 1-5, tag o38s_pc_{arm}_n4000_s{seed}_T8; stage 2: v1/mt/mst/teo/chc/
shape x seeds 1-5, tag o38s_{arm}_n4000_s{seed}_T8; --runs narrows to a subset, e.g. the stage-2 smoke run):
  complete   .json, .npz, _similarity.json, _verdict.json present and parseable; npz has counts, block_keys, lfp_pA
  identity   recorded seed == tag seed; recorded layout == the arm's frozen layout; T_s == 8; verdict r_ab finite
  provenance recorded git_sha == HEAD given; git_dirty false; jaxfne_file under the verified export
  stage 2 only:
  exec       {tag}_exec.json (open38_exec_manifest): right tag, 32-hex exec_id, ended with rc 0, output hashes equal
             the files on disk
  run checks engine version and export, realized ratio within 1 cell, finite arrays and scores (frozen
             open38_stage1_check.run_checks); its stop state is reported, not failed (spec §7: stop states are
             reported, not counted)
Across runs:
  provenance_unique  no two runs share an execution record: wall_s (construct, simulate, total) and, in stage 2, the
             exec_id. A shared record is evidence of SUSPECTED duplication (one execution under two tags), not proof:
             timings can collide or be copied. It fails this check, and per owner ruling 2026-11-16 no scientific
             result is rejected on it without independent review.
  propagation (stage 2) the stage log has a line "start TAG LAYOUT exec=ID" for each run, with the manifest's exec_id
  identical_outputs  runs sharing npz bytes or the counts array are listed; with distinct execution records this is
             a WARNING, not a failure (identical outputs can occur legitimately; owner 2026-11-16)
  no_extra   no run of the stage outside the expected set (stage 2 ignores o38s_pc_* and the two smoke-test tags)
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
import open38_exec_manifest as em  # noqa: E402
import open38_stage1_check as s1  # noqa: E402
import open38_smoke_check as sc  # noqa: E402

BASE = "e2_3unit_n4000_dens_k1_kd1_sst15"
LAYOUT = {"ref": BASE, "flat": BASE + "_rhof0p794", "rev": BASE + "_rhor"}
LAYOUT2 = {  # spec §3 table, frozen 2026-11-16
    "v1": BASE + "_rhouS1p688_rhouM0p558_rhouD1p584",
    "mt": BASE + "_rhouS1p280_rhouM0p599_rhouD1p014",
    "mst": BASE + "_rhouS1p220_rhouM0p615_rhouD1p103",
    "teo": BASE + "_rhouS1p409_rhouM0p695_rhouD1p153",
    "chc": BASE + "_rhouS1p345_rhouM0p595_rhouD1p178",
    "shape": BASE + "_rhouS1p095_rhouM0p485_rhouD0p942",
}
ARMS2 = tuple(LAYOUT2)
SMOKE_TAGS = {"o38s_ref_n4000_s1_T8", "o38s_nc_n4000_s1_T8"}
T_S = 8.0


def tag2(arm: str, seed: int) -> str:
    return f"o38s_{arm}_n4000_s{seed}_T8"


STAGES = {1: {"arms": s1.ARMS, "tag": s1.tag, "layout": LAYOUT, "glob": "o38s_pc_*_T8.json"},
          2: {"arms": ARMS2, "tag": tag2, "layout": LAYOUT2, "glob": "o38s_*_n4000_s*_T8.json"}}


def check_run(d: Path, arm: str, seed: int, head: str, stage: int = 1) -> dict:
    cfg = STAGES[stage]
    t = cfg["tag"](arm, seed)
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
    if rec.get("layout") != cfg["layout"][arm]:
        out["problems"].append(f"wrong arm: recorded layout {rec.get('layout')!r}, expected {cfg['layout'][arm]!r}")
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
    if stage == 2:
        out["exec_id"], probs = em.check_manifest(d, t)
        out["problems"] += probs
        rc = s1.run_checks(d, t)
        if not (rc.get("engine") and all(v["ok"] for v in rc.get("ratio", {}).values()) and rc.get("finite")):
            out["problems"].append(f"run checks: engine {rc.get('engine')}, ratio {rc.get('ratio')}, "
                                   f"finite {rc.get('finite')}")
        out["stop"] = rc.get("stop")
    out["ok"] = not out["problems"]
    return out


def main(out: str, d: Path, head: str, stage: int = 1, runs: list[tuple[str, int]] | None = None,
         stage_log: Path | None = None) -> int:
    cfg = STAGES[stage]
    pairs = runs or [(a, s) for a in cfg["arms"] for s in s1.SEEDS]
    res = [check_run(d, a, s, head, stage) for a, s in pairs]
    keys = ("exec_record", "npz_sha256", "counts_sha256") + (("exec_id",) if stage == 2 else ())
    dup = {}
    for key in keys:
        seen: dict = {}
        for r in res:
            h = r.get(key)
            if h:
                seen.setdefault(h, []).append(r["tag"])
        dup[key] = [v for v in seen.values() if len(v) > 1]
    expected = {cfg["tag"](a, s) for a, s in pairs}
    found = {re.sub(r"\.json$", "", p.name) for p in d.glob(cfg["glob"])}
    if stage == 2:
        found = {t for t in found if not t.startswith("o38s_pc_") and t not in SMOKE_TAGS}
    extra = sorted(found - expected)
    shared = dup["exec_record"] + dup.get("exec_id", [])
    checks = {"runs": {"ok": all(r["ok"] for r in res), "failed": {r["tag"]: r["problems"] for r in res if not r["ok"]}},
              "provenance_unique": {"ok": not shared, "shared_execution_record": dup["exec_record"],
                                    **({"shared_exec_id": dup["exec_id"]} if stage == 2 else {}),
                                    "meaning": "suspected duplication; independent review required before rejection"},
              "no_extra": {"ok": not extra, "extra": extra}}
    if stage == 2:
        log = stage_log.read_text(encoding="utf-8") if stage_log and stage_log.is_file() else ""
        lay = {cfg["tag"](a, s): cfg["layout"][a] for a, s in pairs}
        miss = [r["tag"] for r in res
                if not (r.get("exec_id") and f" start {r['tag']} {lay[r['tag']]} exec={r['exec_id']}\n" in log)]
        checks["propagation"] = {"ok": bool(log) and not miss, "stage_log": str(stage_log), "missing": miss}
    warnings = {"identical_outputs": {"npz_sha256": dup["npz_sha256"], "counts_sha256": dup["counts_sha256"]}}
    if stage == 2:
        warnings["stop_states"] = {r["tag"]: r["stop"] for r in res if r.get("stop")}
    ok = all(c["ok"] for c in checks.values())
    report = {"pass": ok, **({"stage": 2} if stage == 2 else {}), "head": head, "checks": checks, "warnings": warnings,
              "runs": res}  # stage-1 report keeps its 2026-11-17 byte layout
    Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"pass": ok, "failed": [k for k, c in checks.items() if not c["ok"]],
                      "identical_output_groups": len(dup["npz_sha256"]) + len(dup["counts_sha256"])}))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--head", required=True)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--runs", nargs="+", help="ARM:SEED subset (default: the stage's full matrix)")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "laminar_e2"))
    ap.add_argument("--stage-log", help="stage-2 driver log (default: RESULTS_DIR/o38s_s2_stage.log)")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    sub = [(x.split(":")[0], int(x.split(":")[1])) for x in a.runs] if a.runs else None
    if sub and any(arm not in STAGES[a.stage]["arms"] or s not in s1.SEEDS for arm, s in sub):
        sys.exit(f"--runs outside stage {a.stage}: {a.runs}")
    sl = Path(a.stage_log) if a.stage_log else rd / "o38s_s2_stage.log"
    sys.exit(main(a.out, rd, a.head, a.stage, sub, sl))

"""OPEN-51 frozen protocol (5d586da): the full re-run batch, in order, with its stop rules.

  python -B scripts/open51_rerun_batch.py OUT_DIR

Steps: R1 (E0 seed 1), R2 (E0 seed 1 again), R3 (E0 seeds 2-5), R4a (E1 rho 4 seed 1, twice), R4 (E1 rho 0.25, 0.5,
2 x seeds 1-3 and rho 4 seeds 2-3). Each run is one scripts/open51_rerun.py process under suite_lock.py. A failed
mandatory gate stops every dependent run; nothing is retried. batch_state.json is rewritten after every run.
Only one jax candidate met the protocol's rule (C:/Python314: jax 0.10.1), so an R1 failure stops everything.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
PY = Path("D:/venvs/_temp_o51_rerun/Scripts/python.exe")
LOCK = Path.home() / ".claude/bin/suite_lock.py"
WT = {"e0": "D:/repos/_work/_temp_o51_wt_e0", "e1": "D:/repos/_work/_temp_o51_wt_e1"}
OUT = Path(sys.argv[1])
STATE = OUT / "batch_state.json"
state = {"schema": "jomission.open51_batch.v1", "protocol": "OPEN51_RERUN_PROTOCOL_2026-11-17.md@5d586da",
         "jax_candidates": ["jax 0.10.1 / jaxlib 0.10.1 / numpy 2.4.6 (C:/Python314, only candidate)"],
         "started_utc": None, "steps": [], "stopped": None, "classification": {}}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(OUT / "batch.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save():
    STATE.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8", newline="\n")


def run(step, kind, seed, rho=None):
    out = OUT / step
    args = ["--worktree", WT[kind], "--kind", kind, "--seed", str(seed), "--out", str(out)]
    if rho is not None:
        args += ["--rho", str(rho)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["JAX_COMPILATION_CACHE_DIR"] = str(OUT / "jaxcache")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [str(PY), "-B", str(LOCK), "--wait-s", "14400", "--label", f"jomission-open51-{step}", "--",
           str(PY), "-B", str(LANE / "scripts/open51_rerun.py"), *args]
    log(f"{step} start: {kind} seed {seed} rho {rho}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=LANE, env=env, capture_output=True, text=True)
    (OUT / f"{step}.stdout.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    stem = f"e0_seed{seed}" if kind == "e0" else f"e1_rho{rho:g}_seed{seed}"
    rp = out / f"{stem}.receipt.json"
    rec = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    entry = {"step": step, "run": stem, "exit": p.returncode, "elapsed_s": round(time.time() - t0, 1),
             "receipt": str(rp) if rec else None, "outcome": rec and _outcome(rec, p.returncode)}
    if rec:
        r = rec.get("result", {})
        entry.update(raster_sha256=r.get("raster_sha256"), edge_table_sha256=r.get("edge_table_sha256"),
                     counts_array_sha256=r.get("counts_array_sha256"),
                     counts_equal=rec.get("counts_comparison", {}).get("equal"),
                     json_equal=rec.get("json_comparison", {}).get("equal"))
    state["steps"].append(entry)
    save()
    log(f"{step} end: exit {p.returncode}; {entry['outcome']}")
    return entry


def _outcome(rec, code):
    return {0: "counts equal, JSON equal", 2: "R0 gate failed", 3: "counts differ",
            4: "counts equal, JSON differs"}.get(code, f"crash or lock exit {code}")


def stop(reason):
    state["stopped"] = {"at_utc": now(), "reason": reason}
    log(f"STOP: {reason}")
    classify()
    save()
    sys.exit(1)


def same_realization(a, b):
    return all(a[k] and a[k] == b[k] for k in ("raster_sha256", "edge_table_sha256", "counts_array_sha256"))


def classify():
    """Protocol classification table; a run not reached is INCOMPLETE."""
    st = {s["step"]: s for s in state["steps"]}
    r0r1 = st.get("R1", {}).get("exit") == 0
    r2 = "R2" in st and st["R2"]["exit"] == 0 and same_realization(st["R1"], st["R2"])
    r4a = ("R4a_2" in st and st["R4a_1"]["exit"] == 0 and st["R4a_2"]["exit"] == 0
           and same_realization(st["R4a_1"], st["R4a_2"]))
    cls = {}
    for s in state["steps"]:
        if s["step"] in ("R2", "R4a_2"):
            continue
        e1 = s["run"].startswith("e1")
        if s["exit"] == 3:
            c = "NOT REPRODUCED"
        elif s["exit"] == 0 and r0r1 and r2 and (r4a or not e1):
            c = "COUNT-VALIDATED RECONSTRUCTION"
        elif s["exit"] == 0:
            c = "UNCLASSIFIED (counts equal; a determinism gate failed or was not reached)"
        else:
            c = f"INCOMPLETE (exit {s['exit']})"
        cls[s["run"]] = c
    for seed in (1, 2, 3):
        if f"e0_seed{seed}" in cls:
            cls[f"e1_rho1_seed{seed}"] = cls[f"e0_seed{seed}"] + " (inherited from E0, byte-identical counts)"
    state["classification"] = cls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state["started_utc"] = now()
    state["lane_head"] = subprocess.run(["git", "-C", str(LANE), "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()
    save()
    r1 = run("R1", "e0", 1)
    if r1["exit"] != 0:
        stop(f"R1 failed ({r1['outcome']}); no other jax candidate; every run NOT REPRODUCED or INCOMPLETE")
    r2 = run("R2", "e0", 1)
    if r2["exit"] != 0 or not same_realization(r1, r2):
        stop("R2 failed: environment nondeterministic or run failed; no raster is classified")
    for seed in (2, 3, 4, 5):
        e = run(f"R3_s{seed}", "e0", seed)
        if e["exit"] != 0:
            stop(f"R3 seed {seed} failed ({e['outcome']}); R4 does not start")
    a1 = run("R4a_1", "e1", 1, 4.0)
    if a1["exit"] != 0:
        stop(f"R4a first run failed ({a1['outcome']}); no E1 raster is classified")
    a2 = run("R4a_2", "e1", 1, 4.0)
    if a2["exit"] != 0 or not same_realization(a1, a2):
        stop("R4a repeat not byte-identical or failed; no E1 raster is classified")
    for rho, seed in [(0.25, s) for s in (1, 2, 3)] + [(0.5, s) for s in (1, 2, 3)] + \
                     [(2.0, s) for s in (1, 2, 3)] + [(4.0, 2), (4.0, 3)]:
        e = run(f"R4_rho{rho:g}_s{seed}", "e1", seed, rho)
        if e["exit"] != 0:
            stop(f"R4 rho {rho:g} seed {seed} failed ({e['outcome']})")
    classify()
    state["finished_utc"] = now()
    save()
    log("batch complete")


if __name__ == "__main__":
    main()

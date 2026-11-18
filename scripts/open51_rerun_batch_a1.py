"""OPEN-51 protocol 5d586da with amendment A1 (adopted by the owner 2026-11-18): the full re-run batch.

  python -B scripts/open51_rerun_batch_a1.py OUT_DIR

Differs from scripts/open51_rerun_batch.py only in (A1.1-A1.8):
- the interpreter is the fixed A1 venv;
- batch-level R0 checks before anything runs: exact pip freeze, pip check, the JaxFNE identity check;
- R0c, a compile gate: E0 seed-1 jit_scan captured without simulation (YNN 7, dynamic_variables 4, strict HLO equal to
  the original executable's);
- R2 and R4a_2 run with a fresh, empty JAX cache (recompile); on a byte-identity failure the suspected cause is
  recorded from the two cached jit_scan executables;
- a sidecar <step>.a1.json per run and the A1 tag in batch_state.json (the run driver's receipt keeps 5d586da);
- the selection rule: E0 seed 1 is reported as conditional if R3 or R4 fails.
Order, stop rules, classes and the run driver scripts/open51_rerun.py are unchanged. Nothing is retried.
"""
import gzip
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
PY = Path("D:/venvs/_temp_o51_jax0102/Scripts/python.exe")
LOCK = Path.home() / ".claude/bin/suite_lock.py"
WT = {"e0": "D:/repos/_work/_temp_o51_wt_e0", "e1": "D:/repos/_work/_temp_o51_wt_e1"}
DIAG = LANE / "docs/laminar_motif/open51_diag"
AMEND = "docs/laminar_motif/OPEN51_RERUN_PROTOCOL_AMENDMENT_A1_2026-11-18.md"
PINS = ["jax==0.10.2", "jaxfne @ git+file:///D:/repos/_ref/jaxfne.git@e0160dfc643551f1f1501fac40bd88b7c7fcc8c3",
        "jaxlib==0.10.2", "ml_dtypes==0.5.4", "numpy==2.4.6", "opt_einsum==3.4.0", "scipy==1.18.0"]
SIDE_SITE = "E:/repos/_wt/jaxfne-0.5.9-site"
OUT = Path(sys.argv[1])
STATE = OUT / "batch_state.json"


def sh(*a, **kw):
    return subprocess.run([str(x) for x in a], capture_output=True, text=True, **kw)


AMEND_SHA = sh("git", "-C", LANE, "log", "-1", "--format=%h", "--", AMEND).stdout.strip()
TAG = f"OPEN51_RERUN_PROTOCOL_2026-11-17.md@5d586da+A1@{AMEND_SHA}"
ENV_NOTE = "reconstructed jax 0.10.2 environment (A1)"
state = {"schema": "jomission.open51_batch_a1.v1", "protocol": TAG, "environment": ENV_NOTE, "interpreter": str(PY),
         "started_utc": None, "r0": {}, "steps": [], "stopped": None, "classification": {}}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(OUT / "batch.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save():
    STATE.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8", newline="\n")


def r0_checks():
    """A1.2: exact pins, pip check, JaxFNE identity. A1.3: compile gate R0c."""
    freeze = sorted(sh(PY, "-m", "pip", "freeze").stdout.split("\n"))
    freeze = [x for x in freeze if x.strip()]
    state["r0"]["pip_freeze"] = freeze
    if freeze != sorted(PINS):
        stop("R0 pip freeze differs from the A1.1 list")
    chk = sh(PY, "-m", "pip", "check")
    state["r0"]["pip_check"] = chk.stdout.strip()
    if chk.returncode != 0:
        stop("R0 pip check failed")
    site = Path(PY).parents[1] / "Lib/site-packages"
    ident = sh(PY, "-B", LANE / "docs/laminar_motif/open51_rerun/jaxfne_identity_check.py", SIDE_SITE,
               "D:/repos/_ref/jaxfne.git", f"export:{site.as_posix()}")
    state["r0"]["jaxfne_identity"] = ident.stdout.strip().splitlines()[-1:] + ident.stderr.strip().splitlines()[-2:]
    if ident.returncode != 0 or "differing 0 of 84" not in ident.stdout:
        stop("R0 JaxFNE identity check failed")
    save()
    log("R0 checks pass (pins, pip check, JaxFNE identity)")
    cap = OUT / "R0c"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "JAX_COMPILATION_CACHE_DIR")}
    p = sh(PY, "-B", LOCK, "--wait-s", "14400", "--label", "jomission-open51a1-R0c", "--",
           PY, "-B", LANE / "scripts/open51_compile.py", "--worktree", WT["e0"], "--label", "capture", "--out", cap,
           cwd=LANE, env=env)
    (OUT / "R0c.stdout.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    rp = cap / "capture/receipt.json"
    rec = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    res = rec.get("result", {})
    for name, gz in (("capture.hlo.txt", cap / "capture/jit_scan.hlo.txt.gz"),
                     ("original.hlo.txt", DIAG / "original_e0s1.hlo.txt.gz")):
        if gz.exists():
            (cap / name).write_bytes(gzip.decompress(gz.read_bytes()))
    d = sh(PY, "-B", DIAG / "hlodiff_strict.py", cap / "original.hlo.txt", cap / "capture.hlo.txt")
    state["r0"]["r0c"] = {"exit": p.returncode, "receipt": str(rp), "ynn_fusion_count": res.get("ynn_fusion_count"),
                          "dynamic_variables": res.get("dynamic_variables"),
                          "edge_equal_r1": rec.get("precondition", {}).get("equal"),
                          "strict_hlo": d.stdout.strip().splitlines()[:1]}
    save()
    ok = (p.returncode == 0 and res.get("ynn_fusion_count") == 7 and res.get("dynamic_variables") == 4
          and "identical True" in d.stdout)
    if not ok:
        stop("R0c compile gate failed")
    log("R0c pass (YNN 7, dynamic_variables 4, strict HLO identical to the original)")


def run(step, kind, seed, rho=None, fresh_cache=False):
    out = OUT / step
    args = ["--worktree", WT[kind], "--kind", kind, "--seed", str(seed), "--out", str(out)]
    if rho is not None:
        args += ["--rho", str(rho)]
    cache = OUT / (f"jaxcache_{step}" if fresh_cache else "jaxcache")
    if fresh_cache and cache.exists():
        stop(f"{step}: fresh cache directory already exists")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [str(PY), "-B", str(LOCK), "--wait-s", "14400", "--label", f"jomission-open51a1-{step}", "--",
           str(PY), "-B", str(LANE / "scripts/open51_rerun.py"), *args]
    log(f"{step} start: {kind} seed {seed} rho {rho} cache {cache.name}")
    before = set(cache.glob("jit_scan-*-cache"))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=LANE, env=env, capture_output=True, text=True)
    (OUT / f"{step}.stdout.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    stem = f"e0_seed{seed}" if kind == "e0" else f"e1_rho{rho:g}_seed{seed}"
    rp = out / f"{stem}.receipt.json"
    rec = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    entry = {"step": step, "run": stem, "exit": p.returncode, "elapsed_s": round(time.time() - t0, 1),
             "receipt": str(rp) if rec else None, "outcome": rec and _outcome(rec, p.returncode),
             "jax_cache": str(cache),
             "new_jit_scan_entries": sorted(str(x) for x in set(cache.glob("jit_scan-*-cache")) - before)}
    if rec:
        r = rec.get("result", {})
        entry.update(raster_sha256=r.get("raster_sha256"), edge_table_sha256=r.get("edge_table_sha256"),
                     counts_array_sha256=r.get("counts_array_sha256"),
                     counts_equal=rec.get("counts_comparison", {}).get("equal"),
                     json_equal=rec.get("json_comparison", {}).get("equal"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{step}.a1.json").write_text(json.dumps({"protocol": TAG, "environment": ENV_NOTE, "interpreter": str(PY),
                                                     "receipt": entry["receipt"], "jax_cache": str(cache)}, indent=1)
                                         + "\n", encoding="utf-8", newline="\n")
    state["steps"].append(entry)
    save()
    log(f"{step} end: exit {p.returncode}; {entry['outcome']}")
    return entry


def _outcome(rec, code):
    return {0: "counts equal, JSON equal", 2: "R0 gate failed", 3: "counts differ",
            4: "counts equal, JSON differs"}.get(code, f"crash or lock exit {code}")


def suspected_cause(a, b):
    """A1.5: compare the jit_scan executables each run of a failed determinism pair added to its cache."""
    ea, eb = a["new_jit_scan_entries"], b["new_jit_scan_entries"]
    if len(ea) != 1 or len(eb) != 1:
        return f"unknown: expected one new jit_scan entry per run, found {len(ea)} and {len(eb)}"
    p = sh(PY, "-B", DIAG / "objcode.py", f"first={ea[0]}", f"second={eb[0]}")
    same = "all equal True" in p.stdout
    return {"objcode": p.stdout.strip().splitlines(),
            "suspected": "execution (executables equal)" if same else "compile variation (executables differ)"}


def stop(reason, cause=None):
    state["stopped"] = {"at_utc": now(), "reason": reason, "suspected_cause": cause}
    log(f"STOP: {reason}")
    classify()
    save()
    sys.exit(1)


def same_realization(a, b):
    return all(a[k] and a[k] == b[k] for k in ("raster_sha256", "edge_table_sha256", "counts_array_sha256"))


def classify():
    """Protocol classification table; a run not reached is INCOMPLETE. A1 selection rule on E0 seed 1."""
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
    oos_failed = any(s["step"].startswith(("R3", "R4")) and s["exit"] != 0 for s in state["steps"])
    if oos_failed and cls.get("e0_seed1", "").startswith("COUNT-VALIDATED"):
        cls["e0_seed1"] += " (conditional: E0 seed 1 selected the A1 environment and R3 or R4 failed)"
    for seed in (1, 2, 3):
        if f"e0_seed{seed}" in cls:
            cls[f"e1_rho1_seed{seed}"] = cls[f"e0_seed{seed}"] + " (inherited from E0, byte-identical counts)"
    state["classification"] = cls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state["started_utc"] = now()
    state["lane_head"] = sh("git", "-C", LANE, "rev-parse", "HEAD").stdout.strip()
    state["driver_status"] = sh("git", "-C", LANE, "status", "--porcelain", "--", "scripts/open51_rerun_batch_a1.py",
                                "scripts/open51_rerun.py", "scripts/open51_compile.py", "scripts/open51_viz.py",
                                "docs/laminar_motif/open51_diag",
                                "docs/laminar_motif/open51_rerun/jaxfne_identity_check.py", AMEND).stdout
    save()
    if state["driver_status"] or not AMEND_SHA:
        stop("drivers uncommitted or amendment not committed")
    r0_checks()
    r1 = run("R1", "e0", 1)
    if r1["exit"] != 0:
        stop(f"R1 failed ({r1['outcome']}); A1 admits no other environment")
    r2 = run("R2", "e0", 1, fresh_cache=True)
    if r2["exit"] != 0 or not same_realization(r1, r2):
        stop("R2 failed: not byte-identical or run failed; no raster is classified", suspected_cause(r1, r2))
    for seed in (2, 3, 4, 5):
        e = run(f"R3_s{seed}", "e0", seed)
        if e["exit"] != 0:
            stop(f"R3 seed {seed} failed ({e['outcome']}); R4 does not start")
    a1 = run("R4a_1", "e1", 1, 4.0)
    if a1["exit"] != 0:
        stop(f"R4a first run failed ({a1['outcome']}); no E1 raster is classified")
    a2 = run("R4a_2", "e1", 1, 4.0, fresh_cache=True)
    if a2["exit"] != 0 or not same_realization(a1, a2):
        stop("R4a repeat not byte-identical or failed; no E1 raster is classified", suspected_cause(a1, a2))
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

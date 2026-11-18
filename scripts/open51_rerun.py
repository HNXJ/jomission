"""OPEN-51 frozen protocol (docs/laminar_motif/OPEN51_RERUN_PROTOCOL_2026-11-17.md at 5d586da): one re-run.

  python -B scripts/open51_rerun.py --worktree WT --kind e0|e1 --seed N [--rho R] --out DIR [--smoke-T-s T]

Runs from the fresh venv with no PYTHONPATH. Repeats caminos_e0.run() (construct, simulate, block, labels, sizes,
spikes_to_counts) from the worktree's own module, unchanged, and adds the protocol's two saves: the full spike raster
as sparse (step, cell) int32 pairs and the realized edge table (model.static["adex"]). R0 assertions (clean worktree,
import origin, comparator sha256) stop the run with exit 2 before any simulation. A full run (T 12 s) compares its
counts with the original F: counts (np.array_equal) and its analyze_run with the committed run JSON; exit 0 equal,
3 counts differ, 4 counts equal but JSON estimands differ. --smoke-T-s runs a short simulation that checks the
pipeline only and is never compared or classified (exit 0 when it completes).
"""
import argparse
import ast
import ctypes
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
SALVAGE = Path("F:/repo_salvage_2026-11-10/jomission/E_repos__wt_jomission-handoff.loose/results")
P, T_S, DT = 0.07, 12.0, 1e-4
JSON_COMMIT = {"e0": "ea249f1", "e1": "8200fac"}
CHECK_KEYS = ("spikes_kept", "peak_freq_E_Hz", "peak_excess_E", "oscillates", "lag_PV_SOM_ms")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(wt, *a):
    return subprocess.run(["git", "-C", str(wt), *a], capture_output=True, text=True, check=True).stdout


def sha_table():
    """The sha256 table in scripts/open51_viz.py, read as a literal (no import: it would load the lane's jomission)."""
    tree = ast.parse((LANE / "scripts/open51_viz.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SHA":
            return ast.literal_eval(node.value)
    raise SystemExit("SHA table not found in scripts/open51_viz.py")


def arr_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def peak_rss_bytes():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong)
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return f"unavailable: winerror {ctypes.get_last_error()}"
        return int(c.PeakWorkingSetSize)
    except Exception as e:  # recorded, never fatal
        return f"unavailable: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--kind", choices=("e0", "e1"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rho", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-T-s", type=float)
    a = ap.parse_args()
    wt = Path(a.worktree).resolve()
    stem = f"e0_seed{a.seed}" if a.kind == "e0" else f"e1_rho{a.rho:g}_seed{a.seed}"
    tag = stem + ("_smoke" if a.smoke_T_s else "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "jomission.open51_rerun.v1", "protocol": "OPEN51_RERUN_PROTOCOL_2026-11-17.md@5d586da",
           "run": stem, "smoke": bool(a.smoke_T_s), "kind": a.kind, "seed": a.seed, "rho": a.rho, "p": P,
           "T_s": a.smoke_T_s or T_S, "started_utc": now(), "argv": sys.argv,
           "label": "count-validated reconstructed raster if classified COUNT-VALIDATED RECONSTRUCTION; "
                    "never the recovered original raster"}
    receipt = out / f"{tag}.receipt.json"

    def finish(code, **kw):
        rec.update(kw, finished_utc=now(), exit_code=code, peak_working_set_bytes=peak_rss_bytes())
        receipt.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        print(f"{tag}: exit {code} {kw.get('outcome', '')}")
        sys.exit(code)

    # --- R0: worktree, environment, import origin, comparator --------------------------------------------------
    rec["driver"] = {"lane_head": git(LANE, "rev-parse", "HEAD").strip(),
                     "driver_status": git(LANE, "status", "--porcelain", "--", "scripts/open51_rerun.py",
                                          "scripts/open51_viz.py"),
                     "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if rec["driver"]["driver_status"] and not a.smoke_T_s:
        finish(2, outcome="R0 FAIL: driver or sha table uncommitted")
    status = git(wt, "status", "--porcelain")
    rec["worktree"] = {"path": str(wt), "head": git(wt, "rev-parse", "HEAD").strip(), "status_porcelain": status}
    if status:
        finish(2, outcome="R0 FAIL: worktree not clean")
    if os.environ.get("PYTHONPATH"):
        finish(2, outcome=f"R0 FAIL: PYTHONPATH set ({os.environ['PYTHONPATH']})")
    sys.path.insert(0, str(wt))
    import jax
    import numpy as np
    import jaxfne as jtfne
    from jomission.laminar import caminos_analysis as an
    from jomission.laminar import caminos_e0 as c

    origin = {"caminos_e0": c.__file__, "caminos_analysis": an.__file__, "jaxfne": jtfne.__file__,
              "sys_prefix": sys.prefix}
    rec["import_origin"] = origin
    if not (Path(c.__file__).resolve().is_relative_to(wt) and Path(an.__file__).resolve().is_relative_to(wt)):
        finish(2, outcome="R0 FAIL: jomission not imported from the worktree")
    jf = Path(jtfne.__file__).resolve()
    if not jf.is_relative_to(Path(sys.prefix).resolve()) or "jaxfne-0.5.9-site" in str(jf):
        finish(2, outcome="R0 FAIL: jaxfne not imported from the fresh venv")
    if jtfne.__version__ != "0.5.9":
        finish(2, outcome=f"R0 FAIL: jaxfne {jtfne.__version__}")
    direct = json.loads(md.distribution("jaxfne").read_text("direct_url.json") or "{}")
    cfg_vals = {k: str(v) for k, v in sorted(jax.config.values.items())}
    rec["environment"] = {
        "python": sys.version, "executable": sys.executable,
        "packages": {d.metadata["Name"]: d.version for d in md.distributions()},
        "jaxfne_direct_url": direct,
        "jax_config": cfg_vals,
        "env": {k: os.environ.get(k) for k in ("XLA_FLAGS", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                                                "JAX_COMPILATION_CACHE_DIR", "JOMISSION_JAX_CACHE",
                                                "JAX_PLATFORMS", "JAX_ENABLE_X64")},
        "jax_devices": [str(d) for d in jax.devices()],
        "cpu": platform.processor(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
    }
    if not a.smoke_T_s:
        rel = f"laminar_{a.kind}/{stem}_counts.npy"
        raw = (SALVAGE / rel).read_bytes()
        cmp_sha = hashlib.sha256(raw).hexdigest()
        want = sha_table()[rel]
        rec["comparator"] = {"path": str(SALVAGE / rel), "sha256": cmp_sha, "expected": want}
        if cmp_sha != want:
            finish(2, outcome="R0 FAIL: comparator sha256 differs from the recorded table")

    # --- run(): body of caminos_e0.run, unchanged, plus the two saves -----------------------------------------
    n_sst = c.rho_to_nsst(a.rho) if a.kind == "e1" else None
    T = a.smoke_T_s or T_S
    t0 = time.time()
    spec = c.build_spec(P, a.seed) if a.kind == "e0" else c.build_spec(P, a.seed, n_sst=n_sst)
    model = jtfne.construct(spec)
    sim = c.build_sim(P, a.seed, T)
    if str(sim.resolved_runtime.actual_dtype) != "float32":
        finish(2, outcome=f"dtype {sim.resolved_runtime.actual_dtype}")
    signals = model.simulate(sim)
    jax.block_until_ready(signals.spikes)
    labels = tuple(model.params["emitter"].labels)
    sizes = {t: labels.count(t) for t in c.POPS}
    spikes = np.asarray(signals.spikes)
    counts = c.spikes_to_counts(spikes, sizes)
    wall = time.time() - t0
    steps, cells = (x.astype(np.int32) for x in np.nonzero(spikes))
    adex = model.static["adex"]
    edge_keys = [k for k in ("pre", "post", "weight_nS", "receptor") if k in adex]
    edges = {k: np.asarray(adex[k]) for k in edge_keys}
    np.save(out / f"{tag}_counts.npy", counts)
    np.savez(out / f"{tag}_raster.npz", step=steps, cell=cells, shape=np.array(spikes.shape, np.int64))
    np.savez(out / f"{tag}_edges.npz", **edges)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (out / f"{tag}_counts.npy", out / f"{tag}_raster.npz", out / f"{tag}_edges.npz")}
    res = {"wall_s": wall, "steps": int(sim.n_steps), "sizes": sizes, "raster_shape": list(spikes.shape),
           "n_spikes": int(steps.size), "counts_array_sha256": arr_sha(counts),
           "raster_sha256": arr_sha(steps, cells), "edge_keys": edge_keys,
           "edge_table_sha256": arr_sha(*[edges[k] for k in edge_keys]), "n_edges": int(edges["pre"].size)
           if "pre" in edges else None, "output_files_sha256": files}
    rec["result"] = res
    if a.smoke_T_s:
        orig = np.load(SALVAGE / f"laminar_{a.kind}/{stem}_counts.npy")
        n = counts.shape[0]
        rec["diagnostic_prefix_equal_to_original"] = bool(np.array_equal(counts, orig[:n]))
        finish(0, outcome="SMOKE complete (pipeline only; not compared, not classified)")

    # --- comparison -------------------------------------------------------------------------------------------
    orig = np.load(SALVAGE / f"laminar_{a.kind}/{stem}_counts.npy")
    eq = orig.dtype == counts.dtype and orig.shape == counts.shape and bool(np.array_equal(orig, counts))
    diff = {"dtype": [str(orig.dtype), str(counts.dtype)], "shape": [orig.shape, counts.shape]}
    if orig.shape == counts.shape:
        d = counts.astype(np.int64) - orig.astype(np.int64)
        bad = np.nonzero(np.any(d != 0, axis=1))[0]
        diff.update(n_steps_differ=int(bad.size), first_step_differ=int(bad[0]) if bad.size else None,
                    max_abs_diff=int(np.abs(d).max()), total_spikes_new=counts.sum(0).tolist(),
                    total_spikes_orig=orig.sum(0).tolist())
    rec["counts_comparison"] = {"equal": eq, **diff, "original_array_sha256": arr_sha(orig)}
    committed = json.loads(git(LANE, "show", f"{JSON_COMMIT[a.kind]}:results/laminar_{a.kind}/{stem}.json"))
    got = an.analyze_run(counts, DT, sizes)
    jc = {k: {"new": got[k], "committed": committed["analysis"][k]} for k in CHECK_KEYS}

    def same(x, y):
        if isinstance(y, (bool, dict)) or y is None:
            return x == y
        return abs(float(x) - float(y)) <= 1e-9 * max(1.0, abs(float(y)))

    json_ok = all(same(v["new"], v["committed"]) for v in jc.values())
    rec["json_comparison"] = {"equal": json_ok, "json_commit": JSON_COMMIT[a.kind], "values": jc}
    if not eq:
        finish(3, outcome="counts differ from the original")
    if not json_ok:
        finish(4, outcome="counts equal, committed JSON estimands differ")
    finish(0, outcome="counts equal the original at every step; JSON estimands equal")


if __name__ == "__main__":
    main()

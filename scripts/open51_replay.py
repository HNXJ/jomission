"""OPEN-51 diagnostic replay (owner 2026-11-18): run E0 seed 1 through the unchanged construct/simulate path with the
`jit_scan` executable taken from a given persistent-cache entry instead of a fresh compile.

The entry's bytes are written into a fresh, empty JAX cache directory under R1's cache key, so the re-run environment
loads it in place of compiling. The run aborts before simulating when:
- the worktree is unclean, PYTHONPATH is set, or the import origins are wrong;
- the entry fails to deserialize, or its YNN fusion count differs from --expect-ynn;
- the edge table differs from R1's (input precondition);
- JAX tries to compile `jit_scan` (the cache key differs, so the entry would not be used).

Diagnostic only: no classification under the frozen protocol. Writes counts, raster and a receipt to --out.
Usage: python -B open51_replay.py --worktree WT --entry CACHEFILE --expect-ynn N --label NAME --out DIR
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
SALVAGE = Path("F:/repo_salvage_2026-11-10/jomission/E_repos__wt_jomission-handoff.loose/results")
R1_DIR = Path("D:/tmp/_temp_o51_rerun/batch/R1")
R1_KEY = "jit_scan-decb6ecbb937bae38fe4684d91638ce70256e2a41f515d511855f468f04cf080-cache"
R1_EDGE_SHA = "06f6c60fd6e8474e0de99d5be68042c3621ef6a83439916ed18bb217c0c757c7"
P, T_S, SEED = 0.07, 12.0, 1


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(wt, *a):
    return subprocess.run(["git", "-C", str(wt), *a], capture_output=True, text=True, check=True).stdout


def arr_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def first_diff(a, b):
    import numpy as np
    d = a.astype(np.int64) - b.astype(np.int64)
    bad = np.nonzero(np.any(d != 0, axis=1))[0]
    return {"equal": bool(bad.size == 0), "n_steps_differ": int(bad.size),
            "first_step_differ": int(bad[0]) if bad.size else None,
            "diff_at_first": d[bad[0]].tolist() if bad.size else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--expect-ynn", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wt = Path(a.worktree).resolve()
    out = Path(a.out) / a.label
    out.mkdir(parents=True, exist_ok=False)
    entry = Path(a.entry).read_bytes()
    rec = {"schema": "jomission.open51_replay.v1", "label": a.label, "started_utc": now(), "argv": sys.argv,
           "entry": {"path": str(a.entry), "sha256": hashlib.sha256(entry).hexdigest(), "installed_as": R1_KEY},
           "driver": {"lane_head": git(LANE, "rev-parse", "HEAD").strip(),
                      "driver_status": git(LANE, "status", "--porcelain", "--", "scripts/open51_replay.py"),
                      "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    receipt = out / "receipt.json"

    def finish(code, outcome, **kw):
        rec.update(kw, outcome=outcome, finished_utc=now(), exit_code=code)
        receipt.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        print(f"{a.label}: exit {code} {outcome}")
        sys.exit(code)

    if rec["driver"]["driver_status"]:
        finish(2, "driver uncommitted")
    rec["worktree"] = {"path": str(wt), "head": git(wt, "rev-parse", "HEAD").strip(),
                       "status_porcelain": git(wt, "status", "--porcelain")}
    if rec["worktree"]["status_porcelain"]:
        finish(2, "worktree not clean")
    if os.environ.get("PYTHONPATH"):
        finish(2, "PYTHONPATH set")
    cache = out / "jaxcache"
    cache.mkdir()
    (cache / R1_KEY).write_bytes(entry)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    sys.path.insert(0, str(wt))

    import jax
    import numpy as np
    from compression import zstd
    from jax._src import compiler, xla_bridge

    import jaxfne as jtfne
    from jomission.laminar import caminos_e0 as c

    rec["import_origin"] = {"caminos_e0": c.__file__, "jaxfne": jtfne.__file__, "jax": jax.__version__,
                            "sys_prefix": sys.prefix}
    if not Path(c.__file__).resolve().is_relative_to(wt) or jtfne.__version__ != "0.5.9":
        finish(2, "import origin or jaxfne version")

    # Smoke: the entry deserializes on this jaxlib and still holds the expected YNN fusion count.
    body = zstd.decompress(entry)
    try:
        exe = xla_bridge.get_backend("cpu").deserialize_executable(body[4:], jax.devices("cpu")[:1], None)
        hlo = "\n".join(m.to_string() for m in exe.hlo_modules())
    except Exception as e:
        finish(5, f"entry does not deserialize: {type(e).__name__}: {str(e)[:300]}")
    ynn = hlo.count("ynn_fusion")
    rec["smoke"] = {"deserialized": True, "ynn_fusion_count": ynn, "expected": a.expect_ynn}
    del exe
    if ynn != a.expect_ynn:
        finish(5, "YNN fusion count differs from expectation")

    # Guard: a compile of jit_scan means the installed entry was not used.
    orig_compile = compiler._compile_and_write_cache

    def guarded(backend, computation, executable_devices, compile_options, host_callbacks, module_name, cache_key):
        if module_name.startswith("jit_scan"):
            raise RuntimeError(f"jit_scan compile requested (cache key {cache_key}); entry not used")
        return orig_compile(backend, computation, executable_devices, compile_options, host_callbacks,
                            module_name, cache_key)

    compiler._compile_and_write_cache = guarded

    t0 = time.time()
    spec = c.build_spec(P, SEED)
    model = jtfne.construct(spec)
    adex = model.static["adex"]
    keys = [k for k in ("pre", "post", "weight_nS", "receptor") if k in adex]
    edge_sha = arr_sha(*[np.asarray(adex[k]) for k in keys])
    rec["precondition"] = {"edge_table_sha256": edge_sha, "r1": R1_EDGE_SHA, "equal": edge_sha == R1_EDGE_SHA}
    if edge_sha != R1_EDGE_SHA:
        finish(6, "edge table differs from R1's")
    sim = c.build_sim(P, SEED, T_S)
    if str(sim.resolved_runtime.actual_dtype) != "float32":
        finish(2, f"dtype {sim.resolved_runtime.actual_dtype}")
    try:
        signals = model.simulate(sim)
        jax.block_until_ready(signals.spikes)
    except RuntimeError as e:
        if "jit_scan compile requested" in str(e):
            finish(7, str(e))
        raise
    labels = tuple(model.params["emitter"].labels)
    sizes = {t: labels.count(t) for t in c.POPS}
    spikes = np.asarray(signals.spikes)
    counts = c.spikes_to_counts(spikes, sizes)
    wall = time.time() - t0
    steps, cells = (x.astype(np.int32) for x in np.nonzero(spikes))
    np.save(out / "counts.npy", counts)
    np.savez(out / "raster.npz", step=steps, cell=cells, shape=np.array(spikes.shape, np.int64))
    r1_counts = np.load(R1_DIR / "e0_seed1_counts.npy")
    r1_rast = np.load(R1_DIR / "e0_seed1_raster.npz")
    orig = np.load(SALVAGE / "laminar_e0/e0_seed1_counts.npy")
    rec["result"] = {
        "wall_s": wall, "n_spikes": int(steps.size), "counts_array_sha256": arr_sha(counts),
        "raster_sha256": arr_sha(steps, cells),
        "vs_original_counts": first_diff(counts, orig),
        "vs_r1_counts": first_diff(counts, r1_counts),
        "raster_equal_r1": bool(np.array_equal(steps, r1_rast["step"]) and np.array_equal(cells, r1_rast["cell"])),
    }
    finish(0, "replay complete (diagnostic; not classified)")


if __name__ == "__main__":
    main()

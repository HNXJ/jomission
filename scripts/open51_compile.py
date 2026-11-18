"""OPEN-51 compile-only check (owner 2026-11-18): build E0 seed 1 through the unchanged construct/simulate path in the
running environment, capture the `jit_scan` executable at its compile, and stop before it executes.

Writes, under --out/--label: the serialized executable as a cache entry (zstd: 4-byte BE compile time + bytes, the
format of the JAX persistent cache), its optimized HLO text (gzip), and a receipt with the jax version, the YNN fusion
count, the loop backend_config field names, and the edge-table sha256 (precondition: equal to R1's).
No simulation runs: the compile hook raises after capturing, before the executable is called.
Usage: python -B open51_compile.py --worktree WT --label NAME --out DIR
"""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
R1_EDGE_SHA = "06f6c60fd6e8474e0de99d5be68042c3621ef6a83439916ed18bb217c0c757c7"
P, T_S, SEED = 0.07, 12.0, 1


class Captured(Exception):
    pass


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wt = Path(a.worktree).resolve()
    out = Path(a.out) / a.label
    out.mkdir(parents=True, exist_ok=False)
    rec = {"schema": "jomission.open51_compile.v1", "label": a.label, "started_utc": now(), "argv": sys.argv,
           "driver": {"lane_head": git(LANE, "rev-parse", "HEAD").strip(),
                      "driver_status": git(LANE, "status", "--porcelain", "--", "scripts/open51_compile.py"),
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
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    sys.path.insert(0, str(wt))

    import importlib.metadata as md

    import jax
    import numpy as np
    from compression import zstd
    from jax._src import compiler

    import jaxfne as jtfne
    from jomission.laminar import caminos_e0 as c

    rec["environment"] = {"python": sys.version, "executable": sys.executable,
                          "packages": {d.metadata["Name"]: d.version for d in md.distributions()}}
    rec["import_origin"] = {"caminos_e0": c.__file__, "jaxfne": jtfne.__file__}
    if not Path(c.__file__).resolve().is_relative_to(wt) or jtfne.__version__ != "0.5.9":
        finish(2, "import origin or jaxfne version")

    orig_compile = compiler._compile_and_write_cache

    def capture(backend, computation, executable_devices, compile_options, host_callbacks, module_name, cache_key):
        exe = orig_compile(backend, computation, executable_devices, compile_options, host_callbacks,
                           module_name, cache_key)
        if module_name.startswith("jit_scan"):
            raise Captured((exe, backend))
        return exe

    compiler._compile_and_write_cache = capture

    model = jtfne.construct(c.build_spec(P, SEED))
    adex = model.static["adex"]
    keys = [k for k in ("pre", "post", "weight_nS", "receptor") if k in adex]
    edge_sha = arr_sha(*[np.asarray(adex[k]) for k in keys])
    rec["precondition"] = {"edge_table_sha256": edge_sha, "r1": R1_EDGE_SHA, "equal": edge_sha == R1_EDGE_SHA}
    if edge_sha != R1_EDGE_SHA:
        finish(6, "edge table differs from R1's")
    sim = c.build_sim(P, SEED, T_S)
    try:
        model.simulate(sim)
    except Captured as e:
        exe, backend = e.args[0]
    else:
        finish(7, "simulate returned without a jit_scan compile")
    ser = backend.serialize_executable(exe)
    entry = zstd.compress((0).to_bytes(4, "big") + ser)
    (out / "jit_scan.cache").write_bytes(entry)
    hlo = "\n".join(m.to_string() for m in exe.hlo_modules())
    (out / "jit_scan.hlo.txt.gz").write_bytes(gzip.compress(hlo.encode("utf-8"), mtime=0))
    rec["result"] = {"jax": jax.__version__, "ynn_fusion_count": hlo.count("ynn_fusion"),
                     "dynamic_variables": hlo.count('"dynamic_variables"'),
                     "dynamic_variable_tuple_indices": hlo.count("dynamic_variable_tuple_indices"),
                     "serialized_bytes": len(ser), "entry_sha256": hashlib.sha256(entry).hexdigest()}
    finish(0, "captured; not executed")


if __name__ == "__main__":
    main()

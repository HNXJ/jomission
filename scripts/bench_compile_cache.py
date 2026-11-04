"""Reproducible benchmark for the persistent compile cache (Sol review 2026-11-04).

Each cold run uses a fresh cache directory; each warm run reuses the directory filled by the
preceding cold run. Every run is a separate process. Writes docs/receipts/compile_cache_benchmark.json.
Usage: python scripts/bench_compile_cache.py [repeats]
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD = r"""
import time
t0 = time.time()
import jomission.network.builder as b
t1 = time.time()
b.build_jomission_model(n_per_area=100, seed=0)
print(f"{t1 - t0:.4f} {time.time() - t1:.4f}")
"""


def run_once(env: dict) -> tuple[float, float]:
    out = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-800:])
    imp, build = out.stdout.strip().split()[-2:]
    return float(imp), float(build)


def main(repeats: int = 3) -> None:
    base = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    base.pop("JAX_COMPILATION_CACHE_DIR", None)
    results = {"cold": [], "warm": []}
    for _ in range(repeats):
        cache = tempfile.mkdtemp(prefix="jomission-jcc-bench-")
        env = dict(base, JAX_COMPILATION_CACHE_DIR=cache)
        results["cold"].append(run_once(env)[1])
        results["warm"].append(run_once(env)[1])
    import statistics as st
    summary = {k: {"build_s": v, "median_s": round(st.median(v), 3), "min_s": round(min(v), 3)} for k, v in results.items()}
    receipt = {
        "schema": "jomission.compile_cache_benchmark.v1",
        "workload": "build_jomission_model(n_per_area=100, seed=0), separate processes",
        "repeats": repeats,
        "settings": {"min_compile_time_secs": 0, "min_entry_size_bytes": -1},
        "machine": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
        "results": summary,
        "note": "Measured by this script; build time excludes import.",
    }
    dest = ROOT / "docs" / "receipts" / "compile_cache_benchmark.json"
    dest.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)

"""Reproducible benchmark for the persistent compile cache (Sol review 2026-11-04).

Each cold run uses a fresh cache directory; each warm run reuses the directory filled by the
preceding cold run. Every run is a separate process. The receipt goes to OUT.json (argv[1], relative to the
working directory); an existing OUT.json is refused (exit 2) and nothing is written. Temp cache dirs are removed.
Usage: python scripts/bench_compile_cache.py OUT.json [repeats]
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
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


def package_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def git_sha() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not 1 <= len(args) <= 2:
        print("usage: python scripts/bench_compile_cache.py OUT.json [repeats]", file=sys.stderr)
        return 2
    dest = Path(args[0])
    repeats = int(args[1]) if len(args) == 2 else 3
    if dest.exists():
        print(f"REFUSING: {dest} exists; nothing written", file=sys.stderr)
        return 2
    try:
        sha = git_sha()
    except subprocess.CalledProcessError as e:
        print(f"FAIL: git rev-parse HEAD in {ROOT}: {e.stderr.strip()}", file=sys.stderr)
        return 1

    base = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    base.pop("JAX_COMPILATION_CACHE_DIR", None)
    results = {"cold": [], "warm": []}
    for _ in range(repeats):
        cache = tempfile.mkdtemp(prefix="jomission-jcc-bench-")
        try:
            env = dict(base, JAX_COMPILATION_CACHE_DIR=cache)
            results["cold"].append(run_once(env)[1])
            results["warm"].append(run_once(env)[1])
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    import statistics as st
    summary = {k: {"build_s": v, "median_s": round(st.median(v), 3), "min_s": round(min(v), 3)} for k, v in results.items()}
    receipt = {
        "schema": "jomission.compile_cache_benchmark.v2",
        "workload": "build_jomission_model(n_per_area=100, seed=0), separate processes",
        "repeats": repeats,
        "settings": {"min_compile_time_secs": 0, "min_entry_size_bytes": -1},
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "jax_version": package_version("jax"),
            "jaxlib_version": package_version("jaxlib"),
            "jaxfne_version": package_version("jaxfne"),
            "git_sha": sha,
        },
        "results": summary,
        "note": "Measured by this script; build time excludes import.",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("x", encoding="utf-8") as f:  # "x": refuse to overwrite even if the file appeared during the run
            f.write(json.dumps(receipt, indent=1) + "\n")
    except FileExistsError:
        print(f"REFUSING: {dest} appeared during the run; nothing written", file=sys.stderr)
        return 2
    print(f"receipt: {dest}")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

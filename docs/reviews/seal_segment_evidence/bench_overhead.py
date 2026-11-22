"""Per-call overhead of the seal write guard at commit 83b83a3 (jomission/harness/seals.py).

Usage: <venv>/bin/python bench_overhead.py bare|guard
  bare  : seals not installed (builtins.open is the original)
  guard : seals.install() called; ROOT/REGISTRY rebound to measure/bench_root with a copy of the
          real 396-entry manifests/sealed_artifacts.json (copied, the clone is only read)

Each condition runs one process. Inside it, 5 repetitions (rep) of three variants, interleaved per rep:
  os_write_128B_x20000   : os.write(fd, 128 B) x 20000 on an fd opened before timing (os.write is NOT patched)
  open_ab_inside_x20000  : with open(p, "ab") as f: f.write(128 B) x 20000, p inside ROOT (unsealed)
  open_ab_outside_x20000 : same, p outside ROOT (guard returns after resolve(), before reading registry)
Per-op time = wall time / 20000 via time.perf_counter_ns(). Reported: median over the 5 reps.
The file for the open variants is truncated before each rep (not timed).
"""

import importlib.util
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

CLONE = Path(__file__).resolve().parents[3]
MEAS = Path(tempfile.gettempdir()) / "seal_bench"
N = 20000
REPS = 5
BUF = b"\x5a" * 128

mode = sys.argv[1]
assert mode in ("bare", "guard"), mode

root = (MEAS / "bench_root").resolve()
if root.exists():
    shutil.rmtree(root)
(root / "manifests").mkdir(parents=True)
(root / "results").mkdir()
shutil.copyfile(CLONE / "manifests" / "sealed_artifacts.json", root / "manifests" / "sealed_artifacts.json")

outside_dir = MEAS / "bench_outside"
outside_dir.mkdir(parents=True, exist_ok=True)
inside_bin = root / "results" / "bench_unsealed.bin"
outside_bin = outside_dir / "bench_outside.bin"

spec = importlib.util.spec_from_file_location("seals", CLONE / "jomission/harness/seals.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
S.ROOT = root
S.REGISTRY = root / "manifests" / "sealed_artifacts.json"
n_entries = len(S.sealed_paths())
if mode == "guard":
    assert S.install() is True
    assert S.active()
else:
    assert not S.active()


def t_os_write():
    fd = os.open(str(inside_bin), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    t0 = time.perf_counter_ns()
    for _ in range(N):
        os.write(fd, BUF)
    t1 = time.perf_counter_ns()
    os.close(fd)
    return (t1 - t0) / N / 1000.0


def t_open(path):
    path.write_bytes(b"")
    t0 = time.perf_counter_ns()
    for _ in range(N):
        with open(path, "ab") as f:
            f.write(BUF)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / N / 1000.0


variants = {
    "os_write_128B_x20000": lambda: t_os_write(),
    "open_ab_inside_x20000": lambda: t_open(inside_bin),
    "open_ab_outside_x20000": lambda: t_open(outside_bin),
}
samples = {k: [] for k in variants}
for _ in range(REPS):
    for name, fn in variants.items():
        samples[name].append(fn())

summary = {}
for name, vals in samples.items():
    summary[name] = {
        "median_us_per_call": round(statistics.median(vals), 4),
        "min_us": round(min(vals), 4),
        "max_us": round(max(vals), 4),
        "reps_us": [round(v, 4) for v in vals],
    }

print(json.dumps({
    "mode": mode,
    "guard_active": S.active(),
    "registry_entries_read": n_entries,
    "n_per_rep": N,
    "reps": REPS,
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "machine": platform.machine(),
    "variants": summary,
}, sort_keys=True))

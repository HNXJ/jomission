"""E1b batch (docs/laminar_motif/E1B_SEAL.md): the 27-run matrix, at most --jobs runs at once, one retry each.

  python -B scripts/e1b_batch.py --out results/laminar_e1b [--jobs 3]

A run whose receipt reads "complete" is skipped, so the batch resumes after an interruption. Each finished run
appends one line to batch.log in --out. Exit 0 when all 27 are complete, 1 otherwise.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = (6, 7, 8)
MATRIX = ([("A", r, s) for r in (0.25, 0.5, 1.0, 2.0, 4.0) for s in SEEDS]
          + [("B", r, s) for r in (0.25, 0.5, 2.0, 4.0) for s in SEEDS])
assert len(MATRIX) == 27


def tag(arm, rho, seed):
    return f"e1b{arm}_rho{rho:g}_seed{seed}"


def done(out, t):
    p = out / f"{t}.receipt.json"
    return p.exists() and json.loads(p.read_text(encoding="utf-8")).get("outcome") == "complete"


def log(out, msg):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out / "batch.log", "a", encoding="utf-8") as f:
        f.write(f"{stamp} {msg}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    queue = [(arm, r, s, 0) for arm, r, s in MATRIX if not done(out, tag(arm, r, s))]
    log(out, f"start: {len(queue)} of 27 to run, jobs {a.jobs}")
    running = []
    while queue or running:
        while queue and len(running) < a.jobs:
            arm, r, s, tries = queue.pop(0)
            t = tag(arm, r, s)
            cmd = [sys.executable, "-B", str(REPO / "scripts/e1b_run.py"), "--arm", arm, "--rho", f"{r:g}",
                   "--seed", str(s), "--out", str(out)]
            with open(out / f"{t}.stdout.txt", "a", encoding="utf-8") as fo:
                p = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=str(REPO))
            running.append((p, arm, r, s, tries))
            log(out, f"launch {t} try {tries + 1} pid {p.pid}")
        time.sleep(20)
        for item in list(running):
            p, arm, r, s, tries = item
            if p.poll() is None:
                continue
            running.remove(item)
            t = tag(arm, r, s)
            ok = p.returncode == 0 and done(out, t)
            log(out, f"end {t} rc {p.returncode} {'complete' if ok else 'FAILED'}")
            if not ok and tries == 0:
                queue.append((arm, r, s, 1))
    n = sum(done(out, tag(*m)) for m in MATRIX)
    log(out, f"finish: {n} of 27 complete")
    sys.exit(0 if n == 27 else 1)


if __name__ == "__main__":
    main()

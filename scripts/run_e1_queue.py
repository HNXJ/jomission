"""Run remaining E1 runs under suite_lock, one per lock hold (docs/laminar_motif/E1_SEAL.md).

Iterates through all 12 remaining (rho, seed) pairs, acquiring and releasing the machine's
suite_lock once per run so other workspaces are not starved. Updates e1_result.json after
each completed run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE_LOCK = Path.home() / ".claude" / "bin" / "suite_lock.py"
PYTHON = sys.executable

RHOS = (0.25, 0.5, 2.0, 4.0)
SEEDS = (1, 2, 3)


def main() -> int:
    out_dir = ROOT / "results" / "laminar_e1"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rho in RHOS:
        for s in SEEDS:
            rec_path = out_dir / f"e1_rho{rho:g}_seed{s}.json"
            if rec_path.exists():
                print(f"Skipping already completed {rec_path.name}")
                continue
            print(f"Queueing run rho={rho:g} seed={s} under suite_lock...")
            cmd = [
                PYTHON,
                str(SUITE_LOCK),
                "--label",
                f"jomission-e1-rho{rho:g}-s{s}",
                "--",
                PYTHON,
                str(ROOT / "scripts" / "run_e1_scored.py"),
                "run",
                str(rho),
                str(s),
            ]
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            if rc != 0:
                print(f"Run rho={rho:g} seed={s} failed with exit code {rc}; stopping.")
                return rc
            # Update score after each finished run
            subprocess.run([PYTHON, str(ROOT / "scripts" / "run_e1_scored.py"), "score"], cwd=ROOT)
            print(f"Finished rho={rho:g} seed={s}; released suite_lock.")

    # Final score
    subprocess.run([PYTHON, str(ROOT / "scripts" / "run_e1_scored.py"), "score"], cwd=ROOT)
    print("All E1 runs completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

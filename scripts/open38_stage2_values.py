"""OPEN-38 stage 2: per-run values for the closure receipt, read through the frozen stage-1 `run_checks`.

Reporting only: no gate, no threshold, no pooling and no interpretation (spec §6 analysis is separate). Writes one
record per arm-seed tag with its execution ID and prints the receipt table rows.

    python -B scripts/open38_stage2_values.py OUT_JSON [--dir results/laminar_e2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open38_stage1_check as s1  # noqa: E402
from open38_stage1_integrity import ARMS2, tag2  # noqa: E402


def main(out: str, d: Path) -> int:
    rows = {}
    for arm in ARMS2:
        for seed in s1.SEEDS:
            t = tag2(arm, seed)
            r = s1.run_checks(d, t)
            r["exec_id"] = json.loads((d / f"{t}_exec.json").read_text(encoding="utf-8")).get("exec_id")
            rows[t] = r
            rt = r["e_rates_hz"]
            print(f"| {arm} | {seed} | {r['r_ab']:.4f} | {r['r_gm']:.4f} | {r['r_proto']:.4f} | "
                  f"{str(r['crossing_2se']).lower()} | {str(r['pass']).lower()} | "
                  f"{' / '.join(f'{rt[u]:.2f}' for u in ('S', 'M', 'D'))} | {r['stop'] or 'none'} | {str(r['ok']).lower()} |")
    Path(out).write_text(json.dumps(rows, indent=1, default=str) + "\n", encoding="utf-8")
    shas = {r["git_sha"] for r in rows.values()}
    print(f"runs {len(rows)}; ok {sum(r['ok'] for r in rows.values())}; git_sha {sorted(shas)}; "
          f"dirty {sorted({str(r['git_dirty']) for r in rows.values()})}; unique exec_id {len({r['exec_id'] for r in rows.values()})}")
    return 0 if all(r["ok"] for r in rows.values()) and len(shas) == 1 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dir", default="results/laminar_e2")
    a = ap.parse_args()
    sys.exit(main(a.out, Path(a.dir)))

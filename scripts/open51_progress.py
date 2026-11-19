"""Snapshot an OPEN-51 batch_state.json into a committed, path-free progress file for the dashboard.

Usage: python -B open51_progress.py BATCH_STATE_JSON OUT_JSON
Keeps protocol, start/stop/finish times, the classification (empty until the batch ends) and, per step: step, run,
exit, outcome, counts_equal, json_equal, elapsed_s. Drops paths. Not a classification: the batch writes that.
"""
import json
import sys
from pathlib import Path

s = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
keep = ("step", "run", "exit", "outcome", "counts_equal", "json_equal", "elapsed_s")
out = {"protocol": s.get("protocol"), "environment": s.get("environment"), "started_utc": s.get("started_utc"),
       "finished_utc": s.get("finished_utc"), "stopped": s.get("stopped"),
       "r0c": {k: (s.get("r0") or {}).get("r0c", {}).get(k) for k in ("exit", "ynn_fusion_count", "dynamic_variables",
                                                                         "strict_hlo")},
       "steps": [{k: st.get(k) for k in keep} for st in s.get("steps", [])],
       "classification": s.get("classification", {})}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"{len(out['steps'])} steps; stopped {bool(out['stopped'])}; classified {len(out['classification'])}")

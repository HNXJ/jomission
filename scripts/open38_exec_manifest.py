"""OPEN-38 stage-2 execution manifest (owner 2026-11-17): one unique execution ID per run, bound to the run's outputs.

Written by the driver around the unchanged scientific chain (scripts/run_o38s_scored.sh); no scientific code reads it.

Usage:
  python scripts/open38_exec_manifest.py begin TAG --head SHA [--results-dir D]   prints the new execution ID
  python scripts/open38_exec_manifest.py end TAG RC [--results-dir D]

begin  refuses if {TAG}_exec.json exists (no reuse, no overwrite); writes tag, exec_id (uuid4 hex), head, host,
       driver pid, start time.
end    refuses unless the manifest is begun and not ended; adds rc, end time and the sha256 of every run output
       (OUTPUTS; null when absent), which binds the execution ID to those bytes.
check_manifest(d, tag) is the reader the integrity checker uses.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
import uuid
from pathlib import Path

SCHEMA = "o38s_exec/1"
OUTPUTS = (".json", ".npz", "_verdict.json", "_similarity.json", "_motif.json")
ID_RE = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "results" / "laminar_e2"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def path(d: Path, tag: str) -> Path:
    return d / f"{tag}_exec.json"


def begin(d: Path, tag: str, head: str) -> str:
    p = path(d, tag)
    if p.exists():
        raise SystemExit(f"refused: {p.name} exists")
    rec = {"schema": SCHEMA, "tag": tag, "exec_id": uuid.uuid4().hex, "head": head, "host": platform.node(),
           "driver_pid": os.getppid(), "start_utc": _now()}
    with p.open("x", encoding="utf-8") as f:  # exclusive create: a concurrent second begin fails
        json.dump(rec, f, indent=1)
    return rec["exec_id"]


def end(d: Path, tag: str, rc: int) -> dict:
    p = path(d, tag)
    if not p.is_file():
        raise SystemExit(f"refused: {p.name} not begun")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if "end_utc" in rec:
        raise SystemExit(f"refused: {p.name} already ended")
    rec.update(rc=rc, end_utc=_now(), outputs={s: _sha(d / f"{tag}{s}") for s in OUTPUTS})
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def check_manifest(d: Path, tag: str) -> tuple[str | None, list[str]]:
    """Return (exec_id, problems) for one run's manifest against the files on disk."""
    p = path(d, tag)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return None, [f"exec manifest: {type(e).__name__}: {e}"]
    probs = []
    eid = rec.get("exec_id")
    if rec.get("schema") != SCHEMA:
        probs.append(f"exec manifest: schema {rec.get('schema')!r}")
    if rec.get("tag") != tag:
        probs.append(f"exec manifest: tag {rec.get('tag')!r} != {tag!r}")
    if not (isinstance(eid, str) and ID_RE.match(eid)):
        probs.append(f"exec manifest: bad exec_id {eid!r}")
    if "end_utc" not in rec or rec.get("rc") != 0:
        probs.append(f"exec manifest: not ended with rc 0 (rc {rec.get('rc')!r})")
    for s in OUTPUTS:
        want, have = (rec.get("outputs") or {}).get(s), _sha(d / f"{tag}{s}")
        if want is None or want != have:
            probs.append(f"exec manifest: {tag}{s} hash {'absent' if want is None else 'differs'}")
    return eid, probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=("begin", "end"))
    ap.add_argument("tag")
    ap.add_argument("rc", nargs="?", type=int)
    ap.add_argument("--head")
    ap.add_argument("--results-dir", default=str(DEFAULT_DIR))
    a = ap.parse_args()
    d = Path(a.results_dir)
    if a.op == "begin":
        if not a.head:
            sys.exit("begin needs --head")
        print(begin(d, a.tag, a.head))
    else:
        if a.rc is None:
            sys.exit("end needs RC")
        end(d, a.tag, a.rc)

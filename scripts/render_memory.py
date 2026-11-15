"""Render MEMORY.md from the verified lessons store.

Deterministic. MEMORY.md is a compact orientation view for compaction injection:
the global autocompact-memory plugin carries it into checkpoint summaries. The
plugin's injection limit is not verified (no plugin config found, 2026-11-14),
so the view keeps only id, trigger and repair. Cause, evidence and scope stay in
manifests/harness/lessons.json, by id. Never store current state, secrets or
unverified claims here.

    python scripts/render_memory.py            # rewrite MEMORY.md in place
    python scripts/render_memory.py --check    # exit 1 if stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "MEMORY.md"
BUDGET = 4000  # compact view ceiling; the injection limit is unverified (see docstring)


def load(rel: str):
    return json.load(open(ROOT / rel, encoding="utf-8"))


def render() -> str:
    lessons = load("manifests/harness/lessons.json")["lessons"]
    out = ["# MEMORY.md (jomission) — compact orientation, verified lessons",
           "",
           "Full fields (cause, evidence, scope) by id: manifests/harness/lessons.json.",
           "Never store current state, secrets or unverified claims here.",
           "",
           "## Lessons",
           ""]
    out += [f"- {le['id']}: {le['trigger']} -> {le['repair']}" for le in lessons]
    text = "\n".join(out) + "\n"
    if len(text) > BUDGET:
        raise SystemExit(f"MEMORY view is {len(text)} chars, over BUDGET {BUDGET}: "
                         "shorten the view; never drop a lesson silently")
    return text


def is_current() -> bool:
    return MEMORY.is_file() and MEMORY.read_text(encoding="utf-8") == render()


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if is_current() else 1)
    MEMORY.write_text(render(), encoding="utf-8", newline="\n")

"""Render MEMORY.md from the verified lessons store.

Deterministic. MEMORY.md is a read-optimized view for compaction injection:
the global autocompact-memory plugin carries it verbatim into checkpoint
summaries. Canonical truth stays in manifests/; this file holds verified
reusable lessons only, never current state, secrets, or unverified claims.
Budget matches the plugin's project read budget (6000 chars).

    python scripts/render_memory.py            # rewrite MEMORY.md in place
    python scripts/render_memory.py --check    # exit 1 if stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "MEMORY.md"
BUDGET = 6000


def load(rel: str):
    return json.load(open(ROOT / rel, encoding="utf-8"))


def render() -> str:
    lessons = load("manifests/harness/lessons.json")["lessons"]
    out = ["# MEMORY.md (jomission) — verified reusable lessons only",
           "",
           "Scope: this project. Global lessons live in the global MEMORY.md.",
           "Never store current state, secrets, or unverified claims here.",
           "Canonical truth stays in manifests/; this file is a compaction-injection view.",
           "Format per lesson: {trigger, cause, repair, evidence, scope}.",
           "",
           "## Lessons",
           ""]
    for le in lessons:
        out.append(f"- {le['id']}: trigger: {le['trigger']} / cause: {le['cause']} / "
                   f"repair: {le['repair']} / evidence: {le['evidence']} / scope: {le['scope']}")
    text = "\n".join(out) + "\n"
    if len(text) > BUDGET:
        keep = [ln for ln in out if ln.startswith("- ")]
        while keep and len("\n".join(out[:9] + keep) + "\n") > BUDGET:
            keep.pop(0)
        text = "\n".join(out[:9] + keep) + "\n… (trimmed to budget; oldest lessons dropped first)\n"
    return text


def is_current() -> bool:
    return MEMORY.is_file() and MEMORY.read_text(encoding="utf-8") == render()


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if is_current() else 1)
    MEMORY.write_text(render(), encoding="utf-8", newline="\n")

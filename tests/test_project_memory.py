"""Tier 0: project memory is present, within injection budget, and current.

MEMORY.md is the compaction-injection view of manifests/harness/lessons.json
(the global autocompact-memory plugin carries it into checkpoint summaries).
Currency is mechanical: render the file and compare, exactly like the
handoff state block.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 6000


def test_project_memory_current_and_within_budget():
    mem = ROOT / "MEMORY.md"
    assert mem.is_file(), "MEMORY.md missing; run: python scripts/render_memory.py"
    text = mem.read_text(encoding="utf-8")
    assert len(text) <= BUDGET, f"MEMORY.md {len(text)} chars exceeds injection budget {BUDGET}"
    r = subprocess.run([sys.executable, "scripts/render_memory.py", "--check"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, "MEMORY.md stale; run: python scripts/render_memory.py"


def test_project_memory_covers_lessons():
    text = (ROOT / "MEMORY.md").read_text(encoding="utf-8")
    lessons = json.load(open(ROOT / "manifests" / "harness" / "lessons.json"))["lessons"]
    assert lessons, "lessons store empty"
    for le in lessons:
        assert le["id"] in text, f"lesson {le['id']} missing from MEMORY.md"

"""Checks for scripts/build_dashboard.py and the page it writes, docs/DASHBOARD.html."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "DASHBOARD.html"
CHECKLIST = ROOT / "checklist-score.md"
ROW = re.compile(r"^\| (\d+) \| (.*?) \| (.*?) \| ([\d.]+) \| (.*?) \|$", re.M)


def test_rebuild_is_byte_identical():
    before = PAGE.read_bytes()
    try:
        subprocess.run(
            [sys.executable, "scripts/build_dashboard.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        after = PAGE.read_bytes()
    finally:
        PAGE.write_bytes(before)
    assert after == before


def test_open_list_matches_checklist():
    rows = ROW.findall(CHECKLIST.read_text(encoding="utf-8"))
    assert len(rows) == 50, f"expected 50 checklist rows, parsed {len(rows)}"
    expected = sum(1 for row in rows if float(row[3]) < 2)
    html = PAGE.read_text(encoding="utf-8")
    assert html.count("<li>item ") == expected


def test_inventory_counts():
    expected = len(list((ROOT / "results").glob("*.json")))
    html = PAGE.read_text(encoding="utf-8")
    assert html.count("<tr><td>results/") == expected


def test_open_stack_rows_shown():
    n = sum(1 for line in (ROOT / "docs/OPEN.md").read_text(encoding="utf-8").splitlines() if line.startswith("| OPEN-"))
    html = PAGE.read_text(encoding="utf-8")
    assert n > 0 and html.count("<tr><td>OPEN-") == n

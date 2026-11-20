"""CLI contract of scripts/bench_compile_cache.py that needs no benchmark run: usage errors and the overwrite refusal
exit 2 before any work, and an existing receipt is left byte-identical."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_compile_cache.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


def test_no_arguments_is_a_usage_error():
    r = run()
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "usage" in r.stderr


def test_existing_output_is_refused_and_unchanged(tmp_path):
    out = tmp_path / "receipt.json"
    sentinel = b"sentinel: must survive\n"
    out.write_bytes(sentinel)
    assert out.read_bytes() == sentinel, "fixture: OUT must exist with sentinel bytes"
    r = run(str(out), "1")
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "REFUSING" in r.stderr
    assert out.read_bytes() == sentinel

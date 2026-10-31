"""Check the git-ignored local inputs against their sha256 in manifests/local_inputs.json.

    python scripts/check_local_inputs.py

Prints one line per input and one line per missing entry, then LOCAL_INPUTS_PASS or
LOCAL_INPUTS_FAIL. Exit 0 when every input exists and matches; exit 1 otherwise.
Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "local_inputs.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["inputs"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            print(f"MISSING {entry['id']} {entry['path']}")
            failures.append(entry["id"])
            continue
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            print(f"HASH_MISMATCH {entry['id']} {entry['path']} {actual}")
            failures.append(entry["id"])
            continue
        print(f"OK {entry['id']} {entry['path']}")
    for entry in manifest.get("missing", []):
        print(f"NOT_HELD {entry['id']}: {entry['status']}")
    if failures:
        print(f"LOCAL_INPUTS_FAIL: {', '.join(failures)}")
        return 1
    held_out = len(manifest.get("missing", []))
    print(f"LOCAL_INPUTS_PASS ({held_out} entries not held, listed above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

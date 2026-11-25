"""Classify every tracked JSON file by area, sealing status and parse state (checklist item 34 denominator).

Reads only git objects at HEAD and manifests/sealed_artifacts.json. Writes a TSV (one row per tracked JSON) and
a summary to stdout. It judges no file's quality: a row records presence, not sufficiency.

Usage: python3 scripts/classify_json_provenance.py [--tsv PATH]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_KEYS = ("script", "generator", "runner", "module", "source_code", "code_path")
HASH_KEYS = ("sha256", "hash", "digest", "blob")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def decode(raw: bytes) -> tuple[str, str]:
    """UTF-8 first; cp1252 for files the owner ruled sealed as cp1252 (2026-11-03). Otherwise 'undecodable'."""
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return "", "undecodable"


def area_of(path: str) -> str:
    parts = path.split("/")
    if parts[0] == "docs" and len(parts) > 1 and parts[1] == "receipts":
        return "docs/receipts"
    if parts[0] in ("results", "manifests", "tests", "scripts", "paper", "site-src", "jomission"):
        return parts[0]
    return "other"


def has_key(obj, needles: tuple[str, ...]) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(n in k.lower() for n in needles):
                return True
            if has_key(v, needles):
                return True
    elif isinstance(obj, list):
        return any(has_key(v, needles) for v in obj)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=None)
    a = ap.parse_args()

    registry = json.loads(git("show", "HEAD:manifests/sealed_artifacts.json"))
    sealed = {e["path"]: e for e in registry["artifacts"] if "path" in e}

    rows = []
    for path in git("ls-files", "*.json").split("\n"):
        if not path:
            continue
        blob = git("rev-parse", f"HEAD:{path}").strip()
        text, enc = decode(git_bytes("show", f"HEAD:{path}"))
        try:
            obj = json.loads(text) if enc != "undecodable" else None
            parse = "ok" if obj is not None else "fail"
        except (json.JSONDecodeError, ValueError):
            obj = None
            parse = "fail"
        area = area_of(path)
        if path in sealed:
            status = "sealed"
            reg_ok = "match" if sealed[path]["blob"] == blob else "blob_differs"
        else:
            status = "unsealed"
            reg_ok = "n/a"
        gen = has_key(obj, GENERATOR_KEYS) if obj is not None else False
        hsh = has_key(obj, HASH_KEYS) if obj is not None else False
        rows.append((path, area, status, reg_ok, parse, "y" if gen else "n", "y" if hsh else "n", enc))

    header = ("path", "area", "registry", "blob_check", "parse", "names_generator", "has_hash", "encoding")
    text = "\t".join(header) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
    if a.tsv:
        a.tsv.write_text(text, encoding="utf-8")

    from collections import Counter

    print(f"tracked JSON: {len(rows)}")
    for key, idx in (("area", 1), ("registry", 2), ("parse", 4), ("blob_check", 3)):
        print(f"by {key}:", dict(Counter(r[idx] for r in rows)))
    print("area x registry:", dict(Counter((r[1], r[2]) for r in rows)))
    print("area x parse fail:", dict(Counter(r[1] for r in rows if r[4] == "fail")))
    print("area x names_generator (of parsed):",
          dict(Counter((r[1], r[5]) for r in rows if r[4] == "ok")))
    print("area x has_hash (of parsed):", dict(Counter((r[1], r[6]) for r in rows if r[4] == "ok")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

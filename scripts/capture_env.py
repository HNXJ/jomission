"""Record the environment a Jomission run used: interpreter, platform, package versions, git identity.

    python3 scripts/capture_env.py OUT.json

Stdlib only. Writes one JSON object to OUT.json (path relative to the working directory). A package not installed in
this interpreter is recorded as null, never guessed; nothing is installed. `git_dirty` is true when `git status
--porcelain` in this repo lists any change, untracked files included. Refuses to overwrite an existing OUT.json.
Exit 0 when written; 2 on bad usage or an existing OUT; 1 when git cannot read HEAD.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "jax_version": "jax",
    "jaxlib_version": "jaxlib",
    "numpy_version": "numpy",
    "scipy_version": "scipy",
    "jaxfne_version": "jaxfne",
}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def package_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python3 scripts/capture_env.py OUT.json", file=sys.stderr)
        return 2
    out = Path(args[0])
    if out.exists():
        print(f"REFUSING: {out} exists; nothing written", file=sys.stderr)
        return 2

    try:
        sha = git("rev-parse", "HEAD").strip()
    except subprocess.CalledProcessError as e:
        print(f"FAIL: git rev-parse HEAD in {ROOT}: {e.stderr.strip()}", file=sys.stderr)
        return 1

    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        **{key: package_version(dist) for key, dist in PACKAGES.items()},
        "git_sha": sha,
        "git_dirty": bool(git("status", "--porcelain").strip()),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:  # "x": refuse to overwrite even if the file appeared since the check
        f.write(json.dumps(record, indent=1) + "\n")
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

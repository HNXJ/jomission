"""Run tier 0 against a clone's OWN source in an isolated venv, so the test proves the clone was exercised.

Why this exists: `project_check.py` run inside a clone reports one failure, because the editable install in the
ambient environment points at the original checkout, so `import jomission` resolves outside the clone under test
(`tests/test_environment_provenance.py::test_jomission_import_resolves_to_this_checkout`). That failure is honest,
but it also means the other 257 results in that run do not prove the clone's files were the ones imported. This
script removes the ambiguity: it builds a venv that can see jax/jaxfne/numpy from the system but has no jomission of
its own, installs the clone editable into it, asserts the import resolves INSIDE the clone, then runs tier 0 with
that interpreter. Zero failures is then a real receipt for the clone.

    python3 scripts/verify_clone.py [--clone DIR] [--venv DIR] [--keep]

Default clone: a fresh `git clone --shared` of HEAD at _temp_verify_clone/ next to the repo, at the current HEAD.
Exit 0 only when the import resolves inside the clone and tier 0 reports no failures.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = (
    "from pathlib import Path; import jomission; "
    "p = Path(jomission.__file__).resolve(); c = Path({clone!r}).resolve(); "
    "print('RESOLVED', p); print('INSIDE_CLONE', p.is_relative_to(c))"
)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-6:]
    for line in tail:
        print(f"  {line}", flush=True)
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clone", type=Path, default=ROOT.parent / "_temp_verify_clone")
    ap.add_argument("--venv", type=Path, default=Path("/tmp/jomission_verify_venv"))
    ap.add_argument("--keep", action="store_true", help="keep the clone and venv")
    a = ap.parse_args(argv)

    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()[:7]
    if a.clone.exists():
        shutil.rmtree(a.clone)
    print(f"clone {ROOT} @ {sha} -> {a.clone}")
    if subprocess.run(["git", "clone", "-q", "--shared", str(ROOT), str(a.clone)]).returncode != 0:
        print("FAIL: clone")
        return 1

    if a.venv.exists():
        shutil.rmtree(a.venv)
    # --system-site-packages makes jax, jaxfne, numpy and pytest visible without installing jomission anywhere
    if run([sys.executable, "-m", "venv", "--system-site-packages", str(a.venv)]).returncode != 0:
        print("FAIL: venv")
        return 1
    py = a.venv / "bin" / "python"

    inst = run([str(py), "-m", "pip", "install", "--no-deps", "-e", str(a.clone)])
    if inst.returncode != 0:
        print("FAIL: pip install -e clone")
        return 1

    probe = run([str(py), "-c", PROBE.format(clone=str(a.clone.resolve()))], cwd="/tmp")
    if "INSIDE_CLONE True" not in probe.stdout:
        print("FAIL: the clone's own jomission is not what imports; tier 0 would not be testing the clone")
        return 1

    tests = json.loads((a.clone / "manifests/test_tiers.json").read_text())["tiers"]["0"]["tests"]
    t0 = run([str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=str(a.clone))
    summary = [l for l in (t0.stdout or "").splitlines() if "passed" in l or "failed" in l]

    print("\n--- receipt ---")
    print(f"clone HEAD          : {sha}")
    print(f"import inside clone : True")
    print(f"tier 0              : {summary[-1] if summary else '(no summary)'}")

    if not a.keep:
        shutil.rmtree(a.clone, ignore_errors=True)
        shutil.rmtree(a.venv, ignore_errors=True)
    ok = t0.returncode == 0
    print(f"VERIFY_CLONE_{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

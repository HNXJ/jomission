"""Run tier 0 against a clone's OWN source in an isolated venv, so the test proves the clone was exercised.

Why this exists: `project_check.py` run inside a clone reports one failure, because the editable install in the
ambient environment points at the original checkout, so `import jomission` resolves outside the clone under test
(`tests/test_environment_provenance.py::test_jomission_import_resolves_to_this_checkout`). That failure is honest,
but it also means the other 257 results in that run do not prove the clone's files were the ones imported. This
script removes the ambiguity: it builds a venv that can see jax/jaxfne/numpy from the system but has no jomission of
its own, installs the clone editable into it, asserts the import resolves INSIDE the clone, then runs tier 0 with
that interpreter. Zero failures is then a real receipt for the clone.

    python3 scripts/verify_clone.py [--clone DIR] [--venv DIR] [--keep] [--offline]

Default clone and venv: under the system temp directory (`tempfile.gettempdir()`), not beside the other repos.
The script refuses to delete a non-empty directory it did not create, marking its own with
`.verify_clone_target`. `--offline` installs with `--no-build-isolation`, so no build requirement is fetched from
PyPI; without it, pip may need the network to resolve the build backend.
Exit 0 only when the import resolves inside the clone and tier 0 reports no failures.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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


def _check_target(path: Path, what: str) -> bool:
    """Refuse to delete a directory that is not one of this script's own outputs.

    v1 (OPEN-45): `shutil.rmtree` on a caller-supplied path with no guard could destroy any directory, including a
    checkout. A target is acceptable only if it is empty, absent, or carries this script's marker file.
    """
    marker = path / ".verify_clone_target"
    if not path.exists():
        return True
    if not path.is_dir():
        print(f"REFUSING: --{what} {path} exists and is not a directory")
        return False
    if not any(path.iterdir()):
        return True
    if marker.exists():
        return True
    print(f"REFUSING: --{what} {path} is a non-empty directory not created by this script "
          f"(no {marker.name} inside). Nothing was deleted.")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clone", type=Path, default=Path(tempfile.gettempdir()) / "jomission_verify_clone")
    ap.add_argument("--venv", type=Path, default=Path(tempfile.gettempdir()) / "jomission_verify_venv")
    ap.add_argument("--keep", action="store_true", help="keep the clone and venv")
    ap.add_argument("--offline", action="store_true",
                    help="install with --no-build-isolation so no build requirement is fetched from the network")
    a = ap.parse_args(argv)

    # v1: never delete a directory this script did not create
    if not _check_target(a.clone, "clone") or not _check_target(a.venv, "venv"):
        return 2

    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()[:7]
    if a.clone.exists():
        shutil.rmtree(a.clone)
    a.clone.mkdir(parents=True, exist_ok=True)
    (a.clone / ".verify_clone_target").write_text("created by scripts/verify_clone.py\n", encoding="utf-8")
    print(f"clone {ROOT} @ {sha} -> {a.clone}")
    if subprocess.run(["git", "clone", "-q", "--shared", str(ROOT), str(a.clone / "repo")]).returncode != 0:
        print("FAIL: clone")
        return 1
    repo = a.clone / "repo"

    if a.venv.exists():
        shutil.rmtree(a.venv)
    # --system-site-packages makes jax, jaxfne, numpy and pytest visible without installing jomission anywhere
    if run([sys.executable, "-m", "venv", "--system-site-packages", str(a.venv)]).returncode != 0:
        print("FAIL: venv")
        return 1
    py = a.venv / "bin" / "python"

    # v3: --offline skips build isolation so nothing is fetched from PyPI
    inst_cmd = [str(py), "-m", "pip", "install", "--no-deps", "-e", str(repo)]
    if a.offline:
        inst_cmd.insert(5, "--no-build-isolation")
    inst = run(inst_cmd)
    if inst.returncode != 0:
        print("FAIL: pip install -e clone" + (" (offline)" if a.offline else ""))
        return 1

    probe = run([str(py), "-c", PROBE.format(clone=str(repo.resolve()))], cwd="/tmp")
    if "INSIDE_CLONE True" not in probe.stdout:
        print("FAIL: the clone's own jomission is not what imports; tier 0 would not be testing the clone")
        return 1

    tests = json.loads((repo / "manifests/test_tiers.json").read_text())["tiers"]["0"]["tests"]
    t0 = run([str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=str(repo))
    summary = [l for l in (t0.stdout or "").splitlines() if "passed" in l or "failed" in l]

    print("\n--- receipt ---")
    print(f"clone HEAD          : {sha}")
    print(f"import inside clone : True")
    print(f"tier 0              : {summary[-1] if summary else '(no summary)'}")

    if not a.keep:
        shutil.rmtree(a.clone, ignore_errors=True)
        shutil.rmtree(a.venv, ignore_errors=True)
    ok = t0.returncode == 0

    # v4: write a tracked receipt under docs/receipts/, so the result does not live only in a commit message or a
    # chat log. Not under results/: every tracked results/*.json must be a registered scientific result
    # (tests/test_harness_guards.py::test_results_coverage_registered), and this is a harness receipt.
    receipt = ROOT / "docs" / "receipts" / "verify_clone.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "kind": "verify_clone_receipt",
        "sha": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip(),
        "import_inside_clone": True,
        "tier0_summary": summary[-1] if summary else None,
        "tier0_returncode": t0.returncode,
        "tests_run": len(tests),
        "offline": bool(a.offline),
        "verdict": "VERIFY_CLONE_PASS" if ok else "VERIFY_CLONE_FAIL",
    }, indent=1) + "\n", encoding="utf-8")
    print(f"receipt: {receipt.relative_to(ROOT)}")

    print(f"VERIFY_CLONE_{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

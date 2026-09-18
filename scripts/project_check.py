"""Pre-work check for a fresh agent. No simulation.

    python scripts/project_check.py [--allow-dirty] [--offline] [--no-tests]

Prints one line per check and ends with PROJECT_CHECK_PASS or
PROJECT_CHECK_FAIL: <first failing check>: <detail>.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SSH_REMOTE = "git@github.com:HNXJ/jomission.git"


def git(*a):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true", help="do not fail on uncommitted changes")
    ap.add_argument("--offline", action="store_true", help="skip remote comparison")
    ap.add_argument("--no-tests", action="store_true", help="skip the tier-0 pytest subset")
    args = ap.parse_args()

    from jomission.harness import validate as v

    state, todo = v.load("manifests/current_state.json"), v.load("manifests/todo.json")
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(f"{name}: {detail}")

    # workspace identity
    _, top, _ = git("rev-parse", "--show-toplevel")
    check("workspace", Path(top).resolve() == ROOT, f"git toplevel {top}")
    _, branch, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    _, head, _ = git("rev-parse", "--short=7", "HEAD")
    print(f"      branch {branch} @ {head}")
    _, dirty, _ = git("status", "--porcelain", "--untracked-files=no")
    check("tree clean", args.allow_dirty or not dirty,
          f"{len(dirty.splitlines())} modified tracked files" + (" (allowed)" if args.allow_dirty and dirty else ""))
    main_head = state["scientific_head"]["main"]
    check("scientific head ancestry", v.is_ancestor(main_head, "HEAD"), f"main {main_head} in HEAD history")
    if not args.offline:
        rc, out, err = git("ls-remote", SSH_REMOTE, "refs/heads/main", f"refs/heads/{branch}")
        remote = dict(reversed(line.split("\t")) for line in out.splitlines()) if rc == 0 else {}
        rmain = remote.get("refs/heads/main", "")
        if rc != 0 or not rmain:
            check("remote main", False, f"ls-remote failed: {err[:120]}")
        elif not v.commit_exists(rmain):
            check("remote main", False, f"origin main {rmain[:7]} not in local objects; git fetch {SSH_REMOTE} main")
        else:
            check("remote main", v.is_ancestor(main_head, rmain), f"origin main {rmain[:7]} contains {main_head}")
        rbr = remote.get(f"refs/heads/{branch}")
        _, full, _ = git("rev-parse", "HEAD")
        print(f"      origin {branch}: {rbr[:7] if rbr else 'absent'}"
              + ("" if rbr == full else " (local differs: unpushed or behind)"))

    # execution identity
    want = state["jaxfne_version"]
    got = version("jaxfne")
    check("jaxfne identity", got == want, f"installed {got}, execution authority {want}")
    env = {k: val for k, val in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run([sys.executable, "-c", "import jomission; print(jomission.__file__)"],
                           cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    where = Path(r.stdout.strip().splitlines()[-1]).resolve() if r.returncode == 0 and r.stdout.strip() else None
    check("editable install", where is not None and where.is_relative_to(ROOT),
          f"jomission -> {where}" if where else r.stderr[-200:]
          + f"; fix: {sys.executable} -m pip install --no-deps -e {ROOT}")

    # manifests
    for name, errs in (("todo", v.validate_todo(todo)),
                       ("current_state", v.validate_current_state(state, todo)),
                       ("sealed artifacts", v.validate_sealed_registry(v.load("manifests/sealed_artifacts.json"))),
                       ("lineage registry", [e for rec in v.load("manifests/lineage_registry.json")["lineages"]
                                             for e in v.validate_lineage(rec)]),
                       ("skills", v.validate_skills())):
        check(name, not errs, "; ".join(errs[:3]) + (f" (+{len(errs) - 3})" if len(errs) > 3 else ""))

    # sealed-write guard: cheap, no mutation. A bare full-suite pytest must not be able to
    # overwrite registered sealed evidence, so check the guard denies a sealed path and that
    # tests/conftest.py installs it for every session.
    from jomission.harness import seals

    sealed_present = [r for r in sorted(seals.sealed_paths()) if (ROOT / r).exists()]
    guard_ok, guard_detail = False, "no sealed artifact present to probe"
    if sealed_present:
        probe = sealed_present[0]
        before = (ROOT / probe).read_bytes()
        try:
            seals.check_write(ROOT / probe)
            guard_detail = f"{probe} is not denied by jomission.harness.seals.check_write"
        except seals.SealedArtifactWriteError:
            installed = "seals.install()" in (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
            guard_ok = installed and (ROOT / probe).read_bytes() == before
            guard_detail = (f"denies {probe}; installed by tests/conftest.py" if guard_ok
                            else "tests/conftest.py does not call seals.install()")
    check("sealed-write guard", guard_ok, guard_detail)

    print(f"      next authorized task: {state['next_authorized_task']}")
    print(f"      current gate: {state['current_gate']['id']} = {state['current_gate']['status']}")

    if not args.no_tests:
        tier0 = v.load("manifests/test_tiers.json")["tiers"]["0"]["tests"]
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tier0],
                           cwd=ROOT, capture_output=True, text=True, timeout=900)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
        check("tier-0 tests", r.returncode == 0, tail)

    if failures:
        print(f"PROJECT_CHECK_FAIL: {failures[0]}")
        return 1
    print("PROJECT_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

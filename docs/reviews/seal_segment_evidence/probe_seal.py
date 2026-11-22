"""Probe the seal write guard (jomission/harness/seals.py) at commit 83b83a3, read-only on the clone.

Each case runs in its own child process so the global guard state (builtins.open patch) is fresh.
Each case gets its own temp root under measure/roots/<case> with a temp registry; seals.ROOT and
seals.REGISTRY are rebound to that root before install(). The clone is only read.

Run:  <venv>/bin/python probe_seal.py            (all cases, writes seal_probe_results.json)
      <venv>/bin/python probe_seal.py <case>     (one case, prints one JSON line)
"""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import stat
import sys
from pathlib import Path

sys.dont_write_bytecode = True

CLONE = Path(__file__).resolve().parents[3]
MEAS = Path(tempfile.gettempdir()) / "seal_probe"
ENV = "JOMISSION_ALLOW_SEAL_WRITE"
ORIG = b"SEALED-ORIGINAL-BYTES-0123456789"
NEW = b"OVERWRITTEN-BY-PROBE"

# case -> (registry entries, sealed path relative to root, path whose bytes are observed, install?)
CASES = [
    "builtin_open_w_positive",
    "builtin_open_w_not_installed_control",
    "builtin_open_unsealed_negative",
    "env_authorized_control",
    "io_fileio_w",
    "dir_rename_parent",
    "os_replace_onto_sealed",
    "os_open_o_trunc",
    "fd_opened_before_guard",
    "path_touch_sealed",
    "os_unlink_sealed",
    "path_write_text",
    "symlink_alias_to_sealed",
    "registered_name_is_symlink",
    "registry_dot_form",
    "registry_absolute_form",
    "atomic_write_json_onto_sealed",
    "persist_phase_snapshot_onto_registered",
    "append_trial_snapshot_onto_registered",
]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, CLONE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case_root(case):
    return (MEAS / "roots" / case).resolve()


def child(case):
    root = case_root(case)
    if root.exists():
        shutil.rmtree(root)
    (root / "manifests").mkdir(parents=True)
    (root / "results").mkdir()

    sealed_rel = "results/sealed.json"
    entries = [sealed_rel]
    observed = root / sealed_rel
    if case == "registry_dot_form":
        entries = ["./results/sealed.json"]
    elif case == "registry_absolute_form":
        entries = [str(root / sealed_rel)]
    elif case == "append_trial_snapshot_onto_registered":
        sealed_rel = "results/exposure_trials.jsonl"
        entries = [sealed_rel]
        observed = root / sealed_rel
    elif case == "persist_phase_snapshot_onto_registered":
        sealed_rel = "results/pre_snapshot.json"
        entries = [sealed_rel]
        observed = root / sealed_rel
    elif case == "registered_name_is_symlink":
        (root / "outside").mkdir()
        target = root / "outside" / "real.json"
        target.write_bytes(ORIG)
        link = root / "results" / "sealed.json"
        os.symlink(target, link)
        observed = target
    elif case == "symlink_alias_to_sealed":
        pass  # alias created below, after the sealed file

    sealed = root / sealed_rel
    if not (case == "registered_name_is_symlink"):
        sealed.write_bytes(ORIG)
    (root / "manifests" / "sealed_artifacts.json").write_text(
        json.dumps({"artifacts": [{"path": p} for p in entries]}), encoding="utf-8")
    if case == "symlink_alias_to_sealed":
        os.symlink(root / "results" / "sealed.json", root / "results" / "alias.json")

    S = load("seals", "jomission/harness/seals.py")
    A = load("atomic_save", "jomission/simulation/atomic_save.py")
    S.ROOT = root
    S.REGISTRY = root / "manifests" / "sealed_artifacts.json"

    install_first = case not in ("builtin_open_w_not_installed_control", "fd_opened_before_guard")
    if install_first:
        S.install()

    extra = {}
    raised, msg = None, None
    try:
        if case in ("builtin_open_w_positive", "builtin_open_w_not_installed_control",
                    "env_authorized_control"):
            with open(sealed, "w") as f:
                f.write("X")
        elif case == "builtin_open_unsealed_negative":
            new = root / "results" / "new_unsealed.json"
            with open(new, "w") as f:
                f.write("X")
            extra["new_file_created"] = new.exists()
        elif case == "io_fileio_w":
            f = io.FileIO(sealed, "w")
            f.write(b"X")
            f.close()
        elif case == "dir_rename_parent":
            os.rename(root / "results", root / "results_moved")
            observed = root / "results_moved" / "sealed.json"
            extra["registered_path_exists_after"] = sealed.exists()
        elif case == "os_replace_onto_sealed":
            tmp = root / "results" / "tmp_new.json"
            tmp.write_bytes(NEW)
            os.replace(tmp, sealed)
        elif case == "os_open_o_trunc":
            fd = os.open(sealed, os.O_WRONLY | os.O_TRUNC)
            extra["size_right_after_open"] = os.fstat(fd).st_size
            os.close(fd)
        elif case == "fd_opened_before_guard":
            fd = os.open(sealed, os.O_WRONLY)
            S.install()
            extra["guard_active_before_write"] = S.active()
            os.write(fd, NEW)
            os.close(fd)
        elif case == "path_touch_sealed":
            os.utime(sealed, (1_000_000_000, 1_000_000_000))
            from pathlib import Path as P
            P(sealed).touch()
            extra["mtime_changed"] = sealed.stat().st_mtime != 1_000_000_000
        elif case == "os_unlink_sealed":
            os.unlink(sealed)
        elif case == "path_write_text":
            Path(sealed).write_text("X", encoding="utf-8")
        elif case == "symlink_alias_to_sealed":
            with open(root / "results" / "alias.json", "w") as f:
                f.write("X")
        elif case == "registered_name_is_symlink":
            with open(root / "results" / "sealed.json", "w") as f:
                f.write("X")
            extra["write_went_to"] = str(Path(os.path.realpath(root / "results" / "sealed.json")).relative_to(root))
        elif case == "registry_dot_form":
            with open(root / "results" / "sealed.json", "w") as f:
                f.write("X")
        elif case == "registry_absolute_form":
            with open(root / "results" / "sealed.json", "w") as f:
                f.write("X")
        elif case == "atomic_write_json_onto_sealed":
            A.atomic_write_json(sealed, {"probe": 1})
            extra["mode_replaced_sealed"] = oct(stat.S_IMODE(sealed.stat().st_mode))
            newf = root / "results" / "fresh_atomic.json"
            A.atomic_write_json(newf, {"probe": 2})
            extra["mode_fresh_atomic_file"] = oct(stat.S_IMODE(newf.stat().st_mode))
        elif case == "persist_phase_snapshot_onto_registered":
            A.persist_phase_snapshot(root / "results", "pre", (0, 1), {"probe": 1})
        elif case == "append_trial_snapshot_onto_registered":
            A.append_trial_snapshot(root / "results", "exposure", 0, {"probe": 1})
        else:
            raise SystemExit(f"unknown case {case}")
    except Exception as e:  # noqa: BLE001
        raised, msg = type(e).__name__, str(e)[:90]

    if case == "os_unlink_sealed":
        present = observed.exists()
        extra["file_present_after"] = present
        unchanged = None
    elif case == "dir_rename_parent":
        unchanged = observed.exists() and observed.read_bytes() == ORIG
    else:
        if observed.exists():
            data = observed.read_bytes()
            unchanged = data == ORIG
            extra["size_after"] = len(data)
        else:
            unchanged = False
            extra["observed_exists"] = False

    if case == "append_trial_snapshot_onto_registered" and observed.exists():
        extra["size_after"] = observed.stat().st_size

    out = {
        "case": case,
        "guard_installed": S.active(),
        "raised": raised,
        "msg": msg,
        "sealed_bytes_unchanged": unchanged,
        **extra,
    }
    print(json.dumps(out, sort_keys=True))


def parent(cases):
    results = []
    for case in cases:
        env = dict(os.environ)
        env.pop(ENV, None)
        if case == "env_authorized_control":
            env[ENV] = "results/sealed.json"
        cp = subprocess.run([sys.executable, __file__, case], env=env,
                            capture_output=True, text=True, timeout=120)
        line = cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else ""
        if cp.returncode != 0 or not line.startswith("{"):
            results.append({"case": case, "error": cp.stderr[-400:], "rc": cp.returncode})
        else:
            results.append(json.loads(line))
    out = MEAS / "seal_probe_results.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    for r in results:
        print(json.dumps(r, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        parent(CASES)

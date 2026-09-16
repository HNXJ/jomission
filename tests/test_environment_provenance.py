"""Environment provenance: `import jomission` outside pytest resolves to this checkout.

pytest's `pythonpath = ["."]` masks the installed package, so the check runs in a
subprocess from a foreign cwd with PYTHONPATH cleared -- the same conditions as a
standalone analysis script. A stale editable install pointing at another clone
fails here instead of silently supplying old code to scientific runs.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_jomission_import_resolves_to_this_checkout():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory() as cwd:
        out = subprocess.run(
            [sys.executable, "-c", "import jomission; print(jomission.__file__)"],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    resolved = Path(out.stdout.strip().splitlines()[-1]).resolve()
    assert resolved.is_relative_to(ROOT), (
        f"jomission imports from {resolved}, not {ROOT}; "
        f"reinstall: {sys.executable} -m pip install --no-deps -e {ROOT}")


def test_jaxfne_matches_execution_authority():
    import json
    from importlib.metadata import version
    want = json.load(open(ROOT / "manifests" / "current_state.json"))["jaxfne_version"]
    assert version("jaxfne") == want, f"jaxfne {version('jaxfne')} != execution authority {want}"
    pin = f'"jaxfne=={want}"'
    assert pin in (ROOT / "pyproject.toml").read_text(), "pyproject pin differs from execution authority"

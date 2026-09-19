"""Tier 0: one canonical gate definition.

1. The historical profile reproduces every sealed V2.1/V2.1b `checks` dict exactly.
2. The prospective profile encodes amendment 1 (evaluated on the sealed low_sm2p0
   cell it fails VIP band and E ISI, the recorded SCI-V21-BACKGROUND blocker) and
   amendment 2 (active fraction per class, which that cell passes).
3. Outside the frozen executed batteries, any Python file naming these checks
   imports jomission.harness.gates instead of carrying its own thresholds.
"""

import json
import re
from pathlib import Path

from jomission.harness import gates

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["results/v21_low.json", "results/v21_mid.json", "results/v21_high.json"] + [
    f"results/v21b_{r}_{s}.json" for r in ("low", "mid", "high") for s in ("sm2p0", "sm1p0", "sm0p5")]


def test_historical_profile_reproduces_sealed_checks():
    for rel in CELLS:
        res = json.load(open(ROOT / rel))
        got = gates.evaluate("historical_v21_battery", gates.windows_from_v21_result(res), res["drift"])
        assert got == res["checks"], (rel, got, res["checks"])


def test_prospective_profile_on_low_sm2p0():
    res = json.load(open(ROOT / "results/v21b_low_sm2p0.json"))
    isi = json.load(open(ROOT / "results/v21b_isi_classes.json"))
    windows = []
    for r, cv, s, cls in zip(res["rates"], res["cv"], res["sync"], isi["per_window_class"]):
        windows.append({"rates": r, "rate_cv": {"E": cv["E_rate_CV"]}, "sync": s,
                        "isi_fraction": {c: cls[c]["frac_in"] for c in ("E", "PV", "SST", "VIP")},
                        "active_fraction": {c: cls[c]["cv_evaluable"] / cls[c]["n_cells"]
                                            for c in ("E", "PV", "SST", "VIP")}})
    got = gates.evaluate("prospective_v2", windows, res["drift"])
    assert got["class_rate_bands"] is False  # VIP ~130 Hz > 80
    assert got["isi_cv_by_class"] is False   # E 0.683-0.750, VIP 0.0
    assert got["active_fraction_by_class"] is True  # amendment 2: min 31/32 SST (window 3)
    spec = gates.load_gate()["profiles"]["prospective_v2"]["checks"]["class_rate_bands"]["rate_hz"]
    assert spec == {"E": [2.0, 30.0], "PV": [1.0, 80.0], "SST": [1.0, 80.0], "VIP": [1.0, 80.0]}


def test_no_duplicate_gate_implementations():
    gate = gates.load_gate()
    frozen = set(gate["frozen_literal_implementations"]) | {"tests/test_gates_single_source.py"}
    names = set()
    for prof in gate["profiles"].values():
        names |= set(prof["checks"])
    names = {n for n in names if "_" in n}  # "sync"/"drift" are also ordinary metric keys
    pat = re.compile(r"""["'](%s)["']""" % "|".join(sorted(map(re.escape, names))))
    offenders = []
    for base in ("jomission", "tests", "scripts"):
        for f in (ROOT / base).rglob("*.py"):
            rel = f.relative_to(ROOT).as_posix()
            if rel in frozen or rel == "jomission/harness/gates.py":
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            if pat.search(text) and "jomission.harness" not in text:
                offenders.append(rel)
    assert offenders == [], offenders


def _tracked_py():
    import subprocess
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    return [p for p in out if (ROOT / p).is_file()]


def test_operation_batteries_use_canonical_estimators():
    """A non-frozen V2 operation battery computes windows only through jomission.harness.operation."""
    frozen = set(gates.load_gate()["frozen_literal_implementations"])
    uses = re.compile(r"from jomission\.harness import [^\n]*\boperation\b"
                      r"|from jomission\.harness\.operation import|import jomission\.harness\.operation")
    batteries = [p for p in _tracked_py() if re.fullmatch(r"tests/test_v2\w*operation\w*\.py", p) and p not in frozen]
    offenders = [p for p in batteries if not uses.search((ROOT / p).read_text(encoding="utf-8"))]
    assert offenders == [], offenders
    assert uses.search("from jomission.harness import gates, operation") and not uses.search("from jomission.harness import gates")


HISTORICAL_PROFILE_USERS = {
    "tests/test_gates_single_source.py",
    "tests/test_v2_operation_prospective.py",
    # Renders sealed V2.1 results, whose checks ARE the frozen semantics. It reproduces
    # them for display and evaluates no new execution, which is what this guard protects.
    "jomission/visualization/regime_atlas.py",
}


def test_historical_profile_is_reproduction_only():
    """Only the reproduction tests may evaluate the frozen V2.1 semantics; new work uses prospective_v2."""
    offenders = [p for p in _tracked_py() if p not in HISTORICAL_PROFILE_USERS
                 and "historical_v21_battery" in (ROOT / p).read_text(encoding="utf-8", errors="ignore")]
    assert offenders == [], offenders

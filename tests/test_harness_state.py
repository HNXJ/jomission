"""Tier 0: current-state and TODO manifests are valid and mutually consistent."""

import copy

from jomission.harness import validate as v


def test_current_state_and_todo_valid():
    state, todo = v.load("manifests/current_state.json"), v.load("manifests/todo.json")
    assert v.validate_todo(todo) == []
    assert v.validate_current_state(state, todo) == []


def test_validators_reject_defects():
    state, todo = v.load("manifests/current_state.json"), v.load("manifests/todo.json")
    bad_todo = copy.deepcopy(todo)
    bad_todo["items"][0]["status"] = "MAYBE"
    del bad_todo["items"][1]["acceptance"]
    errs = v.validate_todo(bad_todo)
    assert any("bad status" in e for e in errs) and any("missing acceptance" in e for e in errs)

    bad = copy.deepcopy(state)
    bad["latest_verdict"]["V2.1"]["status"] = "PROBABLY_FINE"
    bad["retired_axes"].append("NOT_DEFINED_AXIS")
    bad["next_authorized_task"] = "V2.2-RECURRENCE"
    errs = v.validate_current_state(bad, todo)
    assert any("not in vocabulary" in e for e in errs)
    assert any("NOT_DEFINED_AXIS" in e for e in errs)
    assert any("not an OPEN TODO item" in e for e in errs)


def test_preserved_science_todo():
    # Durable invariants only; current statuses live in the manifests, not in this test.
    items = {i["id"]: i for i in v.load("manifests/todo.json")["items"]}
    live = {"OPEN", "OPEN_AFTER_HARNESS", "BLOCKED", "DONE"}
    for k in ("SCI-V21C-BOUNDARY", "SCI-V21-GATE-REPAIR", "SCI-V21-BACKGROUND"):
        assert items[k]["status"] in live, k
    assert all(items[k]["status"] == "LOCKED" for k in ("V2.2-RECURRENCE", "BLIND-OMISSION"))


def test_verdict_status_follows_evidence_not_tests():
    # Built on copies of a sealed gate entry (V2.1b); never on the current hold state.
    state, todo = v.load("manifests/current_state.json"), v.load("manifests/todo.json")
    bad = copy.deepcopy(state)
    bad["latest_verdict"]["V2.1b"]["status"] = "PASS"  # e.g. copied from a green pytest run
    assert any("contradicts evidence verdict V2_LOCAL_OPERATION_FAIL" in e for e in v.validate_current_state(bad, todo))
    held = copy.deepcopy(state)
    held["latest_verdict"]["V2.1b"]["status"] = "UNRESOLVED"
    held["latest_verdict"]["V2.1b"]["review_override"] = {
        "evidence_reads": "V2_LOCAL_OPERATION_FAIL", "held_status": "UNRESOLVED", "reason": "probe",
        "authority": "probe", "lift": "probe"}
    assert not [e for e in v.validate_current_state(held, todo) if "V2.1b" in e]
    wrong = copy.deepcopy(held)
    wrong["latest_verdict"]["V2.1b"]["review_override"]["authority"] = ""
    assert any("review_override missing" in e for e in v.validate_current_state(wrong, todo))
    wrong = copy.deepcopy(held)
    wrong["latest_verdict"]["V2.1b"]["review_override"]["evidence_reads"] = "V2_LOCAL_OPERATION_PASS"
    assert any("!= evidence verdict" in e for e in v.validate_current_state(wrong, todo))


def test_test_tiers_consistent():
    import re
    from pathlib import Path
    root = Path(v.ROOT)
    tiers = v.load("manifests/test_tiers.json")["tiers"]
    t3 = set(tiers["3"]["tests"])
    t1 = {x.split("::")[0] for xs in tiers["1"]["by_subsystem"].values() if isinstance(xs, list) for x in xs}
    for t in set(tiers["0"]["tests"]) | t1 | t3:
        assert (root / t).exists(), t
    assert not t3 & set(tiers["0"]["tests"])
    writes = re.compile(r"json\.dump|np\.savez|open\([^)]*[\"']w[\"']")
    for f in sorted((root / "tests").glob("test_*.py")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(root).as_posix()
        if re.search(r"[\"']results[/\"']|RESULTS", text) and writes.search(text) and rel not in ("tests/test_harness_guards.py", "tests/test_seal_write_guard.py"):
            assert rel in t3, f"{rel} writes results but is not tier 3"


def test_handoff_state_block_current():
    from jomission.harness import handoff
    assert handoff.is_current(), "run: python -m jomission.harness.handoff"

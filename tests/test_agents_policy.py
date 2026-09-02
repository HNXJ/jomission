"""Deterministic harness test for Jomission Agent Reliability Policy (AGENTS.md)."""

import pathlib
import pytest

REQUIRED_HEADINGS = [
    "Optimization & Epistemic Discipline",
    "Action & Scope Discipline",
    "Execution Grammar: W = P(RG)^N S",
    "Context Discipline",
    "Harness Adaptation & Maintenance",
    "Communication & Delivery",
    "Review & Evidence Discipline",
]

CRITICAL_RULES = [
    "Priority: correctness > evidence > clarity > speed",
    "claim ∈ {observed, derived, inferred, assumed, unknown}",
    "execution ≠ verification",
    "PASS requires observed empirical receipts matching claim scope",
    "H1 External review is hypothesis generation, not authority",
    "H2 Hard-gate claims require receipts",
    "H3 Reconcile arithmetic before Seal",
    "H4 Serialization is an epistemic boundary",
]


def test_agents_policy_exists():
    root = pathlib.Path(__file__).resolve().parent.parent
    agents_file = root / "AGENTS.md"
    assert agents_file.is_file(), f"AGENTS.md must exist at repository root ({agents_file})"
    assert agents_file.stat().st_size > 500, "AGENTS.md must not be empty or truncated"


def test_agents_policy_headings_and_rules():
    root = pathlib.Path(__file__).resolve().parent.parent
    agents_file = root / "AGENTS.md"
    content = agents_file.read_text(encoding="utf-8")

    # Verify header
    assert "Jomission Agent Reliability Policy" in content, "Missing title in AGENTS.md"
    assert "Authority: Project-level harness policy" in content, "Missing authority statement"

    # Verify required headings
    for heading in REQUIRED_HEADINGS:
        assert f"## {heading}" in content, f"Missing required heading: ## {heading}"

    # Verify critical rules
    for rule in CRITICAL_RULES:
        assert rule in content, f"Missing critical rule text: {rule}"

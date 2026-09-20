"""Deterministic harness test for Jomission Agent Reliability Policy (AGENTS.md).

AGENTS.md is a thin router: project rules plus the owner's Agent Operating
Contract and the project evidence rules. The generic block that duplicated the
global CLAUDE.md was retired with checkpoint closure (2026-09-19); this test
asserts durable content, not current statuses.
"""

import pathlib
import pytest

PROJECT_RULE_KEYS = [
    "python scripts/project_check.py",
    "next_authorized_task",
    "Seal before running",
    "Do not retune after results",
    "Evidence class",
    "sealed_artifacts.json",
    "Omission firewall",
    "Visualization contract",
]

CONTRACT_HEADINGS = [
    "## 0. AIM",
    "## 1. MODEL",
    "## 2. TRUTH + ACTION",
    "## 3. PRGS",
    "## 4. SIMPLICITY + CAPABILITY",
    "## 5. HARNESS",
    "## 6. NEVER",
    "## 7. OUTPUT",
]

CRITICAL_RULES = [
    "correctness > evidence > clarity > speed",
    "Classify material claims: observed | derived | inferred | assumed | unknown",
    "execution != verification",
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

    # Verify project rules
    for key in PROJECT_RULE_KEYS:
        assert key in content, f"Missing project rule: {key}"

    # Verify the owner operating contract sections
    for heading in CONTRACT_HEADINGS:
        assert heading in content, f"Missing contract heading: {heading}"

    # Verify critical rules
    for rule in CRITICAL_RULES:
        assert rule in content, f"Missing critical rule text: {rule}"

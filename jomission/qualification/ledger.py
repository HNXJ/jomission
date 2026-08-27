"""Modification/Causal Ledger — append-only JSONL with hash chain.

Spec: scratch/gen2_qualification_skeleton_A.json modification_causal_ledger +
      scratch/gen2_qualification_skeleton_A.md §5 +
      scratch/gen2_parent_integration_v0.1.md §7

File: manifests/gen2_modification_ledger.jsonl (append-only, write-once per entry)
Invariants:
  - reason_before_results precedes observed_generic_effect (temporal)
  - parent_model hash before change
  - append-only never mutate prior line
  - evidence_state monotonic SPECIFIED → IMPLEMENTED → TESTED → OBSERVED

Topologically immutable: one principal modification per Ledger edge G0→G1→...
Ledger entry types: scaffold, config, RF, observability — each needs B-gate target.

Post-B12 extension (§5.2 b12_extension) appends T1_T7_delta etc. to same change_id
without overwriting pre-B12 fields (extend_ledger_after_B12).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Ledger file is fixed per spec
LEDGER_PATH = Path(__file__).resolve().parents[2] / "manifests" / "gen2_modification_ledger.jsonl"
LEDGER_JSON_PATH = Path(__file__).resolve().parents[2] / "manifests" / "gen2_modification_ledger.json"

Provenance = Literal["LITERATURE_PRIOR", "MODEL_ASSUMPTION", "DERIVED", "ENGINE_DEFAULT", "EMPIRICAL_FIT"]
EvidenceState = Literal["SPECIFIED", "IMPLEMENTED", "TESTED", "OBSERVED"]

VALID_EVIDENCE_STATES: tuple[EvidenceState, ...] = ("SPECIFIED", "IMPLEMENTED", "TESTED", "OBSERVED")
VALID_PROVENANCE: tuple[Provenance, ...] = (
    "LITERATURE_PRIOR",
    "MODEL_ASSUMPTION",
    "DERIVED",
    "ENGINE_DEFAULT",
    "EMPIRICAL_FIT",
)
EVIDENCE_ORDER = {s: i for i, s in enumerate(VALID_EVIDENCE_STATES)}


@dataclass
class ParentModel:
    config_hash: str
    hp_hash: str
    rf_hash: str | None = None
    dt_ms: float = 0.1
    numerical_config_hash: str | None = None


@dataclass
class ResultingModel:
    config_hash: str
    hp_hash: str
    rf_hash: str | None = None
    dt_ms: float = 0.1
    numerical_config_hash: str | None = None
    artifact: str | None = None


@dataclass
class LedgerDelta:
    factor: float | None = None
    absolute: float | str | None = None
    description: str | None = None


@dataclass
class LedgerEntry:
    """One immutable Ledger record matching A §5.2 required_fields.

    Required pre-B12 fields:
      change_id, parent_model, parameter_path, pathway, old_value, new_value,
      delta, reason_before_results, provenance, qualification_target,
      predicted_generic_effect, observed_generic_effect, unexpected_effects,
      tests_passed, tests_failed, resulting_model, evidence_state,
      timestamp, author, gate (+ result artifact hash)

    Post-B12 (via extend_ledger_after_B12):
      b12_extension {T1_T7_delta, Q1_Q15_delta, polarity_transitions, unchanged, attribution, impossible_remains}
    """

    change_id: str  # GEN2_C000, GEN2_C001, ...
    parent_model: dict[str, Any]  # {config_hash, hp_hash, rf_hash, dt_ms, ...}
    parameter_path: str  # e.g. "network.builder.within_gain" or "scaffold" for C000
    pathway: str  # mechanistic pathway, e.g. "H→b_eff" or "ledger.init"
    old_value: Any
    new_value: Any
    delta: dict[str, Any]  # {factor, absolute, description}
    reason_before_results: str  # must precede observed_generic_effect temporally
    provenance: Provenance
    qualification_target: str  # e.g. "B0 provenance audit" or "scaffold"
    predicted_generic_effect: str | dict[str, Any]
    observed_generic_effect: str | dict[str, Any] | None
    unexpected_effects: list[str] = field(default_factory=list)
    tests_passed: list[str] = field(default_factory=list)
    tests_failed: list[str] = field(default_factory=list)
    resulting_model: dict[str, Any] = field(default_factory=dict)
    evidence_state: EvidenceState = "SPECIFIED"
    timestamp: str = ""  # ISO8601 UTC
    author: str = "W1.0 scaffold"
    gate: str = "B0"  # B0..B12

    # Post-B12 extension, appended later without mutating pre-B12 line
    b12_extension: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=False)

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []
        if not self.change_id.startswith("GEN2_C"):
            issues.append(f"change_id {self.change_id} must start with GEN2_C")
        if self.provenance not in VALID_PROVENANCE:
            issues.append(f"provenance {self.provenance} not in {VALID_PROVENANCE}")
        if self.evidence_state not in VALID_EVIDENCE_STATES:
            issues.append(f"evidence_state {self.evidence_state} invalid")
        if not self.gate.startswith("B"):
            issues.append(f"gate {self.gate} must be B0..B12")
        if "config_hash" not in self.parent_model:
            issues.append("parent_model missing config_hash")
        if not self.reason_before_results:
            issues.append("reason_before_results empty — must precede observed")
        return {"valid": not issues, "issues": issues}


def ledger_sha256_16(ledger_path: Path | str = LEDGER_PATH) -> str:
    """Hash chain over ledger JSONL (excluding comment lines starting with #).

    Returns first 16 hex chars of sha256 over concatenated non-comment lines + newline.
    Stored in manifests/gen2_freeze.json as ledger_sha256_16 for tamper evidence.
    """
    p = Path(ledger_path)
    if not p.exists():
        return hashlib.sha256(b"").hexdigest()[:16]
    h = hashlib.sha256()
    for line in p.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        h.update((stripped + "\n").encode())
    return h.hexdigest()[:16]


def validate_ledger_entry(entry_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw ledger dict against required_fields and enums."""
    from jomission.qualification.gen2_gates import SPECIFIED_GATES  # lazy

    issues: list[str] = []
    required = [
        "change_id",
        "parent_model",
        "parameter_path",
        "pathway",
        "old_value",
        "new_value",
        "delta",
        "reason_before_results",
        "provenance",
        "qualification_target",
        "predicted_generic_effect",
        "observed_generic_effect",
        "tests_passed",
        "tests_failed",
        "resulting_model",
        "evidence_state",
        "timestamp",
        "author",
        "gate",
    ]
    for k in required:
        if k not in entry_dict:
            issues.append(f"missing required field: {k}")
    prov = entry_dict.get("provenance")
    if prov and prov not in VALID_PROVENANCE:
        issues.append(f"provenance {prov} invalid")
    es = entry_dict.get("evidence_state")
    if es and es not in VALID_EVIDENCE_STATES:
        issues.append(f"evidence_state {es} invalid")
    gate = entry_dict.get("gate", "")
    if gate and gate not in SPECIFIED_GATES and gate != "B0":
        # allow B0..B12
        if not gate.startswith("B"):
            issues.append(f"gate {gate} not B0..B12")
    return {"valid": not issues, "issues": issues}


def append_ledger(entry: LedgerEntry | dict[str, Any], ledger_path: Path | str = LEDGER_PATH) -> dict[str, Any]:
    """Append-only JSONL write. Never mutates prior lines.

    - Validates entry
    - Checks change_id not already present (write-once)
    - Appends single JSON line (no pretty print) for hash-chain stability
    - Also regenerates manifests/gen2_modification_ledger.json aggregate (array of all entries)
    Returns {"ok": bool, "ledger_sha256_16": str, "issues": [...]}
    """
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(entry, LedgerEntry):
        d = entry.to_dict()
        v = entry.validate()
    else:
        d = dict(entry)
        v = validate_ledger_entry(d)
        # coerce missing optional
        d.setdefault("unexpected_effects", [])
        d.setdefault("tests_passed", [])
        d.setdefault("tests_failed", [])

    if not v["valid"]:
        return {"ok": False, "issues": v["issues"], "ledger_sha256_16": ledger_sha256_16(p)}

    # Enforce change_id uniqueness (write-once) — scan non-comment lines
    if p.exists():
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
                if obj.get("change_id") == d.get("change_id"):
                    return {
                        "ok": False,
                        "issues": [f"change_id {d['change_id']} already exists — append-only violation"],
                        "ledger_sha256_16": ledger_sha256_16(p),
                    }
            except json.JSONDecodeError:
                continue
        # Evidence-state monotonicity: prior entries for same gate should not go backwards
        # (light check: last entry's evidence_state index <= new if same gate? we just enforce valid enum)

    # Append single line
    line = json.dumps(d, sort_keys=False)
    with p.open("a") as f:
        f.write(line + "\n")

    # Regenerate aggregate JSON array (for convenience, not authoritative; jsonl is authoritative)
    try:
        entries: list[dict[str, Any]] = []
        for ln in p.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            entries.append(json.loads(s))
        LEDGER_JSON_PATH.write_text(json.dumps(entries, indent=2) + "\n")
    except Exception:
        pass

    return {"ok": True, "issues": [], "ledger_sha256_16": ledger_sha256_16(p), "change_id": d.get("change_id")}


def extend_ledger_after_B12(
    change_id: str,
    b12_extension: dict[str, Any],
    ledger_path: Path | str = LEDGER_PATH,
) -> dict[str, Any]:
    """Append B12 extension for an existing change_id.

    Per spec, post-B12 fields (T1_T7_delta, Q1_Q15_delta, polarity_transitions,
    unchanged, attribution, impossible_remains) are appended without overwriting
    pre-B12 fields. Implemented as an append-only companion record with same
    change_id + kind="b12_extension" so hash chain remains append-only.

    The resulting ledger will have two lines for the same change_id; queries
    must group by change_id. This preserves immutability of the original line.
    """
    p = Path(ledger_path)
    if not p.exists():
        return {"ok": False, "issues": ["ledger does not exist"], "ledger_sha256_16": ""}

    # Verify change_id exists
    found = False
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
            if obj.get("change_id") == change_id and obj.get("kind") != "b12_extension":
                found = True
                break
        except json.JSONDecodeError:
            continue
    if not found:
        return {"ok": False, "issues": [f"change_id {change_id} not found — cannot extend"], "ledger_sha256_16": ledger_sha256_16(p)}

    # Check not already extended
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
            if obj.get("change_id") == change_id and obj.get("kind") == "b12_extension":
                return {
                    "ok": False,
                    "issues": [f"change_id {change_id} already has b12_extension"],
                    "ledger_sha256_16": ledger_sha256_16(p),
                }
        except json.JSONDecodeError:
            continue

    # Validate extension has at least one expected key
    allowed_keys = {"T1_T7_delta", "Q1_Q15_delta", "polarity_transitions", "unchanged", "attribution", "impossible_remains"}
    if not any(k in b12_extension for k in allowed_keys):
        return {
            "ok": False,
            "issues": [f"b12_extension must contain one of {sorted(allowed_keys)}"],
            "ledger_sha256_16": ledger_sha256_16(p),
        }

    extension_record = {
        "change_id": change_id,
        "kind": "b12_extension",
        "b12_extension": b12_extension,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "author": "B12 extension",
        "ledger_sha256_16_at_extension": ledger_sha256_16(p),
    }
    line = json.dumps(extension_record, sort_keys=False)
    with p.open("a") as f:
        f.write(line + "\n")

    # Update aggregate json
    try:
        entries: list[dict[str, Any]] = []
        for ln in p.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            entries.append(json.loads(s))
        LEDGER_JSON_PATH.write_text(json.dumps(entries, indent=2) + "\n")
    except Exception:
        pass

    return {"ok": True, "issues": [], "ledger_sha256_16": ledger_sha256_16(p), "change_id": change_id}


__all__ = [
    "LEDGER_PATH",
    "LEDGER_JSON_PATH",
    "LedgerEntry",
    "ParentModel",
    "ResultingModel",
    "LedgerDelta",
    "ledger_sha256_16",
    "append_ledger",
    "extend_ledger_after_B12",
    "validate_ledger_entry",
    "VALID_PROVENANCE",
    "VALID_EVIDENCE_STATES",
]

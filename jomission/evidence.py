"""Typed EvidenceRef / configuration provenance — distinguishes evidence classes."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal
import hashlib
import json

EvidenceClass = Literal["CANONICAL_CONFIRMATORY", "DEVELOPMENTAL", "SUPPLEMENTARY", "MECHANISTIC", "SENSITIVITY"]
Namespace = Literal["canonical_confirmatory", "pilot_developmental", "supplementary", "mechanistic", "sensitivity"]


@dataclass(frozen=True)
class EvidenceRef:
    code_sha: str
    parent_run: str | None
    config_hash: str
    numerical_config_hash: str
    hp_hash: str
    dt_ms: float
    seed: int
    network_realization: str
    phase: str
    initial_state_hash: str | None
    namespace: Namespace
    evidence_class: EvidenceClass
    estimand_version: str
    generated_owner: str
    artifact_hash: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def distinct_from_canonical(canonical_config_hash: str, candidate_config_hash: str) -> bool:
        return canonical_config_hash != candidate_config_hash

    @staticmethod
    def hash_of(obj: dict) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]

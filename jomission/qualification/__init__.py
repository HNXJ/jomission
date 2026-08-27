"""Qualification package — Gen-2 biophysical gates and ledger.

This package is topology+heterogeneity+observability infrastructure only (W1).
No dynamics tuning is exposed here; see ledger for change history.
"""

from jomission.qualification.gen2_gates import SPECIFIED_GATES, FORBIDDEN_TERMS_B10
from jomission.qualification.ledger import LedgerEntry, append_ledger, ledger_sha256_16
from jomission.qualification.b1_targets import B1_CLASS_TARGETS

__all__ = [
    "SPECIFIED_GATES",
    "FORBIDDEN_TERMS_B10",
    "LedgerEntry",
    "append_ledger",
    "ledger_sha256_16",
    "B1_CLASS_TARGETS",
]

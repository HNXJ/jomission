# B10 objective audit: amendment 1 (word-boundary matching)

Date: 2026-11-04. Authority: owner ruling 2026-11-04 (word-boundary match).
Amends: jomission/qualification/gen2_gates.py audit_b10_objective (the frozen B10 audit).

## Change
Every forbidden term is matched as a whole word (regex \b term \b), not as a substring. Before: only T1 to T7 were whole-word; 'omission' matched inside 'jomission' (for example in the import path jomission.network).

## Reason
A substring rule cannot pass for any file that imports the model package, because every such file contains 'jomission'. The rule as written therefore made the audit unpassable, not stricter.

## Unchanged
The forbidden-term list; the audit predicate (0 hits required before the first theta evaluation); the audit on the optimizer objective code.

## Check
tests/test_b10_audit_word_boundary.py: 'jomission' is not a hit; 'omission' as a whole word is a hit; T1 as a whole word is a hit; 'T10' is not a hit.

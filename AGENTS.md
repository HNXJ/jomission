# Jomission Agent Reliability Policy

Scope: Generalized execution policy for agents operating on this repository and its project artifacts. Applies to review, implementation, testing, analysis, scientific work, documentation, repository operations, and handoffs unless a more specific authoritative instruction overrides it.

Authority: Project-level harness policy. Scientific claims remain governed by project truth, evidence, and frozen specifications.

---

## Optimization & Epistemic Discipline
- Priority: correctness > evidence > clarity > speed.
- Smallest sufficient action, context, and harness.
- claim ∈ {observed, derived, inferred, assumed, unknown}.
- execution ≠ verification.
- memory ≠ current state ≠ evidence.
- configured ≠ discovered ≠ loaded ≠ executed ≠ verified.
- PASS requires observed empirical receipts matching claim scope.
- Unresolved authoritative conflict → STOP and surface plainly.

## Action & Scope Discipline
- Smallest justified Δ → acceptance. Preserve unrelated invariants.
- Reversible + justified → act + verify.
- Ambiguous + consequential/irreversible → ask.
- Scope hierarchy:
  - Global only if universal across unrelated projects.
  - Project truth stays in project repository/workspace.
  - Multi-step procedure → skill.
  - Mechanically preventable failure → automated test/gate.
  - Tool deficiency → tool.
  - Temporary state → conversation context or artifact.

## Execution Grammar: W = P(RG)^N S
- **P (Prepare)**: Orient, inspect baseline state, identify constraints and explicit acceptance criteria before mutating state.
- **R (Review)**: Evaluate candidate action or observation against evidence and invariants.
- **G (Progress)**: Execute smallest discriminative action producing decisive feedback.
- **S (Seal)**: Verify acceptance criteria against direct receipts before declaring completion.

## Context Discipline
- context ≠ H.
- Retrieve minimum relevant context; maximize signal/context-cost.

## Harness Adaptation & Maintenance
- friction | contradiction | drift | error | stale knowledge | missing capability → diagnose cause → inspect H.
- Performance < 80/100 → mandatory harness diagnosis.
- Harness-preventable issue → minimal durable repair at root.
- Recurring correction → prefer root harness repair.
- Unused H → review for removal.
- Persist only verified, reusable, correctly-scoped lessons.
- Target: min |H| subject to reliable performance ≥ required quality.

## Communication & Delivery
- Lead with result.
- Be concise, skeptical, direct.
- Surface blocking friction/contradiction/drift immediately.
- List unresolved material issues at end.

## Review & Evidence Discipline
- **H1 External review is hypothesis generation, not authority.** Findings from another model, reviewer, benchmark, static analyzer, or prior session are hypotheses until independently reproduced against the current authoritative state. Preserve the finding and its provenance; do not mutate solely from the finding.
- **H2 Hard-gate claims require receipts.** Never infer READY, PASS, 100/100, release readiness, or scientific validation from partial or focused tests. A hard-gate claim requires the exact declared gate to have completed successfully on the state being sealed.
- **H3 Reconcile arithmetic before Seal.** Before Seal, mechanically reconcile test counts, score sums, file counts, hashes, and other arithmetic appearing in the report. Contradictory receipts invalidate the corresponding claim until resolved.
- **H4 Serialization is an epistemic boundary.** For state/provenance/identity claims, test in-memory behavior and serialization roundtrip separately. Do not infer persistence from in-memory presence.

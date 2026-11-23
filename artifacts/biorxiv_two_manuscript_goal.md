# Jomission Solmind — two-paper bioRxiv goal and execution queue

Status: PROPOSED; not a replacement for canonical `manifests/todo.json` or sealed criteria. Source reviews: `artifacts/lansa_biorxiv_stage15_review.md`, `artifacts/fixationboat_biorxiv_stage15_review.md`. Base `jomission@dev@c4c9c6b17a45b4d925b2d72a8bb02f2515850c70`. Planning branch `_temp_solmind`.

## Goal

Prepare independently defensible LANSA and Fixation Boat manuscripts for bioRxiv. A manuscript receives **submission-readiness 100/100** only if all fifteen stage obligations, strict scientific and presentation gates, provenance, owner-supplied declarations, and independent preflight pass; actual submission is reserved for Hamm. Preserve the canonical 15-stage historical completion score separately: its stage 15 literally requires submission. No invented stage credits.

## Current scores and evidence

| Manuscript | Declared stage | Provisional stage-credit | Quality | Critical limit |
|---|---:|---:|---|---|
| LANSA | 10 | 63.3/100 (9 completed + active 0.5, unverified stage prerequisites) | historical 51/100, 20% full coverage | M3 UNKNOWN, figures/methods defects, lock pending |
| Fixation Boat / Plasticity Boat | 2 | 10.0/100 (stage 1 + active 0.5) | UNKNOWN | identity/scope and experimental protocol unresolved |

The separate quality rubric for LANSA is frozen. Do not substitute the workflow score for scientific quality. For Fixation Boat, establish a rubric before scoring, with numerical integrity, claim calibration, statistics, provenance, figures, methods, prose and hard gates; no post-outcome threshold changes.

## Execution routing

| Work package | Tasks | Ready work | Owner gate |
|---|---|---|---|
| L-A source reconciliation | L01-L03,L13-L14 | SHA, result, number and source audit | E1b scientific reinterpretation |
| L-B figures/methods | L04-L12,L16-L17 | figures, captions, methods, intervals, main/supp | methodological change beyond existing seals |
| L-C manuscript seal | L15,L18-L23 | style, critique, reproducible build | methodological lock, declarations, submission |
| B-A scientific identity | B01-B05,B11-B12 | conflict packet and frozen-criteria comparison | central question, restoring force, new protocol |
| B-B task and control validation | B06-B10,B13-B16 | synthetic tests, receipts audit, controls | confirmatory experiment criteria |
| B-C science and manuscript | B17-B24 | run only authorized protocol, then figures/text/review | owner lock, metadata, submission |

## Agent execution contract

1. Read latest Jarness `main` `rules/start.md`, jomission `AGENTS.md`, `README.md`, `manifests/current_state.json`, `manifests/todo.json`, sealed artifacts and applicable skills. Check Git HEAD, dirty state, open blockers and authoritative goal before work. These planning files do not authorize work that conflicts with `next_authorized_task`; obtain reviewer/owner task authorization first.
2. At most three concurrent lanes including reviewers; one writer per file set. Delegate each packet with base SHA, read/write scope, exact test/receipt, falsifier, control, stopping condition and maximum effort. Lead owns scientific decisions and integration; independent non-writer reviewer verifies.
3. Preserve stage ordering. LANSA remains stage 10 until scientific issues and mock review are closed and Hamm authorizes the lock. Fixation Boat remains stage 2 until its skeleton is accepted. From stage 11, scientific or methodological changes are prohibited without returning to an authorized earlier stage.
4. For every result: verify exact SHA, input hashes, seeds, units, n, uncertainty, negative controls, analysis designation, multiplicity and figure provenance. Source absence = UNKNOWN, not PASS. Preserve failed and inconclusive outcomes.
5. Execute ready tasks until no authorized work remains; blocked owner items do not block unrelated tasks. Record findings in canonical open/todo homes only when authorized, avoiding parallel trackers. Never silently edit frozen results or source manifests.
6. Run `python scripts/project_check.py` and appropriate tier from `manifests/test_tiers.json`, LaTeX build and rendered-page checks, claim-to-source audit and independent critique. A local test pass is not an independent review.
7. Close with one report per manuscript: stage and score with evidence coverage, quality score and caps, exact commit, gates, open blockers, owner decisions, next authorized task. Submit nothing. At submission-readiness 100, request Hamm's explicit final go.

## Immediate queue

- LANSA: reconcile E1b and source SHAs; finish M3, C6, C7 and C3 findings before any stage-10 lock request.
- Boat: resolve B01 central paper identity, B03 passive restoring force and B04 trial-completion semantics; meanwhile audit available control receipts without using held-out starts.
- Both: keep central scientific evidence in main figures, detailed sensitivity and secondary controls in supplement; captions must be independently interpretable; use `match-writing-style` with author's publication benchmarks and avoid unnecessary abbreviations.

## Stop states

`READY_FOR_OWNER_GO`: every required pre-submission check PASS, independent reviewers ACCEPT, no mandatory UNKNOWN, full package compiled and checked, author metadata approved. `BLOCKED`: no authorized ready task, exact owner decisions listed. `SUBMITTED`: only after Hamm authorizes actual bioRxiv submission and receipt is verified.

This document is a planning handoff, not a dispatch receipt.

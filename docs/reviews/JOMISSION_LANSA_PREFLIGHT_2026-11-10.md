# Jomission-LANSA: independent preprint preflight (2026-11-10)

Evidence: uploaded `jomission-dev.zip` (1498 entries, no `.git` metadata), inspected `paper/jomission_lansa/{sections,main.md,data}`, `docs/laminar_motif/SEAL_T8.md`, `jomission/dynamics/hdp.py`, and `site-src/pages/omission.html`. This is a source/receipt audit, not an independent rerun. **No release, publication, or canonical state edit authorized.**

## Decision

**80/100 = owner-proposed readiness, not independently certified.** Scientific result is suitable for a carefully scoped negative-result preprint after the blockers below. Do not describe T8 as identifying a unique cell-type mechanism. Current evidence supports the specific three-unit, assumed-connectivity model under stationary drive.

## Major checks

| ID | Evidence / defect | Required closing check |
|---|---|---|
| P0-1 | `sections/backmatter.tex` still contains `[MISSING]` for LANSA access terms, run archive/DOI, code archive/DOI, acknowledgements, contributions, competing interests, funding, AI disclosure. | Owner supplies facts; archive accessible, checksums match, release SHA and environment recorded. Do not invent disclosures. |
| P0-2 | Snapshot has no `.git`; manuscript cites multiple commits and an unreleased JaxFNE 0.5.9. | In real checkout: `git status`, branch, HEAD, source and engine SHA, test/receipt inventory; reconcile citation to frozen release. |
| P1-1 | T8 sealed 5 arms x 3 seeds, paired contrasts, descriptive uncorrected decision. Density arms change counts and in-degree; ratio arms change subtype counts and incoming inhibition. Single-unit broadband dominance in controls. | Report all paired seed differences, rate shifts, unit spectral shares, and lack of band specificity. No claims of unique density or SST causal mediation. |
| P1-2 | Joint readout file shows `map_mid_full` feasible at tolerance 0.02, r=0.84515, mid=0.82805 (lower boundary), eigenvalues near zero; `y_mid_full` infeasible in reported optimizer. | Specify objective, constraints, PSD parameterization, feasibility tolerance, convergence/multistart; rerun deterministic synthetic known-answer and independent grid bound. Optimizer output is not a global proof. |
| P1-3 | Readout capacity LOAO trains on ten observed areas, not on cell densities. | Call it cross-area generalization of a fitted observation model, not prediction from cellular data. Include the 10-area-template baseline 0.93 +/- 0.06. |
| P1-4 | Reference alpha-beta r 0.83-0.86 on selection seeds, T1 winner 0.32-0.47 on selection and 0.35-0.47 on fresh seeds. | Lead with T1 FAIL and distinguish baseline, selection, and confirmation. |
| P1-5 | Model peaks 27-46 Hz isolated, 43/11/12 Hz in stack; scored gamma 50-150 Hz. | Distinguish a missing fast oscillatory mode from a misplaced source and from a frequency-normalization artifact. |
| P1-6 | T8 runner launched 15 jobs simultaneously and one visualization was generated after a verdict was read (noted in `main.md`). | Confirm sealed computation unaffected; disclose visualization-order deviation, avoid claiming a strictly prospective visual firewall. |
| P2-1 | Several figure labels are very small; main text still contains source-path-like labels in places. | Render final PDF and inspect all figure panels at publication size, tables, references, hyperlinks. |
| P2-2 | One timed event-delivery pair: 1387 vs 140 s, 22/22 arrays equal; T8 reports 3-seed engine equivalence. | Report exact layouts/SHAs, scope of equality and speedup; no universal equivalence claim. |

## T8 interpretation

The manuscript plan reports ref-minus-control alpha-beta differences: density-flat +0.004 (no consistent effect), density-reversed +0.13, ratio-flat +1.56, ratio-reversed +1.58; all four controls lose the crossing (0/3 versus reference 2/3). This supports a **construction-dependent phenotype**, not separable gamma-vs-alpha-beta cell-class causality: interventions also change E rates and unit spectral dominance. The seal explicitly calls n=3 an uncorrected descriptive screen.

## Acceptance

1. Complete all P0 rows; source and archive checks pass.
2. Close P1-2 numerical-capacity verification and retain all negative T8 results without post-hoc threshold changes.
3. Cross-check abstract, Results, Discussion, figures and tables against machine-readable receipts.
4. Render final preprint and verify figure readability; owner approves metadata, authorship, disclosure, and submission.

No checks executed against the full simulator in this review. ZIP contents alone do not establish current remote branch or git cleanliness.

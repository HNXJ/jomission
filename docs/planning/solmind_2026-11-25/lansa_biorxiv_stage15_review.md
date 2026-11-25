# LANSA bioRxiv readiness review — Solmind

Baseline: jomission dev c4c9c6b17a45b4d925b2d72a8bb02f2515850c70; source `paper/jomission_lansa/`. Planning only; no manuscript files changed. Jarness main `rules/reference/manuscript-stages.md` governs stages. Stage score convention: 1 completed, 0.5 active, 0 unstarted; 100*credit/15. This is workflow progress, not manuscript quality.

## Score and scientific verdict

Current declared stage 10; provisional stage-credit 9.5/15 = **63.3/100**, assuming stages 1-9 were completed to enter stage 10. Their full acceptance was not independently rerun here. Last frozen quality rubric `paper/jomission_lansa/quality_rubric.md`: **51/100**, coverage **20/100**, evaluated on d289583, with M3 numerical trace UNKNOWN. Later M3 repairs and E1b are not a rescored quality result. A 35-page latexmk build at d289583 was reported PASS; no new build run in this review.

The central finding is negative: the tested three-unit density-informed circuit does not reproduce the primate spectrolaminar motif under the stated readout and stationary drive. Do NOT turn it into a positive mechanism claim. E0 is a pipeline equality test; E1 inconclusive at rho=4; E1b fixed-PV control needs its actual sealed result integrated with exact status; T1 fresh-seed confirmation fails; T8 n=3, uncorrected, cannot establish band-specific causal mechanism. The conditional observation-model capacity and depth-placement confound are central, not a validation of biological generators.

## Detailed priority review and closing checks

| ID | Priority | Finding / work | Closing evidence |
|---|---|---|---|
| L01 | P0 | Reconcile current branch/HEAD, E1b result, and stage-10 round-2 report | Source SHA, all result receipts, no stale claim |
| L02 | P0 | Finish full M3 number trace across abstract, Results, Discussion, tables, captions, supplement | Every quantitative token traced to hashed source; no UNKNOWN |
| L03 | P0 | Resolve eight remaining M3 UNKNOWN readings/literature/process statements | Measured, cited, removed or explicitly unavailable |
| L04 | P0 | Repair Fig S1 internal filenames and agent-choice text | Rendered figure has no internal path |
| L05 | P0 | Explain Fig S2 difference colors and scale | Standalone caption checked against image |
| L06 | P0 | Letter Fig 8 phi/CSD panels and align caption | Panel-to-measure mapping unambiguous |
| L07 | P0 | Align Fig 8 conductivity, lateral distance, z offsets, and bandpass with Methods | Exact source and units stated in Methods |
| L08 | P0 | Remove or quantify Fig S4 >35 Hz visual claim | Source-derived measurement or deletion |
| L09 | P1 | Cite S1 and S3 in main narrative where relevant | All supplement figures findable |
| L10 | P0 | State trough-contact selection and avoid circular validation | Prespecified selection; independent evaluation |
| L11 | P0 | State T1 search size (21 levels, 63 level-seed scores) and selection/multiplicity | Fresh-seed inferential interpretation bounded |
| L12 | P0 | Report T8 and fresh-seed effect sizes, n, sidedness and uncertainty; n=3 limitations | Honest interval or explicit inability; no false precision |
| L13 | P0 | Reconcile D 21-ms loss wording between Results and Discussion | Same causal and selection scope |
| L14 | P0 | Pin C15-C18 and any revision-dependent source identity | Resolvable commit/hash for every receipt |
| L15 | P1 | Remove unasked contrasts and unnecessary abbreviations | Style checker and non-writer author-voice review |
| L16 | P1 | Check all figures at final print size; captions define panel, metric, n, uncertainty, tests | Rendered-page visual review; no pending placeholders |
| L17 | P1 | Main versus supplement audit: central negative, T1/T8/readout identifiability in main; per-run atlases, sensitivities, detailed controls in supplement | Claim-to-figure map, no central evidence hidden |
| L18 | P0 | Resolve data provenance, access terms, code archive/DOI, author declarations only at stage 13 | Owner-sourced metadata; no invented declaration |
| L19 | P0 | Fresh skeptical mock review of complete updated draft | Independent ACCEPT with all major findings resolved |
| L20 | P0 | Request methodological lock only after L01-L19 scientific items close | Hamm explicit lock; no stage-11 scientific edits |
| L21 | P0 | Stages 11-12: strict text/format/reference/style check | Reproducible clean TeX build, DOI/reference audit, complete figure QA |
| L22 | P0 | Stage 13 metadata; stage 14 bioRxiv submission package | Owner-approved author fields, disclosures, licensing, source files and strict preflight |
| L23 | P0 | Stage 15 READY_FOR_OWNER_GO; submission owner-reserved | Independent reviewer ACCEPT, all mandatory checks PASS, no required UNKNOWN |

## Release gate

100/100 *submission readiness* requires all 15 stage obligations satisfied through a complete stage-15 preflight packet; submission itself is a separate owner action. The historical 15-stage convention counts actual submission at stage 15, so report the readiness indicator separately from literal workflow completion until Hamm explicitly amends that convention. Do not silently grant the last stage point. Negative science is allowed; unsupported claims, untraceable numbers, missing mandatory declarations, or unreviewed figures are not.

## Agent handoff

Read Jarness main start; jomission AGENTS.md, README, manifests/current_state.json, manifests/todo.json, `paper/jomission_lansa/main.md`, quality_rubric.md, STAGE10_R2_VERIFICATION.md, current main.tex and sections, receipts and sealed evidence. Existing scientific gates and stage 10 lock outrank this proposal. At most 3 lanes including reviewers. Work on ready items; blocked owner fields do not halt independent QA. Use original repo tests, render and visually inspect. Record SHA, command, output, figures and independent verdict. Do not execute unapproved simulations or retune frozen criteria.

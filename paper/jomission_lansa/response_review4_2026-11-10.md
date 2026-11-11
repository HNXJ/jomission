# Response to review 4 and the preflight (2026-11-10)

Reviews: the pasted "Peer review — Jomission-LANSA, latest draft" (minor scientific revision, major submission
preparation) and `docs/reviews/JOMISSION_LANSA_PREFLIGHT_2026-11-10.md`. The HDP proposal in the same archive is
filed at `docs/designs/OMISSION_SPECTROLAMINAR_HDP_PROPOSAL_2026-11-10.md`; it is marked unsealed and was not run.

Status per comment: **done** (text or evidence changed), **partial**, **owner** (needs facts only the authors hold).

## Scientific

| Comment | Status | Change |
|---|---|---|
| T8 has 3 seeds and 4 primary comparisons, uncorrected: keep descriptive | done | Results T8 states the comparisons are descriptive and uncorrected; decision-rule wording only in Methods |
| Joint optimum at the lower feasibility boundary: numerical solution, not global optimum (preflight P1-2: objective, constraints, PSD parameterization, tolerance, multistart, known-answer test, independent bound) | done | New Methods "Readout capacity" states all of these. New check (C15, D10): 4 synthetic targets recovered feasible at r >= 0.9999; unattainable m = 1.2 returned infeasible; Fibonacci grid of 2e5 rank-one C gives 0.8450 vs optimizer 0.8452 (sigma 0.15), 0.9775/0.9814 vs 0.9779/0.9814 (0.20); 2e5 random full-rank L at most 0.8423. Text: "numerical solution, not a proven global maximum" |
| T8 controls change rates, in-degrees and counts together | done | Kept; construction-rule interpretation stated in Results and Discussion |
| Stationary vs peri-stimulus target: more prominent | done | Leads the Discussion Limits paragraph, with the consequence (part of the mismatch may reflect drive) |
| Gamma deficit is relative | done | Abstract "of its per-frequency maximum"; Results R3 opens its decomposition paragraph with it; Discussion states it is not a measured absolute shortfall |
| "Failure lies in the circuit's spectra" is conditional on the observation operator | done | Abstract, Introduction, Results and Discussion now say "under / within this observation model" |
| P1-3: LOAO is cross-area generalization of a fitted observation model; template baseline 0.93 +- 0.06 | done | Renamed in text and table caption; baseline in both |
| P1-4: lead with T1 FAIL; separate baseline, selection, confirmation | done | Results R4 opens with the failure and the fresh-seed loss of r_ab, then names the three run kinds |
| P1-5: missing fast mode vs misplaced source vs normalization artifact | done | Results R3 names the three accounts: cancellation excluded (cross terms 0-12%); normalization: deficit is M relative to S, not separable from the data; missing mode favoured (unit peaks 43/11/12 Hz; isolated module 27-46 Hz; M-scaling bound 0.615); source geometry treated in R6 |
| P1-6: no strictly prospective visual firewall | done (already) | Methods T8 deviation lists the V1 rasters drawn after one score file was read; no text claims a prospective firewall |
| P2-2: event delivery, exact layouts and SHAs | done | Moved to Methods: layouts `kd1p41` s1 and the reference s1-3, JaxFNE 351f6048, negative controls, one timed pair, configuration-specific |

## Writing and order

| Comment | Status | Change |
|---|---|---|
| Abstract: shorter, narrative | done | Rewritten from the suggested text, 249 words; added the assumed density-to-connectivity rule, the imposed ramp, the fresh-seed loss and the readout qualification |
| Results order: reference, failure, interventions, readout identifiability | done | R3 reference; R4 size/T1/weights; R5 T8; R6 readout |
| Move optimizer diagnostics out of R3 | done | PSD parameterization, eigenvalues, controls, penalty and multistart, distributed model, kernel-width rationale now in Methods; circuit-vs-optimum table kept in R6 |
| R2.5 lead with the alpha-beta/gamma trade-off | done | R4 title and first sentences |
| R2.7 lead with Fig 6 and the opposing dominance | done | R5 title and second paragraph; statistics paragraph follows |
| Fewer audit terms (sealed, winner, arm, verdict) in Results | done | "pre-registered", "selected setting", "condition"; audit terms kept in Methods |
| Discussion -20 to 25% | done | 1459 to 941 words; limits and the rate-matched M-gamma prediction kept; repeated numbers removed |

## Figures

| Comment | Status | Change |
|---|---|---|
| Fig 4: compact r_ab vs r_gm scatter with acceptance region; atlases to supplement | done | New Fig 4 (C16): 93 runs, 7 groups, shaded both >= 0.6, drawn at print size (7 pt); T1 atlases now Fig S4 |
| Fig 1E crowded | partial | Caption explains reading bars against same-colour lines; panel not redrawn |
| Fig 2C dense, small annotations | partial | Caption names the shaded scored bands; panel not redrawn |
| Fig 7 CSD units and source geometry | done | Split into Fig 7 (probe geometry) and Fig 8 (CSD), each at full width so labels stay >= 6 pt; the CSD caption gives source offsets, return currents, sigma_e 0.3 S/m and lateral distance 0.05 (both assumed) and why units are arbitrary |

## Owner items (not generated)

Authors, affiliations, ORCIDs, corresponding author; data availability and LANSA access terms; code and run-record
archive with DOI; funding, competing interests, contributions, acknowledgements, AI-use statement. All remain
`[MISSING]` in `sections/backmatter.tex` and `main.tex`.

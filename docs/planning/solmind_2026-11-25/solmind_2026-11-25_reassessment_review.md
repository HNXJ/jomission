# Solmind reassessment: verified review and proposed whole-program roadmap (2026-11-25)

Status: PROPOSAL. Not canonical. Does not change `manifests/`, sealed artifacts, the LANSA rubric, the E1/E1b/T1 verdicts or any goal, open or todo stack.
Baselines (OBSERVED): canonical `dev` = `ef0a766`. Advisory and planning files = `_temp_solmind` at `735eab3`. The advisory's own base is `ef0a766`. The planning goal file (`artifacts/biorxiv_two_manuscript_goal.md`) cites `c4c9c6b`, which is stale. The reviewer's local checkout (`_temp_integ_20261124`, 48 commits ahead of `origin/main`) is not used as a baseline.
Evidence scope: every claim below was read from `dev` sources or from `_temp_solmind` at the SHAs above. Nothing was run.

Evidence classes: **E** established (source read), **I** inferred, **P** proposed, **B** blocked (needs owner or a missing input).

## 1. Verdict

1. The advisory's central premise is supported by the sources: E1 slope and peaks match the records, E1 is INCONCLUSIVE, E1b arm A is INCONCLUSIVE and arm B is PASS, and E1b was designed after a re-read of E1 counts. The advisory is right to keep the frequency slowing as an association.
2. The proposed factorial atlas is a **new lineage**, not an extension of E1. It cannot be used to rescue, reinterpret or re-seal E1, E1b or T1. Any criterion change, including a widened peak band to pass r = 4, is a new pre-registered lineage with owner approval.
3. The dendritic stage is not supported by the current model. The engine's multicompartment support is unverified. A forward-model geometry assumption already exists in the CSD code and could confound any LFP claim.
4. The plan conflicts with the canonical state in several places (section 3, C1–C10). The owner must rule on the boat's identity, the programme scope and the stage workflow before any stack is edited.
5. Recommendation: split the work. LANSA keeps its current, frozen scope and verdicts. The regime atlas is a separate prospective workstream with its own lineage. The dendritic extension is conditional on the atlas. The Plasticity Boat stays separate.

## 2. Claim audit of the advisory

| # | Advisory claim | Status | Evidence |
|---|---|---|---|
| 1 | E1 peaks 46.0, 39.7, 34.0, 27.0 Hz at r = 0.25, 0.5, 1, 2; slope −6.27 Hz/oct, n = 3 | E | `results/laminar_e1/e1_result.json` (mean_slope −6.2667; per-seed −6.6, −5.9, −6.3); `e1_rho*_seed*.json` peak_freq_E_Hz |
| 2 | E1 verdict INCONCLUSIVE; r = 4 criterion not met | E | `e1_result.json` verdict INCONCLUSIVE. Refinement: at r = 4 the three seed peaks are 21, 22 and 21 Hz, at the 20 Hz band edge. Excess 3.04 and 3.26 are below 4 in two seeds (`docs/laminar_motif/E1B_SEAL.md` line 10). The advisory's wording "not met in enough seeds" should name the band edge |
| 3 | E1b arm B oscillated at r = 4, 24–26 Hz; arm A INCONCLUSIVE | E for verdicts (`e1b_result.json` lines 701 INCONCLUSIVE, 936 PASS). Reviewer read-only check: arm B r = 4 peaks 24, 26, 24 Hz | `e1b_result.json` |
| 4 | E1b designed after inspecting E1 | E | `E1B_SEAL.md` line 10; `methods.tex:86` |
| 5 | Total inhibitory population varies in E1b arm B | E (as stated in the advisory's own caveat) | `E1B_SEAL.md`, `methods.tex:86` |
| 6 | Do not call the trajectory bell-shaped | E (advisory guidance; consistent with the 4 points available) | — |
| 7 | T1 FAIL preserved: current column did not meet the full motif | E | `results.tex:246` (T1 fails as pre-registered; r_proto 0.377, 0.365, 0.377) |
| 8 | Point-neuron E/PV/SOM column; no dendritic compartment | E | No `apical`/`dendrit` in `jomission`, `scripts`, `tests`, `results`. Only `scripts/e2_csd.py:51-53` and `docs/laminar_motif/PLAN_PAPER_JOMISSION_LANSA.md:116` |
| 9 | Dendritic apical SOM model would explain the motif | P (hypothesis) | Not tested |
| 10 | Factorial paths A–F with fixed/co-varying quantities | P | Paths A and E correspond to E1 and E1b; B, C, D, F are new |
| 11 | Run cost | E for one measured point; I for the rest | `e1_rho1_seed1.json` wall_s 2128.8; E1 runs of 12 s took 1275–1285 s (`E1B_SEAL.md` line 93). Reviewer read-only check: rho = 1 records 2129–2157 s (these reuse E0 seeds, `results.tex:250`); rho = 4 records 1476–1510 s. The cause of the spread is not recorded (unknown); E1b arm estimate 21–33 min per arm, 27 runs ≈ 11–13 CPU-h serial, 2.82 GB peak RSS per run (`E1B_SEAL.md` line 92) |
| 12 | Frequency slowing is an association; mechanism open | E (advisory) and consistent with records | — |

## 3. Conflicts and gaps (each with both sides)

**C1. Boat identity.** Planning file `artifacts/fixationboat_biorxiv_stage15_review.md` titles the subject "Fixation Boat / Plasticity Boat" and lists B01 as open: "fixation acquisition/encoding vs eight-rule plasticity competition". The programme files (`_temp_plasticity_boat/…`) and the boat gap analysis `docs/subprojects/plasticity-boat/GAP_ANALYSIS.md` (on dev, commit 7cb8a57) treat the boat as the plasticity competition. `dev` also holds `docs/subprojects/fixation-sailor/REVIEW.md`, a third subject. **Owner ruling needed (B01).** Until then the boat is not merged into this plan.

**C2. Score meanings.** The planning file gives LANSA "63.3/100 provisional stage-credit". The frozen quality rubric gives the same manuscript a capped quality score of at most 49 (`docs/receipts/lansa_audit_2026-11-25_provisional.md`). These are different measures. They must not be combined or quoted as one score.

**C3. Programme state.** `manifests/current_state.json` on `dev` is dated 2026-11-14 and says the current program is "Closeout" with the V2 chain and coupled-background program retired. The advisory and planning files assume an active LANSA/boat program. The state file is stale or the premise is. **Owner ruling needed** before any stack is edited.

**C4. Stage workflow.** The planning files cite jarness `rules/reference/manuscript-stages.md`. No stage reference file exists under `/mnt/d/repos/jarness` (searched for `*stage*`, none found). Not searched under `~/.claude`. Plasticity Boat `HANDOFF.md` says the owner's 15-stage workflow is not in repo or jarness (O-7). The stage list is therefore unverified and stage numbers remain provisional.

**C5. Stale baseline.** `artifacts/biorxiv_two_manuscript_goal.md` cites base `c4c9c6b`. `dev` is now `ef0a766`. The stage-credit figures were computed on the old base and must be recomputed.

**C6. Task ids.** The planning files define L01–L23 and B01–B24. `manifests/todo.json` on `dev` holds about 60 items (statuses DONE, RETIRED, OPEN, LOCKED, BLOCKED); `PB-DEV100-V2-VALIDATION` is BLOCKED, and none of the L or B ids appear. The planning ids are not canonical and must not be used as todo references until the owner adds them.

**C7. Frozen criteria.** E1 is sealed (`E1_SEAL.md`). A widened band for r = 4, or any fit that excludes r = 4, changes a sealed criterion and starts a new lineage. The advisory does not propose a band change, but the atlas will face the same band-edge problem at r ≥ 4. The band must be fixed before the atlas run, in its own sealed record.

**C8. Readout geometry.** `scripts/e2_csd.py:51-53` already assigns the SST kernel to a fixed apical depth (`z_u − 0.15`) and the AMPA kernel to basal and apical halves. So the forward-model "apical SOM" geometry is an assumption in the existing CSD code, not a new mechanism. Any LFP-based dendritic claim is confounded with this kernel choice. The advisory's readout-geometry critique must be run against this code first.

**C9. Engine support.** `jaxfne` contains "compartment" or "dendrit" strings in `connectivity.py`, `_model_simulate.py`, `bridges.py`, `_construct_hh.py`, `emitters_hh.py`. Whether it supports coupled soma and apical compartments with separate perisomatic and apical inhibition is **unverified** (B). Checked in the installed jaxfne 0.5.8 package (`.venv`), file list only. The working copy at `/mnt/e/repos/jaxfne` is not pinned to a version and was not used as evidence.

**C10. Stale findings note.** `docs/laminar_motif/FINDINGS_2026-11-10.md:166` gives 36 Hz for weeM2; the paper gives 54 Hz, which matches the record. Correct the note before any atlas uses it as an anchor.

## 4. Proposed whole-program structure

Phases follow the advisory's order, with a reconciliation phase added in front.

| Phase | Question | Falsifier | Controls | Seeds, n | Units / estimand | Cost (E or I) | Dependencies | Acceptance / stop |
|---|---|---|---|---|---|---|---|---|
| P0 Reconcile (no runs) | Is the baseline and each record authoritative? | A receipt hash or SHA that does not match its source | Hash check of `sources_sha256.txt`; resolve D17 (`details.json`) | — | — | Hours (I) | Owner rulings C1, C3, C4; run-record decision (D16 in the LANSA audit) | All records resolvable at one SHA; C1–C5 ruled |
| P1 Smoke | Does the factorial pipeline run at all? | Any silent or failed call (record, do not drop) | One seed, two ratios | 1 seed × 2 | peak, excess, rate, failures | ≈ 1 h (I, from 21–35 min per run) | P0; a fixed band (owner) | Every call logged; no unexplained silence |
| P2 First controlled experiment: path A, fresh seeds | Does the frequency slowing with r hold at fixed I on seeds not used in E1 or E1b? | Slope CI spans 0, or a negative slope in fewer than 8 of 10 seeds | Fixed I (path A); band fixed before the run; r = 4 reported with its band-edge status | 10 fresh seeds × 5 ratios = 50 runs | slope of peak frequency on log₂ r, seed-level, t-interval; excess; failures | ≈ 50 × 21–35 min ≈ 18–30 CPU-h serial (I) | P1 passed; owner approves band and seeds | Accept if the slope CI excludes 0 and the slope is negative in at least 8 of 10 seeds; otherwise report the null with its interval. Stop if the band is changed after seeing data |
| P3 Path B and E, matched controls | Do the axes differ in effect, and does total inhibition explain them? | Path E slope equals path A within CI, or total-I covariance explains the slope | Total-I held fixed; matched in-degree and drive | new seeds | as P2 | Per path ≈ 20–30 CPU-h (I) | P2 | Effects of ratio separated from effects of total I, with CIs |
| P4 Causal perturbation | Do selective PV→E and SOM→E perturbations shift the peak? | No shift with a perturbation larger than the seed spread | Sham perturbation; rate-matched surrogate | predeclared | Δpeak per perturbation | Not estimated (B: needs P3) | P3 | Predeclared shift threshold |
| P5 Minimal dendritic model (conditional) | Does an apical SOM compartment change dynamics beyond the point model? | Dynamics identical to the point model within seed spread | No-dendrite, passive dendrite, SOM-targeted dendrite, matched input | predeclared | spike/oscillation metrics; forward-modelled LFP/CSD | Not estimated (B: engine support, C9) | P4 result; engine verification (C9) | Compartment equations verified against an independent numerical oracle |
| P6 Held-out LFP tests | Does any mechanism predict held-out areas, depths or conditions? | Held-out prediction no better than the simpler readout baseline | Readout-geometry-only baseline (C8) | held-out | prediction error vs baseline | Not estimated | P5 | Beat the baseline on held-out data |
| P7 Manuscripts | — | — | — | — | — | — | P2–P6 | Owner decides split; no submission without the owner |

The factorial atlas (path set A–F, adaptive q × r grid) is deliberately **not** scheduled until P2 shows a robust first effect. Its cost is 6 paths × a grid × seeds: 100 runs ≈ 35–59 CPU-h serial (I).

## 5. First experiment (recommended)

Path A, fixed I = 1500, fresh seeds, five ratios (0.25, 0.5, 1, 2, 4), predeclared band, 10 seeds. Run a one-seed, two-ratio smoke first (P1). Predeclare: the estimand (slope of peak frequency on log₂ r, seed-level), the interval (t, as E1), the acceptance rule above, the band, and the seeds. Record every failed or silent call.

Why this one: it re-tests the only effect with a measured direction, on seeds not used to design E1b. It also forces the band-edge question to be settled before any larger atlas.

## 6. Critic challenges (for the independent review the advisory requires)

1. **Identifiability.** q, r, I and N are not independent at fixed N. At fixed I, changing r changes both N_PV and N_SOM, so the effect of composition is confounded with population size. The advisory's own caveat says so; P2 must show that it does not change the conclusion.
2. **Selection.** E1b was designed after E1 was re-read. Any atlas designed after E1b inherits that selection. Only fresh seeds in P2 can test it.
3. **Total inhibition.** Fixed-PV arm B varied total inhibition. The sign of the E1b effect cannot be attributed to ratio until the total-I control in P3 is run.
4. **Readout geometry.** The CSD kernel (C8) can produce a depth-resolved motif without any dendrite. A dendritic claim must beat that baseline on held-out data.
5. **Band-edge bias.** Peaks at the 20 Hz edge (r = 4) suggest the band truncates the spectrum. A band widened after seeing r = 4 is post hoc.

## 7. Owner decisions (ranked, one question each)

1. **B01, boat identity** (C1): DECIDED 2026-11-25 by owner: plasticity competition; fixation acquisition is subordinate. Record: `dev:docs/subprojects/plasticity-boat/B01_decision.md` (commit 6af28d3). Open within it: the definition of "competitive" (its §4.1).
2. **Programme state** (C3): is the active program LANSA + the atlas, or the closeout the state file records?
3. **Run records** (LANSA audit D16): commit R3–R10, R12 to this repo or seal them.
4. **Authoritative `details.json`** (LANSA audit D17).
5. **Peak band for the atlas** (C7): fixed before P2; the current band is 20–80 Hz.
6. **Stage workflow** (C4): supply the 15-stage reference, or confirm the stages are provisional.
7. **Dendritic stage** (P5): proceed only after the atlas, or start an engine-verification task now.

## 8. Blockers

- B-1: owner ruling 2 (programme state, C3) blocks stack edits and manuscript scope. Ruling 1 (B01) is decided (section 7).
- C4 (stage workflow) and C9 (engine compartment support) remain unresolved by owner instruction.
- B-2: C9 (engine compartment support) blocks P5 estimates.
- B-3: the run records (D16) block any M3 or C4 claim about LANSA from this repo.

## 9. Scope of this document

Does not rewrite the LANSA abstract, frozen criteria or verdicts. Does not absorb the Plasticity Boat. Does not add todo items. No runs were executed. Costs are estimates from recorded wall times, not measured for new configurations.

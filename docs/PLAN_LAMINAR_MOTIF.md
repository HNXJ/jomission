# Plan: spectrolaminar motif column (side plan)

Status: **authorized by the owner in chat, 2026-11-05** ("i authorize once you fix this", after the AdEx and HDP corrections, now made). `manifests/todo.json` has entry `SCI-LAMINAR-MOTIF-1`; `next_authorized_task` stays `SCI-AGSDR-CALIBRATION`.
Progress (no score change until the seal is approved): item 2 digest complete, 97 of 97 values matched, including the synaptic Q tables and mean-field T (receipt and script in `docs/laminar_motif/`); the T25-or-T26 set choice remains UNSET. Item 3 seal structure approved by the owner 2026-11-05 (`docs/laminar_motif/ITEM03_SEAL_PROPOSAL.md`); all thresholds remain UNSET, so item 3 is not closed and runs needing a threshold are blocked. Todo entry `SCI-LAMINAR-MOTIF-1` added (item 1), `next_authorized_task` unchanged. Plan numbers below that came from a source are references, not thresholds.
Minimal-model track (2026-11-06, owner): `docs/laminar_motif/MINIMAL_MODEL_SPEC.md` (draft, unsealed) states the hypothesis as E/I ratio, SST/PV ratio and one-direction SST-to-PV wiring, with experiments E0 to E6 and their falsifiers. It is the minimal-scale route through items 6, 7, 11, 15, 17 and 21; no item is added. Figure plan: `docs/FIGURE_PLAN.json` (7 main figures per series). CAMINOS is the mechanism name of Tahvili, Vinck and di Volo, which JaxFNE already ports.
Skill: `.claude/skills/jaxfne-cortical-laminar-microcircuit`. Agent: `.claude/agents/laminar-microcircuit-builder.md`.

## Goal

An Izhikevich-TFNE cortical column in JaxFNE, four types (E, PV, SST, VIP), wiring and drives tuned so the simulated LFP reproduces the spectrolaminar motif almost identically: gamma (31-150 Hz) superficial, alpha-beta (8-30 Hz) deep, L4 crossover. The H-state carries an adaptation term (AdEx-like, through HDP) and a balance term (E/I). The tuned column is then the building block of the omission paradigm (separate lineage, blind to omission here).

Hypothesis [H]: PV and SST density peak superficially while SST/PV peaks deep, so gamma (35-90 Hz) is generated mainly superficially and alpha-beta (10-25 Hz) deep.

Prior art (deck v4): motif r 0.44, held-out r 0.30, gamma above alpha-beta in 5 of 11 areas (10 of 11 observed), Q² 0.37 (observed 0.42). Targets for this plan are declared and sealed in item 3, not here.

## Scoring

Score = 4 points per item, 25 items, **0/100 now**. An item scores 4 only when its closing check was run on the sealed state and its receipt is in the repo; otherwise 0 (no partial credit).
Done = two independent skeptic `reviewer` runs (separate contexts, neither sees the other's score nor the builder's reasoning) each score 100/100 against this list and the work. Any item either reviewer fails goes back to the builder; the final score is the lower of the two.

## Checklist

| # | Item | Closing check | Builder | Rev A | Rev B |
|---|---|---|---|---|---|
| 1 | Authorization and lineage: owner or reviewer names this task; todo entry, parent authority, frozen list, one principal delta per edge | entry in `manifests/todo.json`; `project_check` PASS | 4 | 0 | 0 |
| 2 | Source digest verified: every number in the skill re-read from the PDFs; T25 vs T26 parameter choice recorded with reason | table of number, page, source; no unmatched row | 0 | 0 | 0 |
| 3 | Pre-execution seal: targets (motif r, band peak depths, PCA split, LOAO r, Q²), parameter brackets, scoring rule, stop states, seeds | sealed spec committed and pushed before any item 11+ run | 0 | 0 | 0 |
| 4 | Anatomy to column: LANSA densities, 11 areas, 21 bins; type fractions per bin | max abs fraction error computed and reported; deck v4 reached 0.021 (source value, threshold UNSET) | 0 | 0 | 0 |
| 5 | Hypothesis profiles: PV, CB(SST), CR(VIP), SST/PV per bin and area; hypothesis restated against the data with its falsifier | profile figure and table; peak depths of PV, SST, SST/PV reported | 0 | 0 | 0 |
| 6 | Four Izhikevich parameter sets with units, dt, rheobase per type, mapped to the Tahvili AdEx roles | rheobase table; each type fires above its rheobase at its drive (closes P-043) | 0 | 0 | 0 |
| 7 | Wiring declared (explicit mode): type-pair and layer rules, Tahvili asymmetry (SST→PV strong, PV→SST ~0, SST→SST ~0), VIP→SST | wiring file; JaxFNE realized edge counts equal configured | 0 | 0 | 0 |
| 8 | H-state specified in TFNE form `x : A : y` with adaptation `h_w` and balance `h_b`; HDP channel mapping stated | spec file; JaxFNE accepts it; `validate()` valid | 0 | 0 | 0 |
| 9 | Adaptation via HDP matches AdEx `w` (a, b, τw) on one neuron | separate tests, each failing on pre-fix code first: (a) spike-triggered jump size and sign; (b) continuous decay equals τw within dt discretization; (c) subthreshold `a (v − EL)` coupling sign; (d) `drive_bias` enters as a negative current, units mapped to pA by a stated factor; (e) no unintended plasticity: `edge_weight` and all other coordinates unchanged, bias returns to start after activity, fresh per run, no clipping at ±50 in range; (f) trace error vs AdEx reference below a sealed tolerance (UNSET) | 0 | 0 | 0 |
| 10 | Balance term bounded and switchable; H moves measurably (not 1.0 to 1.0024) | H range reported; HDP-off ablation changes output | 0 | 0 | 0 |
| 11 | Tahvili replication on an isolated E-PV-SST module: source: ~35 Hz, PV leads SST ~7 ms (T26, T25), SST/PV ratio up lowers frequency | peak frequency and phase delay measured over the sealed seed count (UNSET), thresholds UNSET | 0 | 0 | 0 |
| 12 | Physiological regime: E, PV, SST, VIP all active after onset, rates in the sealed band | per-type rates and CV; no silent type | 0 | 0 | 0 |
| 13 | Drive and gain scan on a sealed grid, circuit selected by held-out area (leave-one-area-out) | selection table; fold agreement count | 0 | 0 | 0 |
| 14 | Analysis port verified on observed data: multitaper 2 Hz, relative power, 21 bins, PCA, PLSR (4 comp), Procrustes | reproduces manuscript PCA 90.93/99.87%, Q² 0.42, r 0.71 within sealed tolerance | 0 | 0 | 0 |
| 15 | Simulated motif: gamma peak superficial, alpha-beta peak deep, L4 crossover, per area | peak bin per band per area with interval; count of areas with gamma peak superficial to alpha-beta peak, against observed 10/11 and deck v4 5/11 (threshold UNSET) | 0 | 0 | 0 |
| 16 | Frequency attribution: gamma 35-90 Hz traced to the superficial E-PV loop, alpha-beta 10-25 Hz to deep SST-dominated activity | band power by type-resolved current per depth bin | 0 | 0 | 0 |
| 17 | Causal test of the hypothesis: flatten or reverse the SST/PV profile at fixed inhibitory count | deep alpha-beta peak moves or vanishes; if not, hypothesis recorded as falsified | 0 | 0 | 0 |
| 18 | Structure-function stats on simulated LFP: conservation, LOAO r, Q², PCA split vs observed (72/28) | numbers beside deck v4 baseline and observed | 0 | 0 | 0 |
| 19 | Visualization contract: V0 schematic, V1 initial 1 s raster, V2 final raster, V3 atlas | four paths named in the record and report; V0, V1 precede tuning | 0 | 0 | 0 |
| 20 | Seeds and held-out robustness: sealed seed counts (UNSET), tuning and evaluation seeds disjoint, per-area spread, uncertainty and effect sizes | spread table; no result rests on one seed | 0 | 0 | 0 |
| 21 | Ablations: H off, VIP off, PV off, SST off, adaptation off | per-ablation motif change table | 0 | 0 | 0 |
| 22 | Known residuals handled: edge bins (0.48 vs 0.86), alpha-beta depth, irregular firing (E CV 0.14) | each residual fixed, or logged open with its number | 0 | 0 | 0 |
| 23 | Column block interface frozen for omission: inputs, outputs, H-state, parameters; no omission information anywhere upstream | interface file and firewall grep clean | 0 | 0 | 0 |
| 24 | Receipts: result JSON with `verdict`, SHA, config, seeds, JaxFNE version; tier-0 test added; sealed files immutable | `project_check` PASS; `test_sealed_artifacts_unchanged` PASS | 0 | 0 | 0 |
| 25 | Final report: arithmetic reconciled (H3), claims tagged, open issues in a table, negatives stated as negatives | report passes both reviewers' recount | 0 | 0 | 0 |

Current score (2026-11-05): builder **4/100**, reviewer A 0/100, reviewer B 0/100 (no reviewer has run). The goal is met only at 100/100 from both reviewers.

Builder basis: item 1 closed (entry `SCI-LAMINAR-MOTIF-1` in `manifests/todo.json`; `project_check --offline` PASS and tier-0 216 passed on the pushed commit a460ebd). Not scored, with reason:

| Item | State | Why it scores 0 |
|---|---|---|
| 2 | digest 97/97 matched | the T25-or-T26 set choice is UNSET (owner) |
| 3 | structure approved | every numeric threshold UNSET; spec not yet amended with values |
| 8 | rule implemented | no TFNE `x : A : y` spec file and no balance term `h_b` yet |
| 9 | tests a-e pass (8 tests, 6 of 6 mutants killed) | (f) AdEx-reference trace tolerance UNSET and untested; full `project_check --offline` on the commit carrying the rule not yet run (suite-lock wait timed out 2026-11-05, requeued); unpushed |

## Order

1-3 (authority, sources, seal) first. 4-10 build the column and the H-state, no tuning. 11-13 regime. 14-18 motif and hypothesis. 19-22 visuals, robustness, residuals. 23-25 handoff. Items 4 and 14 can run in parallel; 9 and 10 share the HDP files.

## Open points for the owner

| Point | Why it matters |
|---|---|
| Task authorization (item 1) | `AGENTS.md` limits work to `next_authorized_task` or a reviewer-named task |
| T25 vs T26 AdEx parameters differ: SST Δ 1.5/2 mV, SST b 25/80 pA, E EL -60/-65 mV, E b 130/50 pA (SST EL and all PV values equal) | pick one set; recorded in item 2 (OPEN-36) |
| Cell-type map CB~SST, CR~VIP, NG=E is an assumption; the Disney papers on the overlap are not in the komposer repository | alters the density-to-fraction step in item 4; owner to supply the PDFs |
| Neuron class for the minimal model: AdEx emitter (validated CAMINOS reference) or Izhikevich-TFNE (OPEN-40) | decides where E0 can be checked and whether item 6 is on the critical path |
| Native-to-pA factor for `drive_bias` and the H units of `h_w`, `h_b` | `drive_bias` is the chosen path (rule built, tests a-e pass); the factor is a design choice still to record in item 8 |

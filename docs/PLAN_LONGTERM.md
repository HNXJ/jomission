# Long-term plans (10)

Date: 2026-10-31. Status: proposed; no task is authorized (`next_authorized_task` is null).
Objective chain (`docs/HANDOFF_CURRENT.md`): generic cortical operation → functional recurrence →
zero-tonic active basin → propagation → T_R → functional hierarchy → Gen-2 freeze → blind omission.
Minimal-model track: `docs/PLAN_MINIMAL_MODELS.md` (scoreboard 75/100 at 6dfd540; MIN-02, MIN-04,
MIN-05 open; MIN-06 OPEN, not authorized).

## Rules every plan follows

- Loop: simulate → analyze → reason → optimize → evaluate → back to simulate. "Optimize" means a
  pre-sealed, log-spaced search over declared parameters, one lineage per search. No retuning after
  results (`NO_RETUNE_AFTER_RESULTS`).
- Each cycle: spec committed before the run; results committed separately; gate via
  `jomission.harness.gates` (`prospective_v2`); `jomission-lineage-review` before any result is cited.
- Interpretation and any estimand choice stay with the main model and the owner. Workers do sweeps,
  gate runs, regenerations and claim checks (`rules/worker-tier.md`).
- Visualization contract V0 → V1 → (science) → V2 → V3 for every model lineage.
- Stop at the first failed gate or at an owner decision. A blocked plan step scores or reports
  nothing; it does not get a workaround.

## Summary

| # | Plan | Objective | Blocked on | Lane |
|---|---|---|---|---|
| P1 | Minimal model core | S-13 PASS (VIP and PV silence fixed) and N = 50/100/200 rate bands | owner: silence criterion (S-13 wording), MIN-02 mapping | A |
| P2 | Checklist A (habituation, adaptation) | CK-A 8 pts: items 1–10 each with a receipt | owner: habituation schedule, trial order, seed (MIN-04) | A |
| P3 | Checklist B (spectrolaminar 90%) | CK-B 7 pts: PCA cumulative variance ≥ 90% (2 PCs) vs chance floor | owner: LANSA densities into the repo (MIN-05) | B |
| P4 | Literature and parameter provenance | every weight and probability traces to a passage (S-11, S-10) | ingest Tremblay 2016, Pi 2013 (MIN-02) | B |
| P5 | PV state dependence (V2.1 background) | explain PV_STATIONARITY / MICROSTATE FAILs; one mechanism lineage | owner choice: background design route | B |
| P6 | E-I regime mechanism (SCI-EI-REGIME-1) | counterfactual interventions on PV recruitment, VIP veto, SST advantage | reviewer direction after REPLAY-058 | C |
| P7 | Whole-system propagation and T_R | WHOLE_SYSTEM_PROPAGATION from UNRESOLVED to PASS/FAIL; burst structure | P6 result or reviewer order | C |
| P8 | Engine and numerics | JaxFNE 0.5.8 identity, noise default, float32 floors, test tiers | none (engine lane) | C |
| P9 | Gen-2 freeze and hierarchy | freeze criteria for the functional hierarchy; blind-omission readiness | P5, P7 PASS; reviewer authorization | (after A–C) |
| P10 | Governance, state and visualization | stacks current, V0–V3 complete per lineage, atlas unavailable entries closed | none | docs lane |

## Plans

### P1 Minimal model core
- Objective: S-13 each class fires, none silent (smoke: 1 VIP neuron silent, 0 PV silent); S-14 class
  rates against bands (PV 12–25, VIP 8–15, SST 6–12 Hz). Current smoke: PV and SST outside bands,
  reported not gated.
- Loop: smoke run (N = 100) → per-class rate and silence report → log-spaced sweep of the one drive
  parameter (DC input current, units read from the JaxFNE convention, not assumed) → receipt → review.
- Gate: S-13 and S-14 with the ratified criteria; S-12 finite at 1000 ms; S-15 H bounded.
- Owner decisions: whether S-14 bands gate (decision 3); the S-13 silence definition.
- Lane A. Worker runs the sweeps; main model reads the rates.

### P2 Checklist A: habituation and adaptation
- Objective: CK-A (items 1–10, 0.8 each). Item 4 needs the schedule: stimulus sequence, trial order,
  seed, ≥ 1000 s exposure.
- Loop: naive vs habituated arms on one seed set (only exposure history differs) → perturb each new
  parameter at 0, boundary and non-default → realized-change check → omission windows and slot
  check (`jomission/paradigm/epochs.py`).
- Gate: `prospective_v2`; perturbation receipts; omission firewall (no omission outcome informs
  construction).
- Owner decision: the schedule values (or the rig log path). Nothing runs until they are sealed.
- Lane A, after P1 sweeps are sealed.

### P3 Checklist B: spectrolaminar 90% pattern
- Objective: CK-B (items 1–10, 0.7 each). Statistic fixed by the LANSA pipeline: PCA cumulative
  explained variance ≥ 90% (2 PCs), against a surrogate chance floor. Source:
  `local/inputs/lansa/LANSA_Analysis_Pipeline.m` lines 218–222 (copy; see `manifests/local_inputs.json`).
- Loop: pin LANSA input by blob → build 11 areas × 21 bins → check realized E/PV/SST/VIP fractions
  per bin against densities → canonical wiring → PCA on simulated and observed profiles → surrogate
  null → held-out areas.
- Gate: reference check of the ported PCA against MATLAB output on observed data before use (item 8);
  SL2 rule: no H tuning to reproduce the pattern. F4 is a positive control only.
- Blocked until the LANSA cell densities are in the repo (MIN-05).
- Lane B.

### P4 Literature and provenance
- Objective: every weight and probability in `docs/minimal/connectivity.json` cites a passage (S-10,
  S-11). Open: CB→SST and CR→VIP marker mapping (MIN-02); Tremblay 2016 and Pi 2013 not ingested;
  PV→VIP probability 0.0556 is a rule-of-three bound, not an estimate.
- Loop: ingest source → extract the passage → compare to the value → flag mismatch → owner ruling.
- Gate: claim check by a worker against the passage text; mismatches go to the owner, not edited.
- Lane B. Owner decides every value that changes.

### P5 PV state dependence (V2.1 background)
- Objective: explain why PV state depends on history at identical tonic (V2.1_PV_MICROSTATE and
  PREDECESSOR UNRESOLVED; PV_STATIONARITY FAIL; V2.1_BACKGROUND FAIL, held).
- Loop: transplant tests (M1 FAIL, M2 PASS recorded) → microstate measurement → one mechanism
  hypothesis written as a new lineage → counterfactual run → evaluate against the sealed negatives.
- Gate: sealed negatives stay negative; a changed criterion is a new lineage. Lineage review before
  any PASS.
- Owner choice: whether a new background design is authorized (SCI-V21-BACKGROUND).
- Lane B, after P4 (shared minimal and PV code paths).

### P6 E-I regime mechanism (SCI-EI-REGIME-1)
- Objective: test the sealed mechanism (PV recruitment deficit, VIP intrinsic state veto, SST state
  advantage) with interventions declared before running.
- Loop: baseline replay (REPLAY-058 reproduced) → one intervention per lineage (e.g. PV drive,
  VIP adaptation, SST gain) → classify each against the sealed labels → reconcile arithmetic (H3).
- Gate: `prospective_v2`; ONE_DELTA per lineage; interventions sealed before they run.
- Blocked on reviewer direction after REPLAY-058 (disposition open).
- Lane C.

### P7 Whole-system propagation and T_R
- Objective: WHOLE_SYSTEM_PROPAGATION (UNRESOLVED; historical receipt QUARANTINED); BURST_STRUCTURE
  UNRESOLVED; then T_R.
- Loop: propagation probe on the current engine → stimulus latency per area → comparison against the
  periodic-mode and intrinsic-generator PASS receipts → sweep → evaluate.
- Gate: stale-evidence check (receipts at the result commit; engine version 0.5.8); no quarantined
  receipt cited as PASS.
- Lane C, after P6 (shared network code).

### P8 Engine and numerics
- Objective: JaxFNE identity and behaviour at 0.5.8. Known: ≥ 0.5.6 native-noise coefficient differs
  (1.1180 vs 0.5 at dt 0.1 ms); probe arrays differ under 0.5.5 and 0.5.7; float32 cross-process
  floor ~1e-4 on motif means.
- Loop: reproduce the migration receipt at 0.5.8 → pin every default that changed → determinism check
  (same seed, same bytes) → regression on the sealed probes.
- Gate: tier 0 and the migration receipt; the battery names its tier (`manifests/test_tiers.json`).
- Lane C. Engine work runs one suite at a time (suite lock).

### P9 Gen-2 freeze and hierarchy
- Objective: freeze criteria for the functional hierarchy, then blind omission readiness. Not started:
  it needs P5 and P7 PASS, or an owner order that accepts a different entry.
- Loop: define freeze spec → sweep the hierarchy parameters declared in the spec → freeze receipt →
  blind-test authorization request to the reviewer.
- Gate: a new blind test needs a reviewer-authorized lineage (BLIND-OMISSION retired 2026-09-19).
- Blocked by the reviewer. No work before authorization.

### P10 Governance, state and visualization
- Objective: `current_state.json` and `HANDOFF_CURRENT.md` current; every result lineage has V0–V3
  artifacts; the three UNAVAILABLE atlas entries (REPLAY-058 manifest) are closed or justified.
- Loop: after each lane's seal → run `python -m jomission.harness.handoff` → `project_check` →
  regenerate V0–V3 → update stacks.
- Gate: `project_check` PASS; manifest counts equal the record.
- Docs lane. Runs after each seal; no science decisions.

## Execution

- Lanes A, B, C run in parallel, at most three subagents at a time (`CLAUDE.md` §6). Each writer has
  its own worktree. Lanes share no file: A owns `jomission/minimal/` and `docs/minimal/`; B owns
  the LANSA and PV code; C owns the engine and network code. P10 runs after each seal.
- Dispatcher checks every worker diff and re-runs its acceptance command before integrating.
- A reviewer reads every science result before it is cited.
- Nothing starts without an authorized `next_authorized_task`; this plan authorizes none.

## Open owner decisions

1. Habituation schedule values or rig log path (P2).
2. LANSA cell densities into the repo (P3).
3. Whether S-14 class bands gate (P1).
4. Whether a new background design is authorized (P5).
5. Reviewer direction after REPLAY-058 (P6, P7, P9).

# Jomission handoff — current state (2026-09-16, revised after V2.1b class-resolved ISI)

## 1. Authority (verify before mutating)

```text
main HEAD:                commit carrying this revision; parent 972ffcb
                          (merge v21b-isi-classes). Verify with §10.
scientific authority:     V2.1  934a718 via merge 5e8911b (V2_LOCAL_OPERATION_FAIL)
                          V2.1b c01bc01 (seal) -> c01913a (V2_LOCAL_OPERATION_FAIL) via merge 18697db
                          V2.1b ISI 8e61fe2 (rule) -> 3f7aa5f (E_NOT_IRREGULAR) via merge 972ffcb
spec:                     results/generic_substrate_v2_spec.json rev2 + amendment 1 (class-resolved
                          ISI gate, full rate bands; prospective only)
design lineage (merged):  v21b-symmetry @ b4dc4f1 via merge bb409c7 (H5 PRIVATE_STOCHASTIC_DRIVE_JUSTIFIED)
provenance correction:    results/v21_provenance_correction.json (V2.1 plant carries native noise; §13)
V2 spec branch:           generic_substrate_v2 (rev2 spec). origin = ae49af2; a local ref
                          may point at fd56085 (drift, unused)
working tree:             clean (verify: git status --short -> empty)
JaxFNE:                   0.4.24 (C:\Python314\Lib\site-packages, wheel install)
jomission install:        editable -> E:\repos\jomission (guarded by tests/test_environment_provenance.py)
environment:              win32, Python 3.14.3, CPU plant with JaxFNE native noise (seeded, reproducible)
CI:                       verify with gh (§10)
Pages:                    enabled (build_type=workflow); live https://hnxj.github.io/jomission/ ; no V2.1b entry yet
```

Receiving Actor: re-run the verification commands (§10) and confirm every
mutable identity above before any mutation. If anything differs, STOP.

## 2. Project objective

```text
generic cortical operation -> functional recurrence -> zero-tonic active basin
  -> propagation -> T_R -> functional hierarchy -> Gen-2 freeze -> blind omission
```

Omission stays downstream because Gen-2 may not use omission-success
outcomes to construct the substrate (omission firewall); physiology is the
question asked after the substrate passes, never a tuning target before.

## 3. Current gate

`V2_LOCAL_OPERATION_FAIL` for V2.1 (static column, 3 tonic regimes, native
noise 0.5*N(0,1)/step) and for V2.1b (V2.1 + mean-controlled private shot
noise, 9 predeclared cells).

OBSERVED V2.1 (results/v21_{low,mid,high}.json):

```text
LOW:  E/PV/SST exactly 0.0 Hz; VIP 2.5 Hz
MID:  E/SST/VIP ~10.7 Hz; PV exactly 0.0 Hz
HIGH: all classes ~20-22 Hz, drift <3%, sync guard clean
ISI-CV gate fraction = 0 (all regimes); E rate-CV <= 0.004
```

OBSERVED V2.1b (results/v21b_*.json, v21b_lineage.json):

```text
realization: 9/9 cells, class mean ratio within 1%, sigma/mu within 0.7%
E rate-CV:   <= 0.053 in all 9 cells (gate 0.3)
ISI-CV (pooled over classes) >= 80% only where E > 30 Hz:
             mid_sm2p0 E 34.7; high_sm2p0 E 61.1; high_sm1p0 E 36.3
every cell with E in 2-30 Hz fails ISI-CV (best: low_sm2p0, 0.675-0.734)
sync guard clean in all cells
```

OBSERVED V2.1b class-resolved ISI, low_sm2p0 (results/v21b_isi_classes.json):

```text
E   frac_in 0.683 / 0.733 / 0.750 (gate 0.8); CV median 0.52 (q10 0.47, q90 0.58)
PV  1.0 (CV ~0.73)    SST 0.78 / 0.91 / 0.90    VIP 0.0 (CV ~1.7, ~130 Hz)
```

Status (reviewer disposition 2026-09-16):

```text
H5 private stochastic drive:  PRIVATE_STOCHASTIC_DRIVE_JUSTIFIED -> EXECUTED -> INSUFFICIENT_ALONE
E_TEMPORAL_IRREGULARITY:      PARTIAL
E_RATE_HETEROGENEITY:         FAIL
```

Interpretation (INFERRED): private independent noise raises temporal
irregularity partially and leaves mean firing propensity identical across E
cells; temporal irregularity != rate heterogeneity. Established scope:
mean-controlled shot noise alone is insufficient over the sealed family; other
shot parameters are untested. VIP at ~130 Hz with CV ~1.7 in low_sm2p0 exceeds
the spec VIP band (amendment 1): the shot background is not class-generic.

## 4. Falsified / retired (do not rediscover)

| lineage | question | verdict | why stopped | artifact/commit | reopen? |
|---|---|---|---|---|---|
| 7 recurrent families | scalar/motif/slow-I/slow-E/delay/modular/tonic-transfer | falsified | closed==cut / basin dominance | results/gen2_seal.json | NO |
| global HDP scale (s=2..36) | does amplitude fix geometry? | SCALE_INSUFFICIENT | E/I co-scaling preserves silent-only | hdp-efficacy-scale fe79fd2 | NO |
| pathway-selectivity inverse | min selectivity for cortical root? | FEASIBLE-minimal (E-out gain 54-61, I fixed) | solved; implemented next | selectivity-inverse c35a4ac | superseded by S6-S9 |
| selective native HDP | transfer to native? | PASS (shape, bitwise-I, common mod) | passed | s6-selective-native 14fdf99 | kept as qualified |
| coarse geometry/stability | S7 roots + S8 14-state linear | PASS as reductions | native manifold failure (below) | s7-geometry 23c56d5; s8-stability ac4422e | as reductions only |
| native basin S9 | finite active basin? | ACTIVE_BASIN_FAIL | silence/sync dominate; FP unreachable | s9-basin ef32fcc | NO |
| basin access B | tiny vs off-manifold? | OFF_NATIVE_MANIFOLD | zero linger, direct jumps | b-basin-access 34996af | NO |
| native geometry N1-N4 | intermediate native region? | NO_INTERMEDIATE | 99.9% corner dwell; 235/235 divergent coarse matches | n-native-geometry b5b1417 | NO |
| microstate N5 | causal fate owner? | NONE_IDENTIFIED | no flip; KW0/UCLAMP modulatory only | n5-microstate b239f4e | NO |
| C-min family | repair further? | STOOD_DOWN (scope-bounded) | above arc; redesign cheaper | results/cmin_family_seal.json (main) | NO |
| propagation (old plant) | FF/FB transmission? | STRUCTURAL_ONLY | closed==cut everywhere | results/propagation.json | stale after plant change |
| V2.1 operation | ordinary cortical regime? | FAIL (above) | regular, homogeneous despite native noise | v2-1-operation 934a718 | superseded by V2.1b |
| V2.1b shot drive | does private shot noise fix V2.1? | FAIL (above) | E rate-CV <=0.053; ISI-CV only above E band | v21b-shot-drive c01913a | NO further noise brackets |
| duration scratch | sustained drive | VOID (tonic-contaminated) | never evidence | results/duration_postmortem.json | NEVER |

## 5. Remains valid

JaxFNE 0.4.24 migration PASS; generic finite-state HDP capability;
HDP = Hidden-state Dependent Plasticity; qualified plastic/history
machinery; configured->realized->executed->effective harness;
drive-additivity invariant; estimator invariant (bin width, aggregation,
baseline, sign); continuation/checkpoint semantics; Gen-1 immutable
baseline; omission firewall; Pages machinery (fetch-depth:0 full clone
required for lineage checks); all negatives above; H1 single-cell E
limit-cycle bound (noise-free numpy assays).

## 6. H5 decision evidence (results/h_sensitivity.json, results/h_history.json)

Unchanged from the H5 seal except where §13 supersedes wording. The private
stochastic column is now tested (V2.1b): temporal irregularity partially
yes (pooled, class-unresolved), rate heterogeneity no. Intrinsic
heterogeneity is the next H5 mechanism class: H1 shows E detuning is
available (E-a gain 0.845, zero rheobase shift) but no E ISI mechanism
(max 0.001). Secondary hypothesis retained: SST-c/VIP-c single-cell bursting.

## 7. Next authorized actions (in order; V2.2 locked)

```text
1. DONE: class-resolved ISI check -> E_NOT_IRREGULAR (merge 972ffcb)
2. intrinsic-heterogeneity design checkpoint (single-cell numerics only; no
   V2.1 battery / plant execution):
   hypothesis: fixed low_sm2p0 shot background + one zero-mean intrinsic
               dispersion -> V2.1 gates (class-resolved ISI per amendment 1)
   H1 per parameter p: S_r = dr/dp, S_rh = dI_rheobase/dp, S_CV = dCV_ISI/dp;
      strong rate authority, limited class-mean rate/rheobase shift, no
      bursting/silence pathology, native biological meaning
   H2 derive the width needed to raise E rate-CV 0.053 -> >=0.3 without
      lowering E ISI frac_in below 0.68-0.75, from single-cell sensitivity
      under the measured shot background
   H3 one parameter, one deterministic zero-mean distribution; width bracket
      {below, at, modestly above} the predicted requirement
   H4 next execution gates E ISI directly (>=80% of active E in [0.5,1.5]);
      class viability enforced for all classes
   stop: INTRINSIC_HETEROGENEITY_{FEASIBLE, INFEASIBLE, UNRESOLVED}
3. STOP for review. Only FEASIBLE authorizes another V2.1 execution.
forbidden: retuned or additional noise, joint multi-parameter dispersion,
           HDP rescue, topology changes, tonic re-derivation, V2.2 recurrence
```

Known conflict for step 3: low_sm2p0 VIP ~130 Hz exceeds the VIP [1,80] band
(amendment 1). E-only dispersion is unlikely to move VIP (E->VIP mean input
~0.16 vs VIP tonic 22.67; INFERRED). Reviewer must decide which rate gate
applies before any execution.

## 8. Invariants and traps

Drive schedules add to emitter tonic; pure perturbation means
schedule-only delta; every transient estimator carries bin width,
aggregation, baseline, sign; no parameter movement equals memory;
static fixed point does not imply native basin; current magnitude does
not imply causal authority; structural connectivity does not imply
functional propagation; configured, realized, executed, effective are
distinct; population irregularity does not imply per-cell ISI
irregularity; pooled ISI-CV fraction does not imply E irregularity;
temporal irregularity does not imply rate heterogeneity; sealed negatives
are not retuned away; one principal delta per lineage edge (274c9a9
recorded exception, results/lineage_notes.json); unknown is not PASS;
F-table interp overestimates near threshold jumps — use exact assays for
tonic derivation; cross-process float32 floor ~1e-4 on active-motif w/I
means (rates exact).

JaxFNE `compile_step_fn(kernel="baseline")` without `noise_scale` adds
0.5*N(0,1) per neuron per step; "deterministic" needs `noise_scale=0`.
The V2.1 battery (tests/test_v21_operation.py) is frozen and blob-pinned by
tests/test_v21b_operation.py; its docstring is stale (§13) and stays
unedited. The V2.1 battery holds ~32 GB peak RSS per regime process.
Scripts must import jomission from this checkout
(tests/test_environment_provenance.py).

## 9. Repository map (smallest authoritative set)

```text
results/v21_{low,mid,high,lineage}.json   V2.1 evidence + verdict
results/v21_vectors.json                  sealed tonic vectors
results/v21_provenance_correction.json    native-noise correction (§13)
results/v21b_vectors.json                 sealed V2.1b shot bracket
results/v21b_{low,mid,high}_sm{2p0,1p0,0p5}.json, v21b_lineage.json   V2.1b evidence + verdict
results/v21b_isi_rule.json, v21b_isi_classes.json   class-resolved ISI (low_sm2p0)
results/h_defect.json                     H0 defect seal (wording superseded in part, §13)
results/h_history.json                    H3 history + viable region (wording superseded in part, §13)
results/h_sensitivity.json                H1 64-assay table
results/h_lineage.json                    H5 verdict
results/generic_substrate_v2_spec.json    V2 rev2 spec (branch generic_substrate_v2)
jomission/qualification/symmetry.py       H1 assay code
tests/test_v21_operation.py               V2.1 battery (frozen)
tests/test_v21b_operation.py              V2.1b battery (shot proxy over frozen run_regime)
tests/test_v21_provenance.py              reproduces native-noise finding (~20 s)
tests/test_v21b_symmetry.py               H0-H5 drivers (rewrites h_sensitivity/h_lineage when run)
docs/PROJECT_HANDOFF.md                   historical (superseded; Poisson context §63-64)
site-src/manifest.json                    publication allowlist + lineage edges
```

## 10. Verification commands (from E:\repos\jomission)

```text
git log --oneline -1 && git status --short && git branch --show-current
git merge-base --is-ancestor c01913a HEAD && git merge-base --is-ancestor b4dc4f1 HEAD && git merge-base --is-ancestor 3f7aa5f HEAD && echo lineages-merged
python -c "import json; print(json.load(open('results/v21b_lineage.json'))['verdict'])"
pip show jaxfne | Select-String Version
python -m pytest tests/test_environment_provenance.py tests/test_pages_build.py -q -p no:cacheprovider
python -m pytest tests/test_v21b_operation.py -q -p no:cacheprovider -k "moments or seal_consistent or wiring"
gh run list --repo HNXJ/jomission --limit 1 --json conclusion,status,headSha
```

Full suite NOT required (slow plant batteries write results/*.json); run
targeted tests only unless the next task changes a shared surface.

## 11. Open uncertainties (material, unresolved)

* Class-rate gate for the next execution: battery lower bound only vs spec
  bands (low_sm2p0 VIP ~130 Hz) — §7.
* Silent E cells escape the ISI gate but raise E rate-CV (amendment 1 open item).
* VIP->E pathway is omitted from the V2.1 battery's recorded currents (8400 edges).
* PV single-cell silence at I=2.8 vs F-interp 10 Hz: threshold lies
  between grid points; tonic derivation must use exact assays.
* HDP re-entry point is unspecified (only: after basin gate).
* Whether intrinsic dispersion that spreads E rates preserves shot-driven
  temporal irregularity (untested; §7 step 2 designs it).
* Shot parameters outside the sealed 9-cell family untested.
* Cross-process 1e-4 floor on active-motif w/I (rates exact).
* Whether SST-c/VIP-c bursting entrains E irregularity (secondary lineage).
* Pages has no V2.1b entry.

## 12. Fresh-Actor instruction

> Reconstruct the state from repository evidence before acting. Treat this handoff as a routing map, not authority over executed artifacts. Independently challenge the current conclusion before implementing the next step. If repository evidence contradicts this handoff, STOP and surface the conflict. Make the smallest authorized scientific delta, verify it, seal it, and stop at the first failed gate.

## 13. Superseded statements (kept for provenance)

The V2.1 plant as executed carries JaxFNE native noise
(results/v21_provenance_correction.json, reproduced by
tests/test_v21_provenance.py). These statements are false as stated:

```text
40fb739 handoff §3:   "V2_LOCAL_OPERATION_FAIL (static deterministic column, 3 tonic regimes)"
40fb739 handoff §3:   "uniform tonic drowns heterogeneous weights and identical cells fire identically"
40fb739 handoff §7:   "HDP detached, deterministic baseline otherwise"
40fb739 handoff §6:   private stochastic drive "untested here"
934a718 message:      "no irregularity source"
test_v21_operation.py docstring: "deterministic (no manufactured noise)"
h_defect.json:        "Static deterministic column ... No irregularity/heterogeneity source exists in the frozen candidate."
h_history.json:       "the C-min V2.1 plant itself was never tested with noise"; "V2.1 frozen candidate measured deterministic"
```

Current reading: V2.1 contained weak private stochastic input and remained
highly regular and homogeneous.

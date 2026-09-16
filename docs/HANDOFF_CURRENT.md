# Jomission handoff — current state (2026-09-16)

## 1. Authority (verify before mutating)

```text
main HEAD:                fd56085 (Pages: V2.1 OBSERVED/FAIL entry)
scientific authority:     934a718 content via merge 5e8911b (V2.1 V2_LOCAL_OPERATION_FAIL)
open branch:              v21b-symmetry @ b4dc4f1 (H0-H5 sealed, pushed, UNMERGED)
H artifacts live ONLY on v21b-symmetry (not on main): results/h_defect.json,
  results/h_history.json, results/h_sensitivity.json, results/h_lineage.json,
  jomission/qualification/symmetry.py, tests/test_v21b_symmetry.py.
  Access via `git show v21b-symmetry:<path>` (§10). Do not duplicate them onto
  main outside a merge.
V2 spec branch:           generic_substrate_v2 (rev2 spec, pushed)
working tree:             clean (verify: git status --short -> empty)
JaxFNE:                   0.4.24 (C:\Python314\Lib\site-packages, wheel install)
environment:              win32, Python 3.14.3, deterministic CPU plant
CI:                       green — pages workflow success on fd56085, 5e8911b, 55549df
Pages:                    enabled (build_type=workflow); live https://hnxj.github.io/jomission/ (verified serving)
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

`V2_LOCAL_OPERATION_FAIL` (static deterministic column, 3 tonic regimes).
H5 design verdict: `PRIVATE_STOCHASTIC_DRIVE_JUSTIFIED` (unmerged branch).

OBSERVED V2.1 phenotype (results/v21_{low,mid,high}.json):

```text
LOW:  E/PV/SST exactly 0.0 Hz; VIP 2.5 Hz
MID:  E/SST/VIP ~10.7 Hz; PV exactly 0.0 Hz
HIGH: all classes ~20-22 Hz, drift <3%, sync guard clean
ISI-CV gate fraction = 0 (all regimes)
E rate-CV ~= 0.003 (HIGH)
```

Interpretation (separate from observation): INSUFFICIENT_SYMMETRY_BREAKING —
uniform tonic drowns heterogeneous weights and identical cells fire
identically. Not established as a noise deficit (see §6).

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
| V2.1 operation | ordinary cortical regime? | FAIL (above) | symmetry defect | v2-1-operation 934a718 | superseded by H5 next step |
| duration scratch | sustained drive | VOID (tonic-contaminated) | never evidence | results/duration_postmortem.json | NEVER |

## 5. Remains valid

JaxFNE 0.4.24 migration PASS; generic finite-state HDP capability;
HDP = Hidden-state Dependent Plasticity; qualified plastic/history
machinery; configured->realized->executed->effective harness;
drive-additivity invariant; estimator invariant (bin width, aggregation,
baseline, sign); continuation/checkpoint semantics; Gen-1 immutable
baseline; omission firewall; Pages machinery (fetch-depth:0 full clone
required for lineage checks); all negatives above.

## 6. H5 decision evidence (results/h_sensitivity.json, results/h_history.json)

64 single-cell assays (E/PV/SST/VIP x a/b/c/d x 0.8/0.9/1.1/1.2 at HIGH
tonic) + sealed history. Claim classes marked.

| | intrinsic heterogeneity | private stochastic drive | structural heterogeneity |
|---|---|---|---|
| break population synchrony | DERIVED yes (E-a gain 0.845, zero rheobase shift: phases dispersible) | INFERRED yes (independent per-neuron fluctuations) | INFERRED yes (sparse asymmetric coupling desynchronizes) |
| per-cell temporal irregularity | OBSERVED no (E ISI-CV max 0.001 all params; limit-cycle bound) | INFERRED yes (fluctuating drive -> ISI variance; untested here) | UNKNOWN (balanced-chaos possible; no evidence sought) |
| mean-rate preservation | OBSERVED single-cell shifts bounded except E-b/SST-b (rheo shift 0.75/3.0 — excluded) | INFERRED preservable via mean-controlled design (H3 prescription) | UNKNOWN (rebalancing cost unestimated) |
| confound risk for memory assays | low (static dispersion) | MEDIUM: continuous stochastic support must not become the memory explanation; isolate by design | low-medium |
| prior evidence | none (untested at network) | AGAINST the failed regime only: 2kHz/amp2.0 -> CV 0.344 (<0.5 bar); high-rate limit inferred clock-like; prescribed mean-vs-variance sys-id never executed | none |
| new degrees of freedom | 1 (one param dispersion width) | 2 (rate x amplitude at fixed mean) | many (topology) |
| uncertainty | can it reach E ISI? OBSERVED no | calibration untested; H3 bounds the viable region (lower-rate/larger-amp shot regime, sigma/mu~1+) | highest |

Second hypothesis retained (not selected): SST-c/VIP-c single-cell
bursting (ISI 0.89/1.35, bounded means) as a later intrinsic lineage;
it cannot reach E's gate without network proof. H4 rule was repaired
mid-course to require E-reachability (history shows both versions).

## 7. Next authorized action (ONE task only; V2.2 locked)

Goal: re-execute the V2.1 battery with calibrated private stochastic
drive; everything else frozen (column, cells, static s=1 efficacies,
tonic vectors LOW/MID/HIGH, HDP detached, deterministic baseline otherwise).

```text
parent commit:        main HEAD at execution time (verify; expected fd56085+)
single delta:         ADD private per-neuron stochastic drive (shot-noise,
                      independent across neurons); no other change
bracket:              NOT YET PREDECLARED — next Actor derives a mean-
                      controlled rate x amplitude grid (H3 sys-id: hold
                      mu = lambda*A*tau fixed per class operating point;
                      avoid 2kHz/amp2.0 and the high-rate continuous limit;
                      private Bernoulli/Poisson per neuron, fixed seeds),
                      seals it in results/v21b_vectors.json pre-execution
frozen:               V2.1 plant + tonic vectors + HDP detached + no V2.2
measurements:         V2.1 gates verbatim (rates, ISI-CV>=80%, E rate-CV,
                      sync guard, drift, currents recorded-not-tuned)
acceptance:           V2_LOCAL_OPERATION_{PASS,FAIL,UNRESOLVED}; subset
                      reported, no regime selected
stop states:          PASS -> stop, V2.2 still locked until review
                      FAIL/UNRESOLVED -> stop, no retuning (G3/G4)
forbidden:            HDP rescue, cell retuning, topology changes, tonic
                      re-derivation after results, V2.2 recurrence work
```

Starting point (INFERENCE, not authority — derive independently):
lambda in {50, 200, 800} Hz x amplitude at matched per-class mean;
verify mean preservation vs deterministic V2.1 rates before gating.

## 8. Invariants and traps

Drive schedules add to emitter tonic; pure perturbation means
schedule-only delta; every transient estimator carries bin width,
aggregation, baseline, sign; no parameter movement equals memory;
static fixed point does not imply native basin; current magnitude does
not imply causal authority; structural connectivity does not imply
functional propagation; configured, realized, executed, effective are
distinct; population irregularity does not imply per-cell ISI
irregularity; sealed negatives are not retuned away; one principal
delta per lineage edge (274c9a9 recorded exception, results/
lineage_notes.json); unknown is not PASS; F-table interp overestimates
near threshold jumps — use exact assays for tonic derivation;
cross-process float32 floor ~1e-4 on active-motif w/I means (rates exact).

## 9. Repository map (smallest authoritative set)

```text
results/v21_{low,mid,high,lineage}.json   V2.1 evidence + verdict (on main)
results/v21_vectors.json                  sealed tonic vectors (on main)
results/h_defect.json                     H0 defect seal (v21b-symmetry ONLY)
results/h_history.json                    H3 Poisson history + viable region (v21b-symmetry ONLY)
results/h_sensitivity.json                H1 64-assay table (v21b-symmetry ONLY)
results/h_lineage.json                    H5 verdict (v21b-symmetry ONLY)
results/generic_substrate_v2_spec.json    V2 rev2 spec (branch generic_substrate_v2)
jomission/qualification/symmetry.py       H1 assay code (v21b-symmetry ONLY)
tests/test_v21_operation.py               V2.1 battery (frozen, on main)
tests/test_v21b_symmetry.py               H0-H5 drivers (v21b-symmetry ONLY)
docs/PROJECT_HANDOFF.md                   historical (superseded; Poisson context §63-64)
site-src/manifest.json                    publication allowlist + lineage edges
```

## 10. Verification commands (from E:\repos\jomission)

```text
git log --oneline -1 && git status --short && git branch --show-current
git show v21b-symmetry:results/h_lineage.json | python -c "import json,sys; print(json.load(sys.stdin)['verdict'])"
pip show jaxfne | Select-String Version
python -c "import json; [json.load(open(f)) for f in ['results/v21_lineage.json','results/v21_vectors.json']]; print('JSON valid')"
python -m pytest tests/test_pages_build.py -q -p no:cacheprovider
gh run list --repo HNXJ/jomission --limit 1 --json conclusion,status,headSha
```

Full suite NOT required (slow plant batteries); run targeted tests only
unless the next task changes a shared surface.

## 11. Open uncertainties (material, unresolved)

* PV single-cell silence at I=2.8 vs F-interp 10 Hz: threshold lies
  between grid points; tonic derivation must use exact assays.
* s61 boundary-marginal statuses (historical C-min line; retired with it).
* HDP re-entry point is unspecified (only: after basin gate).
* VIP viability under stochastic drive untested.
* Cross-process 1e-4 floor on active-motif w/I (rates exact).
* v21b-symmetry unmerged (intentional: merge only after review).
* Stochastic bracket un-derived (next Actor's first deliverable).
* Whether SST-c/VIP-c bursting entrains E irregularity (secondary,
  needs its own lineage if stochastic fails).

## 12. Fresh-Actor instruction

> Reconstruct the state from repository evidence before acting. Treat this handoff as a routing map, not authority over executed artifacts. Independently challenge the H5 conclusion before implementing it. If repository evidence contradicts this handoff, STOP and surface the conflict. Make the smallest authorized scientific delta, verify it, seal it, and stop at the first failed gate.

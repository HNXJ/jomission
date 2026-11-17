# OPEN-38 stage 2: construction-gate ruling request (2026-11-17)

Self-contained packet for the owner's ruling. It holds no Allen-derived parameter or argument. Details and full
receipts: `STAGE2_O38S_2026-11-17.md` (sections "Smoke mode" and "Independent review of the failure").

## 1. Gate specification (frozen text, quoted)

- Spec `SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md` §5, negative control (a), line 102: "For rules into or out of
  PV/SOM, realized edge count within the binomial 99.9% interval of p · N_pre · N_post for that arm."
- §7, line 137, study stop: "construction invariance failing on E cell count, E parameters, per-cell drive or the
  PV/SOM binomial check".
- §7: "Not retuned: arms, layouts, seeds, estimands, rule fixed at seal." §5: thresholds "are not revised after any
  o38s outcome is seen".
- Implementation `scripts/open38_construction_check.py` 9-11 and 104-111: `binom.interval(0.999, n, p)` per rule
  (two-sided, 0.05 % per tail); pass only if every rule is inside. There is no correction for the number of rules
  tested. The spec text is silent on multiplicity.

## 2. Evidence

| Item | Value |
|---|---|
| Run | `run_o38s_stage2.sh smoke` at HEAD 584922c; receipt `results/laminar_e2/o38s_s2_construction.json` (sha256 2f1eb000…), log `o38s_s2_stage.log` |
| Scope | 6 arms × seeds 1-5 = 30 arm-seed pairs × 26 PV/SOM rules = 780 binomial tests; the other 5 gates (sizes, E→E, E parameters, drive, geometry) pass for all 30 pairs |
| Failure | shape, seed 1, `e_pv_S` (local E→PV in unit S): 421 511 edges; interval [417 543, 421 496]; z +3.32. The weights of that rule equal ref's |
| Outcome under the frozen gate | **FAIL**, and the receipts exist, so the outcome is not UNKNOWN. The driver stopped; no stage-2 simulation, execution manifest or verdict exists |
| Reproducibility | The construction is a deterministic function of the seed: the independent reviewer rebuilt shape seed 1 in a separate process and counted 421 511 again. A rerun under the frozen gate fails again |
| Independent review (read-only) | Builder, counting, expected value, variance and spec ambiguity excluded. The sampler is Bernoulli per pair, so the count is exactly binomial. p 0.139996605 was recomputed from the spec independently. Arms at one seed share the rule's random stream, which leaves about 130 independent tests per stage. Class: chance excursion under uncorrected multiplicity. False-stop probability of the frozen gate with a correct builder: about 12-18 % per stage |
| Stage 1 | 260 tests, 0 outside; passes under the frozen gate and under any correction below |

## 3. Decision requested

"Proceed under the existing frozen gate, without waiver or redefinition" gives FAIL, and stage 2 stays stopped,
because the failing count is deterministic. The ruling needed is one of:

| Option | Gate | Stage-2 consequence | Detects |
|---|---|---|---|
| A | frozen, unchanged | stop stands; stage 2 does not run (no reseeding: seeds are frozen) | — |
| B | add a family-wise level, 5 % over all 780 tests: per-test α 6.4e-5, \|z\| ≤ 4.00 (Bonferroni); independent of the inferred stream structure | 421 511 passes; smoke reruns at a new freeze | p, N or wiring errors ≥ about 0.6 % relative on this rule (80 % power); not smaller shifts, and not an offset shared by all arms |
| C | as B over the about 130 effective streams: α 3.9e-4, \|z\| ≤ 3.55 | passes; smoke reruns | about 0.5 %; relies on the inferred stream structure |
| D | the shape arm (or shape seed 1) is "not evaluable" under the frozen gate; the other 5 arms proceed | the primary family loses one comparison; Holm runs over 7 | — |

B, C and D change how the frozen gate is applied after a failure was seen, which spec §5 and §7 forbid without an
owner ruling. Recommendation: B, recorded as a post-hoc owner amendment, with stage 1 re-checked under it.

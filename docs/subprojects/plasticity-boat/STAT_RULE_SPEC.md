# Plasticity Boat: statistical-rule specification for O-6 (DRAFT, not frozen)

Status: DRAFT, 2026-11-25. Not frozen. O-6 stays open until the items in §4 are decided and §3 items are checked.
Base: `dev@a25ebf9`. `protocol_v0.md` is unchanged (status PROPOSED, 2026-11-19). Labels below:
- **P** determined by the protocol (cited).
- **O** owner-approved new criterion, 2026-11-25; not in the protocol text; must be labelled as new when frozen.
- **U** undetermined; escalated in §4. Not inferred.

## 1. Specification

| Element | Rule | Label | Source |
|---|---|---|---|
| Unit of inference | the start | P | `protocol_v0.md` §5 line 50; `stage02_skeleton.md` line 34 |
| Outcome per start | occupancy = fraction of intervals in the final 20% of the episode (errors[480:600], pre-step, K = 600) with distance ≤ 1.0. Integer count k out of 120 | P | `protocol_v0.md` §4 line 42; `stage02_skeleton.md` line 32 |
| Pairing | same starts, same circuit initialisation and seed for every strategy | P | `protocol_v0.md` §2 item 4; `stage02_skeleton.md` line 33 |
| Contrast | for strategy S_k (S2–S8): the per-start difference occupancy(S_k) − occupancy(S1), same start index | P (paired); statistic U1 | `protocol_v0.md` §5 line 50 |
| Resampling | bootstrap over starts: resample start indices with replacement, apply the same indices to both arms of the contrast | P (bootstrap over starts), derived (joint resampling is what "paired" requires) | `protocol_v0.md` §5 line 50 |
| Resamples and generator | 10 000, `numpy.random.default_rng(0)` | P for count and seed; stream handling U5 | `protocol_v0.md` §5 line 50 |
| Interval | 95% percentile interval of the bootstrap distribution | P | `protocol_v0.md` §5 line 50 |
| Effective-variation gate | at least 0.9·N distinct values per contrast, else degenerate and no CI. N = number of starts in the run (20 dev, 40 held-out). Distinctness on the integer counts k (see U4) | P (rule); U4 (definition) | `protocol_v0.md` §5 line 51; `stage02_skeleton.md` line 34 |
| "Changes occupancy" | the paired 95% CI excludes 0, and the gate passed | O (approved 2026-11-25) | `B01_decision.md` §4.2 |
| Stopping | none. A negative result is reported unchanged | P | `stage02_skeleton.md` line 34 |
| Splits | development 2025 (debugging and falsification only, never to choose a strategy); held-out 2026, 40 starts, used once after the freeze and all §2 controls pass | P | `protocol_v0.md` §5 lines 48–49 |

### 1.1 Consequence for the owner's rule

- A contrast is reported with its interval whether or not the gate passes. The "changes occupancy" verdict in the O row is defined only where the gate passes.
- The verdict is per contrast. With seven contrasts, the family-wise error is not controlled by the O rule as stated (U2).

## 2. Executability

- Executable now: counts k per start (from the recorded distance trajectories), paired differences, the gate as a count of distinct differences, and the percentile CI. Nothing in §1 requires a parameter not in the protocol, except the items below.
- Not yet executable: the statistic (U1), the multiplicity treatment (U2), the stream convention (U5) and the gate's distinctness definition (U4). Writing code for these now would choose them.
- Check to run before freezing: a synthetic positive (a known shift of 0.2 in occupancy, gate passes, CI excludes 0) and a synthetic negative (identical arms, CI contains 0). Both must be reproduced exactly with the chosen U-decisions. This is a test of the code, not evidence about the boat.

## 3. Checks required before freezing (from the owner)

| Check | Status |
|---|---|
| CI construction, sampling unit, multiplicity defined | Partly: construction and unit are P; multiplicity U2 is undecided |
| Effective-variation gate defined and executable | Partly: rule P; definition U4 undecided; not yet executable |
| S4 isolates channel identity without unrelated changes | Not verifiable: no S4 or S5 code exists, and the F3 parameters are not in the repo (§5) |
| S8 unsupervised mechanism and controls verified against frozen definitions | Not verifiable: no S8 code exists, and the STDP parameters are not in the repo (§5) |

## 4. Escalations (owner decisions; each needs a choice, not an inference)

- **U1 Statistic.** The contrast statistic is the mean of the per-start differences, the median, or another summary. The protocol names the contrast but not its statistic.
- **U2 Multiplicity.** The protocol states no correction across the seven contrasts S2–S8 vs S1. Options: none (each contrast at 95%), Holm, Bonferroni, or another. The owner's O rule, as recorded, is uncorrected. This choice sets what "excludes 0" means at family level.
- **U4 Gate definition.** "Distinct values" on integer counts k with exact equality, or after another convention? And N = 40 for the held-out run only, or N = the run's starts in every case?
- **U5 Stream convention.** One `default_rng(0)` shared across the seven contrasts (order-dependent), or a fresh generator with seed 0 per contrast? The protocol gives the seed but not how it is used.
- **U6 Scope of the gate.** Is the gate applied to each contrast separately, or to the whole family? The protocol says "per contrast" (§5 line 51), so the default is per contrast. Confirm.

## 5. S4 and S8: verification status and blockers

**Finding 1. Most strategies are unimplemented.** `circuit100.py` states "no plasticity, no competition". It runs the static circuit, which corresponds to S1 (a baseline run, not a contrast). The programme code contains no update rule at all, so S2, S3, S4, S5, S6, S7 and S8 are unimplemented in the tracked code. The protocol (§3 line 37) names only S4 and S8. That understates the gap: S2, S3, S5, S6 and S7 are also missing. The protocol's status must be corrected before freezing (escalated, E6 below).

**Finding 2. The parameters are not in the repo.** S5 is defined as "channel-specific three-factor eligibility, reward −Δe (F3)". S6 is "running-mean reward baseline (F3b)"; S7 "shuffled distance signal (F5)"; S2 "F2", S3 "F6". These F-labels refer to the 10-neuron rule definitions in "contract §13", which is cited in several files but is not tracked in `dev`. Without it, no learning rate, eligibility time constant, reward form or bound is available. S8 ("reward-free pair-based STDP") needs STDP amplitudes and time constants, which appear in no tracked file.

**What would verify S4 and S8 (acceptance tests, not run):**
- S4 isolates channel identity: a test that S4 and S5 produce identical updates when the channel assignment of the eligibility is the only difference, with identical inputs, seeds and reward. Any other difference in the update function fails the test.
- S8 is reward-free: the update is invariant to any reward or distance signal (perturb the reward; updates are bit-identical). The STDP sign convention and time windows must match the frozen definition.
- Both must pass the protocol §2 controls (zero drive, stationary input, positive control) before use.

**Blockers:** the contract §13 F-rule definitions and the S8 STDP parameters, which the owner must supply or approve. No parameter was chosen here.

## 5a. Owner rulings, 2026-11-25

- **U2 Multiplicity: Holm** across the seven contrasts S2–S8 vs S1. The construction of Holm with CIs is not yet decided (§5b).
- **U1 Statistic: mean** of the per-start paired differences.
- **E5 Parameters: the owner will supply contract §13** and the S8 STDP values. Pending. Nothing is built until they are in the repo.
- **E6 Protocol correction: done** (`protocol_v0.md` line 37, factual, no criterion changed).
- Still open: U4 (gate definition), U5 (stream convention), U6 (gate scope), U7 (Holm step-down tie handling, §5b).

## 5b. Holm construction: DECIDED by owner, 2026-11-25

Chosen: Holm step-down on two-sided paired-bootstrap p-values, p = 2·min(P*(d ≤ 0), P*(d ≥ 0)), across the seven contrasts S2–S8 vs S1. A contrast changes occupancy if its Holm-adjusted p < 0.05.

**Consequence to confirm at freeze.** The approved rule was written as "the paired 95% CI excludes 0". With Holm on p-values, the family-level verdict is the adjusted p, not the unadjusted CI. The per-contrast 95% percentile CI is still reported with every contrast. This follows from the choice of p-value construction; it is not a separate decision. Confirm it when §5 is frozen.

Not decided here: the order of the step-down follows the observed p-values, and ties are not specified (U7, below).

## 6. Escalations, consolidated

| ID | Decision | Blocks |
|---|---|---|
| U1 | Contrast statistic | freeze |
| U2 | Multiplicity method | freeze; meaning of "changes occupancy" |
| U4 | Gate definition | executable gate |
| U5 | Stream convention | reproducibility |
| U6 | Gate scope | confirmation only |
| U7 | Holm tie handling and step-down order | freeze of the verdict |
| E5 | Source of F3, F3b, F5, F6 parameters (contract §13) and S8 STDP parameters | S4, S5, S6, S7, S8, S2, S3 |
| E6 | Correct protocol §3 line 37 to list all unimplemented strategies | freeze |

## 7. Not decided here

No freeze. No run. No parameter value. No change to the protocol, criteria, seals or the O rule as approved.

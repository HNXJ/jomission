# B01 decision record: central question of the Plasticity Boat programme

Status: DECIDED by owner, 2026-11-25 ("Plasticity competition is B01"). Scope decided 2026-11-25 (strategies, not mechanisms). Wording closed 2026-11-25 (§4.5). §4.2 and O-6 open. §4.3 and §4.4 reconciled 2026-11-25 (see each).
Scope: recorded here only. No todo item, no paper, no run, no change to sealed criteria or the frozen protocol.
Verification: `dev` commit `5db3b7a` was checked against the tree at `ef0a766`: only this file and `GAP_ANALYSIS.md` changed; `protocol_v0.md` is byte-identical.

## 1. Decision

- Central question is plasticity competition. Fixation acquisition is a candidate mechanism or measurable consequence, not a replacement.
- The competition B01 investigates is between plasticity strategies, not within a mechanism (§4.1).
- Programme framing retained (`_temp_plasticity_boat/artifacts/programme/manuscript/plasticity_boat/main.md`; `protocol_v0.md` §1 and §3).
- Do not create separate papers. Do not expand scope without a demonstrated scientific need.
- Identifier: B01 is the planning-file id (`_temp_solmind:artifacts/fixationboat_biorxiv_stage15_review.md` line 21). It appears in no programme file and is a label only.

## 2. Central question

Approved wording, 2026-11-25 (owner ruling on §4.5; "competition" follows protocol line 4, "primary eight-strategy competition"):

In the eight-strategy competition, which candidate plasticity strategies (S2–S8) change fixation occupancy relative to the fixed-weight reference S1, under the programme's directional-observability requirements?

Superseded: "Does competition between plasticity mechanisms explain the emergence, stabilization, or suppression of learned behavior, beyond what simpler noncompetitive mechanisms predict?" (owner text, 2026-11-25). Within-mechanism competition is out of scope (§4.1). Kept in git history of this file.

## 3. Estimand and outcome

- Estimand: each candidate strategy's paired contrast with S1 on fixation occupancy (`protocol_v0.md` §5, line 50). No ranking, no selection (§4.5).
- Outcome: fixation occupancy, threshold 1.0, final 20% of intervals (`protocol_v0.md` §4, proposed; EX-20). Final distance is reported with it (§4).
- Unit of inference: the start. Bootstrap, 10 000 resamples, `default_rng(0)`, 95% percentile CI. Effective-variation gate: at least 0.9·N distinct values per contrast, otherwise degenerate with no CI (§5, line 51).
- Starts: development 2025 (never used to choose a strategy); held-out 2026 (40 starts, used once after the owner freeze and all §2 controls pass).

## 4. Open items

### 4.1 Scope. RESOLVED IN PART, 2026-11-25 (owner)

B01 investigates competition between plasticity strategies (the eight-strategy comparison), not competition within a plasticity mechanism.
- S5 vs S4 is a comparison of rules. It is not evidence that either mechanism implements competition for a constrained resource, and it is not labelled "competitive plasticity."
- Competition within a mechanism needs a mechanistic definition, matched controls, a protocol freeze and a new lineage, all owner-approved. Out of scope for B01.

### 4.5 Central-question wording. CLOSED 2026-11-25

Owner chose "contrast vs S1" (§2). Checked against protocol §5 (paired contrasts vs S1, no selection rule) and line 4 (competition term). The proposed "best supports" was not adopted because it implies a selection rule the protocol does not define. Its wording, which is a proposal and not a verified quotation, is kept in git history.

### 4.2 Falsifier. OPEN: one owner decision

Owner falsifier (proposal, not a verified quotation): "competitive plasticity fails to improve held-out predictions or produces no distinctive experimental signature."

Reconciled to the strategy scope: no candidate strategy S2–S8 changes fixation occupancy relative to S1 on the held-out starts (seed 2026).

Protocol check:
- The protocol contains no falsifier and no rejection rule for any contrast. It reports paired CIs (§5) and an effective-variation gate (degenerate contrasts get no CI). Its only other falsification role is the development-start rule (§5, line 48).
- It has no held-out predictive model, so "predictions" has no referent in the protocol.
- "Changes" needs an operational rule, which the protocol does not give. Options for the owner: (a) the 95% CI excludes 0; (b) the CI excludes 0 and the effect is at least a stated threshold; (c) another rule.
- Any of these adds a rejection criterion. Not applied here.
- F-b and F-c in `GAP_ANALYSIS.md` §5 are my earlier proposals, not protocol text. They are not part of B01's falsifier.

### 4.3 Control mapping. RECONCILED 2026-11-25, pending owner confirmation of the noncompetitive control

Labels are the protocol's own (§2, §3, §4). Corrections to my earlier draft are marked.

| Item | Protocol label | Role in B01 | Pass condition or purpose | Implemented |
|---|---|---|---|---|
| S1 | static weights, baseline | reference of every contrast | fixed-weight dynamics | not stated in protocol |
| Zero drive; stationary input | negative controls (§2.3), same starts (§4) | must keep the boat at R = 5 | artifact check; failure means the pipeline is broken | F0 holds at 5.000 (protocol §2.3; model not named) |
| Directional positive control | §2.2 | must reach p* from every dev start | failure: no competition runs | 10-neuron fixture; EX-9 |
| S2 | H effect without weight change | control | separates drive effect from plasticity | not stated |
| S3 | inhibitory plasticity (H-scaled PV→E, F6) | candidate, not a control | corrected: my earlier draft called it an ablation | not stated |
| S4 | ablates channel identity | channel-identity ablation of S5 | corrected: my earlier draft called it "matched noncompetitive"; no protocol strategy is labelled noncompetitive | no (protocol §3: must be built and pass §2 controls) |
| S5 | primary rule | the rule under test | — | no (100-neuron circuit not built) |
| S6 | reward-baseline control | control | tests running-mean reward baseline | not stated |
| S7 | reward-structure control | control | tests shuffled distance signal | not stated |
| S8 | unsupervised control | control | reward-free STDP | no (protocol §3: must be built and pass §2 controls) |

Falsifiers per control: none in the protocol. Each control's pass or failure is its protocol check above; a control that changes the contrast has no defined consequence.

Open: the owner's requirement "matched noncompetitive plasticity" has no protocol strategy with that label. Under the strategy scope, the question is answered by S1 and S2–S8. Whether a noncompetitive control is still needed is an owner decision. Not mapped further here.

### 4.4 Subordinate hypothesis. RESOLVED 2026-11-25 (no new estimand)

"Fixation acquisition" is not defined in the protocol. The protocol reports fixation occupancy and final distance (§4). Therefore:
- Fixation acquisition is subordinate. It is not a primary estimand and has no separate metric.
- Any claim about acquisition is descriptive, drawn from the same runs, and not confirmatory.
- An acquisition-latency metric would be new and needs owner approval.

## 5. O-6 (strategy list and primary metric). OPEN, narrowed

Remaining before freeze:
1. Owner freeze of protocol §3–§5 (protocol §6: locked on owner freeze).
2. The §4.2 falsifier decision.
3. S4 and S8 built and passing §2 controls before use (protocol §3, line 37).

Already in place: the strategy list (§3, marked proposed) and the metric (fixation occupancy, threshold 1.0, proposed, EX-20).

## 6. Consequences

- Plasticity Boat stays one programme. The eight-strategy protocol remains PROPOSED. This record does not freeze it.
- C4 (stage workflow, unlocated) and C9 (engine compartment support, unverified) remain unresolved.
- Solmind's fixation-boat planning items (B01–B24) are not canonical. Only the owner can make them so.

## 7. Not decided here

- The falsifier operational rule (§4.2) and the noncompetitive control (§4.3).
- The O-6 freeze, any run, any paper or submission.

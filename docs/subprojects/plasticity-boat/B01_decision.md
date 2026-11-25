# B01 decision record: central question of the Plasticity Boat programme

Status: DECIDED by owner, 2026-11-25 ("Plasticity competition is B01"). Formulation PROVISIONAL (see section 4).
Scope: recorded here only. No todo item, no paper, no run, no change to sealed criteria or the frozen protocol.

## 1. Decision

- Central question is plasticity competition. Fixation acquisition is a candidate mechanism or measurable consequence, not a replacement.
- Second decision, 2026-11-25 (owner): the competition B01 investigates is between plasticity strategies, not within a mechanism (section 4.1).
- Programme's existing scientific framing is retained (`manuscript/plasticity_boat/main.md`, message and claim C2; `protocol_v0.md` §1 and §3).
- Do not create separate papers. Do not expand scope without a demonstrated scientific need.
- Identifiers: B01 is the planning-file id (`_temp_solmind:artifacts/fixationboat_biorxiv_stage15_review.md` line 21). It does not appear in any programme file. It is kept as a label only.

## 2. Central question (owner text)

Does competition between plasticity mechanisms explain the emergence, stabilization, or suppression of learned behavior, beyond what simpler noncompetitive mechanisms predict?

## 3. Scientific requirements (owner text, mapped to the programme)

| Requirement | Programme counterpart | Status |
|---|---|---|
| Primary estimand: effect of competitive vs noncompetitive plasticity on the programme-defined behavioral outcome | Behavioral outcome: fixation occupancy, threshold 1.0, final 20% of intervals (`protocol_v0.md` §4, proposed; O-6 pending) | Outcome defined; contrast not defined (see 4.1) |
| Control: matched noncompetitive plasticity | Candidate: S4 (one shared reward for all E→motor synapses; channel identity ablated) | Candidate mapping, not confirmed. Owner to confirm |
| Control: fixed-weight dynamics | S1 (static weights) | Direct match |
| Control: relevant single-mechanism ablations | S2 (H-scaled drive, no weight change), S3 (H-scaled PV→E plasticity), S6 (running-mean reward baseline), S7 (shuffled distance signal), S8 (reward-free STDP) | Candidate mapping. Owner to confirm which are ablations of the competitive mechanism |
| Falsifier: competitive plasticity fails to improve held-out predictions, or shows no distinctive signature | Closest: F-b (S1 and zero drive indistinguishable from S5) and F-c (S4 exceeds S5) in the gap analysis §5 | Differs: owner's falsifier names held-out prediction and a signature; F-b/F-c name the primary metric only. Owner to choose |
| Held-out evaluation | Seed 2026, 40 starts, used once (`protocol_v0.md` §5) | Matches |
| Fixation acquisition as subordinate hypothesis | Fixation occupancy is already the primary metric; it is the outcome, not a separate mechanism | Clarify in the claim map: subordinate hypothesis about a mechanism or consequence |

## 4. Open items (owner rulings)

4.1 **RESOLVED IN PART, 2026-11-25 (owner): B01 investigates competition between plasticity strategies** (the eight-strategy comparison), not competition within a plasticity mechanism. Consequences:
- The primary contrast is between strategies on fixation occupancy. S1 (fixed weights) is the reference control.
- S5 vs S4 is a comparison of rules. It is not evidence that either mechanism implements competition for a constrained resource, and it is not labelled "competitive plasticity."
- Competition within a mechanism (a constrained resource or allocation) is out of scope for B01. Starting it needs a new mechanistic definition, matched controls, a protocol freeze and a new lineage, approved by the owner.
- Still open: the exact wording of the central question (see 4.5), and the O-6 strategy list and metric.

4.5 **Central-question wording. OPEN: check done, one mismatch.** The owner approved strategy wording (2026-11-25) and gave this proposal, which is not a verified quotation: "Which of the eight candidate plasticity strategies best supports fixation occupancy under the programme's directional-observability requirements, relative to the fixed-weight reference S1?"

Check against `protocol_v0.md` (dev, ef0a766 tree):
- §5 (line 50): unit of inference is the start; "Paired contrasts against S1"; bootstrap CI. The estimand is each strategy's paired contrast with S1. There is no ranking.
- §5 (line 48): development starts "never used to choose a strategy". Held-out starts are used once.
- "Best" implies a selection among strategies. The protocol defines no selection rule, no held-out ranking and no multiplicity rule for one. Adding one changes the estimand, which the owner's scope excludes.
- "Directional-observability requirements" matches protocol §1 items 1–2 (directional encoder; positive control must pass).
- "Fixation occupancy" is the proposed primary metric (protocol §4). O-6 is open, so the question depends on it.

Options for the owner (one decision):
- A (recommended): "Which candidate strategies (S2–S8) change fixation occupancy relative to the fixed-weight reference S1, under the programme's directional-observability requirements?" Estimand: paired contrast vs S1, as §5. No new selection rule.
- B: keep "best". Requires a written selection rule, held-out procedure and multiplicity rule, which is a new estimand and a protocol freeze.

Until the owner chooses, the §2 text stays as recorded. §4.5 closes when the chosen wording is written into §2 and checked against this list.

4.2 **Falsifier wording.** Choose between the owner's held-out-prediction falsifier and F-b/F-c, or combine them.

4.3 **Control mapping.** Confirm S4, S2, S3, S6, S7, S8 as the controls and ablations named above.

4.4 **Subordinate hypothesis.** Confirm that fixation acquisition is tested through the fixation occupancy metric and a mechanism or consequence claim, not a separate study.

## 5. Consequences

- Plasticity Boat stays one programme. The eight-strategy protocol remains PROPOSED; it is not frozen by this record.
- O-6 (strategy list and primary metric) is still open. Section 4.1 is the blocker for it.
- Solmind's fixation-boat planning items (B01–B24) are not canonical. Only the owner can make them so.
- Unchanged: C4 (stage workflow, unlocated) and C9 (engine compartment support, unverified) remain unresolved.

## 6. Not decided here

- The competitive contrast (4.1), the falsifier (4.2), the controls (4.3).
- Any run, any freeze, any paper or submission.

# B01 decision record: central question of the Plasticity Boat programme

Status: DECIDED by owner, 2026-11-25 ("Plasticity competition is B01"). Formulation PROVISIONAL (see section 4).
Scope: recorded here only. No todo item, no paper, no run, no change to sealed criteria or the frozen protocol.

## 1. Decision

- Central question is plasticity competition. Fixation acquisition is a candidate mechanism or measurable consequence, not a replacement.
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

4.1 **"Competitive" is not defined in the programme.** The protocol compares eight named strategies. S5 is "primary rule," with channel-specific eligibility; no file calls any strategy competitive or noncompetitive. The primary estimand cannot be computed until the owner defines the contrast (for example: which strategy is competitive and which is its matched noncompetitive control). Options for the owner: (a) the existing S5 vs S4 contrast; (b) a new strategy that adds an explicit resource or normalization constraint, matched to S5; (c) another definition. This is a scientific choice; not decided here.

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

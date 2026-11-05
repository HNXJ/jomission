# AGSDR stages 1 and 2 under amendment 4 (pre-registered; sealed on commit of this file)

Authority: owner ruling Q24 (2026-11-05): proceed to stage 1 with the 4 feasible candidates. Design `SCI-AGSDR-CALIBRATION_design.md` steps 3 to 5, amendments 1 to 4, ruling Q18 (H07). Feasible set: AGS_S0_06, AGS_S0_15, AGS_S0_16, AGS_S0_20 (`docs/receipts/agsdr_h07h11/h07_q18_summary.json`).

## Delegated decisions (owner rule 2026-11-05: the agent takes the recommended option and logs it)
- D1 (OPEN-33): stage-1 slots = the feasible count (4 instead of 8). Stage 2 keeps the top 3.
- D2 (OPEN-18, raw vs normalised J): J = the runner's six soft terms as `score_run` computes them. Each term is 0 inside its band and normalised by band width: B1 mu; B2 CV_rate, mean CV, median Fano, mean rho; B3 population fraction (proxy Efrac, tau omitted, as stage 0). Weight 1/6 is the tie-break (soft_distance).
- D3 (OPEN-23, boundary and peeling): the epsilon predicate exactly as ruled (Q6-AGSDR, additive, epsilon 0.05, `<=` for all k and `<` for some k). It is applied to the cross-seed mean J vector. Peeled fronts are not used, because they depend on order. Rank key: (number of feasible candidates that epsilon-dominate it, cross-seed mean soft_distance, id), ascending.
- D4 (hard gates across seeds): a candidate stays feasible only if it passes every hard gate at every seed of the stage. That is tau-weighted H09 and H05 over all E cells; H07 and H11 are structural, so they are seed-independent for this topology. An infeasible candidate is dropped and reported.

## Design
- Stage 1: the 4 feasible candidates on seeds {0, 1}. Each seed builds the network with that seed, as in the stage-0 runner. VIP drive x2.25 (amendment 4); 2000 ms, dt 0.1 ms. That is 8 runs; the seed-0 runs repeat stage 0 under amendment 4.
- Stage 2: the top 3 of the stage-1 ranking on seeds {0, 1, 2, 3}. Seeds 0 and 1 are the stage-1 receipts (deterministic), so 6 new runs. Reported per soft term: mean, 95% CI (t with 3 df), and spread.
- theta* = first in the stage-2 ranking (same rank key on 4-seed means). It is reported only; advancing it to a ledger edge is the owner's decision (design step 5).
- Runner `scripts/agsdr_stage12_a4.py`; receipts `docs/receipts/agsdr_stage12_a4/`.

## Controls (before sealing)
- C1 (exact): AGS_S0_06 at seed 0 reproduces `docs/receipts/agsdr_stage0_a4/AGS_S0_06.json`: metrics, tau-weighted gates, Efrac and H05. It passed before sealing.
- C2: class tau is single-valued per presynaptic class (E 2 ms, I 5 ms) and n_E = 242, read from the built model.

## Stop rules
If fewer than 1 candidate is feasible at all stage-1 seeds: stop, record the null. If fewer than 3 survive, stage 2 runs on the survivors.

## Limits
- n = 2 seeds (stage 1) and 4 (stage 2), as the design fixes.
- The B3 soft term uses the proxy Efrac (tau omitted), as in stage 0. Its tau-weighted version needs per-population currents that `score_run` does not keep.
- The VIP scale was chosen on two of these candidates (AGS_S0_06, AGS_S0_20) at seeds 0 and 1.

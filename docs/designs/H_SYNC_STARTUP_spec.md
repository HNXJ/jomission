# Sync-term start-up transient (pre-registered, sealed on commit of this file)

Question: is the sync-term collapse (`docs/receipts/h_sync_test`) caused by the start-up transient, where the slow trace P_s is about 0 while the initial synchronous burst makes sigma = max(0, P_f - (1+g) P_s) / max(P_s, eps) huge? Follow-up to `H_SYNC_MECH_spec.md` whose window (t >= 3 s) excluded the transient; the supporting evidence so far is exploratory (`docs/receipts/h_sync_mech/exploratory_early.txt`, n = 1).

## Design
- Network, conditions A (tonic 0.6) and B (tonic 0.9), background as `H_ACTIVE_TEST_spec.md`. **Seeds 0, 1, 2, 3.** Duration 4000 ms, dt 0.1 ms. **Analysis window t in [0, 4000) ms** (no burn-in).
- Arms per (condition, seed): baseline (HDP off; gives p_ref = its mean population rate over the window, Hz); balance-only (`activity_balance`, defaults); sync (`activity_balance_sync`, defaults of the sealed sync test); fixref (`activity_balance_sync_fixref`, defaults, p_ref as above). `record_weight_trace` False (recording only).
- Runs: 2 conditions x 4 seeds x 4 arms = 32.
- Per-run quantities: window mean rate (Hz); rate in 250 ms bins; sigma(t) = max(0, P_f - 1.5 P_s) / max(P_s, 1e-3) from the sync arm's `aux_trace` (P_f = mean of the fast trace, P_s = mean of the slow trace); peak sigma in [0, 500) ms; minimum over time of the neuron-mean H (sync and fixref arms); recovery time = first 250 ms bin with sync rate >= 0.8 x balance bin rate (reported as > 4000 if none).

## Hypotheses (each can fail; "all" = every one of the 8 (condition, seed) runs)
- **HS1**: sync window rate <= 0.5 x balance window rate, all 8.
- **HS2**: peak sigma in [0, 500) ms >= 10 in the sync arm, all 8.
- **HS3**: fixref window rate >= 0.8 x balance window rate, all 8.
- **HS4**: minimum neuron-mean H of the sync arm <= 0.2 (H floor is 0.1) and of the fixref arm >= 0.6, all 8.
- Support for the start-up (ratio-blowup) explanation requires HS1 to HS4 all true. Any false: reported as observed; no reclassification. n = 4 seeds per condition, descriptive; no inferential test.

## Controls (all must pass before interpreting)
1. Reproduction: sync window rate over [0, 2000) ms of these runs equals `h_sync_test` receipt `bar_r_Hz` for sync, A and B, seeds 0 and 1 (tolerance 1e-9; exact spikes are deterministic).
2. Null: kappa_s = 0 reproduces balance-only spikes exactly (4000 ms, A seed 0).
3. Determinism: repeating the sync arm (A, seed 0, 4000 ms) gives identical spikes.
4. Unit tests of the balance, sync and fixref rules pass.
Stop rule: any control fails: report, do not interpret.

## Pre-seal note
First control run failed the 1e-9 tolerance because spike sums accumulated in float32 (values agreed to 7 digits, identical counts). Sums are now float64; criterion unchanged. No run data used.

## Limits
- p_ref is from the baseline arm; a probe choice, not a candidate rule.
- Seeds vary the Poisson/engine stream only (network seed fixed at 0); n = 4 per condition.
- Cause of the initial burst is not examined (all neurons fire at start: initial-state property of the engine/network).

## Not decided here
Whether to build a corrected synchrony term; OPEN-27 membership.

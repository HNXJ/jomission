# Hysteresis probe: up and down sweeps with state carried (pre-registered; sealed on commit of this file)

Todo SCI-PV-HYSTERESIS (plan P2, docs/PROGRESS_REVIEW_2026-11-05.md). Question: at a fixed candidate, do the class rates depend on the direction of approach along a drive axis? That is the signature of coexisting attractors (a fold pair). It was suggested by V2.1-coupled-background, where PV at lambda 0.65625 ended at 14.7 Hz or near 0 depending on history.

## Design
- Candidate AGS_S0_06 (stage-0 receipt; built per seed with `agsdr_local_sweep.build_candidate_model`). Seeds 0, 1, 2.
- Axis V: VIP drive factor f in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0), global factor g = 1.
- Axis G: global drive factor g in (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25) multiplying every emitter drive, at f = 2.25 (inside the viable interval of SCI-VIP-DRIVE-PROBE).
- Drive changes act on the emitter drive array only (VIP x f, then all x g, clipped to the builder bounds [1, 8]; the clipped fraction is recorded).
- Arms per (axis, seed), state carried across segments with `jaxfne.simulate(..., continuation=state, return_state=True)`:
  - up: 2000 ms warm-up at the first grid value, then one 2000 ms segment per grid value in ascending order;
  - down: the same in descending order;
  - null_up: as up, with the Poisson seed offset by 1000 (same network, same history, different noise);
  - repeat: warm-up then 4 segments at a fixed value (V: 2.5; G: 0.875).
- Score: class mean rates over the last 1000 ms of each segment (first 1000 ms is relaxation).
- Runner `scripts/hysteresis_probe.py`, one process per seed.

## Detector (fixed before running)
Tolerance per (seed, axis, class): tol_c = max(1 Hz, 3 x the largest change between consecutive repeat segments). A comparison of two arms finds a persistent difference in class c if |difference| > tol_c with the same sign at two adjacent grid values.

## Hypotheses (each can fail)
- HP1 (axis V): up vs down shows a persistent difference at all 3 seeds.
- HP2 (axis G): the same on axis G.
Prediction (lead, stated before running): HP1 false (VIP rate rose smoothly and monotonically with f in the drive probe, with little effect on the other classes); HP2 open.

## Decision rule (per axis)
1. Null control fails (null_up vs up shows any persistent difference at any seed): NULL_CONTROL_FAILED_NO_INTERPRETATION.
2. Hypothesis true: HYSTERESIS_PRESENT, reporting class, interval and sign per seed.
3. Otherwise: NO_HYSTERESIS_DETECTED.
n = 3 seeds: descriptive.

## Controls (before sealing; all must pass)
- C1: continuation is deterministic (same state and model give identical spikes).
- C2: a continued segment differs from a fresh start (state is carried).
- C3: f scales only VIP drive; g scales all drives (with the clip).

## Limits (disclosed)
- One candidate; two axes; seven grid values each. A fold narrower than one grid step can be missed.
- 2000 ms segments: slow drifts (seconds) can mimic or mask history dependence; the repeat arm bounds this only at one value.
- No positive control for the detector in the spiking model (no network with a known fold is available); the null arm is the negative control.
- The g axis also scales the PV boost and tonic drive, as a background-drive analogue; it is not the V2.1 lambda.

## Not decided here
Any change to the calibration or the AGSDR design.

# VIP-drive existence probe (pre-registered; sealed on commit of this file)

Replaces the mean-field plan item P1 (todo SCI-EI-MEANFIELD) as the first test of OPEN-25. Reason: the stage-0 receipts show that VIP is the binding constraint. VIP class mean is 0.4 to 1.7 Hz in 24 of 24 candidates against the band 8 to 15 Hz; the PV band passes in 7 of 24 and the SST band in most. The builder gives PV a class-specific drive boost (x1.7, GEN2_C016) and VIP none: VIP drive after tonic 0.6 is about 1.78 against a VIP rheobase near 4 (builder comments, GEN2_C017). A mean-field model would need noisy transfer functions that the repo does not have (docs/receipts/fi_single_neuron_v1.json is deterministic). A direct spiking probe along the deficit axis has an exact positive control.

Question: does a VIP-only drive scale f bring the network into the viable bands (PV [12, 25], VIP [8, 15], SST [6, 12], E < PV; class means pooled over areas and layers, as in the stage-0 receipts)?
Estimand: class mean rates as a function of f; the set of f with all bands met.

## Design
- Candidates: AGS_S0_06 and AGS_S0_20, the two that pass H05 under the all-E reading (OPEN-26). Model, background and motif gains are taken from their stage-0 seed-0 receipts and built with `agsdr_local_sweep.build_candidate_model`.
- Probe: the emitter drive of each VIP cell is multiplied by f and clipped to the builder bounds [1, 8]. Nothing else changes.
- Grid f in {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}. Seeds 0 and 1. 2000 ms, dt 0.1 ms, stage-0 simulation and scoring (`score_run`). 2 x 2 x 7 = 28 runs.
- Runner: `scripts/vip_drive_probe.py`, one process per (candidate, seed).

## Hypotheses (each can fail)
- HV1 (sign check): VIP class rate increases strictly with f in all 4 (candidate, seed) series.
- HV2 (disinhibition sign): SST class rate at f = 4 is below SST at f = 1 in all 4 series. VIP->SST gain is about 3.5 to 3.9 in these candidates.
- HV3 (existence): for at least one candidate, some f in the grid meets all bands and E < PV at both seeds.
- HV4 (reported, not a pass condition): at each f satisfying HV3, whether all stage-0 hard gates pass at both seeds.

## Decision rule
1. Any control fails: CONTROL_FAILED_NO_INTERPRETATION.
2. HV3 true: VIABLE_BRANCH_ON_VIP_DRIVE_AXIS, reporting the f set and HV4. VIP drive is not an AGSDR design dimension, so adding it is an owner decision; this probe does not change the calibration.
3. HV3 false and VIP in band at some (f, seed): VIP_RECRUITED_BUT_OTHER_BANDS_FAIL, reporting which bands fail.
4. Otherwise: VIP_NOT_RECRUITED_BY_DRIVE_IN_GRID.
n = 2 seeds per candidate: descriptive. No post hoc grid extension in this spec.

## Controls (all must pass before interpreting)
- C1 (positive control, exact): at f = 1, seed 0, the class means equal the stored stage-0 receipt values exactly, for both candidates.
- C2: the probe changes only VIP drive (non-VIP drive arrays identical; VIP drive = clip(f d0)).
- C3: determinism: two 200 ms runs at f = 2 with the same seed have identical spikes; and they differ from f = 1.

## Limits (disclosed)
- One axis (deterministic VIP drive). It does not test VIP synaptic recruitment (E->VIP) or VIP intrinsic parameters.
- The clip at 8 caps f d0; the fraction of VIP cells clipped is reported per f.
- Fresh start from the default initial state; history dependence (OPEN-25, V2.1-coupled-background) is the subject of SCI-PV-HYSTERESIS.
- Class means pool areas and layers (stage-0 definition); per-population limits are in the hard gates (HV4).

## Not decided here
Whether VIP drive enters the AGSDR design (owner); calibration values.

## Pre-seal checks (disclosed)
Controls C2 and C3 passed (200 ms runs). C1 was checked before sealing: f = 1, seed 0 reproduces the stored class means and failed-gate lists exactly for both candidates. Two runner defects were fixed before sealing: the candidate record lacked `mu_total` (now `agsdr_local_sweep.mu_of(values)`), and the bar_r key name. No f > 1 run was made before sealing.

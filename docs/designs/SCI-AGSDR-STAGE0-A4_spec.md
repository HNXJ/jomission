# AGSDR stage 0 under amendment 4 (pre-registered; sealed on commit of this file)

Authority: `SCI-AGSDR-CALIBRATION_amendment_4.md` (owner rulings Q16 and Q17, 2026-11-05). Question: under the frozen VIP drive scale 2.25, which of the 24 stage-0 candidates are hard-feasible, and which meet the B1 class bands?

## Design
- Candidates: the 24 stage-0 candidates (`docs/receipts/agsdr_local/candidate_AGS_S0_*_seed0.json`: values, builder args, gains). Seed 0, 2000 ms, dt 0.1 ms. The stage-0 build, simulation and `score_run` are used unchanged. VIP drive x2.25 is applied after the build (`vip_drive_probe.scale_vip_drive`).
- Hard feasibility uses the rules in effect: the runner gates H01 to H10d; H09 with the tau-weighted Efrac (`agsdr_stage0_correction`, per-candidate tau from `stage0_tau_check.json`, unchanged by the drive change); H05 over all E cells (OPEN-26). H11 is carried from `docs/receipts/agsdr_h07h11` (the amendment changes no edge or metadata). H07 stays NOT_DETERMINED until OPEN-29 is ruled.
- Reported per candidate: class means, B1 band check, every gate, failed gates, runner soft distance.
- Runner `scripts/agsdr_stage0_a4.py`. Receipts `docs/receipts/agsdr_stage0_a4/`. The unamended stage-0 receipts are not edited.

## Prediction (lead, before running)
VIP enters the band for most candidates. In the probe, VIP x2.0 and x2.5 bracket the band for both candidates tested; other classes moved by under 0.5 Hz. Hard feasibility changes little, because H05 (E irregularity) was the binding hard gate and VIP drive barely moves E.

## Controls (before sealing)
- C1 (positive control, exact): at scale 1.0, AGS_S0_06 reproduces the stored stage-0 metrics and gate verdicts. It also reproduces the stage-0 correction values (tau-weighted Efrac, H05 all-E, tau hard pass).

## Output and stop rules
`summary.json`: viable-band set, feasible set before structural gates, set after H11, H07 status, count against the 8 stage-1 slots. If the set after H11 is empty: report and stop. Stage 1 is not authorised by this run; it still needs OPEN-29 (H07), OPEN-18 and OPEN-23 (selection rule), and OPEN-33 (shortfall).

## Limits
One seed, as stage 0. The scale 2.25 was chosen from two candidates at two seeds.

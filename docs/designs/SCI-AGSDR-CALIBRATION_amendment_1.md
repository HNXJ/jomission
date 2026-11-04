# SCI-AGSDR-CALIBRATION: amendment 1 (sampling), additive to the sealed design

Date: 2026-11-04. Authority: owner hint 2026-11-04 (log-scale sweep of free parameters; freezing or restricting any coordinate is allowed).
Amends: docs/designs/SCI-AGSDR-CALIBRATION_design.md (commit 0726c5d). The sealed design is unchanged.

## Change
Stage-0 sampling of the free parameters is log-uniform over each frozen range, not linear. Every free parameter is a positive multiplicative scale or a positive rate, so log-uniform sampling is well defined. The seeded Latin hypercube is applied to log10 of each coordinate, then mapped back to linear values.

Reason: several ranges span a factor of two to three (for example Poisson rate 1000 to 3000 Hz, g_fastEI E->PV 1.4 to 2.0 is near-linear, g_disinh 3.5 to 6.5 spans a factor of about 1.9). Log spacing gives each octave equal weight.

Unchanged: frozen ranges; hard gates; soft objectives; seeds S0 to S2; the Pareto and cross-seed selection; the 52-simulation budget; stop rules.

## Coordinates may be frozen or restricted
Any coordinate may be frozen at its baseline (1.0 for gains; the preset tonic scale for the background) and excluded from sampling. A frozen coordinate is recorded in the stage-0 receipt. This amendment does not freeze any coordinate.

## Stated limitation
The g_rec range (0.25 to 0.45) still excludes the baseline 1.0. It is used as written; see the sealed design.

## Procedure timing
This amendment must be committed before the runner executes stage 0. The runner must print the sampling mode and the log10 bounds it uses.

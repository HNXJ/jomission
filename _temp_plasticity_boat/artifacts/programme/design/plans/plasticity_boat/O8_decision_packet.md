# O-8 decision packet: dev-100 directional gain (2026-11-19)

Decision packet only. No parameter is changed by this file. Source of numbers: `dev100_gate_record.md` (v2 sweep). Uncommitted.

## 1. Model (source)

- `code/dev100_closed_loop.py:139-161` (v1 `make_drive`); `code/dev100_v2.py:129-143` (v2 `make_drive_v2`).
- Signal `s = p + tau*v` (tau = 0: position only). Error `e = ||p||`.
- `z = (-s_x, +s_x, -s_y, +s_y)` for groups (+x, -x, +y, -y). `u = max(0,z) / (max(0,z) + d0)`.
- Non-motor E [20-79]: `g_s * e/(e + d0)`. g_s is the v2 sweep variable.
- Motor E, 4 groups of 5: `G_D * u`. G_D = 24 (v2); v1 used `g * u` with g = 24.
- PV [80-99]: 0.
- Units: rates in Hz (observed). p, d0, tau and currents are not given units in source: unknown. Do not read p as metres.

## 2. Observed (fixed boat, no decoder)

Sweep of g_s at G_D = 24:

| g_s | R2 E (Hz) | R2 | R4 max (Hz) | R4 |
|---|---|---|---|---|
| 24 | 250.0 | FAIL | 325.0 | FAIL |
| 18 | 250.0 | FAIL | 325.0 | FAIL |
| 12 | 189.0 | FAIL | 326.5 | FAIL |
| 9 | 152.2 | FAIL | 300.0 | FAIL |
| 6 | 110.0 | FAIL | 286.0 | FAIL |
| 4.5 | 86.0 | pass | 294.0 | FAIL |
| 3 | 65.0 | pass | 318.5 | FAIL |

Diagnostic sweep of G_D at g_s = 3 (not used by the rule):

| G_D | R3 diff (Hz) | R4 max (Hz) | R4 |
|---|---|---|---|
| 24 | 307.0 | 318.5 | FAIL |
| 12 | 234.0 | 246.5 | FAIL |
| 8 | 144.5 | 155.5 | FAIL |
| 6 | 105.5 | 117.5 | FAIL |
| 4 | 53.5 | 82.2 | pass |

At g_s = 4.5, G_D = 4: R4 fails at 109.4 Hz on a non-motor E neuron (R2).

## 3. Derived

- R2 does not depend on G_D (non-motor E uses g_s only). The R2 values above hold under any G_D.
- At g_s = 3 the R4 pass threshold for G_D lies in (4, 6). The table does not resolve it.
- At G_D = 4: R3 diff 53.5 Hz (rule: >= 5 Hz) and R4 margin 100 - 82.2 = 17.8 Hz.
- G_D 24 -> 4 is a 6x cut in motor drive gain. Force decoding gain changes too (see O-9).

## 4. Not measured

- P1, N1, T05 (criteria C5) under any G_D change: unknown.
- Closed-loop behaviour under any G_D change: unknown.

## 5. Falsifiers (reject the change if any holds)

1. R3 toward minus opposite < 5 Hz at the chosen G_D.
2. Any neuron > 100 Hz in R2 or R3 at the chosen (g_s, G_D).
3. Validation (seed 2025, x20) fails C5 (P1/N1/T05).
4. Held-out seed 2026 used in any way for selecting G_D: lineage invalid.
5. G_D chosen after seeing calibration results and not sealed before C3: post-hoc, reject.

## 6. Frozen-criteria compatibility

- Criteria v2 C1 freezes G_D = 24. Any change is a new labelled criteria version (v3) with owner approval; a changed parameter starts a new lineage.
- Criteria v1 stays as frozen (G = 24, D0 = 1). Not edited.
- 10-neuron fixture: unaffected, unchanged.
- Seeds: C3 calibration default_rng(2027) x10; validation default_rng(2025) x20; held-out default_rng(2026) unused.

## 7. Options (owner)

| # | Option | Effect |
|---|---|---|
| 1 | Criteria v3. Seal a G_D grid inside (4, 6) and g_s = 3 before calibration. Pick G_D, then C3 on seed 2027 and C5 on seed 2025. Seed 2026 stays unused. | Keeps dev-100. New lineage. |
| 2 | Keep v2 as is. | Track stops at C3 step 3 (no (g_s, kappa) passes). Same as stopping. |
| 3 | Drop the dev-100 track. | 10-neuron track continues; dev-100 closed. |

## 8. Related open decision (O-9)

The decoder sums spikes over each 5-neuron motor group, about 5x the force gain of the mean. P1 is reported under the sum. Confirm sum or change to mean (new criteria version). This changes closed-loop gain independently of O-8: decide both before validation.

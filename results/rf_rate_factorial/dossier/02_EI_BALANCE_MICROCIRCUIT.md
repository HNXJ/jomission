# 02 — E/I BALANCE & MICROCIRCUIT (cell A_RFoff_RateRef)

_Subpopulation (E vs PV/SST/VIP inhibitory) firing-rate balance per area, measured from per-unit whole-trial spike counts (`spikes_example_*.npz`) mapped to cell class via the sealed checkpoint neuron metadata. Synaptic-current-level E/I balance is NOT recorded (see §5)._

## 1. Cell-type composition (geometry)

| area | E | PV | SST | VIP | total I |
|---|---|---|---|---|---|
| V1 | 64 | 11 | 9 | 16 | 36 |
| V4 | 62 | 15 | 9 | 14 | 38 |
| FEF | 60 | 13 | 10 | 17 | 40 |
| PFC | 56 | 13 | 14 | 17 | 44 |

Composition is from `ckpt_trial_0010.json` `geometry.neuron_rows` [MEASURED]; 100 neurons per area, 400 total, neuron_id contiguous [MEASURED].

## 2. E vs I mean firing rates (Hz, whole-trial)

**AAAB**
| area | E rate | I rate (PV+SST+VIP) | E/I ratio | tag |
|---|---|---|---|---|
| V1 | 16.003 | 32.848 | 0.487 | MEASURED / DERIVED |
| V4 | 16.003 | 35.683 | 0.448 | MEASURED / DERIVED |
| FEF | 16.003 | 33.651 | 0.476 | MEASURED / DERIVED |
| PFC | 16.003 | 37.527 | 0.426 | MEASURED / DERIVED |

**AXAB**
| area | E rate | I rate (PV+SST+VIP) | E/I ratio | tag |
|---|---|---|---|---|
| V1 | 14.490 | 27.688 | 0.523 | MEASURED / DERIVED |
| V4 | 14.490 | 29.611 | 0.489 | MEASURED / DERIVED |
| FEF | 14.490 | 28.276 | 0.512 | MEASURED / DERIVED |
| PFC | 14.490 | 31.987 | 0.453 | MEASURED / DERIVED |

**BBBA**
| area | E rate | I rate (PV+SST+VIP) | E/I ratio | tag |
|---|---|---|---|---|
| V1 | 15.787 | 32.764 | 0.482 | MEASURED / DERIVED |
| V4 | 15.787 | 35.570 | 0.444 | MEASURED / DERIVED |
| FEF | 15.787 | 33.548 | 0.471 | MEASURED / DERIVED |
| PFC | 15.787 | 37.359 | 0.423 | MEASURED / DERIVED |

E and I rates are MEASURED (per-unit counts → Hz, class-averaged); E/I ratio is DERIVED. These are **spike-rate** proxies of balance, not synaptic current balance.

## 2b. Per-class spike-count structure (population synchrony)

Within a cell class, per-unit whole-trial spike counts are highly degenerate [MEASURED]:

**AAAB**
| class | n | mean rate (Hz) | min | max | #unique rates | silent units | tag |
|---|---|---|---|---|---|---|---|
| E | 242 | 16.003 | 16.003 | 16.003 | 1 | 0 | MEASURED |
| PV | 52 | 43.049 | 42.388 | 43.253 | 5 | 0 | MEASURED |
| SST | 42 | 78.313 | 78.071 | 78.503 | 3 | 0 | MEASURED |
| VIP | 64 | 0.125 | 0.000 | 0.216 | 2 | 27 | MEASURED |

**AXAB**
| class | n | mean rate (Hz) | min | max | #unique rates | silent units | tag |
|---|---|---|---|---|---|---|---|
| E | 242 | 14.490 | 14.490 | 14.490 | 1 | 0 | MEASURED |
| PV | 52 | 32.261 | 31.791 | 32.439 | 4 | 0 | MEASURED |
| SST | 42 | 71.022 | 70.934 | 71.367 | 3 | 0 | MEASURED |
| VIP | 64 | 0.000 | 0.000 | 0.000 | 1 | 64 | MEASURED |

In the intact AAAB trial all 242 E units produce **identical** whole-trial spike counts (74), i.e., a single-valued E-rate distribution; PV/SST are nearly uniform within class; VIP are near-silent (0–1 spikes). This per-class uniformity is MEASURED and indicates strong E-population spike-count degeneracy/synchronization [INFERENCE]. Because spike timestamps are not retained, temporal synchrony (whether the equal counts coincide in time) is `UNRESOLVED`.

Consequence: the per-area E/I ratios above reflect population-averaged rates, but the E population is effectively one synchronized rate, so E-rate 'balance' carries little within-area E heterogeneity [INFERENCE].

## 3. Omission effect on E vs I (AXAB − AAAB, Hz)

| area | ΔE (Hz) | ΔI (Hz) | tag |
|---|---|---|---|
| V1 | -1.514 | -5.160 | DERIVED |
| V4 | -1.514 | -6.072 | DERIVED |
| FEF | -1.514 | -5.374 | DERIVED |
| PFC | -1.514 | -5.539 | DERIVED |

Single-replicate, whole-trial window; descriptive only [DERIVED], no inferential power (design requires n≥4). Slot-resolved omission effect is `UNRESOLVED` from artifacts (no per-ms PSTH retained).

## 4. Layer-resolved E/I rates (Hz)

**V1, AAAB (representative intact trial)**
| layer | nE | nI | E rate | I rate |
|---|---|---|---|---|
| L1 | 0 | 8 | nan | 0.081 |
| L2/3 | 22 | 8 | 16.003 | 35.873 |
| L4 | 9 | 6 | 16.003 | 47.758 |
| L5 | 19 | 8 | 16.003 | 45.631 |
| L6 | 14 | 6 | 16.003 | 40.549 |
Rates MEASURED; note L1 contains no E cells in this realization (all inhibitory) [MEASURED], and some layer/class bins are small (n<10), so per-bin rates are high-variance descriptors.

## 5. E/I synaptic currents — UNRESOLVED

The recording union contains **no** current or conductance traces (no `I_syn`, excitatory/inhibitory current observables are declared or stored). Therefore current-based E/I balance (e.g., E:I conductance ratio, net synaptic drive) is `UNRESOLVED` and is NOT imputed. Only spike-rate E/I balance (above) is measured.

## 6. Across-unit rate variability

Across-unit coefficient of variation of firing rate within a class is `DERIVED` (population spread, not temporal irregularity). Temporal CV (ISI-based) is `UNRESOLVED` — no spike timestamps retained.

## 7. RF input targeting (E vs I)

The RF operator targets V1 L4 E(9) + PV(3) units [MODEL_ASSUMPTION]; drive is applied to these targets. This is an input-side specification, not a measured current.


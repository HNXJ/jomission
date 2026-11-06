# Item 8: H-state specification (draft, 2026-11-06)

Status: DRAFT. Not sealed. Item 8 scores 0 until the owner settles the rows marked UNSET and the closing check below is run. Grammar: `docs/tfne/TFNE_ALGEBRA_PROJECT_SOURCE.md` (TFNE/2). No parser or compiler exists or is authorized, so this file is a specification document, not an executable input.

## Column in TFNE form

```text
x : A : y
C      := {E, PV, SST, VIP}
L      := {L1, L2, L3, L4, L5, L6}          (UNSET: bin-to-layer map, item 4)
A      := CTX[lam]
A.Lk[C]  model := Izhikevich                 (E, PV, SST, VIP: four parameter sets, item 6)
N[A.Lk.c], P[A.Lk]                           (UNSET: from LANSA densities, item 4; sum_c N = N[A])
O[ff], X[lat], X[inh]                        (UNSET: wiring rules, item 7)
x := drive to L4.E and L1/L6 (UNSET, item 12)
y := LFP per depth bin from A.Lk.*.v(t)
```

Realization: `A -> (s, h_0, I)`; execution `(h_t, x_t; s) -> (h_{t+1}, y_t)`.

## Mutable state `h`

| Component | Content | Source of evolution |
|---|---|---|
| `h_dyn` | `v`, `u` (Izhikevich) per neuron | model equations |
| `h_H` | `H`: balance coordinate `h_b` per bin and area (below) | HDP rule, UNSET (row B2) |
| `h_plastic` | `drive_bias` (theta target) per neuron = `-w`; `edge_weight` | `drive_bias`: adaptation rule below; `edge_weight`: frozen, no plasticity in this lineage |
| `h_history` | spike history for the aux mirror | JaxFNE |
| `h_rng` | noise key | JaxFNE |

## Adaptation `h_w` (AdEx-like, through HDP)

Source equations (T25 eq. 1-2, T26 eq. 15-16): `tau_w dw/dt = a (V - E_L) - w + b sum_j delta(t - t_j)`, `I_adapt = -w`.

| TFNE / AdEx | JaxFNE channel (0.5.8) |
|---|---|
| `w` (pA) | `-drive_bias` per neuron, native current units; `aux` mirrors the bias |
| `a` | rule param `a_s` per neuron (per mV), zero for PV |
| `b` | rule param `b` per neuron, applied as rate `-b * spike / dt` (no per-event jump in 0.5.8) |
| `tau_w` | rule param `tau_w_ms` (T25 and T26: 500 ms) |
| `E_L` | rule param `E_L` (mV) |
| feedback into neuron | `current = drive + sched + bias + syn + noise`; bias enters as a negative current |
| unit map native to pA | UNSET (design choice, owner) |

Rule: `jomission/dynamics/adex_hdp.py`, name `adex_adaptation_drive_bias`. It writes only `drive_bias`; no `H`, no `edge_weight`. Per type: PV off (`a_s = 0`, `b = 0`); E and SST on; VIP UNSET (neither source models VIP).

## Balance `h_b` (E/I)

Reference quantity from the manuscript (M): the PLSR proportion term `z(E) / sum z(inhibitory)` per bin. That term is static (density). `h_b` here is a bounded slow state.

| Row | Choice | State |
|---|---|---|
| B1 | `h_b` initial value per bin equals the density term of item 4 | proposed |
| B2 | feedback path: (a) diagnostic only, `h_b` tracks a running E/I current ratio and acts on nothing; (b) `h_b` scales inhibitory gain through the H1b gain path | UNSET (owner): scientific trade-off. Minimal base built first: (a) |
| B3 | bounds `[lo, hi]` and time constant (slower than `tau_w`) | UNSET |
| B4 | off switch | required: HDP off must reproduce the `h_b = h_0` run exactly |

Item 10 requires `h_b` to move measurably (the earlier H range 1.0 to 1.0024 is the failure to beat) and an HDP-off ablation that changes output. Under B2(a) the ablation changes nothing, so item 10 can score only under B2(b) or a revised B2(a) definition. That is the cost of the minimal base.

## Closing check for item 8 (not yet run)

1. This spec accepted by the owner (rows B2, B3, unit map, VIP).
2. JaxFNE accepts the registered rule with `h_b` declared: `validate()` valid on a small model (no TFNE parser, so the mapping above is checked by a test that builds the model from the table values).
3. Receipt in `docs/laminar_motif/`.

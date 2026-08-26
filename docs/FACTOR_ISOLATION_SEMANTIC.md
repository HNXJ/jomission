# Semantic Factor-Isolation Validator — Jomission Factorial v0.2

**Module:** `jomission/ablations/factor_isolation.py`
**Tests:** `tests/test_factor_isolation.py` (6 tests, all catch-or-pass semantics)
**Version:** `factor_isolation.v0.2.0`
**Scope:** fixes the v0.1 defect where the syntactic validator passed on a config
diff despite realized scientific invalidity.

---

## 1. The v0.1 defect

`rf_rate_factorial.v0.1.0` validated interventions by **Δconfiguration only**
(config_hash / hp_hash differences). That predicate passed while the realized
science was broken:

| # | v0.1 failure | Measured facts |
|---|--------------|----------------|
| i | **Cell A misconfigured** | exec_A used hp `f327f9d2ad64cc88` (canonical, `K_HDP=0.003`, RF-ON) with **no `K_HDP=0.0` override**; frozen design A requires `bb8277e7a8e0bca2` (`K_HDP=0.0`, RF-OFF). A failed frozen predicate C1. |
| ii | **185× RF drive energy imbalance** | RFoff uniform all-400 drive: `Σ_{t,i} I_i(t) = 42,480,000` per AAAB trial; RFon V1 L4 E/PV graded: `230,180.8`. Ratio `= 184.6×`. The "RF" factor was a compound of config + graded drive + `K_HDP`, **not an isolated retinotopy toggle**. |
| iii | **Rate effect measured during relaxation** | β_tau was a run-mean contrast over settled (A,B) vs mid-relaxation (C,D) regimes — a transient-sampling artifact — never a measured τ_eff from dense Θ(t). |

The syntactic validator could not see any of these: A's hash *asserted at runtime*
was the canonical hash (so C1's own `CELL_HP_HASHES` check was never wired to the
realized run), the drive-energy difference was invisible to `hp_hash`, and
β_tau was computed from nominal timescale labels.

## 2. What the semantic validator tests

`assert_factor_isolation(cell_pair, realized_inputs, realized_dynamics, strict=False)`
returns a pass/fail report **per semantic check**. It tests **both**:

### (a) Δconfiguration
Checked against the frozen **per-cell** identity, not the canonical label:

- `config_hash_match` — RFoff cells must realize `4f9fdeae7428199a`; RFon cells a
  distinct RF-metadata hash.
- `hp_hash_match` — realized hp_hash must equal `CELL_HP_HASHES[cell]` (catches i).
- `hp_params_match` — realized `K_HDP`, `tau_0_ms`, `K_ctrl`, `K_w_ctrl`, `alpha`,
  `gamma`, H/w bounds vs frozen per-cell params.

### (b) Δrealized inputs/dynamics
Measured from the **generated** drive array (`StimulusSchedule.to_array`) and the
dense Θ(t) trace — not config.

**RF intervention (RFoff→RFon):**
- `target_population` — RFoff drives all-400; RFon targets **V1 L4 E/PV** only
  (12 resolved units, subset of V1).
- `total_input_energy_parity` — `|E_off − E_on|/max ≤ 5%`; isolates retinotopy,
  not energy (catches ii).
- `temporal_envelope` — per-slot drive sums ≈ equal share (±5%), delay/fx slots
  exactly 0, onsets/durations match frozen paradigm.
- `stimulus_identity_energy` — A vs B per-presentation energy within ±5%
  (symmetric blobs).
- `omission_energy` — exactly 0 in the omitted slot (≤1e-6).
- `active_unit_count` — RFoff 400; RFon sparse (6 units at 0.2·max, sparsity 0.25).
- `drive_moments` — internal consistency (mean = energy/(N·steps)) + dispersion
  structure: RFoff std ≈ 0 (uniform), RFon CV ∈ [4, 15] (v0.1 graded CV ≈ 9.4,
  scale-invariant so an energy-normalized RFon schedule still passes).

**Timescale intervention (ref→LONG):**
- `fixed_point_equivalence` — `K_HDP/K_w_ctrl` ratio, `K_ctrl`, `alpha`, `gamma`,
  bounds unchanged; `tau_0_ms` scales rate 200× only (Θ∞ and H∞ invariant).
- `measured_tau_effective` — τ_eff **measured from dense Θ(t)** (scipy curve_fit on
  `Θ(t) = Θ∞ − (Θ∞−Θ0)·exp(−t/τ)`): ref ∈ [2, 8] s (target 4.1 s), LONG ∈
  [400, 1500] s (target 833 s), LONG/ref ≥ 50× (target 203×). Catches iii.
- `relaxation_window` — measurement window must span ≥ 1× τ_eff; a window
  truncated mid-relaxation (e.g. 60 s for τ_eff ≈ 833 s) is flagged.
- `bounds_saturation` — observed H ∈ [0.1, 10], w ∈ [0.01, 10] unchanged.

## 3. Reference numbers are measured, not nominal

All frozen references come from the **real v0.1 cell runs**
(`results/rf_rate_factorial/{A,B}_RF*_RateRef/recording/external_drive_examples.npz`):

- RFoff AAAB total energy `42,480,000`, per-slot `10,620,000`, 400 active units,
  drive mean `2.2967`, std `0` (uniform).
- RFon AAAB total energy `230,180.8`, per-slot `≈57,528`, 6 active units,
  `n_target = 12` (V1 L4 E/PV), sparsity `0.25`, drive mean `0.0124`, std `0.1170`.
- Timescale: ref τ_eff,E ≈ 4.1 s (saturates in ~12 s); LONG τ_eff,E ≈ 833 s
  (76% of asymptote at 1200 s); ratio 203×.

The validator therefore evaluates realized facts — if the realized schedule or
trace disagrees with these measured references or with the isolation invariant,
it fails.

## 4. Test coverage (6 tests)

1. `test_rf_intervention_well_formed_passes` — energy-normalized RF pair passes
   every check (incl. `strict=True`).
2. `test_a_misconfig_hp_hash_caught` — A realized with canonical RF-ON hp →
   `config.A_RFoff_RateStd.hp_hash_match` and `hp_params_match` fail (defect i).
3. `test_energy_imbalance_185x_caught` — reproduced ratio `184.6×` →
   `total_input_energy_parity` fails (defect ii).
4. `test_long_measured_during_relaxation_caught` — LONG τ_eff ≈ 4.1 s (transient
   fit) fails `measured_tau_effective`; truncated 60 s window fails
   `relaxation_window`; fit sanity recovers 4.1/833 s targets (defect iii).
5. `test_omission_energy_must_be_zero` — nonzero omission energy fails.
6. `test_rfon_target_population_required` — RFon driving all-400 fails.

## 5. No frozen configs altered

`jomission/ablations/rf_rate_factorial.py`, `manifests/rf_rate_factorial_*.json`,
and all frozen scientific configs are **read-only**; this validator only reads
frozen identities and adds a semantic layer on top.
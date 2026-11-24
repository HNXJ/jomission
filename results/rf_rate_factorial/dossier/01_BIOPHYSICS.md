# 01 — BIOPHYSICS (cell A_RFoff_RateRef)

_Characterization dossier — firing-rate and HDP/biophysical state characterization. All numbers are measured from the sealed generated-owner arrays under `results/rf_rate_factorial/A_RFoff_RateRef/recording/`. Evidence tags: `MEASURED` (read/computed from a recorded array) · `DERIVED` (from MEASURED) · `MODEL_ASSUMPTION` (design/config-specified) · `INFERENCE` (interpretation) · `UNRESOLVED` (missing from artifacts; never imputed)._

- Cell identity: `A_RFoff_RateRef` [MEASURED] · config_hash `4f9fdeae7428199a` (recorded) [MEASURED] · hp_hash `f327f9d2ad64cc88` (recorded) [MEASURED] · design `rf_rate_factorial.v0.1.0` [MODEL_ASSUMPTION]
- Input integrity vs sealed `artifact_hashes.json`: all 20 consumed artifacts match → `OK` [MEASURED]

## 1. Recording & timing

| quantity | value | tag | source |
|---|---|---|---|
| simulation dt | 0.1 ms | MEASURED | H_t_dense.npz `dt_ms` |
| trial length | 46240 steps = 4.624 s | MEASURED | spikes_example_*.npz `n_steps`; design 4624 ms |
| time base | array index 0 = p1 onset (t=0); fx [-500,0) NOT recorded | INFERENCE | external_drive alignment (drive nonzero at p1 slot) |
| phases (trials) | {"pre": 96, "exposure": 260, "post": 96} | MEASURED | event_ledger.json / continuation_state.json |
| checkpoints | 26 ok / 0 fail | MEASURED | continuation_state.json |
| numerical_valid / stability_issues | true / [] (executor-recorded) | MEASURED | A_RFoff_RateRef_result.json |

Epochs (p1-relative ms): p1[0,531] d1[531,1031] p2[1031,1562] d2[1562,2062] p3[2062,2593] d3[2593,3093] p4[3093,3624] d4[3624,4124] rw[4124,4624] [MODEL_ASSUMPTION from event_ledger.json, which is itself a recorded artifact].

## 2. Firing rates (whole-trial, p1..rw window)

Spike artifact stores per-unit whole-trial spike **counts** only (no timestamps); rate = count / 4.624 s.

| trial | global mean rate (recomputed, Hz) | global rate (stored, Hz) | silent units (0 spikes) | tag |
|---|---|---|---|---|
| AAAB | 23.521 | 23.521 | 27 | MEASURED |
| AXAB | 20.417 | 20.417 | 64 | MEASURED |
| BBBA | 23.343 | 23.343 | 64 | MEASURED |

Global mean rate stored in `rate_summary.json` = 23.5 Hz [MEASURED] (executor mean over full run). Recomputed example-trial means above are single trials, p1..rw window.

Per-area whole-trial population rates (example trials):

| trial | V1 | V4 | FEF | PFC | tag |
|---|---|---|---|---|---|
| AAAB | 22.067 | 23.482 | 23.062 | 25.474 | MEASURED |
| AXAB | 19.241 | 20.236 | 20.004 | 22.189 | MEASURED |
| BBBA | 21.899 | 23.304 | 22.891 | 25.279 | MEASURED |

Pre-exposure per-area rates (rate_summary first-5 pre-trials):

| pre trial | V1 | V4 | FEF | PFC | tag |
|---|---|---|---|---|---|
| AAAB | 22.065 | 23.475 | 23.082 | 25.469 | MEASURED |
| AXAB | 19.425 | 20.420 | 20.192 | 22.284 | MEASURED |
| AAXB | 19.427 | 20.430 | 20.199 | 22.398 | MEASURED |
| AAAX | 19.418 | 20.415 | 20.190 | 22.388 | MEASURED |
| BBBA | 22.072 | 23.469 | 23.073 | 25.474 | MEASURED |

Rates lie within the design acceptance band [1, 80] Hz (C3) [MODEL_ASSUMPTION]. The AXAB (p2-omission) trial shows lower whole-trial rates than AAAB in all areas [DERIVED].

## 3. Spiking statistics

- **ISI / temporal CV:** `UNRESOLVED` — artifacts store whole-trial per-unit spike **counts** only; no spike timestamps/raster are retained, so inter-spike-interval statistics cannot be computed and are not imputed.
- **Across-unit rate CV** (population spread, distinct from temporal CV): `DERIVED` — computable from per-unit counts; see 02 for per-class decomposition.
- **Per-class spike-count degeneracy:** within each class the per-unit whole-trial counts are near-constant (all E units = 74 spikes/trial in AAAB; PV/SST nearly uniform; VIP near-silent) [MEASURED] — see 02 §2b. This implies strong population synchrony and low E-rate heterogeneity [INFERENCE].

## 4. HDP state: H(t) and Θ(t)

| quantity | value | tag | source |
|---|---|---|---|
| H trajectory (5 pts) | all constant = 1.028966 | MEASURED | H_t_dense.npz / H_t.json |
| Θ (w) trajectory (5 pts) | all constant = 0.003267 | MEASURED | H_t_dense.npz / H_t.json |
| H run-level extrema (min/max/mean) | 0.9838 / 1.0901 / 1.0290 | MEASURED | A_RFoff_RateRef_result.json (executor-computed) |
| Θ run-level extrema (min/max/mean) | -0.0194 / 0.1093 / 0.0033 | MEASURED | A_RFoff_RateRef_result.json |

- **Effective H/Θ timescale:** `UNRESOLVED` — the retained H/Θ trajectory has 5 identical points and no time base; the declared dense early-time sampling was **not** materialized on disk, so an exponential-timescale fit is impossible from artifacts.
- **Design-specified rate reference:** τ_eff,E ≈ 4.1 s at Rate=std (τ₀=5.0 ms, cube-law) [MODEL_ASSUMPTION, SPECIFIED_NOT_MEASURED_FOR_THIS_RUN].
- **w-floor bound violation:** recorded Θ = 0.003267 sits below the design floor w_floor=0.01 [MEASURED vs MODEL_ASSUMPTION] → the recorded Θ violates the design bound; flagged `UNRESOLVED` (whether the mean was recorded pre-clamp or from an unclamped intermediate is not adjudicable from artifacts).

## 5. Cell-intervention identity (RF-off) — UNRESOLVED

- The design table assigns cell A (RF off, K_HDP=0.0) hp_hash `bb8277e7a8e0bca2` [MODEL_ASSUMPTION, design doc], but the executed artifact records hp_hash `f327f9d2ad64cc88`, which is the **canonical RF-ON** hash (K_HDP=0.003 per design table) [MEASURED from result.json].
- `exec_A_RFoff_RateRef.py` instantiates `hdp.v1_pfc_aaab_hdp_params()` and asserts hp_hash == `f327f9d2ad64cc88`; no `K_HDP=0` override appears in the executor [MEASURED from source]. The analysis artifact's claim of 'RF-off realized via intervention (K_HDP=0)' is not supported by the visible executor code [INFERENCE].
- The measured Θ ≈ 0.00327 is inconsistent with the K_HDP=0 fixed point (design reference w_baseline ≈ 0.013) and is below w_floor, consistent with active (nonzero) K_HDP dynamics [INFERENCE]. Effective K_HDP during the run: `UNRESOLVED`.

## 6. Inputs (drive, RF operator, visual)

| quantity | value | tag | source |
|---|---|---|---|
| drive AAAB mean / max (a.u.) | 2.2967 / 5.0 | MEASURED | external_drive_examples.npz |
| drive AXAB mean / max (a.u.) | 1.7225 / 5.0 | MEASURED | external_drive_examples.npz |
| omission slot drive (AXAB p2 [1031,1562]) | 0 (timing preserved) | MEASURED | external_drive_examples.npz |
| RF operator weights (400×1024) mean / max / nonzero | 0.000244 / 0.0872 / 0.0372 | MEASURED | RF_operator.npz |
| RF geometry (32×32 lattice, 8°, σ 1.8 px, spacing 3.2 px) | config | MODEL_ASSUMPTION | RF_operator.npz `config` |
| RF targets | V1 L4 E(9) + PV(3) = 12 units | MODEL_ASSUMPTION | RF_operator.npz target_indices + neuron metadata |
| stimulus A/B pixel mean, equal-pixel frac | 0.0383 / 0.0383, 0.0303 | MEASURED | visual_field_patterns.npz |

Stimuli A and B are distinct sparse 32×32 patterns (≈97% of pixels differ) [MEASURED]; their identity as canonical A/B is a MODEL_ASSUMPTION of the paradigm.

## 7. Continuation & numerical stability

- Continuation verified, state identities carried per phase; checkpoints 26/26 [MEASURED from continuation_state.json].
- V_m/spikes finite and V_m mean within criteria asserted by executor with zero stability issues [MEASURED from result.json]. Note: raw V_m traces are not retained in the recording union, so independent V_m statistics are `UNRESOLVED` at the artifact level.

## 8. Number registry

Every number above is listed in the corresponding tables with its evidence tag and source. Quantities for which no recorded data exist (spike timestamps for ISI/CV, current/conductance traces, dense H trajectory, calibrated field amplitude, raw V_m) are reported as `UNRESOLVED` and are never imputed.


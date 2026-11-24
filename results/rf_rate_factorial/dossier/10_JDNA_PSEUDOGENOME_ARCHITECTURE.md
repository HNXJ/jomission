# 10 — JDNA PSEUDOGENOME ARCHITECTURE (ACTUAL IMPLEMENTED GRAMMAR)

_Characterization dossier. This report documents the **actual** encoded "pseudogenome" of the Jomission runs — the machine-readable identity/state grammar that defines a replicate and carries its history — as implemented and sealed in this workspace. **No hypothetical DNA-encoded genome is reported.** A workspace search for `JDNA`/`pseudogenome` returns no implementation; the realized "genome" is the `config_hash` + `hp_hash` + `ContinuationState` + evidence-manifest stack described below. All fields are `MEASURED` from sealed artifacts (`EvidenceRef.json`, `artifact_hashes.json`, `recording/continuation_state.json`, `jomission/simulation/state_replacement.py`)._

---

## 1. Identity genome — three hashes + code/seed (per cell)

From `EvidenceRef.json` (cell A shown; all cells carry the same schema):

| field | value (cell A) | meaning | tag |
|---|---|---|---|
| `code_sha` | 57365e6f9afd | git head | MEASURED |
| `config_hash` | 4f9fdeae7428199a | network configuration (+RF metadata for B/D: e5e331a140ebd37e) | MEASURED |
| `numerical_config_hash` | 1b5fd1cc13cd4547 | numeric variant | MEASURED |
| `hp_hash` | f327f9d2ad64cc88 | hdp_params (plasticity genome) | MEASURED |
| `dt_ms` / `seed` | 0.1 / 0 | integration step & RNG seed | MEASURED |
| `network_realization` | `V1→V4→FEF→PFC 100/area izhikevich edge_list` | topology string | MEASURED |
| `phase` / `namespace` / `evidence_class` | post / canonical_confirmatory / CANONICAL_CONFIRMATORY | run stage & evidence tier | MEASURED |
| `estimand_version` | jomission_comparison_matrix.v0.1.0 | analysis contract | MEASURED |
| `generated_owner` | results/rf_rate_factorial/A_RFoff_RateRef | artifact root | MEASURED |
| `artifact_hash` | 0e529983ae1b39f3 | result.json sha16 | MEASURED |

`hp_hash = sha256(json.dumps(hp, sort_keys=True))[:16]`; the full `hp` dict is stored verbatim in each `*_result.json` (see §3).

## 2. Artifact manifest grammar — `artifact_hashes.json`

Maps every sealed artifact (heartbeat, exec.log, result.json, EvidenceRef, recording/*, checkpoints/*) to a `sha256[:16]` content hash; `dossier_receipt.json` re-verifies each against the sealed value (`hash_verified_all: true` for cell A). This is the **materialized genome ledger** — the frozen, write-once record of what a replicate produced [MEASURED].

## 3. Plasticity genome — `hdp_params` consumed keys (ACTUAL, node path)

Only these keys are consumed by the engine; all are stored verbatim in `*_result.json`/design manifest [MEASURED]:

| key | role | value (rate std / slow) | tag |
|---|---|---|---|
| `tau_0_ms` | base H integration constant (rate knob) | 5.0 / 1000.0 | MEASURED |
| `size_scale_by_cell_type` | cube-law `tau_i = tau_0·size³` | E5.0, PV1.0, Inl1.0, SST1.5, Ing1.5, VIP1.5 | MEASURED |
| `K_ctrl` / `alpha` / `gamma` / `beta` / `delta` | restoring / income / spending / scaffold | 0.15 / 0.05 / 0.5 / 0.0 / 0.0 | MEASURED |
| `K_HDP` | weight drive gain (RF toggle) | 0.0 (off) / 0.003 (on) | MEASURED |
| `K_w_ctrl` | weight restoring | 0.001 | MEASURED |
| `H_min`/`H_max`/`w_floor`/`w_ceiling` | bounds | 0.1 / 10 / 0.01 / 10 | MEASURED |
| `barrier_eps`/`barrier_c`/`barrier_d`/`H_boost_gain`/`rho_passive`/`C_spike` | barrier & scaffold | 0.001 / 0.01 / 0.01 / 4.0 / 0.0 / 0.0 | MEASURED |

**Dead/fabricated keys NOT consumed (must not be emitted):** `tau_theta_s`, `learning_rate`, `bounds_lo/hi`, `channels` (audit, dossier 07 §1) — present in some config sources but not in the consumed `hdp_params` genome.

## 4. State genome — `ContinuationState` carriers (ACTUAL)

The mutable "genome" that carries run history, per `jomission/simulation/state_replacement.py` (`_hash_state`, `REPLACEMENT_SPECS`):

| carrier | field(s) | role | tag |
|---|---|---|---|
| fast dynamic **X** | `v`, `u`, `prev_spikes`, `syn_state` | fast neuronal state | MEASURED |
| resource **H** | `H` | per-neuron plasticity resource (history carrier) | MEASURED |
| weight **Θ** | `w` | adaptive weight / Theta (history carrier) | MEASURED |
| delay **D** | `delay_state` | delay-line state | MEASURED |
| RNG | `prng_key` | matched-RNG preservation (Q8) | MEASURED |
| cursor | `step_index` | time-step position | MEASURED |

Each carrier is hashable (`_hash_array`: sha256 of bytes+shape+dtype; `_combined` state hash = sha256 of the carrier-hash dict, `state_hash`). Q8 ablations replace exactly one carrier class (H, Θ, H+Θ, or fast-X) and verify only the declared carrier changed — this is the causal-intervention read/write grammar of the genome [MEASURED, `state_replacement.py:205-262`].

## 5. Schedule genome — run phases (ACTUAL)

| phase | wall_s | trials | steps | tag |
|---|---|---|---|---|
| initialization | 2.0 | — | 20 000 | MEASURED |
| baseline | 10.0 | — | 100 000 | MEASURED |
| exposure | 1202.24 | 260 | 12 022 400 | MEASURED |
| testing (pre/post) | 443.904 | 96 (12×8) | 4 439 040 | MEASURED |
| recovery | 30.0 | — | 300 000 | MEASURED |
| **total** | **1688.144** | — | 20 900 480 | MEASURED |

`continuation_state.json` records `global_step=20900480`, `state_identity`, `checkpoint_ok/fail`, and H/Θ extrema at terminal [MEASURED].

## 6. Explicit non-genome statement

- **No `JDNA`/`pseudogenome` implementation exists** in the workspace (grep across `.py`/`.md`/`.json` = no matches). The name "JDNA pseudogenome" is not realized as DNA-like sequence encoding.
- The **actual** pseudogenome is the stack in §1–§5: identity hashes (config/hp/code/seed), the `hdp_params` plasticity genome, the `ContinuationState` carrier genome, the schedule genome, and the frozen `artifact_hashes` ledger.
- Anything beyond this (e.g., base-pair / codon encodings, transcriptional rules, heritable sequence) is **hypothetical and `UNRESOLVED`/not implemented** — and is not claimed as `MEASURED`.

## 7. Number registry

All hashes, phases, key lists, and carrier names appear in tables with `MEASURED` provenance and source. Hypothetical JDNA encodings are explicitly flagged as not implemented (`UNRESOLVED`), consistent with the dossier-wide rule that only actually-implemented grammar is reported.

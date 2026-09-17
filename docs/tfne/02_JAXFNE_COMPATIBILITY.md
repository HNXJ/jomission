# TFNE ↔ JaxFNE 0.4.24: compatibility record

**Owner:** Jomission (internal)
**Date:** 2026-09-16
**Engine:** jaxfne 0.4.24, Python 3.14.3, JAX default float32, Windows
**Semantics:** `00_TFNE_ALGEBRA_SPEC.md`
**Questions sent to the JaxFNE team:** kept outside the repository until answered.

```
TFNE parameter compilation through JDNA develop = UNQUALIFIED
TFNE structural normal-form/conformance work    = OPEN
Scientific execution authority                  = NONE
```

This record separates two things that must not be merged:

- **TFNE architecture semantics:** the algebra and its hashes `h_S`, `h_T`, `h_P`, `h_M`, defined without any simulator.
- **Current JaxFNE/JDNA realization semantics:** what 0.4.24 actually builds from a JDNA genome or a NeuronalTensor.

TFNE does not compensate for realization behaviour silently. Every difference becomes a field in a compile receipt.

## 1. Status

| Stage | Status | Basis |
|---|---|---|
| Structural compilation (`h_S`) | may proceed (language side) | spec §4.2; no engine dependency |
| Projection compilation with synthetic motifs (`h_T`, `h_P`) | may proceed (conformance only) | `01_SYNTHETIC_CONFORMANCE.md` |
| Biologically canonical projection expansion | BLOCKED by D1–D4 | spec §6.1 |
| Spectrolaminar `H_SL` qualification through JaxFNE field proxies | not qualified | §5 |
| **TFNE → JDNA parameter compilation** | **UNQUALIFIED** | finding B; open questions B1–B4 |
| TFNE → NeuronalTensor (direct) parameter compilation | not qualified | preserves `w_mech` and `dT_ms` (B), but weights are globally normalized (A) and density is fixed at 1 (D) |
| Density `p < 1` through the tensor bridge | UNSUPPORTED | finding D |
| Changes requested in JaxFNE | none | findings are receipts first |

Effective values (dynamical impact) were **not measured**. No `simulate()` was run; every probe stops at `construct()`.

## 2. Compile receipt

Chain per projection identity:

```
TFNE topology -> raw parameters -> JaxFNE normalization -> configured rule -> realized edges
```

| Field | Content |
|---|---|
| `identity` | `(src population, dst population, mechanism)` — enters `h_T` |
| `raw` | parameters as declared in TFNE (`w`, `p`, `tau_syn_ms`, `delay_ms`) — enters `h_P` |
| `path` | `jdna_develop` or `neuronal_tensor_direct` |
| `lost` | declared fields the path does not carry (e.g. `w_mech`, `H`, `dT_ms` on `jdna_develop`) |
| `normalization` | `{rule: "abs(w_mech*g_mech)/sqrt(N_total)", N_total, g_mech, factor}`; required on every projection |
| `configured` | rule name, weight, probability, mechanism name as recorded in `cfg.metadata["circuit"]["connections"]` |
| `realized` | edge count, expected edge count, kernel-resolved weight, `tau_ms`, `delay_steps` (via `jaxfne.emitters` resolvers) |
| `status` | `EXACT` (realized = raw), `NORMALIZED` (realized = raw × factor), `LOST` (a raw field did not reach realization), `UNSUPPORTED` (the path cannot express the raw value; compile error `E_REALIZATION_UNSUPPORTED`, never a silent substitute) |
| `composition_local` | `false` whenever the factor depends on units outside the projection (finding A) |

The receipt makes the hidden couplings observable. Architecture identity (`h_T`, `h_P`) never depends on it.

**Why `normalization` is a mandatory field.** Under finding A, a topology change alters the realized efficacy of projections it does not touch (`Δh_T ≠ 0 ⇒ Δ realized weight` on unrelated projections). A comparison that changes one topology term therefore changes many realized parameters and silently violates ONE_DELTA. Two receipts are comparable under ONE_DELTA only when every projection outside the intended delta has the same `normalization.factor`.

**Time constant and delay.** TFNE `tau_syn_ms` maps to `EdgeList.tau_ms`. TFNE `delay_ms` maps to `delay_steps`, which no 0.4.24 tensor path sets; a nonzero `delay_ms` is `UNSUPPORTED`. JaxFNE `StaticParams.dT_ms` is realized as `tau_ms` and is never read as a delay because of its name.

## 3. Findings (executed)

Probe scripts are in `scripts/tfne/probes/`. They call `construct()` only and print JSON to stdout; nothing is written into the repository.

```
python scripts/tfne/probes/probe_verify.py            # JSON receipt on stdout
python scripts/tfne/probes/probe_verify.py --check    # exit 1 if a recorded value below changed
python scripts/tfne/probes/probe_density.py
```

### A. Weight normalization by total network size

**Observed.** A unit `U` has layers `L2` and `L4` (20 neurons each; E 0.8, PV 0.2) and one within-unit rule `L2.E → L4.E` (AMPA, `w_mech = 1.0`). Adding an identical, unconnected unit `U2` changes that rule:

| | `U` alone (N_total 40) | `U` + `U2` (N_total 80) |
|---|---|---|
| configured weight | 0.158113883 | 0.111803399 |
| realized weight (kernel-resolved) | 0.158113882 | 0.111803398 |
| realized edges | 256 | 256 |

- Ratio: 0.70710678 = √(40/80).
- Configured weight equals `w_mech/√N_total` in both cases.
- Source: `neuronal_tensor.py:828-839` (weight rule), `:1080` (`total_n` sums all areas), `:1084-1096` (applied to within-area and area connections alike).

**Consequence.** In TFNE, `H Q^2` means "replicate and add Q connectivity". Realized in 0.4.24, it also rescales every pre-existing projection, within-unit circuitry included. A topology change becomes a parameter change at realization, so composition is not local.

**TFNE handling.**
- `h_T` and `h_P` are unaffected.
- The receipt records `normalization` and `composition_local: false`, with status `NORMALIZED`.
- No pre-compensation: the compiler does not multiply by √N_total.
- No JaxFNE change requested yet (question A1–A2).

### B. JDNA `develop` discards area-connection parameters

**Observed.** Two genomes differ only in the parameters of one area connection A.L4.E → B.L4.E (10 → 10 neurons):

| | genome 1 | genome 2 |
|---|---|---|
| declared `w_mech`, `H`, `dT_ms` | 0.5, 1.0, 3.0 | 1.0, 1.0, 0.1 |
| `genome_rules_hash` | `6f7df0e4…` | `34e76a6d…` (different) |
| developed `w_mech`, `H`, `dT_ms` | **1.0, 0.0, 0.1** (class defaults) | 1.0, 0.0, 0.1 |
| `phenotype_sha256` | `a93a70ca…` | `a93a70ca…` (equal) |
| configured weight | 0.2236068 (= 1/√20) | 0.2236068 |
| realized | 100 edges, weight 0.2236068, `tau_ms` 0.1, `delay_steps` 0 | identical |

The same declared values in a directly built NeuronalTensor are kept: configured weight 0.1118034 (= 0.5/√20), realized `tau_ms` 3.0, `delay_steps` 0.

Source: `jdna/genome.py:705-717` copies only endpoints and `mechanism`; `:138` hashes the full connection dicts.

**Consequences.**
- The genome hash distinguishes genomes whose developed phenotypes and realized networks are identical.
- `H` is dropped as well.
- `dT_ms` is realized as a synaptic time constant (`EdgeList.tau_ms`), not as a conduction delay. `delay_steps` stays 0 on both paths, so neither path carries a TFNE delay parameter.
- A connection whose `dT_ms` is lost falls back to a 0.1 ms synaptic time constant.

**TFNE handling.**
- **TFNE → JDNA parameter compilation = UNQUALIFIED** until JaxFNE answers B1–B4.
- Structural compilation (areas, layers, connection identities) proceeds independently.
- On the JDNA path the receipt marks `w`, `H` and time constants `LOST`.
- The direct NeuronalTensor path is not adopted as a silent workaround.

### C. The shipped V1–V4–PFC config is an example, not canonical O

**Observed.** `configs/canonical-v1-v4-pfc-multiarea.json` (sha256 `97242c2f176893ac9c3a38e519543a260d0a6cc4bdb43289d69dcc9b536e6ff0`):

- **Area connections:**

| Route | Weight | `dT_ms` | Mechanism |
|---|---|---|---|
| `V1.L4.E → V4.L4.E` | 1.0 | 3.0 | `monotonic_cable_synapse` |
| `V4.L4.E → PFC.L4.E` | 1.0 | 3.0 | `monotonic_cable_synapse` |
| `PFC.L6.E → V4.L1.E` | 0.5 | 3.0 | `monotonic_cable_synapse` |
| `V4.L6.E → V1.L1.E` | 0.5 | 3.0 | `monotonic_cable_synapse` |

- Areas V1, V4 and PFC load in `explicit` mode with 0 within-area rules.
- Realized: 24,750 edges (= 75·75 + 75·75 + 135·50 + 135·50), **0 within-area edges**.

**TFNE handling.**
- TFNE `O` remains the TFNE canonical hierarchical motif (D1 open; laminar routing settled: L23 → L4 feedforward, L56 → L1 and L56 feedback).
- The shipped config is registered, if used at all, as a named non-canonical motif, `O[motif=jaxfne_v1v4pfc_example]`, with its content and file hash above.
- Structural FF/FB wiring alone does not establish functional hierarchy; this example does not define TFNE biology.

### D. Tensor-bridge connections are all-to-all

**Observed.**
- Every rule compiled from a NeuronalTensor has `probability = 1.0` (`neuronal_tensor.py:881-889`). Realized blocks are complete: 256/256 (16×16), 100/100 (10×10), and 24,750 in C.
- Below the bridge, `Configuration.connections(probability=0.25)` on a 16×16 block realized 64 edges for seed 0 and for seed 1. Whether sampling is exact-fraction or Bernoulli, and which seed controls it, was not determined.

**TFNE handling.**
- Density is a TFNE motif parameter (`O[p=ρ]` or inside a registered motif) and enters `h_P`.
- Through the tensor bridge, `p ≠ 1` is a compiler capability boundary: `E_REALIZATION_UNSUPPORTED`, never a silent `p = 1`.
- The Configuration-level capability is recorded but not used until question D1–D2 is answered.

## 4. Secondary observations (held back; not sent)

| # | Observation | Evidence | TFNE relevance |
|---|---|---|---|
| S1 | Connections without declared `dT_ms` realize `tau_ms = 0.1` ms (the `StaticParams` default). | executed (A: realized `tau_ms` 0.1) | A realization default supplies a physiological-looking number; the receipt must show it. |
| S2 | `genome_rules_hash` and `_tensor_identity_digest` serialize areas and connections in declared order; position sampling keys use `fold_in(seed, group_index)` in neuron-table row order. | source-read (`genome.py:131-141`; `neuronal_tensor.py:794-811, 893-920`); not executed | The emitter must sort, or engine identity becomes spelling-dependent. That row order follows area order is inferred. |
| S3 | `merge_neuronal_tensors` renames colliding areas (`V1` → `V1_1`). | source-read (`neuronal_tensor.py:584-590`) | Never use merge for TFNE composition; label identity would break. |
| S4 | Edge sign comes from the presynaptic class. | source-read (`neuronal_tensor.py:880`); realized storage mode `magnitude_times_presynaptic_sign` observed | A motif mechanism inconsistent with its source class would be realized without warning. |
| S5 | Areas without an explicit pose stack at the same origin. | source-read (`merge_neuronal_tensors` docstring) | Geometry is outside TFNE; field proxies of TFNE-built models need explicit poses. |

## 5. Spectrolaminar tooling (source-read; not executed)

Relevant to the `H_SL` candidate (spec §6.2). Read from `jaxfne/fields/proxy.py` in 0.4.24. No readout was run.

| # | Observation | Source | Consequence for `H_SL` |
|---|---|---|---|
| F1 | `project_laminar_sources` places each neuron as a point source at its soma depth and projects it to contacts with a Gaussian kernel (`width` 0.10 relative depth). The returned mode is `proxy_no_field_solve`. | `proxy.py:148-240` | No dipole or dendritic geometry. A deep alpha-beta mechanism carried by apical-dendrite currents cannot be represented. Output is `lfp_proxy`. |
| F2 | `filtered_spike_source`, the default evidence path, is a 5 ms exponential spike trace signed by class (E +1, PV/SST/VIP −1). | `proxy.py:702-735` | The signal is spike-derived, with no synaptic or transmembrane current. The network-generator and forward-model qualifications cannot be separated on this path. |
| F3 | `cable_filter_tau` / `cable_filter_sources` apply a depth-graded low-pass filter: E tau 1 ms superficial → 5 ms deep, PV 0.5, SST 2, VIP 2 ms, order 2. The docstring reports gamma deep:superficial 0.66 after filtering, and says the unfiltered baseline "shows the same flat ~1.7x deep gain in every band". | `proxy.py:1060-1145` | In that documented case, the superficial-gamma landmark is produced by the readout operator. A filter of this kind is a forward-model assumption and cannot count as generator evidence. |
| F4 | `teaching_control_spectrolaminar_resonance_source` injects fixed 15 Hz and 90 Hz layer-weighted sinusoids (`spectrolaminar_profile_injected: True`, `default_evidence_path: False`). | `proxy.py:738-820` | Excluded from `H_SL` evidence. It may serve only as a positive control for a readout. |
| F5 | `spectrolaminar_readout` uses bands alpha_beta 8–25 Hz and gamma 40–150 Hz, normalizes each channel's PSD by its total power over 1–150 Hz, takes neurons (not contacts) as channels, and sets `contact_depths_m = pos_from_l4 × 0.5`. Without scipy, `spectrolaminar_psd` falls back to a raw FFT silently. | `proxy.py:131-146, 824-948` | Per-channel relative power couples bands: lower alpha-beta at a depth raises relative gamma there with no change in gamma power. Bands differ from the proposal's. Band edges, normalization and PSD estimator must be declared before any run. |

Required before `H_SL` evidence through these tools: a flat-spectrum input control, a null source with no laminar structure passed through the same readout (F3, F5), a positive control (F4) the readout must recover, and ablations of the proposed generators. A readout-created motif is a FAIL of the forward model.

## 6. What would change these statuses

| Status | Changes when |
|---|---|
| UNQUALIFIED (B) | JaxFNE states intent and an execution surface for parameterized long-range projections; a receipt test then shows `EXACT` or `NORMALIZED` status for `w` and the time constant on that surface. |
| `composition_local: false` (A) | Only a versioned JaxFNE normalization change. Until then it is recorded, not fixed. |
| UNSUPPORTED density (D) | A tensor-level density field exists, or the JaxFNE team endorses a Configuration-level route with defined sampling semantics. |
| C | Nothing pending; example registration is final unless JaxFNE redefines the config's status. |

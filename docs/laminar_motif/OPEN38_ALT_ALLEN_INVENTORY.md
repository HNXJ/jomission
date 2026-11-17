# OPEN38-ALT-ALLEN P2: read-only source inventory of the Herrera et al. 2026 mouse V1 model (2026-11-17)

Authorization: owner 2026-11-17, priority P2, after the OPEN-38 gate ruling. The 2026-11-09 replica lineage
(`ALLEN_V1_REPLICA_SPEC.md`) was retired the same day. Read-only work: `gh api` metadata and raw reads of the public
repository. Nothing was built, run or ported (P3 and P4 are not authorized). No Allen-derived value enters OPEN-38.

## Repository identity

| Item | Value | Source |
|---|---|---|
| Repository | `AllenInstitute/oscillations-biorealistic-mouse-v1-model`, public, not archived, BSD-3-Clause | `gh api repos/...` |
| HEAD of `main` | `bd25da66bfa92d097692d1c1c5d3621bcd62bcfe` (2026-10-30, "Merge pull request #1 from beaherrera:main"); last push 2026-10-30T19:10:11Z | `gh api .../commits` |
| Against the 2026-11-09 pin | `compare bd25da66...HEAD`: identical, 0 ahead, 0 behind | `gh api .../compare` |
| Tags, releases | none | `gh api .../tags`, `.../releases` |
| Tree | 1754 blobs at bd25da66 | `git/trees?recursive=1` |
| Licence file | `LICENSE`, blob b9e2cb2d | tree |
| Paper | Herrera et al. 2026, "Circuit and cell-type mechanisms of the spectrolaminar motif of cortical oscillations" (preprint; README). DOI per the retired spec: bioRxiv 10.64898/2026.11.04.763717. No copy is held here, so its hash is UNKNOWN | README; retired spec |

## Model, simulator and composition

| Item | Value | Source (blob or hash) |
|---|---|---|
| Simulator | BMTK: filternet for LGN, PointNet on NEST 3.8 for V1. Networks in SONATA, built by a Snakemake pipeline | README; model README lines 52, 133; `Snakefile` blob c8b8017b |
| Neurons | GLIF point neurons | README ("GLIF neurons and double-alpha synapses") |
| Weights | trained in TensorFlow (`biorealistic-mouse-v1-model-training/`) to match Neuropixels firing rates and OSI/DSI. The `checkpoint` weights were "used for all oscillations-paper analyses" | README; model README line 133 |
| Environment | conda `osc-mouse-v1-model` | `environment.yml` blob dc2de50a |
| Network sizes | `core`: 400 µm radius, about 67 k neurons, the version built, trained and analysed. `full`: 700 µm, about 203 k | model README lines 77-91 |
| Core counts | seed file `pop_core_count` total 66 552. Per layer E / PV / SST / VIP: L1 0/0/0/603; L2/3 16 784/552/400/913; L4 17 403/1 188/635/328; L5 9 365/1 067/941/188; L6 15 053/519/519/94 | `V1model_seed_file.xlsx` sha256 e33da358…31c4 (matches the 2026-11-09 import `allen_v1_blocks.json`) |
| Laminar SST/PV (derived) | L2/3 0.725, L4 0.535, L5 0.882, L6 1.000 | from the counts above |
| Connectivity tables | `pmax_matrix_v1dd.csv` 1446631d…887d, `sigma.csv` 1249c732…38e6, `lgn_params.csv` a23706d6…7be | sha256 of the raw files at bd25da66, `base_props_default/`; all four equal the 2026-11-09 import |
| Model variants | 28 `base_props_*` folders. They include L4 Pvalb→E (`L4-Pvalb2e-as-L6`, `-syn-lat-as-L6`, `-syn-tau-amp-as-L6`), L4 Sst→E and Vip→E analogues, cell-type swaps (`PV-like-Sst`, `Sst-like-PV`, `i4Pvalb-like-i4Sst`, `i4Sst-like-i4Pvalb`) and L4 density/connectivity swaps (`L4-cell-density-as-L6`, `L4-cell-volume-density-as-L6`, `L4-connectivity-as-L6`) | tree |

## Readout and reference outputs

| Item | Value | Source |
|---|---|---|
| Estimand | laminar and cell-type **MUA power spectra** (population firing, normalized per cell), not LFP | README "Analysis"; `analysis_*.py` (e.g. `analysis_dg_layer_power.py` d97933cc), `spectral_analysis2.py` c47e7d57 |
| Stimuli | drifting gratings (8 directions × 10 trials), contrasts, temporal frequencies, gray screen, natural scenes and movies, layer and inhibitory-population silencing | model README lines 266-380 |
| Reference figures in git | 45 files under `paper_figs/`, e.g. `fig1C_dg_power_spectrum_Jul17_2026.pdf` a16a0004, `fig2C_laminar_power_merged_Jun16_2026.pdf` 1749aabe, `fig3_dg_power_inh_pop_silencing_Jul27_2026.pdf` c3389f9b, `fig4_dg_L4sim_power_inh_pop_silencing_Jul27_2026.pdf` 0e00bbaf | tree (git blob SHAs) |
| Reference data (networks, spikes, metrics, checkpoints) | not in git; on a Dropbox folder linked from the README. Not downloaded or hashed here (UNKNOWN). Needed for P3 | README "Data" |

## Does the poster model equal the published release?

INFERRED: consistent with it, but not provable by hash.
- The poster's 400 µm, about 67 k neuron column equals the release's `core` (seed-file total 66 552).
- The poster's MUA-per-cell spectra equal the release's analysis.
- The poster's L4 Pvalb→E timing claim maps onto the release's `L4-Pvalb2e-syn-lat-as-L6` and
  `-syn-tau-amp-as-L6` variants.
- The poster names no commit, and the repository has no tags or releases. Identity can be established only by
  reproducing a poster or paper figure from the Dropbox checkpoint data (P3).

## Consequences for P3 (not authorized; for the owner's decision)

- The native reference must run on BMTK/NEST with GLIF neurons and the `checkpoint` weights, not on JaxFNE AdEx.
- Readout compatibility is the first question: their MUA-power spectra against our relative-power LFP-proxy depth
  profile are different estimands. P3 has to define the comparison before running.
- H-comp and H-intr can both be probed inside the release. The density and connectivity swaps address H-comp; the
  Pvalb→E latency and kinetics variants address H-intr.

Scratch copies of the raw files read (untrusted, not committed): `D:/tmp/_temp_allen_src/`.

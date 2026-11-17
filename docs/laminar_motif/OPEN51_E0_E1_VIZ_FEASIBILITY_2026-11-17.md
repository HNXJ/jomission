# OPEN-51: can E0 and E1 get V0-V3 from original evidence? Findings, 2026-11-17

Scope (owner 2026-11-17, option 1): read-only. This investigation ran no simulation and no construction, drew no
visualization, and changed no lineage record or contract. Scratch checks ran under `D:/tmp/_temp_o51_*.py`, which reads
only.

## Verdicts

| Stage | E0 (SCI-LAMINAR-MOTIF-1) | E1 (SCI-LAMINAR-E1) | Missing evidence |
|---|---|---|---|
| V0 construction schematic | reconstructable at the declared level; realized level unknown | same | a construction receipt or edge table from the original runs; E0's JaxFNE build and the jomission SHA for both lineages |
| V1 initial 1000 ms raster | blocked | blocked | per-cell spike times; only per-step population counts were saved |
| V2 final raster | blocked | blocked | same as V1 |
| V3 atlas | blocked as specified; a partial atlas without a raster is reconstructable | same | the raster (contract line 62); membrane dynamics, I_net, H, A, m(H) were never recorded |

The scientific evidence is intact; what is missing is the visualization inputs. The per-step counts that both
verdicts rest on survive, and each reproduces its committed run record exactly. What was never kept is the per-cell
spike raster: `jomission/laminar/caminos_e0.py` `run()` summed the raster to E/PV/SOM counts in memory (line 130) and
discarded it.

## Evidence

### Original outputs

- Committed run records: `results/laminar_e0/e0_seed{1..5}.json` and `e0_result.json` (commit ea249f1), and
  `results/laminar_e1/e1_rho{0.25,0.5,1,2,4}_seed{1..3}.json` and `e1_result.json` (8200fac). Their keys are
  `analysis, jax_devices, jaxfne_file, jaxfne_version, oscillates, p, schema, seed, spikes_kept, steps, undefined,
  wall_s`. They record no code SHA, no JaxFNE commit, no edge table and no hash of the counts. `E0_SEAL.md` line 25
  requires the code SHA in each run manifest, and none carries one.
- The counts files `*_counts.npy` are gitignored (`.gitignore:21`). They are absent from D:\jomission and were not found
  on D:, E: or C:\Users\nejath. The only copies are in the read-only warehouse,
  `F:/repo_salvage_2026-11-10/jomission/E_repos__wt_jomission-handoff.loose/results/laminar_e{0,1}/`: 20 files, each
  1 440 128 bytes, int32, shape (120000, 3). No raster, `.npz` or run JSON is stored beside them.
- Identity check (`_temp_o51_counts.py`): for each of the 20 files, `spikes_kept` and `rates_Hz` recomputed after the
  2 s discard (20 000 steps) equal its committed JSON exactly.
  - Known negative (`_temp_o51_neg.py`): 17 distinct runs give 17 own-record matches and 0 cross-matches out of 272.
  - E1 rho 1 seeds 1-3 are byte-identical to E0 seeds 1-3; E1 reused E0's runs, as `run_e1_scored.py` declares.

| File | sha256 |
|---|---|
| laminar_e0/e0_seed1_counts.npy (= e1_rho1_seed1) | ac5a4415af817472262c57bc272657770927ed4ff1382b93c7ff94d2de3966a7 |
| laminar_e0/e0_seed2_counts.npy (= e1_rho1_seed2) | f7d2a3b1a68c87ccefa19b374f79de6988053b7c981710d09119661392ac14ae |
| laminar_e0/e0_seed3_counts.npy (= e1_rho1_seed3) | e2b70bdc99ed611e2d732d8fb94a0c40978649f6550a57cfc7821af2bac95e9b |
| laminar_e0/e0_seed4_counts.npy | 9158654942bd56eec654072a26859d24e0409c56783872c0f2a517afdb249be3 |
| laminar_e0/e0_seed5_counts.npy | 5549416ecd38032215409bc359573558c0bad98537b269dcf12a3d4814e97ff2 |
| laminar_e1/e1_rho0.25_seed1_counts.npy | eed03c5eda28078b6ffbf0a640c769deec0fa1a2bce87bff41bbd0dccf0cd9f9 |
| laminar_e1/e1_rho0.25_seed2_counts.npy | fd0e731abc0e3301cc0e02628bcf42d0f9578281089e85825a15d89735cb8fa2 |
| laminar_e1/e1_rho0.25_seed3_counts.npy | 5a50fbe94e76cd1f9e6085579c702f40f1f23d41eab34f4009e5f4a88b06b3eb |
| laminar_e1/e1_rho0.5_seed1_counts.npy | 10b95ab35ad605856ecaa0f0f653e63bff261df3a54dcaf092824ca9a53a28d9 |
| laminar_e1/e1_rho0.5_seed2_counts.npy | 8a8eeaf858da40d482631f59f9a6c0bbcb73a824c0f0b50bb0bbee79bcbd36e2 |
| laminar_e1/e1_rho0.5_seed3_counts.npy | 6f49c57caa1313a8a5cb451a17e53a80368623199bf26f0332446837ef3ab9bf |
| laminar_e1/e1_rho2_seed1_counts.npy | 9ff48508f2ef1ded10218cd9a1d9536b0d47a6d2ee0b3ef281807233857b3417 |
| laminar_e1/e1_rho2_seed2_counts.npy | a3a3e819a0ed30e9817029f2db7046698640fd2243c5dd81dc8f5ab185353738 |
| laminar_e1/e1_rho2_seed3_counts.npy | 5dde113bb719900e5fa86441c394925185bb32c2edbc76009c70e53b558389e2 |
| laminar_e1/e1_rho4_seed1_counts.npy | 024a24a0beece805be80a179217d0185941fe142ec7e4ef711220ad418163ecd |
| laminar_e1/e1_rho4_seed2_counts.npy | 868d748ccde9110213223789521233ad73b2fe841729f66a18d4ee09185c8b1b |
| laminar_e1/e1_rho4_seed3_counts.npy | 0648b217445440ae259ad3afce544a7d23e11cba69f015e5ee8bbd0321303ce7 |

### Execution identity

- JaxFNE. `E0_SEAL.md` Amendment 1 (0465b36) declares tag v0.5.9 = `e0160dfc`. The run records say only
  `jaxfne_file: "side-install"`, that is, `E:/repos/_wt/jaxfne-0.5.9-site`.
  - That directory now holds 84 `.pyc` files (Python 3.14, written 2026-11-07 16:35), 10 configs and `py.typed`. Its
    `.py` source is gone.
  - Check (`_temp_o51_pyc.py`, `_temp_o51_commit.py`, `_temp_o51_marshal.py`): each of the 84 imported modules
    compiled from `e0160dfc` (and from `e26e9ec`, which has the same blobs) under Python 3.14.3 equals its `.pyc`:
    82 byte-identical, and 2 (`optim/core.py`, `optim/gsdr.py`) structurally identical with frozenset constants
    compared as sets. Known negatives: the 351f6048 export differs in 19 modules, among them `_construct_core.py` and
    `_construct_adex.py`; `dd48d104` differs in `_model.py`.
  - E1's non-reused runs were written 2026-11-07 20:38 to 2026-11-08 06:59, after the `.pyc` timestamp. E1's build is
    therefore `e0160dfc` (inferred: CPython rewrites a `.pyc` on import when its source changes).
  - E0's runs were written 2026-11-06 19:17 to 2026-11-07 00:38, before that timestamp. E0's build is declared
    `e0160dfc` and unobserved: the `.pyc` evidence starts after E0 ran.
- jomission code: unrecorded.
  - E0 ran after 3257ffc (2026-11-06 18:18, the E0 runner) and before the next change to the runner files, e381154
    (2026-11-07 17:18).
  - The E1 runs followed e381154, and no later commit up to 8200fac touches `caminos_e0.py`, `run_e0_scored.py`,
    `run_e1_scored.py` or `caminos_analysis.py`.
  - Both identities are inferred from timing. A dirty working tree cannot be excluded.
- Construction is seeded (`build_spec` → `cfg.connectivity(edge_seed=seed)`, `drive_spec(p, seed)`). Whether it is
  deterministic across processes was not run (construction was out of scope) and is unknown.

### Visualization contract (`manifests/visualization_contract.json`)

- V0 requires a block schematic with populations, connections, sign, direction and "connection authority g, or realized
  w where relevant".
  - The declared level (populations, p, g, receptors) follows from the committed spec and code. Drawing it needs no
    simulation and, drawn directly from the spec, no construction (inferred: OPEN-38's V0 drew from a built model).
  - The realized level (edge counts, realized w) needs `construct()` at the inferred code identity. No recorded edge
    table exists to compare against, so a rebuild cannot be verified against the original execution. OPEN-38's V0 was
    accepted only because its edge table equalled the run record.
- V1 and V2 require "unit rows ordered area -> layer -> cell type". Population counts carry no unit rows.
- V3 lists the raster, membrane dynamics, I_net, H, A, m(H), weights, rates, a qualified field and the lineage's own
  diagnostics.
  - From the counts: population rate traces, the population-count spectrum (`f_pk`), the PV-SST lag, and the run
    metrics already in the JSONs.
  - No raster can be included. The OPEN-38 atlas lists its unrecorded items under "Not in this atlas", and it
    contains rasters.

## Comparison with OPEN-38 (SCI-OPEN38-O38S)

| Input | OPEN-38 | E0/E1 |
|---|---|---|
| Per-cell raster sample | 45 `.npz` in the repo working tree, each checked against an integrity receipt `npz_sha256` | none saved |
| Edge table in the run record | yes (V0 accepted on equality) | none |
| Execution SHA per run | yes (`git_sha`, 3 stages) | none; inferred from timing |
| JaxFNE build | 351f6048 export, recorded | `e0160dfc`: observed for E1 by compiled-code equality, declared for E0 |
| Population output | in the `.npz` | counts on F: only, verified against the JSONs |

OPEN-38 met all four stages from recorded data. E0/E1 meet V0 at the declared level and part of V3. The OPEN-38 method
cannot produce V1 or V2 for E0/E1.

## Options for a separate implementation decision

1. Declared-level V0 and a partial V3 for E0 and E1, retrospective and labelled, with V1 and V2 left absent. The records
   would stay incomplete under the contract, because the validator requires all four paths. This needs no simulation.
   It produces the faithful subset, and OPEN-51 stays open for V1/V2.
2. Identity-checked re-execution: rerun E0/E1 at `e0160dfc` and the inferred jomission commits, save per-cell rasters,
   and accept a run only if its counts are byte-identical (sha256 above) to the original.
   - A match makes V1/V2 a verified reproduction of the original execution. A mismatch leaves V1/V2 blocked.
   - Cost: 17 distinct runs; `wall_s` is in each run JSON. Successive E0 counts files are 36-125 min apart.
   - Needs separate authorization for new simulations, and an F: read of the counts as the comparator. Bitwise CPU
     determinism of float32 JAX across processes is unknown, and the E0 build is only declared.
3. Contract amendment: add SCI-LAMINAR-MOTIF-1 and SCI-LAMINAR-E1 to `grandfathered`, with this note as the reason. This
   corrects the false `NOT_A_MODEL_LINEAGE` premise without fabricating anything. It is a contract change, which the
   owner reserved. The validator's premise check (OPEN-51 row) is separate in either case.

Ranked: 2 then 1 if the owner wants V1/V2; 3 if retrospective visuals for E0/E1 carry no decision value. Option 1 alone
leaves the validator failing on V1/V2 for both records.

## Not examined

- `F:/.../docs/laminar_motif/E1B_SEAL.md` (sha256 088a5c34d1c2b53bf96a9f8959d94beb98714898f68d159631a19ef3082dd19a) was
  never committed in any branch. It was not read, being outside E0/E1.

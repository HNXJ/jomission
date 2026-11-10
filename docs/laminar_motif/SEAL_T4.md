# Seal: T4 PV/SST silencing and stimulation (lineage `t4`), 2026-11-10

Under PLAN_PAPER_JOMISSION_LANSA.md v2 §2. Committed and pushed before the first T4 run.

- **Question:** how silencing or driving PV or SST, as a whole population or in one unit, changes the
  spectrolaminar map and the band power of each unit.
- **Route:** constant current through the layout modifiers `ipvUV` / `isomUV` (W6, `tensor_triple_drive`, the
  only I_ext path into the AdEx kernel in JaxFNE 0.5.8). Silencing is a hyperpolarizing current, as in SLM.
- **Reference:** as SEAL_T2_T3.md (T1 winner when frozen, otherwise the L-FS reference, labelled).
- **Step 1, dose calibration (`t4cal`):** whole-population current on PV, then on SOM, at
  -100 × 2^k pA, k = 0..5 (-100 to -3200), seed 1, 2 s, no discard. The silencing dose of a population is the
  smallest |I| whose last 1 s has zero spikes in that population. If -3200 pA does not silence, the population
  is reported as not silenceable by this route and step 2 uses -3200 pA, labelled partial.
  - Observed before this seal (W6 fixture, 0.2 s, small p 0.5 network): -400 pA on all PV cells left 136 of
    240 PV spikes. This motivates the grid; it is not a result.
- **Step 2, conditions (8 s, 2 s discard):**

  | Condition | Cells | Current | Seeds |
  |---|---|---|---|
  | PV off, whole | all PV | PV silencing dose | 1-3 |
  | SST off, whole | all SOM | SOM silencing dose | 1-3 |
  | PV off, per unit | PV of S, M or D | PV silencing dose | 1 |
  | SST off, per unit | SOM of S, M or D | SOM silencing dose | 1 |
  | PV on | all PV | +50 pA | 1-3 |
  | SST on | all SOM | +50 pA | 1-3 |

- **Acceptance of a silencing run:** zero spikes in the silenced cells after the discard, and edge counts equal
  to the reference's (`edges` in the run JSON). A run that fails this is reported as failed silencing.
- **Estimands:** as SEAL_T2_T3.md, plus the change vs the reference (difference map, Pearson r, SSIM) and
  per-unit E rate.
- **Deferred:** graded fractions (10/25/50/75/100%) inside the isolated gamma unit. `tensor_triple_drive`
  drives a whole (area, layer, type) triple, so a fraction needs a sub-population label; not built.
- **Stop states:** as SEAL_T2_T3.md.
- **Visualization:** `results/viz/t4/`. V0: W3 block of the reference. V1: 1 s raster of the PV-off whole run,
  seed 1, before reading any other T4 run. V2: final raster per condition. V3: SL map atlas and difference
  maps per condition.

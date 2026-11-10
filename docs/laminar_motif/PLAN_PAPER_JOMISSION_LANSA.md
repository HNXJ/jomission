# Plan: jomission-LANSA paper draft, v1 (2026-11-10, owner order in chat)

Working title: "Laminar-cellular architecture sets the spectrolaminar motif in a primate model of the cortical
hierarchy". Owner order: integrate the overnight results into a paper with the figure order and style of
Herrera et al. 2026 (bioRxiv 10.64898/2026.11.04.763717, "SLM"). Data, conventions and statistics follow the
LANSA manuscript (Mulvey, ..., Bastos, draft; "LANSA"). Every figure uses JaxFNE models and shows LFP, rasters
and CSD. No cap on supplements. Every model gets four things, in the supplement at least: an architecture
schematic, a raster, LFP and CSD panels, and a details table (cells per type and unit, connectivity,
parameters, and the source of each, e.g. CAMINOS / Kopell lineage). Status: plan; nothing below is a result
unless it cites a file.

## 0. Ground truth already in the repo

- **Observed target:** `docs/laminar_motif/data/lansa_lfp_Y.json`, Y [21 bins x 150 Hz x 11 areas], relative
  power (LANSA Eq. 1). Densities X: `lansa_density_X.json` (PV, CB, CR, NG; 21 bins x 11 areas).
- **Model:** `jomission/laminar/caminos_stack.py`, AdEx E/PV/SOM units in JaxFNE.
  - Modifiers: k, kd, sst, sstd, rhod, dsz, n, dens.
  - The RWS LFP proxy and the 21-contact probe are in `scripts/e2_motif_figure.py`.
  - Similarity (r_proto, r_area, null95) is in `scripts/e2_motif_similarity.py`.
- **Best 3-unit run so far:** kd1 + SST->E 15 ms, r_proto 0.376 (null95 0.26). The owner flags
  kd1_sst15_sstd60 (0.299) as "almost there". The plateau and the r_ab/r_gm trade-off are recorded in
  `PLAN_LANSA_E2.md`.
- **Readout ceiling at default geometry:** 0.854; with incoherent units, 0.693.
- **Allen V1 replica:** `allen_v1_replica.py`. X0 shows no motif, shuffle p 0.29. A separate session with
  rulings handles it.
- **JaxFNE:** the execution authority is 0.5.8. Every E2 run used the 0.5.9 side install, so all of them are
  exploratory and off-authority (OPEN-37). The event-driven delivery branch `_temp_adex-event-delivery` is
  about 7x faster and not merged.

## 1. Figure map (SLM order, LANSA conventions)

Conventions for every figure, taken from LANSA:
- Depth on y as position from the L4 centre, superficial (-0.6) at the top, 21 bins.
- Relative power colour range 0.5-1, parula-like map.
- Alpha-beta 8-30 Hz in blue, gamma 31-150 Hz in red.
- Cell colours: PV blue, SST(CB) orange-red, VIP(CR) amber, E(NG) purple.
- Bands: delta 1-3, theta 4-7, alpha 8-12, beta 13-30, low gamma 31-90, high gamma 91-150.
- Bands are dashed at 4/8/12/30/90 Hz. Uncertainty is shown as ±2 SEM bands.

| Paper fig | SLM analogue | Content (model = JaxFNE) |
|---|---|---|
| 1 | SLM Fig 1 | A: architecture of the reference model (units S/M/D or 21 bins; E, PV, SST, VIP). B: raster, 6 s. C: per-unit, per-cell-type spectra. D: model SL map (21 bins x 1-150 Hz). E: observed LANSA prototype map (11-area mean). F: model and observed alpha-beta/gamma depth profiles, LFP traces, CSD. Text gives r_proto, null95 and the controls. |
| 2 | SLM Fig 2 + LANSA Fig 2 | The hierarchy. 11 area models built from the LANSA densities of each area. Grid of areas (rows, V1 to PFC) x columns: density, model raster strip, model SL map, observed SL map. Cross-area matrices of model-observed r (LANSA S7 analogue). |
| 3 | SLM Fig 3 | Interneuron manipulations: PV and SST silencing and stimulation (whole population and per unit), PV/SST ratio sweep, VIP added. Rasters, spectra, SL maps, delta maps vs the reference. |
| 4 | SLM Fig 4 | Laminar source. Unit silencing, isolated units, graded PV/SST/VIP silencing in the gamma-generating unit. |
| 5 | SLM Fig 5 | Synaptic determinants: SST->E and PV->E decay, E/I balance sweep (inhibitory weight g and N_I/N_E), heat maps of r_proto, r_ab, r_gm over the 2-D grids. |
| 6 | new | Probe geometry: a perpendicular laminar probe vs crossing probes at 30/60/90 degrees and lateral offsets. LFP, CSD and SL maps per probe; motif r against angle. |
| 7 | LANSA Fig 6 analogue (optional) | Schizophrenia-pattern density deficits applied to the model (LANSA Hedges' g multipliers), with the change in high gamma. Only if time remains. |

The supplements give, for every model and condition that appears in any figure: the architecture, a 6 s
raster, a 1 s LFP trace per contact, CSD, the SL map, and the details table. They also carry firing-rate
boxplots per cell type and unit (SLM S3/S5/S7 analogue), the readout ceiling and controls (σ, swapped and flat
controls), seeds, and wall times.

## 2. Tests (owner list) with estimand, brackets and checks

Each test seals a spec before its run: an estimand, a parameter bracket (log grid), stop states and the
acceptance check. These are committed and pushed before the run that uses them. No retune after results; a
changed bracket is a new lineage.

- **T1. ~1000-neuron reference model with the SL motif.**
  - Total about 1000 cells (3 units x about 270 E plus I, scaled in-degree K so the drive per cell is kept).
  - Start from kd1_sst15 and kd1_sst15_sstd60 at n4000. Scale with n, keeping the in-degrees (K_E, K_INTER).
  - Fine-tune on the axes that moved r_proto: kd, SST->E decay, deep SST->E decay, k, deep rho.
  - Use a log grid, 3 seeds per point, fitted on the even-area mean and scored on the odd-area mean.
  - Estimand: r_proto (sum row) on the odd-area mean, with null95 and the flat control.
  - Accept: odd r_proto > max(null95, flat control) at 3 of 3 seeds, D E rate > 0.1 Hz, no unit above
    50 Hz E rate.
  - The 0.95 target is reported as reached or not; never claimed without the check.
- **T2. E/I ratio and the SL motif.**
  - Axes: inhibitory weight scale g ∈ {0.25, 0.5, 1, 2, 4} and N_I/N_E ∈ {0.5, 1, 2} x baseline, on the T1
    reference.
  - Estimand: r_proto, r_ab, r_gm and the per-unit peak frequency.
- **T3. PV/SST ratio.**
  - Global rho scale ∈ {0.25, 0.5, 1, 2, 4}, plus a per-unit ramp from LANSA PV:CB.
- **T4. PV/SST stimulation and suppression.**
  - Current step to all PV or all SST: -200 to +200 pA, 5 levels, log-symmetric around 0.
  - Per unit, plus graded silencing fractions 10/25/50/75/100% (SLM Fig 4E analogue).
  - Silencing is a hyperpolarizing current, as in SLM.
- **T5. Add VIP.**
  - A VIP population per unit: VIP->SST strong, VIP->PV weak.
  - AdEx parameters with cited sources (worker inventory first; no number without a source).
  - Density from LANSA CR.
  - Estimand: change in r_proto and in the gamma/alpha-beta ratio; VIP silencing vs intact (SLM: power drop,
    shape kept).
- **T6. Perpendicular and crossing probe LFP.**
  - Place units in (x, z); the volume-conduction kernel uses 3-D distance.
  - The probe is a line at angle θ ∈ {0, 30, 60, 90}° with lateral offset ∈ {0, 0.25, 0.5}.
  - Report LFP, CSD (second spatial difference of the potential, Vaknin padding) and the SL map per probe.
  - Estimand: r_proto against θ. Prediction: the motif decays with the cosine of the depth span.
- **T7. Hierarchy (Fig 2).**
  - 11 area models, each built from the area's LANSA densities: NG->E, PV->PV, CB->SST, CR->VIP by
    co-expression (LANSA p4; an assumption, labelled ASSUMED).
  - Estimand: per-area r_area, the cross-area r matrix and held-out area prediction.
  - Compare with the LANSA PLSR r_PRED of 0.71.

## 3. Readouts every run writes

- **Raster:** spikes per population.
- **LFP:** the RWS proxy per unit, and the probe potential φ(c, t) = Σ_u K_cu x_u(t) on 21 contacts.
- **CSD:** -∂²φ/∂z² on the probe grid.
- **Spectra:** multitaper, 1 s windows, 2 Hz smoothing.
- **SL map, scores and controls:**
  - SL map: relative power on 21 bins x 1-150 Hz.
  - Motif scores: M, z and crossing, as computed by e2_motif_figure.
  - Similarity: r_proto, r_area, r_ab, r_gm and null95.
  - Flat control and swapped control.
  - E rate per unit.
  - Details table: a JSON with the sizes, edges per rule, parameters and the source of each, the SHA, the
    JaxFNE version and file, and the seed.

## 4. Execution

- **Orchestrator:** the parent session. It keeps the science calls: estimands, brackets and interpretation.
- **Workers:** Haiku 5.5 (worker agent, high effort), one packet each, at most 3 at once, writers in their own
  worktree `_temp_paper_<packet>`.
- **Packets, in dependency order:**
  1. P1 readout library: φ, CSD, details-table writer and a Herrera-style figure kit, with LANSA style and
     tests.
  2. P2 n-scaling of caminos_stack, a ~1000-cell layout `e2_3unit_n270`, and its test.
  3. P3 T1 sweep driver plus scorer.
  4. P4 VIP population with a sourced parameter table.
  5. P5 stimulation/silencing current hooks.
  6. P6 E/I and rho-scale modifiers.
  7. P7 probe-geometry kernel.
  8. P8 area-model builder from LANSA X.
  9. P9 paper skeleton in LaTeX (flow-article skill) with figure stubs.
- **Each packet:** a test that fails first, then passes. An independent reviewer reads the diff before the
  parent merges into dev.
- **Runs:** detached, checkpointed, at most 3 at once (machine load). Smoke with 0.5 s first. The SHA, config
  and seed are recorded.

## 5. Risks and open items

| Item | Risk | Handling |
|---|---|---|
| JaxFNE 0.5.9 off-authority | Paper runs are not on the execution authority | Rerun the final figure set on the authority, or the owner moves the authority (OPEN-37). Recorded; not decided here. |
| 0.95 similarity | The 3-unit ceiling with incoherent units is 0.69 at σ 0.15 | Fix σ from independent data, or use 21 bins. Report the ceiling beside every score. |
| CB/CR to SST/VIP mapping | Assumption | Label ASSUMED; sensitivity with an alternative mapping. |
| σ readout | Wider σ inflates the flat control | σ fixed before the runs. Owner value needed, or derive it from LANSA probe data (UNKNOWN). |
| Omission firewall | None: no omission data are used | - |

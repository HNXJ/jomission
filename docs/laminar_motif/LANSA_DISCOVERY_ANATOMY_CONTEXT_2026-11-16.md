# LANSA discovery sequence and anatomical validation (owner clarification, 2026-11-16)

Status: owner-approved scientific context and proposed future direction; **not** a new simulation seal, scientific verdict, or authorization to run. Applies to SCI-LAMINAR-MOTIF-1. Preserve existing E0/E1/E2/T8 records, sealed results, current gates and the omission firewall.

## Scientific story (ordered)

1. **CAMINOS:** An E–PV–SST cortical circuit demonstrates frequency-dependent oscillatory dynamics. This motivates the mechanism; replication of the port is not proof of the in-vivo mechanism.
2. **Mendoza et al.:** The broadly observed cortical spectrolaminar motif (superficial gamma, deeper alpha/beta) is the empirical target, not a construction assumption. Verify precise publication and quantitative scope before citing in a manuscript.
3. **LANSA inference:** Test whether SST/PV composition can account for the motif. **Distinguish fitted/inferred composition from imposed experimental conditions and directly measured anatomy.** Historical E2 prescribed rho = N_SST/N_PV at superficial/middle/deep = 0.25/1/2, within the E1 oscillatory range; those values were not measured SST/PV densities. E1 manipulated rho, and T8 tested flat/reversed ratio controls. Later optimized LANSA ratios, if any, require parameter-estimation code and receipts before calling them estimated; do not retroactively relabel E2.
4. **Independent anatomical validation:** Compare inferred SST/PV/VIP distributions with macaque layer-resolved measurements (L1–L6, area-specific). Existing LANSA X has PV/CB/CR/NG, **not** direct SST/VIP; CB and CR cannot silently stand in for SST/VIP. Record marker, cell-count denominator, sampled volume, layer registration, area, animal, uncertainty, and assay. No numerical six-layer SST/PV/VIP density matrix has yet been validated here.
5. **Return to autonomous mechanistic modeling:** After an anatomical constraint and evaluation design are independently qualified, test whether the E–PV–SST–VIP model predicts held-out areas/layers without fitting each target spectrum. Distinguish density from synaptic gain/kinetics and readout geometry. Use fixed-selection/held-out evaluation, matched total-inhibition controls, laminar swaps, ablations, shuffled anatomy, and null spectral controls. Predeclare estimands, acceptance, uncertainty and stopping criteria before execution.

## Open scientific questions and closing checks

- **Estimation provenance:** Locate the exact later LANSA optimizer, parameter definitions, objective, training data and outputs; report whether SST/PV was fitted or prescribed in each lineage. Close only with code/config/result receipts.
- **Anatomical constraints:** Obtain quantitative macaque SST/PVALB/VIP counts by layer and area from primary data, with GABA-marker validation and sampling-volume normalization where possible; otherwise label estimates as uncertain or unavailable. Avoid invented cells/mm3.
- **Identifiability:** Separate effects of SST/PV composition, total inhibitory density, synaptic gains, and intrinsic dynamics. Preserve null and negative results.
- **Prediction:** Predeclare an out-of-sample area/layer test and independent reviewer gate. No autonomous runs until the project TODO and sealed-execution authority permit them.

## Provenance and authority

Owner clarification in conversation, 2026-11-16: the narrative is CAMINOS → widespread motif → SST/PV-based model inference → anatomical validation → further autonomous simulation. This note is additive context, not a change to `manifests/current_state.json`, `manifests/todo.json`, frozen tables, scientific verdicts, or the approved execution pointer. Current repository records still explicitly describe E2's prescribed ramp; retain that history.

References for follow-up verification: CAMINOS original model; Mendoza et al. spectrolaminar motif; macaque spatial transcriptomics and Disney et al. marker-specificity work. Bibliographic and quantitative claims must be checked against the primary papers before use as model constraints.

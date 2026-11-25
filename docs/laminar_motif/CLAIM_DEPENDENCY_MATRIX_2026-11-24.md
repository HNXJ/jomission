# Claim-to-run dependency matrix, 2026-11-24

Status: PROVISIONAL. Scientific gate HOLD. No reruns, no simulations, no tracked-file edits.
Inventory: `_temp_claim_inventory.json` (739 rows; provenance from `jaxfne_status`, which holds only PARTIAL or UNKNOWN; VERIFIED records: 0). Runner: UNKNOWN for every claim. Machine-readable mirror: `CLAIM_DEPENDENCY_MATRIX_2026-11-24.json`.

Evidence: OBSERVED (text, fields, file existence); DERIVED (counts); INFERRED (run set mapped to families where the text names no file); UNKNOWN (runners, shift-null script, r_ab M3 definition).

## 1. Matrix

| ID | Claim | Records present / expected | PARTIAL | UNKNOWN | VERIFIED | Distinct jomission SHA | Highest status |
|---|---|---|---|---|---|---|---|
| C01 | Single unit reproduces its published rhythm (E0) | 5 / 5 | 0 | 5 | 0 | 0 | UNSUPPORTED-PROVENANCE |
| C02 | E1: peak frequency falls as rho rises 0.25 to 2 (slope -6.27 Hz/oct); no oscillation at rho 4 (seeds 1-3) | 15 / 15 | 0 | 15 | 0 | 0 | UNSUPPORTED-PROVENANCE |
| C03 | E1b replication on fresh seeds 6-8: arm A negative slope, one of three arm-A runs oscillates at rho 4; arm B five of five rho oscillate, slope -4.3 Hz/oct | 27 / 30 | 0 | 27 | 0 | 1 | UNSUPPORTED-PROVENANCE |
| C04 | T1 search, pre-registered (seeds 1-3, N=4000): axis 1 (B), axis 2 (SOM->E decay all), axis 3 (SOM->E decay deep) selected B=1.41, 15 ms (S,M), 21 ms (D) | 57 / 57 | 18 | 39 | 0 | 11 | UNSUPPORTED-PROVENANCE |
| C05 | T1 pre-registered test fails on fresh seeds 4-6 (12 s): r_proto 0.377, 0.365, 0.377; r_ab 0.35-0.47; no pass | 3 / 3 | 3 | 0 | 0 | 1 | SUPPORTED-PARTIAL |
| C06 | Size ladder N=1000, 2000, 4000 (seeds 1-3): no run passes; r_gm < 0.6 in all runs | 9 / 9 | 0 | 9 | 0 | 8 | UNSUPPORTED-PROVENANCE |
| C07 | Mid-depth relative gamma 0.19-0.31 (observed 0.85) across 12 ladder and confirmation runs; reference 0.23-0.26; T8 controls 0.04-0.23 | 12 / 12 | 3 | 9 | 0 | 9 | UNSUPPORTED-PROVENANCE |
| C08 | r_gm ladder: r_gm >= 0.6 in no ladder run at the declared sigma = 0.15; at sigma 0.20 and 0.25, 9 of 12 and 11 of 12 (not passes) | 12 / 12 | 3 | 9 | 0 | 9 | UNSUPPORTED-PROVENANCE |
| C09 | T8 density and SOM/PV controls: changing the SOM/PV ramp hands dominance to the superficial unit, changing density hands it to the deep unit; no control raises mid-depth gamma; reference r_ab 0.83-0.86 is above controls | 15 / 15 | 15 | 0 | 0 | 1 | SUPPORTED-PARTIAL (control-vs-control outputs); UNSUPPORTED-PROVENANCE for reference-baseline comparisons |
| C10 | Weight runs (seed 1): raising M's E->E weight lifts r_gm and moves alpha-beta to M; adding PV->E restores the crossing at cost of M gamma; M recurrent excitation sets both bands | 14 / 13 | 14 | 0 | 0 | 2 | SUPPORTED-PARTIAL (descriptive table only; single seed, unsealed); mechanism NOT-ESTABLISHED (paper labels it inferred) |
| C11 | Spectral shift null: circular shift of unit spectra with diagonal (incoherent) search does not separate content from geometry (p = (3+1)/(20+1) = 0.19; unshifted best 0.725 from 70 of 1,330 placements) | 0 / 21 | 0 | 0 | 0 | 0 | NOT-ESTABLISHED |
| C12 | Depth-placement readout: moving only the three unit depths raises r_proto from 0.38 (unmoved) to 0.720; 59 of 1,330 placements meet both band thresholds on the selection run | 12 / 12 | 0 | 12 | 0 | 8 | UNSUPPORTED-PROVENANCE |
| C13 | r_ab M3 discrepancy | 0 / n/a | 0 | 0 | 0 | 0 | NOT-ESTABLISHED |

Status counts: SUPPORTED-VERIFIED 0; SUPPORTED-PARTIAL 3 (C05, C09, C10); UNSUPPORTED-PROVENANCE 8 (C01, C02, C03, C04, C06, C07, C08, C12); NOT-ESTABLISHED 2 (C11, C13).

## 2. Per-claim detail

**C01. Single unit reproduces its published rhythm (E0).** Manuscript: abstract.tex l.6-7; results.tex sec 2 (E0).
- Admissibility (matrix sec 2): 0.5.9, E0 seal Amendment 1 (matrix sec 2): ADMISSIBLE.
- Missing or unclear: none in inventory; e0_result.json (aggregate) not counted.
- Notes: All 5 inventory rows are alias_of D:/jomission/results/laminar_e0 with jomission_sha null. Matrix sec 7 does not list them and F8 reports 0 E0 runs in either inventory (conflict, blocker B1).

**C02. E1: peak frequency falls as rho rises 0.25 to 2 (slope -6.27 Hz/oct); no oscillation at rho 4 (seeds 1-3).** Manuscript: abstract.tex l.7-8; results.tex sec 2 (E1, table tab:e0e1).
- Admissibility (matrix sec 2): E1 not in matrix sec 2 governing-seal table (E1_SEAL.md exists on disk; matrix silent). Not covered by ruling..
- Missing or unclear: none in inventory; e1_result.json (aggregate) not counted.
- Notes: All 15 rows are alias_of D:/jomission/results/laminar_e1 (blocker B1). Frequency-drop criterion undefined because rho 4 is silent; E1 verdict INCONCLUSIVE (results.tex). Minimum for the slope: 12 files (rho 0.25, 0.5, 1, 2 x seeds 1-3); rho 4 adds 3 for the no-oscillation statement.

**C03. E1b replication on fresh seeds 6-8: arm A negative slope, one of three arm-A runs oscillates at rho 4; arm B five of five rho oscillate, slope -4.3 Hz/oct.** Manuscript: abstract.tex l.8-9; results.tex sec 2 (E1b arm A, arm B).
- Admissibility (matrix sec 2): 0.5.9, E1B seal (matrix sec 2): ADMISSIBLE.
- Missing or unclear: e1bB_rho1 seeds 6-8 (3 records) absent from inventory: arm B has rho 0.25, 0.5, 2, 4 only; the seed-0 arm-B rho-4 smoke run (Methods) absent.
- Notes: Inventory cites .receipt.json files; the only recorded SHA is the driver git_head 4c99e05 (matrix sec 5). Arm B cannot show five of five rho from the inventory without the rho-1 records.

**C04. T1 search, pre-registered (seeds 1-3, N=4000): axis 1 (B), axis 2 (SOM->E decay all), axis 3 (SOM->E decay deep) selected B=1.41, 15 ms (S,M), 21 ms (D).** Manuscript: results.tex sec 2b (T1, tab:t1); discussion.tex l.5-11; seal SEAL_T1.md.
- Admissibility (matrix sec 2): axis 1 and 2: 0.5.8, ADMISSIBLE (minimum); axis 3: 0.5.9 event, NOT COVERED; reference: 0.5.9, NOT COVERED.
- Missing or unclear: reference seed 3 has no named record (T8 t8_ref_n4000_s3 is the candidate, PARTIAL); inventory holds 6 new levels per axis, the 7th level being the reference; the claim 'B=1.41 appears under three axes' is not visible (kd1p41 files are in axis 1 only).
- Notes: Selection rests on 36 UNKNOWN records (axes 1-2) and 18 PARTIAL (axis 3). The fresh-seed failure (C05) uses the setting this search picked.

**C05. T1 pre-registered test fails on fresh seeds 4-6 (12 s): r_proto 0.377, 0.365, 0.377; r_ab 0.35-0.47; no pass.** Manuscript: results.tex sec 2b (T1 confirmation); discussion.tex l.5-6.
- Admissibility (matrix sec 2): 0.5.9 event delivery: NOT COVERED (matrix sec 2).
- Missing or unclear: verdict files t1_confirm_*_T12_verdict.json cited in results.tex are not in the inventory (existence not checked here).
- Notes: Ceiling covers the failure on these 3 runs only. It is conditional on an owner ruling for the NOT COVERED version. The setting's selection (C04) is UNKNOWN, so any statement that the search chose it is UNSUPPORTED-PROVENANCE.

**C06. Size ladder N=1000, 2000, 4000 (seeds 1-3): no run passes; r_gm < 0.6 in all runs.** Manuscript: results.tex sec 2 (size ladder, tab:ladder); discussion.tex l.5-8.
- Admissibility (matrix sec 2): lfs_*: 0.5.8, ADMISSIBLE (minimum); ref-seed1: 0.5.9, NOT COVERED.
- Missing or unclear: no record named lfs_n4000_s1; the N=4000 seed-1 cell is the ref-seed1 file (sw_n4000_dens_k1_kd1_sst15_s1_T8), matched by mapping, not by name.
- Notes: All 9 records UNKNOWN. Seal SEAL_LFS.md governs lfs_* (matrix sec 2).

**C07. Mid-depth relative gamma 0.19-0.31 (observed 0.85) across 12 ladder and confirmation runs; reference 0.23-0.26; T8 controls 0.04-0.23.** Manuscript: abstract.tex l.9-10; results.tex sec 2 (scored runs, tab:sigma); discussion.tex l.4.
- Admissibility (matrix sec 2): mixed: lfs_* ADMISSIBLE (0.5.8); ref-seed1 NOT COVERED; T1 confirm NOT COVERED.
- Missing or unclear: the T8 control values (0.04-0.23) come from T8 records (PARTIAL) and are not in this 12-run set; their inclusion in the abstract range is not traced here.
- Notes: 9 ladder records UNKNOWN, 3 confirmation records PARTIAL. The sigma rescoring (tab:sigma) uses the same spectra; it adds no run record.

**C08. r_gm ladder: r_gm >= 0.6 in no ladder run at the declared sigma = 0.15; at sigma 0.20 and 0.25, 9 of 12 and 11 of 12 (not passes).** Manuscript: results.tex sec 2 (r_gm, tab:sigma, tab:ladder); abstract.tex l.9-10.
- Admissibility (matrix sec 2): as C07.
- Missing or unclear: same record set as C07.
- Notes: Scorer nulls and ceilings computed at sigma 0.15 only (results.tex l.133). Same records as C07; the claim differs in metric.

**C09. T8 density and SOM/PV controls: changing the SOM/PV ramp hands dominance to the superficial unit, changing density hands it to the deep unit; no control raises mid-depth gamma; reference r_ab 0.83-0.86 is above controls.** Manuscript: results.tex sec 3 (tab:t8, fig:t8); discussion.tex l.17-23.
- Admissibility (matrix sec 2): 0.5.9 event delivery: NOT COVERED (matrix sec 2); SEAL_T8.md on disk.
- Missing or unclear: reference r_ab 0.83-0.86 (seeds 1-3) is taken from the UNKNOWN reference records (C04, C06), not from t8_ref; the claim that t8_ref equals earlier reference runs bitwise (D9) cannot be checked from file hashes (0 duplicate sha256, matrix F9).
- Notes: All 15 PARTIAL, one shared jomission SHA (2afef4d). Paired comparisons are descriptive, uncorrected (results.tex l.342-347).

**C10. Weight runs (seed 1): raising M's E->E weight lifts r_gm and moves alpha-beta to M; adding PV->E restores the crossing at cost of M gamma; M recurrent excitation sets both bands.** Manuscript: results.tex sec 2d (tab:weights, fig:weights); discussion.tex l.29-34.
- Admissibility (matrix sec 2): e2x_*: no governing seal (matrix sec 2, F15); 0.5.9 NOT COVERED.
- Missing or unclear: base row (B=1.41, no scaling) is not in the weight family; use t8_ref_n4000_s1 (PARTIAL) or ref-seed1 (UNKNOWN); 2 extra records (weeM1p41_T2, weeM2_T2) are not in the table.
- Notes: 14 PARTIAL records (13 at jomission 8519540, 1 at 34d9512). Hand-chosen settings, one seed.

**C11. Spectral shift null: circular shift of unit spectra with diagonal (incoherent) search does not separate content from geometry (p = (3+1)/(20+1) = 0.19; unshifted best 0.725 from 70 of 1,330 placements).** Manuscript: results.tex sec 2e (readout, shift null); discussion.tex l.51-52.
- Admissibility (matrix sec 2): not applicable: no run record located.
- Missing or unclear: all 20 shifted draws and the unshifted incoherent search have no inventory record; depth_placement.json is the only related record and is untracked (results.tex l.484); the analysis script is UNKNOWN.
- Notes: Runner UNKNOWN; not inferred from names. The paper calls the weak test 20 draws.

**C12. Depth-placement readout: moving only the three unit depths raises r_proto from 0.38 (unmoved) to 0.720; 59 of 1,330 placements meet both band thresholds on the selection run.** Manuscript: results.tex sec 2e (geometry, l.481-488); abstract.tex l.15-17; discussion.tex l.50-52.
- Admissibility (matrix sec 2): as the source runs (C04, C06).
- Missing or unclear: depth_placement.json is untracked (results.tex l.484); the selection run is not named by seed; the 8 other ladder runs are used.
- Notes: Unmoved r_proto is 0.38 in the abstract and results; the same selected setting shows seed mean 0.361 (tab:t1) and seed-1 0.36 (tab:weights). Reconcile before use (blocker B9). Analysis script UNKNOWN.

**C13. r_ab M3 discrepancy.** Manuscript: none located (no M3 token in paper sections; docs grep in blocker B11).
- Admissibility (matrix sec 2): not applicable.
- Missing or unclear: claim has no definition in the manuscript or docs; no run mapping possible.
- Notes: The M3 tokens in docs are an E1b trace label (HANDOUT_LANSA_STAGE10_E1B_2026-11-19.md l.11) and a PCA split label (ITEM03_SEAL_PROPOSAL.md l.39). Neither is an r_ab item. The owner must define the discrepancy.

## 3. Blockers

| ID | Issue | Effect | Action |
|---|---|---|---|
| B1 | E0/E1 classification conflict. Inventory lists 20 cowork E0/E1 rows as alias_of D:/jomission/results/laminar_e0|e1 (both trees hold the files). Matrix sec 7 lists 9 aliases, none E0/E1, and F8 reports 0 E0 runs. | C01, C02 provenance basis unreconciled | Reconcile alias rules between inventory and matrix |
| B2 | Denominators do not agree. Inventory 739 rows (23 alias_of, 515 unmatched, 47 o38s-repo, 12 other-nonclaim, 165 claim-family rows incl. aggregates). Matrix 201 runs + 9 aliases. Ruling 146 and 158. | Coverage cannot be stated | Reconcile per ruling before any coverage figure |
| B3 | jomission SHA count: inventory 36 distinct; ruling 29; Methods l.44 says 32 distinct. | The ruling's 29 is not reproduced | Reconcile |
| B4 | E1b arm B rho 1 (seeds 6-8): 3 records absent. Seed-0 arm-B smoke run absent. | C03 five-of-five rho unsupported by inventory | Locate records or correct the claim |
| B5 | Reference: no named seed-3 record; seed 1 has two records (ref-seed1 sst15, sweeps kd1_s1). D9 bitwise equality not checkable (0 duplicate sha256). | C04, C06, C09 baseline | Name the reference record per seed |
| B6 | 51 PARTIAL records (T1 axis 3 18, T1 confirm 3, T8 15, weights 14, evchk 1) are 0.5.9 outside E0/E1b seals: NOT COVERED per matrix sec 2. The ruling quoted in matrix sec 1 covers the 0.5.8 minimum and not these. | All PARTIAL ceilings are conditional | Owner ruling on version coverage (not a rerun) |
| B7 | The 51 PARTIAL records name JaxFNE commit 351f6048 in receipts (matrix sec 5), not in the inventory jomission_sha field. Executed-code identity not verified. | Accepted as PARTIAL per ruling; no VERIFIED anywhere | None; recorded |
| B8 | No inventory record for the 20 shifted draws or the unshifted incoherent search. depth_placement.json untracked. Analysis scripts UNKNOWN. | C11, C12 cannot be traced to runs | Track and name the scripts and outputs |
| B9 | Unmoved r_proto: 0.38 (abstract, results) vs 0.361 (tab:t1 seed mean, same setting) vs 0.36 (tab:weights seed 1). | C12 baseline ambiguous | Reconcile |
| B10 | 'B=1.41 appears under three axes': kd1p41 records are in axis 1 only; axes 2 and 3 hold sst and sstd levels. | C04 scored-three-times statement unverifiable | Name the records scored per axis |
| B11 | 'r_ab M3 discrepancy' has no definition in the manuscript or docs. | C13 NOT-ESTABLISHED | Owner to define |
| B12 | The inventory has no runner field; no record maps runs to scripts. | All runners UNKNOWN; none inferred from names | Record runner per run family |

## 4. Rule applied

A claim's ceiling is the weakest provenance class among its required runs, subject to the scientific caveats in its notes. No claim reaches SUPPORTED-VERIFIED. PARTIAL is an accepted classification (ruling of 2026-11-24); it does not clear the scientific gate.

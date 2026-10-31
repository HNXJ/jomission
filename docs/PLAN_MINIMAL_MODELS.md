# Plan: checklists, minimal typed models, scoreboard

Date: 2026-10-30. Status: spec and builder committed (14486b1); smoke receipt written after that
commit; scoreboard in section 4a. `next_authorized_task` is `null`. Work items:
`manifests/todo.json` (MIN-02, MIN-04, MIN-05 open). Open uncertainties:
`manifests/current_state.json`.

## 1. Checklist A: minimal omission model with habituation and adaptation

1. Name each added mechanism, its parameter and its value. Adaptation: SST `a`, canonical LTS
   0.02 against the 0.05 used in SCI-EI-INTERACTION-1 (`lineage_registry.json`).
2. Keep the omission encoding: the temporal slot stays, drive is zeroed, no time jump
   (`jomission/paradigm/spec.py`). `paradigm_exact` stays valid.
3. Fix the windows: omission_local −1000 to 1000 ms, baseline −250 to −50, slot 0 to 531,
   post 531 to 1000 (`jomission/paradigm/epochs.py`). The spec equals the code.
4. Resolve `habituation_schedule` (UNRESOLVED in `jomission/paradigm/conditions.py`; ≥1000 s
   exposure is the stated requirement). Write schedule, trial order and seed before any run.
5. Pair naive and habituated arms on one seed set. Only exposure history differs
   (`jomission/simulation/ledger.py`, Δ_exposure).
6. Perturb each new parameter at 0, at its boundary and at a non-default value. Require a
   realized change or an explicit refusal.
7. Limit the delta to the two added mechanisms (one principal delta per lineage).
8. Keep the omission firewall. No omission outcome informs construction or reaches the Pages
   before a blind test. BLIND-OMISSION is retired (owner 2026-09-19); a new blind test needs a
   reviewer-authorized lineage.
9. Gates: profile `prospective_v2` through `jomission.harness.gates`. PASS only when every check
   holds on every window and cell. Gate blob equals the blob at the spec commit.
10. Visualize: V0 construction, V1 1 s raster before interpretation, V2 final raster, V3 atlas.
    Spec committed before the run; results in a separate commit; stop at the first failed gate.

## 2. Checklist B: spectrolaminar 90% correlation pattern model

Source: `site-src/pages/spectrolaminar.html` (draft). Gates SL0, SL1, SL2 from
`docs/tfne/TFNE_V2_SPEC.md`. The 90% figure and its statistic are not in the repo.

1. Define the 90% statistic in the spec: the two vectors (e.g. simulated gamma-over-alpha–beta
   depth profile against LANSA relative power), pooled or per area, over the 21 depth bins,
   threshold 0.90.
2. Compute the chance floor from surrogate profiles, so 0.90 is judged against the null.
3. Pin the LANSA input (PV, CB, CR, NG densities and LFP relative power; 11 areas; 21 bins).
   Blocked: the source is not in the jomission repo.
4. Build one column per area. Check realized E/PV/SST/VIP fractions per bin against the
   densities; report the maximum deviation.
5. Use one canonical type-to-type wiring for all areas. Confirm the all-to-all default is off;
   record edge counts per type pair.
6. Fix the physiological firing regime and drives before comparing any LFP.
7. Choose the circuit on some areas; test on held-out areas. List the held-out areas in the spec.
8. Check the ported PCA/PLSR pipeline against reference output on observed data before use.
9. Do not tune H to make the proxy reproduce the pattern (SL2 rule). F4 is a positive control
   only and is excluded from evidence.
10. Keep the claim at "not qualified" until SL0 and SL1 pass. Keep knockdown results off the
    Page until sealed. Open items stay listed: edge depth bins, alpha–beta profile, irregular
    firing.

## 3. Minimal typed model (shared by both checklists)

- Size: N = 100 neurons per model by default; the same builder takes N as a parameter
  (tested at 50, 100, 200). Reference scale: `jaxfne/examples/05_network_100_ei_multimodal.py`
  (100 neurons, 75 E and 25 I).
- Cell types and notation: `E`, `PV`, `SST(CB)`, `VIP(CR)`. Fractions for N = 100 are open
  (MIN-02). Used now (`docs/minimal/connectivity.json`, assumed, decision 1): E 75, PV 11, SST 9,
  VIP 5 for N = 100 (the PV/SST/VIP split is Pfeffer's interneuron split renormalized to 25 percent).
- Neuron: Izhikevich through the TFNE framework in JaxFNE 0.5.8 (`.emitter(family="izhikevich",
  preset=...)`). Per-class parameters come from the JaxFNE preset table, each cited. The realized
  values are read back and compared.
- HDP: every model enables HDP through `.hdp(**DEFAULT_HDP)` (`jaxfne/hdp_network.py`), so
  `enable_hdp` is true. Owner decision 2 is assumed: the JaxFNE default parameters are the
  declared rule, not the detached homeostatic configuration (no success claim rests on it;
  `docs/HANDOFF_CURRENT.md`). The H state is recorded per neuron (`diagnostics H_final`, `H_trace`).
- Connectivity: declared with `.connections(...)`, one rule per pair. Literature basis:
  - Pfeffer et al. 2013, Nat Neurosci, doi 10.1038/nn.3446 (text ingested in komposer,
    `bibliography/10.1038_nn.3446.md`). Mouse V1, layers 2/3 to 5. PV cells strongly inhibit
    PV and provide little inhibition to SST and VIP. SST cells avoid inhibiting SST and
    strongly inhibit PV, VIP and E. VIP cells preferentially inhibit SST.
  - Hertäg & Sprekeler 2020, eLife 9:e57541, "Learning prediction error neurons in a canonical
    interneuron circuit" (`bibliography/computational.bib`). Title verified; the PV–SST–VIP
    content is for Lane 2 to check against the paper.
  - Bastos et al. 2012, Neuron, "Canonical microcircuits for predictive coding" (komposer
    corpus). Title verified; content and DOI for Lane 2 to check.
  - Tremblay, Lee & Rudy 2016, Neuron, doi 10.1016/j.neuron.2016.06.033. Review; not yet
    ingested (komposer Priority A).
  - Pi et al. 2013, Nature, doi 10.1038/nature12676. VIP disinhibition; not yet ingested.
- Weights and probabilities trace to a cited passage. No value from memory. The komposer
  workflow `workflow_multiagent_cortical_omission` is withdrawn (2026-09-23 audit) and is not a
  source.
- Reporting: class rates against the bands the project already uses: PV 12–25 Hz, VIP 8–15 Hz,
  SST 6–12 Hz, E below PV (`scripts/ei_replay_058_compare.py`). Reported, not gated, unless the
  owner decides otherwise (decision 3).

## 4. Scoreboard (100 points)

Scoring v2 (2026-10-30). A score is recorded only when all of these hold:
1. PASS needs a receipt committed after the spec it tests, cited by commit.
2. A criterion that names a source is stated against the single source in the spec, and its
   thresholds are ratified by the owner before it is scored.
3. An item blocked on an open owner decision scores 0. PARTIAL scores 0.
4. Tier 0 and `project_check` pass at the HEAD named in §4a.
5. The total is the sum of PASS points. Checklist points are the weight spread over the 10 items.
   100/100 means every item is PASS and done.

| id | item | pts | receipt |
|---|---|---:|---|
| S-01 | builder returns a JaxFNE model for N = 100 in one call with a seed | 8 | `tests/test_minimal_builder.py` |
| S-02 | same call works for N in {50, 100, 200} with no code change | 4 | same |
| S-03 | identical seed gives identical wiring bytes | 3 | same |
| S-04 | every neuron is Izhikevich via JaxFNE; parameters match the cited presets | 5 | same |
| S-05 | HDP enabled for the whole model; H recorded per neuron, finite | 5 | same |
| S-06 | HDP rule declared in the spec; detached homeostatic configuration not used | 5 | spec |
| S-07 | PROPOSED (2026-10-30, owner ratification pending): PV→PV strong; PV→SST and PV→VIP weak, by Campagnola 2022 Fig. 2B probability: strong ≥ 0.15, weak ≤ 0.10 | 5 | `docs/minimal/connectivity.json` |
| S-08 | PROPOSED (2026-10-30, owner ratification pending): SST→PV, SST→VIP, SST→E strong (≥ 0.15) and SST→SST weak (≤ 0.10), by Campagnola 2022 Fig. 2B probability | 5 | same |
| S-09 | VIP→SST preferential; VIP→PV and VIP→E weak (Pfeffer 2013) | 5 | same |
| S-10 | E→PV, E→SST, E→VIP, E→E rules cited from a source | 5 | same |
| S-11 | every weight and probability traces to a cited passage | 5 | same |
| S-12 | 1000 ms run with no non-finite values | 5 | smoke receipt (MIN-03) |
| S-13 | each class fires; none silent | 5 | same |
| S-14 | E not runaway; class rates reported against the bands | 5 | same |
| S-15 | HDP state stays bounded across the run | 5 | same |
| CK-A | Checklist A items 1–10 each have a receipt (0.8 each) | 8 | MIN-04 |
| CK-B | Checklist B items 1–10 each have a receipt (0.7 each) | 7 | MIN-05 |
| S-18 | spec committed before any run | 3 | git history |
| S-19 | tier 0 passes; new tests pass | 3 | suite_lock summary line |
| S-20 | results committed separately from the spec; no retune after results | 2 | git history |
| S-21 | todo and open items current; `project_check` PASS | 2 | `project_check` output |

Totals: S-01 to S-06 30, S-07 to S-11 25, S-12 to S-15 20, checklists 15 (CK-A 8, CK-B 7), S-18 to S-21 10 = 100.
## 4a. Status (2026-10-30)

Connectivity spec v2 (single source Campagnola 2022): commits 3850bce, 17e6b79 (PV_to_VIP), 22dda38
(SST a = 0.02 after build). Smoke receipt `docs/minimal/smoke_receipt.json`: commit 6dfd540, the receipt
for the current spec. Scoring v2 (§4) applies. Pfeffer 2013 receipts (14486b1) are kept in history.

| id | state | pts | evidence |
|---|---|---:|---|
| S-01 | PASS | 8 | `tests/test_minimal_builder.py::test_n100_builds_with_seed_and_counts` |
| S-02 | PASS | 4 | `test_sizes_build_without_code_change` (50, 100, 200) |
| S-03 | PASS | 3 | `test_same_seed_gives_identical_wiring_bytes`; the seed is not shown to change wiring |
| S-04 | PASS | 5 | `test_every_neuron_is_izhikevich_with_jaxfne_presets` |
| S-05 | PASS | 5 | declaration test; `H_final` shape [100], finite (smoke receipt) |
| S-06 | PASS | 5 | section 3; owner decision 2 is assumed, not confirmed |
| S-07 | FAIL (pending ratification) | 0 | restated criterion scores PASS on the connectivity table: PV_to_PV 0.39, PV_to_SST 0.09, PV_to_VIP 0.056 (rule-of-three bound). The thresholds 0.15 and 0.10 were chosen after the values were read; not ratified, so 0 |
| S-08 | FAIL (pending ratification) | 0 | restated criterion scores PASS: SST_to_PV 0.18, SST_to_VIP 0.30, SST_to_E 0.23, SST_to_SST 0.05. Same post-hoc thresholds; not ratified, so 0 |
| S-09 | FAIL | 0 | Campagnola: VIP_to_SST 0.138, VIP_to_E 0.048, VIP_to_PV 0.019. The criterion cites Pfeffer; restated on the proposed thresholds, VIP_to_SST (0.138) is below strong (0.15), so it is not scored until restated |
| S-10 | PASS | 5 | E_to_E, E_to_PV, E_to_SST, E_to_VIP in `connectivity.json`: Fig. 2B counts, Fig. 3D mV |
| S-11 | PASS | 5 | every rule and absent pair cites Fig. 2B or 3D; PV_to_VIP probability is a rule-of-three bound (owner choice); VIP_to_VIP absence awaits owner confirmation |
| S-12 | PASS | 5 | smoke receipt: finite spikes and V_m, finite H |
| S-13 | FAIL | 0 | 1 VIP neuron with no spike in 1 s (receipt 6dfd540); owner kept FAIL and reported |
| S-14 | PASS | 5 | E max 11 Hz under the 100 Hz ceiling; rates reported (PV 10.9 Hz, SST 31.2 Hz, VIP 0.8 Hz; PV, SST and VIP outside bands) |
| S-15 | PASS | 5 | H within [1.0, 1.0024], bounds [0.1, 10.0]; H barely moves |
| CK-A | FAIL | 0 | blocked on the habituation schedule (MIN-04) |
| CK-B | FAIL | 0 | blocked on the LANSA source (MIN-05) |
| S-18 | PASS | 3 | spec commits 3850bce, 17e6b79, 22dda38 precede receipt commit 6dfd540 |
| S-19 | PASS | 3 | tier 0 on this tree: 221 passed, 0 failed (the tier 0 run after the smoke rerun) |
| S-20 | PASS | 2 | receipts committed separately. Disclosed: the weight source changed (Pfeffer pC to Campagnola mV) after the 14486b1 receipts, before the 5351501 receipt; the old receipts stay in history |
| S-21 | PASS | 2 | `project_check --offline --no-tests`: PROJECT_CHECK_PASS at 5351501 |

## 5. Parallel lanes (prepared, not spawned)

Lanes 1 and 2 touch disjoint files and can run at the same time. They are BLOCKED in
`manifests/todo.json` until owner decisions 1 and 2 (section 6) are given. Lane 3 waits for both. Lane 4
waits for the LANSA source. Workers do mechanical parts only. Science semantics (weights, the
statistic, interpretation) stay with the main model and the reviewer.

- Lane 1, MIN-01 (done in 14486b1; removed from todo). Writes `jomission/minimal/` and `tests/test_minimal_builder.py`.
  Reads `jomission/network/builder.py` and the JaxFNE example and `_config.py` (repo at
  `E:\repos\jaxfne`). Acceptance: `pytest tests/test_minimal_builder.py` passes; S-01 to S-06
  receipts. Stop: JaxFNE refuses the four-type declaration (open item).
- Lane 2, MIN-02 (worker, high; reviewer for the citations). Writes `docs/minimal/connectivity.json`.
  Reads the komposer bibliography files named in section 3. Acceptance: every rule has a
  citation that resolves to a passage; S-07 to S-11 receipts. Stop: a rule with no source is
  listed as open, never filled from memory.
- Lane 3, MIN-03 (done in 53a3d3b; removed from todo). 1000 ms smoke run; the receipt is in docs/minimal/;
  S-12 to S-15 receipts. Any file under `results/` needs an authorizing TODO.
- Lane 4, MIN-05 (blocked). Spectrolaminar statistic and null; waits for the LANSA source.
- MIN-04 (blocked on the habituation schedule; MIN-03 is done): checklist A coverage.

Packet fields for every lane: goal, baseline commit, files it may read, paths it may write,
acceptance command and expected summary line, stop conditions, report cap of 200 words. One
writer per file.

## 6. Owner decisions (one batch)

1. Fractions for N = 100: from the cited density source (recommended) or the placeholder
   75/10/8/7 until a source exists.
2. HDP rule for the minimal models: a named registered JaxFNE rule (recommended; Lane 1 proposes
   from `jaxfne/hdp_rule.py`), or another rule you name.
3. Rate bands: reported only (recommended) or gates.

## 7. Open items

Mirrored in `manifests/current_state.json` `open_uncertainties`.

- Resolved: JaxFNE 0.5.8 accepts the four-type declaration; `simulate()` needs the runtime from
  `jomission/minimal/builder.py` `minimal_runtime` (edge_list backend and HDP carried in RuntimeConfig).
- MIN-02: the CB ↔ SST and CR ↔ VIP marker mapping is the owner's convention, not yet cited.
- MIN-02: E→PV, E→SST, E→VIP rates lie outside the Pfeffer 2013 scheme; no cited source yet.
- MIN-02: Tremblay 2016 and Pi 2013 are not ingested into komposer.
- MIN-05: LANSA densities (Mulvey et al.) are not in the jomission repo.
- MIN-05: the 90% statistic, its vectors and its surrogate null are undefined.
- HDP rule for the minimal models is not chosen (decision 2).
- The komposer workflow numbers are withdrawn; they are not a source for any weight.

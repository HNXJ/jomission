# Audit: minimal-model track (MIN-02, MIN-06 and the scoreboard)

Date: 2026-10-31. Branch `_temp_migration-jaxfne-0-5-7`, head `6a6fd74`. Score 75/100 (S-13 FAIL, §4a of
`docs/PLAN_MINIMAL_MODELS.md`). Scope: this session's work and the receipts it produced. The broader
sealed program is not audited here.

Labels: **observed** = read from a receipt or file; **derived** = computed from observed values;
**inferred** = a reading that a test has not yet confirmed. One seed (0) and one 1 s window in every
run: no seed or duration variance has been measured.

## 0. Result

- The receipts show S-12, S-14, S-15 pass at every drive tested; S-13 fails at every point (VIP silent).
- The tree stopped at node 2 (MIN-06) on its FAIL branch. Nodes 1, 3–6 are blocked on owner or reviewer input.
- Spec-before-run held for both lineages: spec `5a2ba8f` before receipt `02e4f51`; spec `45f506a` before
  sweep `6a6fd74`. Tier 0 (196 tests) passed before each spec push.

## 1. Connectivity (node 1, MIN-02)

### 1.1 Rules (observed, `docs/minimal/connectivity.json`, N=100: E 75, PV 11, SST 9, VIP 5)

| rule | p | w (mV) | sign |
|---|---:|---:|---|
| E_to_E | 0.0568 | 0.11 | + |
| E_to_PV | 0.4151 | 0.27 | + |
| E_to_SST | 0.2985 | 0.10 | + |
| E_to_VIP | 0.1618 | 0.45 | + |
| PV_to_E | 0.3462 | 0.44 | − |
| PV_to_PV | 0.3947 | 0.47 | − |
| PV_to_SST | 0.0909 | 0.44 | − |
| PV_to_VIP | 0.0556 | 0.23 | − (probability is a bound) |
| SST_to_E | 0.2321 | 0.16 | − |
| SST_to_PV | 0.1786 | 0.18 | − |
| SST_to_SST | 0.0519 | 0.19 | − |
| SST_to_VIP | 0.2976 | 0.17 | − |
| VIP_to_E | 0.0484 | 0.06 | − |
| VIP_to_PV | 0.0185 | 0.10 | − |
| VIP_to_SST | 0.1379 | 0.17 | − |

VIP_to_VIP is absent (owner to confirm).

### 1.2 Extremes and medians (derived from 1.1)

| quantity | min | median | max |
|---|---|---|---|
| probability | 0.0185 (VIP→PV) | 0.1618 (E→VIP) | 0.4151 (E→PV) |
| abs weight, mV | 0.06 (VIP→E) | 0.18 (SST→PV) | 0.47 (PV→PV) |

### 1.3 Expected presynaptic partners per cell (derived: p × N_pre, N=100)

| post | from E | from PV | from SST | from VIP | total |
|---|---:|---:|---:|---:|---:|
| E | 4.3 | 3.8 | 2.1 | 0.2 | 10.4 |
| PV | 31.1 | 4.3 | 1.6 | 0.1 | 37.2 |
| SST | 22.4 | 1.0 | 0.5 | 0.7 | 24.5 |
| VIP | 12.1 | 0.6 | 2.7 | 0 | 15.4 |

Intuition: PV is the most-connected class; a VIP cell is driven mostly by E (about 12 of 15) and SST
(about 3). Summed mV amplitudes are not a dynamical measure and are not used here.

### 1.4 Open items (observed in the spec; each blocks a node)

- CB→SST and CR→VIP marker mapping: owner convention, not cited.
- PV→VIP: probability is a rule-of-three bound (3/54); weight is the Fig. 3D amplitude. Owner choice.
- VIP→VIP: absent; owner to confirm.
- PV→SST and SST→SST: Campagnola says present, Pfeffer 2013 says absent. Owner chose Campagnola.
- Weights are used as mV directly; the JaxFNE native weight unit is not checked (**gap**: every weight
  magnitude above is unverified in engine units).
- Probability counts are read from figures. The text copy in `local/inputs` does not contain them, and
  PDF rendering needs poppler (not installed). **Not verified against the figure.**

## 2. Cell parameters (observed, probe of the built model, N=100, seed 0)

### 2.1 Izhikevich and drive by class (per-class constant unless noted)

| class | a | b | c (mV) | d | tonic drive | initial u | initial v range (mV) |
|---|---:|---:|---:|---:|---:|---:|---|
| E | 0.02 | 0.20 | −65 | 8 | 5.0 | −13.0 | −68.7 … −0.4 (mean −33.7) |
| PV | 0.10 | 0.20 | −65 | 2 | 3.0 | −13.0 | −55.7 … −13.3 (mean −35.0) |
| SST | 0.02* | 0.25 | −65 | 2 | 3.5 | −16.25 | −64.1 … −0.5 (mean −38.9) |
| VIP | 0.02 | −0.10 | −55 | 6 | 3.0 | +6.5 | −60.6 … −7.6 (mean −30.0) |

\* SST a is the checklist A patch (0.02 applied after construction, `22dda38`).

Intuitions:
- VIP is the only class with a positive initial u (+6.5), a negative b (−0.1), and a reset at −55 mV
  instead of −65 mV. These three together make it the outlier.
- PV has the fastest recovery (a = 0.10) and the smallest d (2).
- Initial v spans most of the range up to near 0 mV for E and SST. This comes from the seeded initial
  state, not from a threshold.

### 2.2 Effective drive (observed)

Tonic plus DC 1.0 on all neurons, read back from receipts:

| class | tonic | effective (DC 1.0) |
|---|---:|---:|
| E | 5.0 | 6.0 |
| PV | 3.0 | 4.0 |
| SST | 3.5 | 4.5 |
| VIP | 3.0 | 4.0 (14.0 with VIP extra 10) |

## 3. Drive response (observed)

### 3.1 Mean rate by class (Hz), 1 s window, seed 0

| run | E | PV | SST | VIP | VIP silent |
|---|---:|---:|---:|---:|---:|
| smoke, DC 0 (`6dfd540`) | 11.00 | 10.91 | 31.22 | 0.80 | 1 |
| DC 1.0 (`02e4f51`) | 13.21 | 23.36 | 37.56 | 0.80 | 1 |
| DC 1.0 + VIP 0.1 | 13.21 | 23.36 | 37.56 | 0.80 | 1 |
| DC 1.0 + VIP 0.3162 | 13.21 | 23.36 | 37.56 | 0.80 | 1 |
| DC 1.0 + VIP 1.0 | 13.24 | 23.46 | 37.56 | 0.80 | 1 |
| DC 1.0 + VIP 3.162 | 13.24 | 23.46 | 37.56 | 0.80 | 1 |
| DC 1.0 + VIP 10 | 13.19 | 22.91 | 37.67 | 0.80 | 1 |

### 3.2 Sensitivity (derived from the table)

- DC 0 → 1.0 (tonic +1 on every class): E +2.2 Hz (+20 %), PV +12.5 Hz (×2.1), SST +6.3 Hz (+20 %).
  PV is the most sensitive class; its drive rises 33 % and its rate doubles.
- The E-below-PV criterion (`E_mean_below_PV_mean`) is false at DC 0 (11.00 vs 10.91) and true at DC 1.0
  (13.21 vs 23.36). It depends on the drive.

### 3.3 The VIP non-response (observed; **main finding**)

- The VIP extra drive is realized: effective VIP drive reads 14.0 at extra 10 (receipt `drive_mean_by_class`).
- VIP mean rate stays at 0.80 Hz at every extra value from 0 to 10, and one VIP cell is silent at every point.
- Mean 0.80 Hz over 5 cells with max 1.0 Hz means exactly four VIP cells fire **once** in the second and one
  fires zero times. The silent-cell count is therefore a 1 s window fact as well as a dynamics fact.
- **Inference:** VIP firing in this range is limited by the cell state (a positive initial u of +6.5, a
  negative b, and reset at −55 mV), not by drive. This matches the earlier SCI-EI-NATIVE-STATE-1 finding
  (VIP locked in deep hyperpolarization with u = +8.58; handoff). The sweep was blind to that.
- **Gap:** the grid starts at 0.1 and stops at 10. No point tests whether VIP fires at drives below 3
  or above 14, nor whether a different initial u or a longer window changes the count.

## 4. Dynamics beyond rates (observed)

- **HDP:** H trace and final values are 1.0 to 1.002408 in both the DC 0 and DC 1.0 runs (identical to six
  decimals). Bounds are 0.1 and 10.0, so H sits 0.9 above the lower bound and about 9.0 below the upper.
- **Inference:** H does not respond to a 20–33 % change in rate. Either the HDP coupling is weak at these
  rates or H is set by something the sweep does not vary. Not tested.
- **Silence:** the smoke receipt notes that one 1 s window at about 1 Hz gives about one spike, so zero is a
  sampling fact. A window of 5 s or more would separate sampling from true silence. Not run (needs a new
  spec and owner approval).
- **Runaway:** no class exceeds 40 Hz mean; max single-cell rate is 38 Hz (SST). S-14 ceiling 100 Hz is far
  above all values; it was set after the first probe, not pre-registered (receipt note).

## 5. Criteria and gates

| item | status | note |
|---|---|---|
| S-12 finite | PASS at all points | |
| S-13 each class fires, none silent | FAIL at all points | VIP one silent cell |
| S-14 runaway < 100 Hz | PASS | ceiling set post hoc (not pre-registered) |
| S-15 H bounded | PASS | H sits near 1.0 |
| band check PV 12–25, VIP 8–15, SST 6–12 | reported, not gated | PV in band at DC 1.0 (23.4); SST 37.6 is 3× the top; VIP 0.8 below band |
| S-07 / S-08 thresholds | ratified 2026-10-30 | |
| MIN-02 | BLOCKED | conflicts with S-07–S-11 scored PASS (§4a) |

Scoreboard: 75/100 (`docs/PLAN_MINIMAL_MODELS.md` §4a, last recorded). No item changed since, because S-13
is FAIL at every receipt and CK-A and CK-B are blocked.

## 6. Process and provenance (observed, from git log)

- Spec before run held for both lineages (see §0). Receipts were written by the committed code.
- Tier 0 passed before each code-changing push (196 tests). The first attempt at the sweep's tier-0 run timed
  out because another agent held the suite lock; it was re-queued and passed.
- The validator change (`44f9514`) makes a null next task valid while items are OPEN. It was needed because
  MIN-06 is OPEN and unauthorized. It has one new test.
- Earlier wording error: I said several times that commits were "not pushed". The branch was pushed
  (`a6ea9b3..5a2ba8f`, then `..6a6fd74`). Corrected here.

## 7. State and documentation inconsistencies (observed)

1. `manifests/todo.json` MIN-06 is still `OPEN` and uncommitted, although its tree stopped on FAIL. Stale.
2. `manifests/current_state.json` has no entry for MIN-06 results. Stale.
3. `AGENTS.md` names a G: Drive clone that is not mounted here. The working clone is `E:/repos/jomission`.
4. Sealed `results/v21_pv_*` JSONs cite `.npz` files in an empty Temp folder (`local_inputs.json`, MISSING).
5. `docs/minimal/connectivity.json` names komposer as its source and a Downloads PDF (not repo-held).
6. `jaxfne` in the shared Python is 0.4.25; the project venv is 0.5.8 (venv used for every run here).

## 8. Owner decisions, in the order the tree needs them

1. Node 1: CB/CR mapping; PV→VIP (keep bound or drop); VIP→VIP absence; PV→SST and SST→SST conflict.
2. Node 2: next lineage for the silent VIP cell. The sweep showed drive is not the binding limit, so the next
   candidate is the initial state (u0) or a longer window. This is a recommendation, not a decision.
3. Node 3: habituation schedule (stimulus sequence, trial order, seed, ≥ 1000 s exposure).
4. Node 4: LANSA cell densities, or confirmation that the local `.mat` is the input.
5. Nodes 5–6: reviewer authorization; SCI-EI-INTERACTION-1 verdict.

## 9. Checks not yet run (proposed; each needs its own spec)

- Seeds 1–4 at DC 1.0 and at VIP extra 0 (seed variance; no run has varied seed).
- 5 s window at DC 1.0 (sampling vs silence).
- VIP initial u at 0 (state test) at DC 1.0 (intrinsic-state test).
- H response: a drive step that changes rates by 2× (HDP sensitivity).
- JaxFNE weight-unit check for the four largest weights (PV_to_PV 0.47, E_to_VIP 0.45, PV_to_E 0.44, PV_to_SST 0.44).
- Figure check: read Fig. 2B and 3D counts from the PDF once poppler is installed.

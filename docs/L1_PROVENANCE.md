# L1 Provenance Audit — Milestone 01 → T2

**Question**: Does `L1 × E/PV/SST/VIP` mechanically create a biologically meaningless `L1_E` population?

**Answer**: No. Cartesian grammar `V1_L1_E` etc. is **declarative only**; actual instantiation is filtered by zero-mass populations.

## Evidence

- Per-area tables in `jomission/network/populations.py:18`:
  ```python
  L1: {"E": 0.00, "PV": 0.00, "SST": 0.00, "VIP": 1.00}
  ```
  for all four areas (V1, V4, FEF, PFC). E/PV/SST fractions explicitly 0.

- Geometry builder `jomission/network/geometry.py:18` skips `n_units==0`:
  ```python
  if n_units == 0: continue
  ```

- Model `neuron_metadata` (400 neurons, `build_jomission_model(n_per_area=100, seed=0)`):
  - V1_L1: 8 VIP, 0 E/PV/SST
  - V4_L1: 8 VIP, 0 E/PV/SST
  - FEF_L1: 8 VIP, 0 E/PV/SST
  - PFC_L1: 8 VIP, 0 E/PV/SST
  - Total L1 = 32 VIP only; no L1_E/PV/SST neurons exist in the instantiated network.
  - Verified via `Counter((area,layer,cell_type))` in `tests/test_l1_audit.py` (added).

- Total counts: 4 areas × 100 = 400 neurons; all L1 neurons are VIP.

## Biological semantics

- L1 in primate cortex is overwhelmingly inhibitory: Cajal-Retzius, VIP+, SST+/NDNF+, PV rare, pyramidal E essentially absent or with apical tufts only. Literature fraction `E≈0` is intentional.
- Representing L1 as a single VIP compartment is a **proxy simplification**, not a full L1 microcircuit. It satisfies:
  - Feedback target geometry (deep→L1 projections need a contact depth)
  - Field source depth continuity (0.00–0.10 proxy)
- Area-specific elaboration is permitted: FEF/PFC L1 could later split VIP/SST if needed, via `AREA_LAYER_CELL_TYPES` override — prohibited to assume identical everywhere (Method Plan prohibits identical microcircuit).

## Remaining debt

- Current L1 is homogeneous VIP; future literature update may add L1_SST (or NDNF) split. Table is replaceable without architectural change.
- Not claiming calibrated L1 physiology; claim level remains `computational_scaffold`, `physical_amplitude_calibrated=False`.

## Verdict

`L1_E` does **not** exist in the sealed network. The Cartesian name exists only as a declaratory key; zero-mass filtering prevents biologically meaningless population. Provenance is explicit in `populations.py` and `neuron_metadata`.

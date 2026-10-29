# Jomission — dense laminar omission simulation

Mechanistic in-silico reconstruction of a multi-area, laminar, hierarchical visual omission experiment. Built as a **separate simulation project with frozen JaxFNE as engine** — simulation output is not evidence for the empirical manuscript or for JaxFNE itself.

## Architecture

```
V1 → V4 → FEF → PFC
X = { X_{a,l,c} }   a=area, l=layer, c={E,PV,SST,VIP}
C_t = (X_t, H_t, Theta_t, D_t)  →  q_t → φ_t → y_t
```

- 4 cortical areas, laminar (L1, L2/3, L4, L5, L6) × E/PV/SST/VIP populations, explicit FF/FB.
- Continuous trajectory: `init → baseline → ≥1000s AAAB/BBBA exposure → omission → recovery` with no reset of `(X,H,Θ,D)`.
- JaxFNE provides neurons, synapses, HDP/RBD, continuation state, delays, source/field/probe.

## Paradigm — authoritative timing (p1-relative)

| epoch | start (ms) | end (ms) | duration |
|-------|------------|----------|----------|
| fx    | -500       | 0        | 500 ms   |
| p1    | 0          | 531      | 531 ms   |
| d1    | 531        | 1031     | 500 ms   |
| p2    | 1031       | 1562     | 531 ms   |
| d2    | 1562       | 2062     | 500 ms   |
| p3    | 2062       | 2593     | 531 ms   |
| d3    | 2593       | 3093     | 500 ms   |
| p4    | 3093       | 3624     | 531 ms   |
| d4    | 3624       | 4124     | 500 ms   |

Full trial = 4624 ms (p1→d4 = 4124 ms, fixation included). `X` = absence of stimulus during the 531-ms slot, **not** deletion of the interval — temporal geometry identical on omission trials.

12 conditions:

```
A: AAAB  AXAB  AAXB  AAAX
B: BBBA  BXBA  BBXA  BBBX
R: RRRR  RXRR  RRXR  RRRX

p2 omission: AXAB  BXBA  RXRR
p3 omission: AAXB  BBXA  RRXR
p4 omission: AAAX  BBBX  RRRX
```

Omission-local windows (reindexed t=0 at expected onset): `[-1000,+1000] ms`, baseline `[-250,-50] ms`, omission slot `[0,531] ms`, post `[531,1000] ms`. Do not pool p2/p3/p4 before testing position dependence.

**UNRESOLVED** (explicitly `None`, never invented): reward schedule, full fixation schedule, exact stimulus metadata (identity A/B/R is canonical; visual parameters pending), block ordering, AAAB:BBBA ratio beyond condition taxonomy.

`PARADIGM_EXACT` = exact for currently authoritative variables; unresolved fields remain `UNRESOLVED`.

## Timescales

- `τ_H ∈ {0.1, 1, 10, 100, 1000} s` — initial design, each H_k has explicit meaning/driver/domain/coupling. HDP `τ_Θ ≫ τ_X`, bounded, switchable.
- `H` = finite-dimensional biophysical/history state, **not** homeostasis.

## Recording

`C_t → q_t → φ_t → y_t` via JaxFNE source/readout. Outputs: `SPK`, `MUAe-like`, `LFP-like`, `CSD-like`, band/TFR. Terminology stays `*-like` until physical calibration.

## Evaluation (frozen, not tuned)

```
T1 sparse omission-linked spiking
T2 higher-order bias
T3 weak V1 population omission spiking
T4 frontal low-gamma omission effect
T5 gamma-rate coupling
T6 weaker field coupling for omission-selective units
T7 absence of strong fixed between-area lead/lag
```

Failures are evidence about the model.

## Modeling roadmap (step by step)

Source: the dissertation proposal deck, Chapter 3 (in-silico) and its roadmap, Google Drive file `1VaZ3xyYGQ1wi1dEtSjPRAVTAuYFZMxZ6rF44r85uyIM`. The Drive export carries no slide numbers, so the stages below are taken from the modeling content, not from page numbers.

| stage | deck status | size (columns × neurons) | areas (deck) | repo status (observed) |
|---|---|---|---|---|
| 5 Omission (initial draft) | draft | — | — | this README |
| 6 Omission-Model-lite | implemented, pre-tuned; "proof of concept; tools validated" | 2 × 50 = 100 (built as listed: 26 + 24 = 50) | V1 (L2/3); FEF/PFC (L4) | built: `jomission/network/lite.py`, structure only, untuned; 4 tests pass under jaxfne 0.5.5 (`tests/test_lite_network.py`). The 4-area builder (`V1, V4, FEF, PFC`, `N_PER_AREA_MIN` = 100) is separate |
| 7 Omission-Model-base | proposed, after lite | 6 × 50 = 300 | V1/V2, FEF/PFC, V3/V4 | not started |
| 8 Omission-Model-augmented | proposed, after base | 20 × 50 = 1000 (8-6-4-2) | adds MT/MST | not started |

Deck column composition (lite): E = 18, PV = 4, SST = 1–2, VIP = 1–2 per column. The counts sum to 24–26, not 50; the deck does not say which variant applies to which column.

Hypotheses (deck, Chapter 3):

- H1: omission mismatch neuron in higher order: P(spiking | omission) > a · P(baseline)
- H2: omission and E/I ratio: P(omission | balance) ? P(omission | imbalance)
- H3: omission effect: P(post-omission) ? P(pre-omission)

The deck does not state the comparison symbol in H2 and H3 (`?`).

Routing rule: Chapter 2 of the deck uses predictive routing (match: feedback α/β; mismatch: feedforward γ/θ). The final modeling slide uses signal-noise routing (match: signal, FF and FB α/β; mismatch: noise, FF γ/θ). The deck does not say which rule the model tests.

Open for the owner:

- Lite size: the deck's 100 neurons against the repo's minimum of 400 (4 areas × 100).
- Area set: the deck's labels (V1/V2, V3/V4, MT/MST) differ from the repo's `V1, V4, FEF, PFC`.
- Comparison symbols in H2 and H3, and the routing rule under test.
- Science is on hold: this roadmap is design only; no stage is executed here.

## First milestone (this pass)

- [x] stale mlxEngine removed (provenance: `a526f58`, tree `dcaf37b`)
- [x] JaxFNE installed (0.4.17 at this milestone; current execution authority in `manifests/current_state.json`)
- [ ] exact paradigm validation passes
- [ ] 4 areas × layer × E/PV/SST/VIP instantiated
- [ ] FF+FB graph instantiated (`W_{(a,l,c)→(a',l',c')}`)
- [ ] short continuous trajectory executes, H/HDP survive trial boundaries
- [ ] omission removes input but preserves timing
- [ ] JaxFNE source/readout path executes
- [ ] deterministic manifest, tests pass

## Structure

```
jomission/
  paradigm/   exact timing + conditions
  network/    populations, geometry, connectivity, builder
  dynamics/   H-state, HDP config
  simulation/ continuous trajectory orchestration
  recording/  probes, LFP/CSD-like
  analysis/   T1–T7 + Δ_exposure
  ablations/  factorial controls
  configs/    area-specific fractions, connectivity tables
  tests/
  manifests/
  results/
```

## Provenance

- Stale tree archived: `/tmp/jomission_stale_a526f58.tar.gz` (HEAD `a526f58d69060c1762fa86c0303c3b10b8dd3754`).
- Engine: `jaxfne` latest (recorded in `manifests/environment.json`).
- History preserved, ordinary commits on `main`.

## Install

```bash
pip install -e .
# or via uv/conda with python>=3.11
```

## Tests

```bash
pytest -v
```

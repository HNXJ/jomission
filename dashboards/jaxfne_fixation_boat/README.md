# Jaxfne Cortical Fixation Boat Task (dashboard)

A single-file browser dashboard. Open `index.html` in a browser; no build step, no network.

## What it models

- A boat steered toward a target by a laminar spiking column.
- Each area is a clone of one per-layer configuration: L1–L6 with E, PV, SST and VIP cells. L1 has no E cells by default.
- Connectivity is a table of routes (see Connectivity). The defaults are: within a layer, E→E, E→PV, E→SST, E→VIP, PV→E (inhibitory), SST→PV (inhibitory, one way), VIP→SST (inhibitory, an assumption); between areas, L5 E of the previous area feeds L2 E, and L5 E of the next area feeds L6 E and L1 VIP.
- Haptic coupling α sends the boat distance into the E cells of every area (0 off, 1 maximum).
- Motor readout: four pools of `motorPerDir` E cells (+x, −x, +y, −y), the first cells of layer `motorLayer` (default L5) in the top area. Force is decoded every `ctrlSteps` × 0.5 ms (20 ms by default) and applied one interval later, as in the fixture.
- Directional cue: always drives the first E cells of L5 in the top area, whatever `motorLayer` is. So `motorLayer` changes only which cells are read out, not the cue. The cue is a labelled addition.

## Parameters

All model and task parameters are one object, `DEFAULTS`, in the script. The page shows it as JSON under **All parameters**:

- Edit the JSON and press **Apply parameters**. Missing keys take the default. Unknown keys, malformed routes and values outside the stated range are rejected with a message; the running network is kept.
- **Reset to defaults** restores `DEFAULTS`. Presets and Reset overwrite unapplied text in the panel.
- Sliders change α, cue, tonic and noise live. Neurons per column, areas and seed take effect on **Apply network**. Unapplied text is kept when a slider moves.
- Presets (`PRESETS` in the script) set α and cue only; their labels record the measured outcome.
- `DT` (0.5 ms) and the 2000 Hz sample rate are fixed in the code, not parameters.

Ranges enforced on Apply parameters:

| key | range |
|---|---|
| `seed` | integer 1–999999 |
| `areas` | integer 1–4 |
| `neuronsPerColumn` | integer 6–1000 |
| `layerFrac` | 6 rows × 4 fractions in 0–1; each row sums to at most 1 |
| `tauMembrane` | greater than 0 |
| `refractory` | integer 0–1000 |
| `tauSyn` | 4 values, each greater than 0 (one per type) |
| `bias` | 4 values in 0–1 (one per type) |
| `layerOffset` | 6 rows × 4 values in −1–1: per-layer drive added to `bias[type]`; zero by default |
| `noise`, `tonic`, `alpha`, `cue`, `hMax` | 0–1 |
| `motorLayer` | integer 0–5; the readout layer (4 = L5, default; 5 = L6). L1 (0) has no E cells by default and is refused in practice |
| `motorPerDir` | integer 1–50; 4 × motorPerDir must not exceed the E cells of layer 5 (cue) and of `motorLayer` (readout) in the top area (refused otherwise) |
| `forceGain` | 0–100 |
| `forceNorm` | 1–1000 |
| `ctrlSteps` | integer 1–10000 |
| `damping` | 0–1000 (the explicit Euler step is 0.0005 s) |
| `fixRadius` | 0 or more |
| `fixWindow` | integer 1–1000000 |
| `start`, `target` | two numbers, each |x| ≤ 50 |
| `routes` | see Connectivity |

`forceNorm` is a free parameter. Its default 20 equals 4 × `motorPerDir` (5), which keeps the default force scale; the page does not enforce that link.

Which neuron counts allow a given `motorPerDir` depends on rounding, so the check is done when you press Apply. At 120 neurons per column the layer-5 E count of the top area is 17, so `motorPerDir` 5 is refused there and `motorPerDir` 4 or less runs. At 240 neurons per column the count is 32, so `motorPerDir` up to 8 runs. These counts are for L5; with `motorLayer` 5 the L6 count applies (16 at 120 and 32 at 240).

## Connectivity (`routes`)

Each route connects a source population to every target cell of one type (in one layer, or in every layer with `"*"`). For each target cell, the routes apply in list order. Each route draws `k` presynaptic cells at random from its source population, each with weight `w`. The list order fixes the random draws, so changing the order changes the network.

| field | meaning |
|---|---|
| `name` | label, optional; shown when the route draws no source cells |
| `from.area` | −1 previous area, 0 same area, 1 next area |
| `from.layer` | 0–5 (L1–L6) or `"same"` (the target's layer) |
| `from.type` | source type: 0 E, 1 PV, 2 SST, 3 VIP |
| `to.layer` | 0–5, or `"*"` for every layer |
| `to.type` | target type: 0 E, 1 PV, 2 SST, 3 VIP |
| `k`, `w` | fan-in (integer 0–100) and weight (−1 to 1); negative `w` is inhibitory |
| `fallbackLayer` | optional, 0–5; used when the source population is empty |

Limits:

- At most 32 routes. The fan-in into any one target class (type and layer, counting `"*"` routes in every layer) is at most 300. A table with four routes of k = 100 into four different types is valid; four routes of k = 100 into one type is refused.
- An inter-area route is skipped where the source area does not exist: area 0 gets no input from a previous area, and the top area gets no input from a next area.
- `fallbackLayer` matters for L1: it has no E cells, so the default PV, SST and VIP routes into L1 draw their E input from layer 3 (index 2). Those L1 connections are therefore not within-layer.
- A route that draws no source cells is listed under "No sources drawn for" after Apply. With one area, the inter-area routes are listed.

The defaults are ten routes in `DEFAULTS.routes`: EE, PVE, E L5 → E L2 (from previous area), E L5 → E L6 (from next area), EPV, SSTPV, ESST, VIPSST, EVIP, E L5 → VIP L1 (from next area). The old `proj` key is refused; use `routes`. The defaults reproduce the wiring of the previous version exactly: the 20 s default run gives the same stats line for line.

## Status of the claims

- This is a reduced current-based LIF model written for the browser. It is not the jaxfne 0.5.8 engine and is not calibrated to LANSA or to any data.
- Measured headless (`headless_check.js`), 60 s, seed 1, 240 neurons per column, 2 areas: distance only (α 0, cue 0) gives e ≈ 3 at 60 s, and haptic α 1 alone drifts further (e ≈ 4.8 at 60 s). A directional cue of 0.2 keeps the boat within the target radius, with or without haptic α 0.5.
- The directional cue is a labelled addition, not the frozen fixture spec, which gives direction-only output.
- Fixation is the share of the last 30 s with e ≤ 0.5; before 30 s have passed, it is the share since the run started.
- Speed: each frame runs at most 30 steps, so at 60 Hz the speed settings run at about 0.9× (1×), and the 2× and 4× settings are not reached. This cap predates the parameter panel.
- The spectrum uses 2–100 Hz on a log grid, with 1024-sample Welch segments (0.512 s) and 50 % overlap.
- Checks: the default run matches the version before the routes table line for line over 20 s at seed 1. `headless_events_check.js` covers the rejections (syntax, unknown keys, out-of-range values, malformed routes, the fan-in limit, the motor-fit refusal), route removal, the seed clamp, and the survival of unapplied text. The per-type fan-in pair (four routes of k = 100 into four types accepted; four into one type refused) is checked by a run: cases 45 and 24 of `headless_events_check.js`, with the expected messages compared. `headless_layer_check.js` checks the readout and cue pools: with `motorLayer` 5 the readout moves to L6 and the cue pool and the wiring do not change; default equals explicit 4; the boundary and refusal cases.
- An independent review by Haiku subagents (equivalence of the default routes with the previous wiring, validation, size, and documentation), with an adversarial refuter per finding, found the issues fixed in this version: the old-key refusal was unreachable, the fan-in limit was global rather than per target, route names contradicted their direction, the README baseline and thresholds were wrong, and the fixation and distance statements did not match the runs.
- Rendering in a browser has not been checked in this version.

## Running the headless checks

```
node headless_check.js <script.js> <seconds>
node headless_events_check.js <script.js> <seconds>
```

Both run the page's own script with a stubbed DOM. `headless_check.js` prints the stats every 5 s. `headless_events_check.js` also fires the page's buttons and fields on a schedule and prints the messages and stats after each action. Extract the script first (the text between `<script>` and `</script>` in `index.html`).

## Building on it

- Change a default by editing `DEFAULTS`; the JSON panel and the sliders read from it. Add the matching range in `parseCfg` when you add a key.
- To add a connection, add a route in the JSON panel (no code change).
- Add a preset by adding an entry to `PRESETS` and a button with `data-preset`.
- Larger networks run slower than real time in a headless check: 240 neurons with 2 areas took about 22 s of wall time for 60 s of frames. 1000 neurons with 4 areas has not been measured.

# Jaxfne Cortical Fixation Boat Task (dashboard)

A single-file browser dashboard. Open `index.html` in a browser; no build step, no network.

## What it models

- A boat steered toward a target by a laminar spiking column.
- Each area is a clone of one per-layer configuration: L1–L6, each with E, PV, SST and VIP cells.
- Connectivity: within a layer, E→E, E→PV, E→SST, E→VIP, PV→E (inhibitory), SST→PV (inhibitory, one way), VIP→SST (inhibitory, an assumption). Between areas, L5 E of area k feeds L2 E of area k+1, and L5 E of area k+1 feeds L6 E and L1 VIP of area k.
- Haptic coupling α sends the boat distance into the E cells of every area (0 off, 1 maximum).
- Motor groups are the first `motorPerDir` E cells of L5 per direction in the top area (4 groups: +x, −x, +y, −y); force is decoded every `ctrlSteps` × 0.5 ms (20 ms by default) and applied one interval later, as in the fixture.

## Parameters

All model and task parameters are one object, `DEFAULTS`, in the script. The page shows it as JSON under **All parameters**:

- Edit the JSON and press **Apply parameters**. Missing keys take the default. Unknown keys, unknown projections and out-of-range values are rejected with a message; the running network is kept.
- **Reset to defaults** restores `DEFAULTS`.
- The sliders change the same values live. **Apply network** rebuilds the network from the current values (needed for neurons per column, areas and seed).
- Presets (`PRESETS` in the script) set α and cue only; their labels record the measured outcome.
- `DT` (0.5 ms) and the 2000 Hz sample rate are fixed in the code, not parameters.

Main keys: `seed`, `areas`, `neuronsPerColumn`, `tauMembrane`, `refractory`, `tauSyn`, `layerFrac` (6×4), `bias`, `noise`, `tonic`, `alpha`, `hMax`, `cue`, `proj` (fan-in `k` and weight `w` per projection: `EE`, `PVE`, `EPV`, `SSTPV`, `ESST`, `VIPSST`, `EVIP`, `interArea`), `motorPerDir`, `forceGain`, `forceNorm`, `ctrlSteps`, `damping`, `fixRadius`, `fixWindow`, `start`, `target`.

## Status of the claims

- This is a reduced current-based LIF model written for the browser. It is not the jaxfne 0.5.8 engine and is not calibrated to LANSA or to any data.
- Measured headless (`headless_check.js`), 60 s, seed 1, 240 neurons per column, 2 areas: distance only (α 0, cue 0) and haptic α 1 alone both drift. A directional cue of 0.2 keeps the boat within the target radius, with or without haptic α 0.5.
- The directional cue is a labelled addition, not the frozen fixture spec, which gives direction-only output.
- The parameter panel was checked headless (`headless_events_check.js`): defaults give the same stats as before the panel was added (20 s, seed 1, compared line by line); alpha 0 with cue 0 drifts to e ≈ 3; invalid JSON, unknown keys, negative fan-in and non-numeric values are rejected with the running network unchanged; a 120-neuron, 1-area network and a moved target and start run without errors.
- Rendering in a browser has not been checked in this version.

## Running the headless checks

```
node headless_check.js <script.js> <seconds>
node headless_events_check.js <script.js> <seconds>
```

Both run the page's own script with a stubbed DOM. `headless_check.js` prints the stats every 5 s. `headless_events_check.js` also fires the page's buttons and fields on a schedule and prints the messages and stats after each action. Extract the script first (the text between `<script>` and `</script>` in `index.html`).

## Building on it

- Change a default by editing `DEFAULTS`; the JSON panel and the sliders read from it.
- Add a preset by adding an entry to `PRESETS` and a button with `data-preset`.
- 1000 neurons per column with 4 areas runs slower than real time in a headless check; the default 240 with 2 areas is fine.

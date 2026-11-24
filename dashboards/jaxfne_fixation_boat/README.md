# Jaxfne Cortical Fixation Boat Task (dashboard)

A single-file browser dashboard. Open `index.html` in a browser; no build step, no network.

## What it models

- A boat steered toward a target by a laminar spiking column.
- Each area is a clone of one per-layer configuration: L1–L6, each with E, PV, SST and VIP cells.
- Connectivity: within a layer, E→E, E→PV, E→SST, E→VIP, PV→E (inhibitory), SST→PV (inhibitory, one way), VIP→SST (inhibitory, an assumption). Between areas, L5 E of area k feeds L2 E of area k+1, and L5 E of area k+1 feeds L6 E and L1 VIP of area k.
- Haptic coupling α sends the boat distance into the E cells of every area (0 off, 1 maximum).
- Motor groups are the first 20 E cells of L5 in the top area; force is decoded every 20 ms and applied one interval later, as in the fixture.

## Status of the claims

- This is a reduced current-based LIF model written for the browser. It is not the jaxfne 0.5.8 engine and is not calibrated to LANSA or to any data.
- Measured headless (`headless_check.js`), 60 s, seed 1, 240 neurons per column, 2 areas: distance only (α 0, cue 0) and haptic α 1 alone both drift. A directional cue of 0.2 keeps the boat within the target radius, with or without haptic α 0.5.
- The directional cue is a labelled addition, not the frozen fixture spec, which gives direction-only output.
- Rendering in a browser has not been checked in this version.

## Running the headless check

```
node headless_check.js <script.js> <seconds>
```

`headless_check.js` runs the page's own script with a stubbed DOM and prints the stats it writes. Extract the script first (the text between `<script>` and `</script>` in `index.html`).

## Building on it

- Parameters are the constants at the top of the script (`BIAS`, `LAYER_FRAC`, the weights in `build`) and the `P` object.
- Presets are in `PRESETS`; their labels in the HTML record the measured outcome.
- 1000 neurons per column with 4 areas runs slower than real time in a headless check; the default 240 with 2 areas is fine.

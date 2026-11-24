# Jaxfne Cortical Fixation Boat Task (dashboard)

A single-file browser dashboard. Open `index.html` in a browser; no build step, no network.

## What it models

- A boat steered toward a target by a laminar spiking column.
- Each area is a clone of one per-layer configuration: L1–L6 with E, PV, SST and VIP cells. L1 has no E cells by default.
- Connectivity: within a layer, E→E, E→PV, E→SST, E→VIP, PV→E (inhibitory), SST→PV (inhibitory, one way), VIP→SST (inhibitory, an assumption). Between areas, L5 E of area k feeds L2 E of area k+1, and L5 E of area k+1 feeds L6 E and L1 VIP of area k.
- Haptic coupling α sends the boat distance into the E cells of every area (0 off, 1 maximum).
- Motor groups are the first `motorPerDir` E cells of L5 in the top area for each direction (4 groups: +x, −x, +y, −y). Force is decoded every `ctrlSteps` × 0.5 ms (20 ms by default) and applied one interval later, as in the fixture.

## Parameters

All model and task parameters are one object, `DEFAULTS`, in the script. The page shows it as JSON under **All parameters**:

- Edit the JSON and press **Apply parameters**. Missing keys take the default. Unknown keys, unknown projection names or fields, and values outside the stated range are rejected with a message; the running network is kept.
- **Reset to defaults** restores `DEFAULTS`.
- Sliders change α, cue, tonic and noise live. Neurons per column, areas and seed take effect on **Apply network**. Unapplied text in the JSON panel is kept when a slider moves.
- Presets (`PRESETS` in the script) set α and cue only; their labels record the measured outcome.
- `DT` (0.5 ms) and the 2000 Hz sample rate are fixed in the code, not parameters.

Ranges enforced on Apply parameters:

| key | range |
|---|---|
| `seed` | integer 1–999999 |
| `areas` | integer 1–4 |
| `neuronsPerColumn` | integer 6–1000 |
| `layerFrac` | 6 rows × 4 fractions in 0–1; each row sums to at most 1 |
| `bias`, `alpha`, `cue`, `tonic`, `noise`, `hMax` | 0–1 |
| `proj` `k` | integer 0–100 per projection (fields `k` and `w` only) |
| `forceGain` | 0–100 |
| `forceNorm` | 1–1000 |
| `damping` | 0–1000 (explicit Euler step is 0.0005 s) |
| `start`, `target` | two numbers, each |x| ≤ 50 |
| `motorPerDir` | integer 1–50, and 4 × motorPerDir ≤ layer-5 E cells of the top area (refused otherwise) |

Other keys: `tauMembrane`, `refractory`, `tauSyn`, `ctrlSteps`, `fixRadius`, `fixWindow`, `bias` (4 values), `tauSyn` (4 values).

`forceNorm` is a separate parameter. Its default 20 equals 4 × `motorPerDir` (5), so keep the two in step when you change `motorPerDir`; otherwise the force scale changes.

With the default `layerFrac`, the layer-5 E count is 32 at 240 neurons per column and 16 at 120. So `motorPerDir` 5 needs about 150 neurons per column or more; the 120 setting is refused with the default `motorPerDir` and runs with `motorPerDir` 4 or less.

## Status of the claims

- This is a reduced current-based LIF model written for the browser. It is not the jaxfne 0.5.8 engine and is not calibrated to LANSA or to any data.
- Measured headless (`headless_check.js`), 60 s, seed 1, 240 neurons per column, 2 areas: distance only (α 0, cue 0) and haptic α 1 alone both drift. A directional cue of 0.2 keeps the boat within the target radius, with or without haptic α 0.5.
- The directional cue is a labelled addition, not the frozen fixture spec, which gives direction-only output.
- The parameter panel was checked headless (`headless_events_check.js`). Defaults give the same stats as before the panel (20 s, seed 1, compared line by line). Alpha 0 with cue 0 drifts to e ≈ 3. Invalid JSON, unknown keys, out-of-range values (start 1e308, damping 5000, a layerFrac row summing to 1000, projection k 1000), malformed projections and the motor-fit refusal are all rejected with the network unchanged. A 120-neuron, 1-area network with `motorPerDir` 3 runs, and a moved target and start run without errors.
- A review by an independent reviewer (adversarial verification of each finding) found the blockers and majors fixed in this version: unbounded start/target froze the tab, unbounded layerFrac and k exhausted memory, unbounded damping produced NaN, and motor groups were silently truncated. The minor findings on caption accuracy (L1, rates, fixation window) are fixed in the text. The run line reads the built network's seed, areas and neurons, so it does not show unapplied topology values.
- Rendering in a browser has not been checked in this version.

## Running the headless checks

```
node headless_check.js <script.js> <seconds>
node headless_events_check.js <script.js> <seconds>
```

Both run the page's own script with a stubbed DOM. `headless_check.js` prints the stats every 5 s. `headless_events_check.js` also fires the page's buttons and fields on a schedule and prints the messages and stats after each action. Extract the script first (the text between `<script>` and `</script>` in `index.html`).

## Building on it

- Change a default by editing `DEFAULTS`; the JSON panel and the sliders read from it. Add the matching range in `parseCfg` when you add a key.
- Add a preset by adding an entry to `PRESETS` and a button with `data-preset`.
- 1000 neurons per column with 4 areas runs slower than real time in a headless check; the default 240 with 2 areas is fine.

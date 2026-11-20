---
name: fixation-boat-jaxfne
description: Runs and explains the interactive Fixation Boat lab, a laminar LIF circuit (areas with L1–L6 E-I layers, L4 input, L6-E output, FF and FB projections, HDP gains) closed in a loop with a 2-D boat, a cued detection task and chaotic environment input. Use when someone asks to open, run, zoom, tune or present the fixation boat or the laminar lab, or to relate it to the jaxfne Crazy Sailor boat contract. Not for claims about the paper's fixation result (the lab is a demonstrator, not evidence) and not for editing the jaxfne sailor code (use the jaxfne repo).
---

# Fixation Boat lab (jaxfne Crazy Sailor, demonstrator)

## What it is

A single HTML page, `references/lif_laminar_lab.html`. Everything runs in the viewer's browser: no server, no data files, no network calls. Nothing is saved between visitors.

## How to run it (anyone with a jomission checkout)

From the repo root:

```bash
cd references && python3 -m http.server 8000
```

Then open `http://localhost:8000/lif_laminar_lab.html`. Or open the file directly in a browser. For other lab members on the same network, bind to `--bind 0.0.0.0` and share the machine's address. Serve only this folder.

## Controls

- **Boat map (top):** mouse wheel or trackpad pinch zooms, drag pans, Zoom +/− and Reset view buttons.
- **Presets:** Balanced, No HDP, Boat off, Single area, Strong feedback, Runaway FF, Heavy chaos.
- **Sliders:** grouped by Areas & cells, Connections, Neuron (LIF), Drive & environment, Task, HDP, Boat (closed loop), Simulation. Areas, neurons per area, f_I, p, p_FF, p_FB, seed, motor count and cue fraction rebuild the network. Everything else applies live.
- **Play/Pause, Reset network.**

## Model (as implemented in the page)

- **Cells:** LIF, V_m with threshold V_th (+ optional adaptive θ), refractory period, noise σ, exponential synapses.
- **Areas:** scalar count, identical parameters, independent random wiring. Each area has layers L1–L6, each with E and I cells at one E:I ratio.
- **Input:** chaotic environment (logistic map r, amplitude A), encoder current and cue S, all into L4 of area 1 only. Baseline drive μ goes to every cell.
- **Projections:** within-layer recurrence; canonical L4 → L2/3 → L5 → L6 inside each area (needed so input can reach L6; weight slider, set 0 to remove); feedforward L5-E area k → L2/3 area k+1; feedback L5-E area k+1 → L1 and L6 area k.
- **Output and boat:** four motor pools of L6-E cells in the top area. Spike counts per control interval set a force on a 2-D point mass (p0 = (5,0), target at origin, dashed circle R = 5). Velocity and position integration follow the jaxfne boat step: v' = v + dt(−β v + F), p' = p + dt(v + v')/2, with F = f0·(n₊ₓ − n₋ₓ, n₊ᵧ − n₋ᵧ) from the previous interval.
- **Encoder:** tonic current g·e/(e + d0) on the input cells, with e the distance to the target. This follows the jaxfne contract `SailorTonicEncoder`, `sailor_tonic_current`, `sailor_decode` in `jaxfne/paradigm.py`.
- **HDP:** each cell has a hidden gain H (neutral 1), with τ_H dH/dt = α·I_syn + β − γ·H·r + K_ctrl(1 − H). Each synapse has a plastic gain m with dm/dt = K_HDP(H_post − H_pre)m + K_w(1 − m), clipped to [0.2, 3]. This is a reduced reading of the repo's HDP guide: the W and ρ terms are dropped and the plasticity is multiplicative per edge.
- **Task:** alternating cue and catch trials; a trial counts as detected when the cue group G's rate in its window reaches θ_task.

## Known limits (state these whenever the lab is shown)

- **The boat does not fixate under the default wiring.** Motor pools are random cells, so their force has no direction that points back at the target. With the loop on, distance grows over time. Matching the repo's F1 wiring and gains is the open step.
- **Parameters are not the repo's.** f0, β, g, d0, μ and the laminar weights are chosen for a stable, visible network. They are not the Crazy Sailor values.
- **One boat only.**
- **Verification:** the simulation core was run in Node (finite values, rates, cue effect against an S = 0 control, HDP bounds). The page itself has not been checked in a browser by the author.

## Invariants

- Do not cite lab numbers as results. Rates, distances and accuracies here are demonstrator output with the defaults above.
- Do not change the page's meaning (what the boat or task represents) without the owner. Parameter defaults may change if the tests above still pass.
- The page must stay static: no server-side code, no data files, no external scripts.

## Verification (to run after edits)

1. Open the page; the boat map, schematic, raster, rates, voltage and task panels should all draw.
2. Set "Boat off" then "Balanced": the boat should freeze, then move.
3. Set "No HDP": mean H should show 1.00.
4. Zoom in and out on the boat map; the grid should cover the visible area.

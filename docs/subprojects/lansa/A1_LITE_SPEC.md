# LANSA A1: lite model spec (draft, not sealed, no run)

Status: DRAFT 2026-11-20. Written from `REVIEW.md` and the owner's brief. Nothing here is sealed. Seal this file before any run that uses it (`AGENTS.md`). Open items are marked OPEN and need an owner ruling; no value is invented.

## 1. Purpose

The lite model is the smallest laminar E-I network on which the spectrolaminar question can be tested before scaling: V1 and FEF, 3 layers, E and I per layer, 100 neurons as a demo, then 1000 per area.

## 2. Architecture

- Areas: V1 and FEF (lite), two areas. Feedforward V1 L5 to FEF L2/3; feedback FEF L5 to V1 L1 and L6 (canonical routing, `docs/subprojects/lansa/REVIEW.md` finding 1).
- Layers: 3 (superficial, middle, deep) per area, each with E and I cells. The owner's brief says "3 layers"; the repo's MINIMAL_MODEL_SPEC uses three depth modules S, M, D. Same count, mapping below.
- Input: L4-equivalent (middle) of V1.
- Output: deep E of FEF (the boat controller population in the Fixation Sailor track).
- Neuron model: AdEx (JaxFNE emitter, as E0). OPEN-C5: confirm, since MINIMAL_MODEL_SPEC decision 1 also names Izhikevich-TFNE as the comparison.

## 3. Composition (from data, not assumed)

Source: `docs/laminar_motif/data/lansa_density_X.json`, shape 21 depth bins x 4 types (PV, CB, CR, NG) x 11 areas. Depth bins run from -0.6 to 0.4 around mid-L4; negative is superficial.

- Layer boundaries: OPEN-C1. Proposed: split the 21 bins into three equal-width ranges (superficial -0.6 to -0.2, middle -0.2 to 0.1, deep 0.1 to 0.4). The boundary values are not in the data file and must be ruled.
- E/I: NG is the pyramidal proxy (`REVIEW.md`); PV, CB and CR are inhibitory. CB maps to SST and CR to VIP only by overlap, and the owner ruling says CB is not SST.
- Per-layer counts: sum NG for E; sum PV, CB and CR for I, within each layer, for V1 and FEF. Normalise to 100 neurons per area for the demo.
- Closing check: a test that recomputes the counts from the JSON and matches the spec table.

## 4. Inhibitory subtypes

The brief asks for E and I per layer. Subtypes (PV, SST, VIP) are not modelled in A1. The SST/PV ratio is an E1 parameter, not an A1 one. OPEN-C2: confirm that A1 uses one I population per layer.

## 5. Parameters

- Demo: 100 neurons per area (V1 and FEF together 200 cells).
- Scaled: 1000 per area (2000 cells). One principal delta: N.
- Duration: 12 s with 2 s discarded (as E0/E1), or 20 s as in PLAN_LANSA_E2 C6. OPEN-C6.
- Engine: JaxFNE source at the pinned SHA once the Sailor pin lands (`docs/subprojects/PLAN.md` D1). Until then, the 0.5.9 tag. Record which.

## 6. Outputs and gates

- V0: schematic of the two areas and three layers (visualization contract).
- V1: initial 1 s raster.
- Run receipt: config, seeds, engine SHA, counts, spikes summary, layer power profile.
- Verdict: declared before the run. Each branch named (no silent branch, cf. E1 C7).
- V2: final raster. V3: atlas.

## 7. Open items (owner)

- OPEN-C1: layer boundaries in depth bins.
- OPEN-C2: one I population per layer in A1.
- OPEN-C5: AdEx or Izhikevich for A1.
- OPEN-C6: duration 12 s or 20 s.
- OPEN-C7: verdict branches for A1.

## 8. Not in A1

Mid model (V1 V4 FEF, 6 layers, E-PV-SST) and high model (adds PFC, VIP). Mouse SST/PV profile. HDP (stays in the motif model). Omission (OXM, blocked).

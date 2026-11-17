# Spec (DRAFT, specification only): omission sensitivity of the LANSA column (lineage `o38o`)

Status: **specification only. No readout is implemented and no omission experiment runs.** The omission firewall
(`AGENTS.md`; `SUBSTRATE_PROGRAM.md` `OMISSION_FIREWALL`) stays in force until the owner approves the estimands
below. Built on `docs/designs/OMISSION_SPECTROLAMINAR_HDP_PROPOSAL_2026-11-10.md` (UNSEALED); that proposal is
not changed. Values marked **[owner]** are proposals awaiting a ruling.

## 0. Firewall and ordering

- The substrate (composition, connectivity, weights, kinetics, drive) is frozen before any omission run, from
  non-omission evidence only (OPEN-38 composition; spectrum spec `SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md`).
  No omission outcome feeds back into it; an omission result that motivates a substrate change starts a new
  lineage.
- Open conflict **[owner]**: `SUBSTRATE_PROGRAM.md` places blind omission after the Gen-2 freeze; the proposal
  calls this a separate lineage from the LANSA preprint. Which governs this lineage is the owner's ruling.
- The current motif model fails the full map scorer; no result here is a physiological omission prediction.

## 1. Omission event (from the canonical paradigm, `jomission/paradigm/spec.py`, `conditions.py`)

- Trial: fx -500 to 0 ms, then p1, d1, p2, d2, p3, d3, p4, d4 (p slots 531 ms, d slots 500 ms; p2 onset 1031,
  p3 2062, p4 3093 ms after p1).
- Omission at position k ∈ {p2, p3, p4}: the slot's sensory drive is zero for its 531 ms at its scheduled onset;
  every other time and drive is identical to the standard trial (AXAB, AAXB, AAAX; BXBA, BBXA, BBBX; RXRR, RRXR,
  RRRX). Omission ≠ time deletion: the slot keeps its duration.
- Event time t0 = scheduled onset of the omitted slot.

## 2. Baseline and windows (fixed from task timing, not from any omission data)

- Baseline B: fx window, -500 to 0 ms before p1, same trial. **[owner]** alternative: last 200 ms of the delay
  before the omitted slot (closer in state, but carries the previous stimulus's offset response).
- Primary window W: t0 to t0 + 531 ms (the whole omitted slot). Secondary: early t0..t0+250, late t0+250..t0+531.
- Spectral quantities: band power averaged over trials of a seed, divided by B, then 10·log10 once (log last).
  Bands as the spectrum spec: alpha-beta 10-30 Hz, gamma 50-150 Hz.

## 3. Arms

Mechanism (from the proposal): HDP off, fast only, slow only, both (h_f, h_s as in the proposal; tau grid sealed
relative to 531 ms slots and multi-trial blocks **[owner]**).
Composition: ref (prescribed) and the OPEN-38 arms passing the spectrum study (at most ref + one pooled data arm
+ flat, to bound cost) **[owner]**. Composition is crossed with mechanism only after the mechanism gates pass.

## 4. Contrasts and estimands

Unit of inference: seed (independent network and noise). Trials within a seed are averaged, never counted as n.

| Id | Contrast | Estimand | Role |
|---|---|---|---|
| E1 (primary) | rare omission vs expected absence: AXAB at low probability in an AAAB block vs AXAB in a block where p2 is always empty | Δ_om = mean E rate over S/M/D in W (rare) − (expected), per seed | within-trial input identical; only history differs, so Δ_om isolates expectation-dependent response |
| E2 | omission vs presented: AXAB vs AAAB, same slot | rate and band-power difference in W per unit and class | sensory withdrawal, reported, not an omission effect |
| E3 | mechanism: Δ_om(arm) − Δ_om(HDP off) | per arm, per seed | attribution to h_f / h_s |
| E4 | laminar structure: Δ_om per unit S/M/D × E/PV/SOM; onset latency (first 10 ms bin where the seed-mean difference exceeds 2 SE of baseline) | profile and sign | secondary; Holm over the 9 strata |
| E5 | composition: E1 under data arms vs ref | difference of Δ_om | only after E1 is established in ref |

Block probability of the rare omission and block length **[owner]** (proposal: omission 20% of trials, blocks of
50 trials; values to be checked against the recorded task design before sealing).

## 5. Null and control conditions

- HDP off (negative control): no history state, so Δ_om must be 0 within seed noise; a nonzero value is an
  estimator or schedule leak and stops the study.
- Shuffled history: trial order permuted within each block (same trial counts); Δ_om null distribution,
  1000 permutations per seed at analysis level when the state allows replay, else 5 shuffled-schedule runs **[owner]**.
- Matched withdrawal: zero drive in the slot with no preceding sequence (single-slot trials), rate- and drive-
  matched; separates withdrawal transients from expectation.
- Timing jitter: omitted slot onset jittered ±50 ms **[owner]** across trials; tests reliance on exact timing and
  separates omission from time deletion.
- Gain-matched rate control: tonic drive scaled so the pre-omission E rate equals the HDP arm's; rules out a
  rate-level explanation.
- Spontaneous periods: input-free runs give the floor of every spectral quantity.
- Positive control: an injected history-dependent current of known amplitude (synthetic h) at t0 must be
  recovered by the estimator (recovered/planted within 10%), with h removed giving 0.

## 6. Replication, uncertainty, acceptance

- Seeds 1-10 per arm **[owner]** (each seed runs both blocks). Per estimand: seed mean, SD, 95% t-interval; paired
  by seed across arms.
- Acceptance (from the proposal, made explicit): in the "both" arm, Δ_om > 0 with the 95% interval excluding 0,
  AND above the 95th percentile of the shuffled-history null, AND E3 > 0 with interval excluding 0 against HDP
  off. Otherwise the result is negative and is reported as such. Two-sided intervals; one-sided claims are not made.
- Implementation gates before any run (proposal §Gates): JaxFNE rule registration, state locality, units, sign,
  saturation; impulse decay and two-pulse tests for tau_f, tau_s; smoke one seed; seal; then independent seeds.
- Not retuned after results: windows, baseline, arms, seeds, thresholds fixed at seal.

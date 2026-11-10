# Response to review (Sol, 2026-11-10, Major revision)

Reviewed draft: commit 12842de. Revised sources: `sections/*.tex`, `main.tex`, `main.md`. New analysis:
`scripts/e2_readout_sensitivity.py` → `data/readout_sensitivity.json` (12 runs: 9 ladder, 3 confirmation).
"Done" = changed in the manuscript; "Deferred" = needs a new sealed simulation, with the reason.

## Summary

| # | Point | Response |
|---|---|---|
| 1.1 | Density-derived overstated | Done: title, abstract and Discussion say "density-informed"; imposed parts named |
| 1.2 | Readout not a validated LFP | Done: derivation from Poisson's equation (Methods), σ sensitivity, decomposition; we defend the proxy as a physically motivated estimate of depth structure, state its assumptions and say what it cannot carry |
| 1.3 | SOM/PV effect unidentified | Conceded in text; flat and reversed controls deferred (sealed as T3, not run) |
| 1.4 | α-β claim needs qualification | Done: in-sample qualifier; winner's loss of α-β on its own selection seeds stated as a primary result |
| 1.5 | Gamma generator is a hypothesis | Done: auto/cross decomposition run; abstract reworded to "consistent with" |
| 1.6 | Density vs kinetics | Done: specificity paragraph; no contradiction with Herrera claimed |
| 2 | Statistics table | Done for all eight rows and the 0.95 target |
| 3.1 | Fig. 1 labels | Done: CB, CR, NG labelled as measured markers |
| 3.2 | Mapping underdetermined | Done: exact mapping equations; p = 0.14 explained; per-unit in-degrees stated |
| 3.3 | Readout vs recorded LFP | Partly: argued in Methods/Discussion; rate-spectrum control deferred |
| 3.4 | Reproducibility | Partly: hashes and script commit given; public archive deferred to submission |
| 3.5 | Figure size | Deferred to the layout pass before submission |

## 1.2 The readout: why a current-based proxy estimates the LFP

We agree the RWS output is not a calibrated potential and no longer call it one. We do not agree that it is an
arbitrary observation operator. The argument now in Methods ("From transmembrane currents to the LFP"):

1. **Physics.** At LFP frequencies (< 500 Hz, tissue dimensions ~mm) the displacement and inductive terms of
   Maxwell's equations are negligible (quasi-static limit), so ∇·(σ_e ∇φ) = −Σ I_n δ(r − r_n). In a homogeneous
   ohmic medium φ(r,t) = (1/4πσ_e) Σ_n I_n(t)/|r − r_n|. This is the forward model of Mazzoni et al. (2015) for
   their multicompartment reference LFP. φ is **linear** in the transmembrane currents.
2. **Cable.** If synaptic inputs act as current sources into passive dendrites, the cable equation is linear and
   each segment current I_n is a linear filter of the synaptic currents (with conductance synapses, to the extent
   driving-force fluctuations are small). Then φ_z(t) = Σ_u Σ_r (h_{z,u,r} * I_{u,r})(t): the LFP is a sum of
   linearly filtered unit-by-receptor synaptic currents.
3. **RWS weights are reported to fit; the separable depth kernel is assumed.** Mazzoni et al. reported that RWS
   (c_AMPA = 1, c_GABA = −1.65, τ_AMPA = 6 ms, τ_GABA = 0) explained > 90% of the variance of the multicompartment
   LFP over the positions they tested, including depths along the column axis (their Fig. 5). It did poorly only
   where the LFP was small, near its inversion point. It predicted the gamma peak frequency (Fig. 7C). They also
   reported that interneurons contributed negligibly (Figs. 3 and 8; closed-field stellate morphology), so pyramidal-cell synaptic currents set the LFP,
   including GABA currents supplied by interneurons. Our L_u uses exactly these currents. Extending RWS to a depth
   profile as h_{z,u,r}(t) = w_{zu} c_r δ(t − τ_r), with one w_zu shared by AMPA and GABA, is our assumption. Real
   AMPA and GABA sources sit at different places on the cell and can differ in sign above and below the soma.
   Methods now says so.
4. **Linearity of the proxy.** Currents are taken at V_ref = −65 mV, so by construction
   L_u = 65·g_AMPA,u(t − 6 ms) − 24.75·g_GABA,u(t) mV. Hence contact power is exactly
   P_z(f) = Σ_{u,v} K_zu K_zv S_uv(f) (your 1.5 expression; K real), checked numerically to 1e-9.
5. **Shared bias cancels.** The scored map is R_z(f) = P_z(f)/max_z P_z(f), and the same normalization is applied
   to the recorded maps. Any gain G(f) common to all units and independent of depth cancels exactly, including the
   proxy's reported spectral bias if it is uniform. Depth-dependent filtering would not cancel; the separable
   kernel already neglects it.
6. **What remains uncalibrated is spatial**: unit depths and kernel width σ (w_zu). These are the parts we now test.
   The defense is therefore that the readout is a physically motivated estimate of the LFP's depth structure, not a
   calibrated potential, with its assumptions stated and its spatial ones tested.

**Sensitivity (new, `data/readout_sensitivity.json`).** σ ∈ {0.10, 0.15, 0.20, 0.25} on all 12 runs. Mid-depth
(−0.2 to 0) relative gamma: 0.10–0.19 at σ = 0.10, 0.19–0.31 at 0.15, 0.31–0.47 at 0.20, 0.44–0.62 at 0.25. The
observed value is 0.85. The deficit holds at every width. r_gm rises with σ (≥ 0.6 in most runs at σ ≥ 0.20), so the
gamma correlation is readout-dependent; we say so (Table `tab:sigma`). Depth placement was already shown to move
r_proto (0.38 → 0.72, in-sample). Conclusion in the text: the latent circuit spectra are constrained more firmly
than the probe-level scores.

## 1.5 Decomposition before attribution

Done as requested (Results, "Where the mid-depth deficit sits"). At −0.2 to 0 over 50–150 Hz, across 12 runs:

- Cross terms: 0–12% of power. The auto-only map gives mid-depth relative gamma 0.19–0.30, so this is not
  cancellation between units (rules out your item 3 as the main cause).
- M supplies 72–81% of the auto power, S the rest.
- The normalizing maximum sits at S (−0.4) at 97–101 of 101 frequencies (your item 2: a strong S matters).
- Scaling M's spectrum by up to 1e4 never reaches 0.85. Once M dominates, the map at those contacts tends to the
  squared kernel of a source at 0, whose mean over −0.2..0 is 0.615 at σ = 0.15 (derived).

So the observed breadth needs gamma generated above 0 or spread in depth, not only more M power (items 1 and 4 are
both in play, with item 4 bounded by the σ result). The abstract now says the deficit is "consistent with" missing
mid-depth 50–150 Hz power and is testable. Item 5 (operating regime) is not addressed beyond the reported rates.

## Other points

**1.1.** Title: "A density-informed three-unit E/PV/SOM column reproduces part of the primate spectrolaminar motif
but leaves mid-depth gamma weak". Methods states which parts come from LANSA (unit sizes, in-degree scaling) and
which are imposed (ρ ramp, E:I, weights, depths, inter-unit coupling). It calls the summed separately-normalized
marker density a proxy that does not preserve absolute abundance.

**1.3.** Agreed. Discussion: SOM/PV, size and in-degree co-vary, so the paper does not claim the ratio gradient
produces the deep α-β. Flat and reversed controls at fixed size and connectivity are sealed (T3,
`docs/laminar_motif/SEAL_T2_T3.md`) and listed as planned, not run.

**1.4.** The reference column's r_ab 0.83–0.86 is labelled in-sample (selection seeds; the reference was not rerun
on fresh seeds). The T1 winner, selected on r_proto, has r_ab 0.32–0.47 on its own selection seeds and 0.35–0.47 on
seeds 4–6. Abstract and Discussion state this drop as a primary result. The confirmatory verdict remains FAIL (0/3).

**1.6.** New Discussion paragraph: the negative result is specific to this column, ramp, SOM kinetics, sizes and
stationary drive. It does not test M-unit PV→E kinetics, so it cannot contradict Herrera et al. (2026).

**2.**
- E1 p is reported as a one-sided Monte Carlo p ≤ 1/10 001.
- E1 stays INCONCLUSIVE.
- Ladder trends are descriptive (Results: "Three seeds per size support a trend, not a test.").
- T1 FAIL is the confirmatory verdict.
- Methods states what each null keeps and destroys.
- Pixel dependence is noted; r_proto is a similarity score, not a test on 3150 samples.
- CSD is "internally consistent; source placement assumed".
- The 9.9× speedup is "one timed pair, one layout".
- 0.95 target: Methods states it is above every r_proto ceiling (0.854), so it could not be reached; the six-part rule
  decides. We did not record when this was recognized beyond the ceiling analysis.

**3.1.** Legend: "NG (pyramidal proxy)", "PV", "CB (not SST-specific)", "CR, not modelled". Results notes that the
measured CB/PV profile runs against the imposed SOM/PV ramp.

**3.2.** Methods gives N_E,u = 3N·D_u/ΣD, K_u = 560(D_u/D̄)^B, p_u = K_u/N_E,u, with D_u the summed marker density
(median over areas) over the unit's bins, and rounding. With B = 1, p = 0.14 in every unit (in-degree about 810/513/356
in S/M/D, mean 560). With B = 1.41, p = 0.163/0.135/0.116. The p = 0.07·8000/N_E rule is stated as the baseline
without the in-degree rule. B is an assumption, scanned in T1.

**3.3, deferred.** A population-rate readout scored identically is a useful control, but rate spectra are not a
model of the extracellular potential (point 1.2). It would bound the readout dependence, not replace it. It needs a
scorer extension and is listed as follow-up.

**3.4.** 92 source hashes (now including the readout-sensitivity script and output); scripts at a named commit; the JaxFNE engine commit is named as unreleased. A public
archive (code, run records, seeds, environment) is a submission item and is not done.

**3.5.** Not done in this revision; a figure-layout pass precedes submission.

## Not changed, with reason

- **Flat/reversed SOM/PV, rate-matched PV kinetics:** new simulations need their own seal (project rule: seal
  before run). Your order (analysis first, then simulations) is followed: items 1–3 of your plan are done.
- **Your title:** we kept "density-informed" rather than "CAMINOS-derived". The density scaling of size and
  in-degree is the tested manipulation, and "minimal" is not defined by a test.

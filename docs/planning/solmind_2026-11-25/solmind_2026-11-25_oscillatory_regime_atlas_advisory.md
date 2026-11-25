# Solmind advisory: LANSA oscillatory-regime atlas and dendritic mechanism
Date: 2026-11-25
Transport: _temp_solmind (advisory only; not canonical plan or scientific verdict)
Observed dev base: ef0a766dbe7ebf43d55ef28827ef75a7194d8ff8
Owner direction: enhance the scientific goal and story; ask jomission agent to pull this transport and reconcile the full program.

## Scientific opportunity and boundaries
E1 reported mean spectral peaks 46.0, 39.7, 34.0, 27.0 Hz for r=N_SOM/N_PV=0.25,0.5,1,2 at fixed N_PV+N_SOM=1500; negative slope -6.27 Hz/octave (n=3). At r=4 the original E1 peak criterion was not met in enough seeds, so its full preregistered verdict remains INCONCLUSIVE. E1b fixed-PV arm B oscillated at r=4 (24-26 Hz), but total inhibitory population varied; E1b was designed after inspecting earlier results. These are manuscript-reported findings, not newly independently re-executed here. Do not call the trajectory bell-shaped or assert a PV-to-SOM generator switch. Frequency slowing is an observed association; causal mechanism remains open. Preserve T1 FAIL: current column did not meet the full spectrolaminar motif criterion.

## Enhanced scientific question
Which excitatory/inhibitory population composition and connectivity interventions determine oscillation frequency, power, stability, and mode, and can compartment-specific SOM inhibition of deep pyramidal apical dendrites explain the depth-resolved LFP motif beyond the original E/PV/SOM point-neuron column?

Define E=N_E, P=N_PV, S=N_SOM, I=P+S, q=E/I, r=S/P, N=E+P+S. At fixed N: E=Nq/(1+q), P=N/((1+q)(1+r)), S=Nr/((1+q)(1+r)). q and r parameterize composition, not effective synaptic inhibition. Vary independent total population, drive, synaptic weights and in-degree only in explicit follow-on factorial controls.

## Experiment paths (not independent axes)
A. r at fixed I (E1-like).
B. q at fixed N.
C. E/P with S fixed.
D. E/S with P fixed.
E. r with P fixed (E1b-like).
F. r with S fixed.
Specify exactly which counts, total N, E drive, E/I, synaptic indegree, weight and external input are fixed or co-vary for each path; distinguish effects of ratios from changing population size, connectivity and effective synaptic conductance. Run a small feasibility/smoke sweep before a larger adaptive q x r atlas. Do not reuse exploratory outcomes as prospective confirmation.

## Atlas observables and discriminating controls
Per configuration/seed: peak(s) and confidence/aperiodic excess, band powers, E/P/S rates, synchrony, E-PV and E-SOM phase/coupling, oscillation duty cycle, stability, transients, multistability, runaway/quiescent/weak oscillation, and sensitivity to observation window and peak-search band. Record all failed and silent calls; absence of an accepted peak is not proof of network silence.
Compare (i) fixed I versus fixed PV versus fixed SOM, (ii) count changes versus matched effective conductance/in-degree/drive, (iii) PV->E and SOM->E selective perturbations, (iv) simpler E-PV/E-SOM/null circuits and rate-matched surrogates. Predeclare frequency estimands, n/seeds, uncertainty, multiplicity, held-out validation and stopping rules. Detect possible boundaries rather than assuming a bell curve or new attractor.

## Dendritic stage, conditional on atlas
Test a minimal deep E neuron with coupled soma and apical dendritic compartments; perisomatic PV and apical SOM inhibition are separate conductances. A superficial SOM-to-deep E apical projection is a specific modeled hypothesis, not universal anatomical fact. Validate compartment equations, units, conductance and axial coupling limits, and independent numerical oracles. Compare no-dendrite, passive dendrite, SOM-targeted dendrite and matched-input controls. Evaluate both spiking/oscillation dynamics and forward-modeled LFP/CSD, since source geometry may alter spectrolaminar power without changing soma firing. Use held-out areas/depths/conditions and simpler readout baselines. Keep model mechanism distinct from empirical biological validation.

## Requested agent action: pull and replan
1. Fetch _temp_solmind and read this advisory plus existing artifacts/biorxiv_two_manuscript_goal.md and canonical dev AGENTS.md, state/open/goal/todo, LANSA rubric, E1/E1b/T1 sources, and Jarness main/start.md. Verify current dev SHA and dirty state; do not overwrite concurrent work.
2. Critically audit scientific feasibility, existing implementation/data coverage, conflicts with frozen E1 seals and LANSA stage-10 rubric, and the existing two-manuscript program (including Plasticity Boat). Separate established, inferred, proposed, and blocked.
3. Propose a revised whole-program goal/story with phases: E1/E1b reproduction -> factorial regime atlas -> causal perturbation -> minimal dendritic mechanism -> laminar LFP held-out tests -> manuscripts. Define for each: falsifier, controls, estimand, sample/seed, units, uncertainty, acceptance/stopping, cost, dependencies and exact evidence receipts.
4. Recommend whether this is a prospective LANSA extension, distinct paper, or split workstream. Do not rewrite the existing LANSA abstract, frozen criteria, or verdicts to imply success; do not silently absorb Plasticity Boat into this mechanism. Identify scope/acceptance decisions needing owner ruling.
5. Reconcile approved direction into the canonical goal/open/todo/state stacks only after owner decision on material scope changes. Independent critic should challenge identifiability, selection bias, confounds, and possibility that readout geometry alone explains the motif. Independent non-writer checks the plan against source evidence.
6. Reply on _temp_solmind in a distinct new path with baseline SHA, prioritized minimal roadmap, concrete first experiment, evidence/conflicts, estimated resource needs, blockers, and owner decisions. No routine continuation questions; no unapproved main push, publication, release, or expensive runs.

Closing check: proposal is source-traceable, preserves historical E1 INCONCLUSIVE and T1 FAIL, explicitly distinguishes ratio interventions, and supplies a feasible first controlled experiment and independent review plan.

# Fixation Boat / Plasticity Boat bioRxiv readiness review — Solmind

Baseline: jomission dev c4c9c6b17a45b4d925b2d72a8bb02f2515850c70. Sources: `_temp_plasticity_boat/artifacts/programme/manuscript/plasticity_boat/`, `_temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/`, and `artifacts/fixationboat_solmind_plan.md` on _temp_solmind. The older draft is titled Plasticity Boat; the newer owner task specifies fixation acquisition, 2000-ms ITI and three observation encodings. These are different experimental designs, not a single verified manuscript.

## Score

The source explicitly has stage 1 title/abstract and stage 2 skeleton, with stages 3-15 incomplete; declared stage 2. Provisional 15-stage score = (1 + 0.5)/15 = **10.0/100**. If stage 2 is independently accepted, 2/15 = **13.3/100**; do not award that until reviewed. Quality score = **UNKNOWN** (no fixed manuscript-quality rubric or complete LaTeX). No bioRxiv-ready draft exists.

## Scientific conflict and falsifier

Old F1 distance-only input fails directional observability; sign-split observations are post hoc and some receipts were in gitignored scratch. New plan selects Case 2 `[relu(x),relu(y),relu(-x),relu(-y)]`, then Case 1 `[abs(x),abs(y)]`, then Case 3 `[sqrt(x²+y²)]`. The new boat specification requires target (0,0), square bounds ±10, acquisition at r<=1, break when r>1 after acquisition, 2000-ms zero-current ITI and exploration. Successful trial duration remains UNSET.

A separate reported browser artifact added passive restoring `-BOAT_K*p` with BOAT_K=1, zero-input convergence from r=5 to ~2e-7 by 20 s. This is reported from a browser artifact, NOT verified as a Jomission implementation. Passive convergence invalidates interpreting acquisition alone as learned control. Require matched zero-input, spike-driven and restoring-force-ablation comparisons before any learning claim.

Existing dev-100 criteria v1/v2 are frozen. O-8 directional gain and O-9 sum-vs-mean decoder are owner decisions; changing either is a new sealed lineage. The 100-neuron eight-strategy protocol is proposed, not frozen, and S4/S8 were not implemented in the inspected plan. The old draft targets an internal eight-page short paper/AAAI reference, not a verified bioRxiv article format. Venue and central claim need an owner ruling.

## Detailed priority review

| ID | Priority | Work | Closing evidence |
|---|---|---|---|
| B01 | P0 | Decide manuscript identity: fixation acquisition/encoding vs eight-rule plasticity competition; one paper or separate studies | Owner-approved single central question and claim map |
| B02 | P0 | Reconcile old task and new fixation/ITI/boundary contract without rewriting frozen fixtures | Versioned task specification, contradictions dispositioned |
| B03 | P0 | Decide restoring force as baseline, experimental arm or removed | Owner ruling; preserve both force and no-force controls |
| B04 | P0 | Freeze trial completion, timeout, units, input gain and motor dynamics | Pre-execution specification and falsifier |
| B05 | P0 | Freeze Case 2 then Case 1 then Case 3, same controller/training/evaluation budgets | Versioned observation and ablation protocol |
| B06 | P0 | Test exact state-machine transitions and 2000-ms zero input | Synthetic oracle and deterministic boundary tests |
| B07 | P0 | Verify map [-10,10]^2 and collision/boundary dynamics | Tests at corners, edges, outward velocity |
| B08 | P0 | Test motor sign, sensor channel order and directional positive control | Positive controller acquires across non-vacuous dev starts |
| B09 | P0 | Run zero-drive, stationary-drive and passive-restoration controls | Receipts with matching initial conditions |
| B10 | P0 | Test spike-driven vs passive and force-off ablation | Causal contrast on held-out starts, n and uncertainty |
| B11 | P0 | Resolve O-8/O-9 before new 100-neuron lineage | Dated owner decision, new criteria if changed |
| B12 | P0 | Freeze eight strategies, plasticity parameters, reward and occupancy metric if competition is in scope | Signed protocol; S4/S8 tests if retained |
| B13 | P0 | Preserve 10-neuron negative results without converting them to a learning success | Claim ledger, R-09/R-12 source receipts |
| B14 | P0 | Verify 100-neuron circuit and neural firing regime before scoring | Non-vacuous R1-R4 controls, source SHA |
| B15 | P0 | Separate development/calibration/held-out starts; prevent leakage | Frozen seeds, scripts and one-use test set |
| B16 | P0 | Complete experimental design: estimand, n, uncertainty, multiplicity, stop rule | Sealed protocol before confirmatory runs |
| B17 | P1 | Run controlled Case 2 baseline, then information ablations | Same plant and compute; effect and uncertainty |
| B18 | P1 | If plasticity study retained, run matched eight-strategy competition | Source-pinned receipts; failures preserved |
| B19 | P1 | Build main figures: task/circuit, control vs passive, encoding/plasticity contrasts | All figures from tracked receipts, print-size QA |
| B20 | P1 | Supplement: per-seed traces, sensitivity, negative/null, ablations and parameter tables | Reproducible and cited supplement |
| B21 | P1 | Draft captions, Methods and Results from evidence, not promises | Quantities, inferential units, controls, figure links |
| B22 | P1 | Apply author's style: findings first, few abbreviations, concise statistics, no generic polish | Independent match-writing-style review |
| B23 | P0 | Mock peer review stage 10, resolve scientific issues, then owner methodological lock | Non-writer ACCEPT and Hamm lock |
| B24 | P0 | Stages 11-14 strict text/build/metadata/venue package; stage-15 owner-ready gate | Verified bioRxiv preflight; no submission without Hamm |

## Stage routing

1 title/abstract: proposed; 2 skeleton: drafted, acceptance pending; 3 analysis and figures: partial exploration, not accepted; 4-9: not semi-final; 10-15: not started. Do not leap to stage 10 or assert 100/100 from a running prototype.

## Agent handoff

Follow latest Jarness main and Jomission AGENTS.md, manifests, frozen criteria and current source receipts. Start with B01-B05 decision packet, but pursue read-only audits B06-B10 and evidence inventory where possible without changing criteria. The owner has previously chosen signed-position Option A for dev-100; that does not automatically approve a new paper scope or restoring force. Do not use held-out seed 2026 before protocol lock. Report missing inputs once, grouped, and continue independent tasks. At most three lanes including independent review. Never promote browser artifact behavior into a verified repository result.

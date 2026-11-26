# Review of the R2 spec draft (6b95a85), 2026-11-25

Reviewer: main-jomission session. Not the writer. Subject: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md` on
`_temp_lansa_r2_spec_20261125` at 6b95a85, based on dev 54048dc. Scope: the Y-independence evidence (spec section 3),
the parameter set (section 4), execution (section 5), the criteria (section 7) and the facts they cite. Verdict: REVISE.

## Verified from git and receipts (observed)

| Claim in the spec | Check | Result |
|---|---|---|
| Y imported at 5992eb7, 2026-11-09 22:25:22; parent cc33b37 at 21:48:56 | `git log` | verified |
| sst15 first at 89a1272, 22:31:47, reporting r_proto 0.38 against Y | `git log` and message | verified |
| SST->E with no sst token uses TAU_I_MS = 8.0 at cc33b37 | `git grep` caminos_e0.py:18, :77 | verified |
| dens and kX modifiers documented at cc33b37 | `git grep` variant docstring | verified |
| X imported before Y (8244f89, 2026-11-06) | `git log` | verified |
| 3-unit depths S -0.4, M 0.0, D +0.3 at cc33b37 | `git grep` caminos_stack.py:117 | verified |
| Readout sigma 0.15 introduced at 432ddd1 (2026-11-09 16:30), before Y | `git log -S` | verified |
| Driver `main(seed, T_s, tag, lay)`, `edge_delivery="event"` at dev | `git grep` run_e2_stack.py | verified |
| T1 confirmation records: T_s 12.0, dt 0.1, edge_delivery "event" | 21b2697 receipt | verified |
| lfs_n4000_s2_T8 simulate 1518.5 s for T_s 8; 12 s scales to about 2278 s (about 38 min) | 21b2697 receipt | verified; scaling is arithmetic |

## Findings

MAJOR 1. The gamma criterion is Y-informed, and the spec does not say so (section 7). `R_MIN = 0.6` and the r_gm leg first
appear in 9681fc5 (2026-11-09 23:58). That commit reports the best Y-based run (kd1_sst15) as failing: "r_gm 0.40 < 0.6".
The threshold is therefore first set in the commit that evaluates the Y result it then judges. The spec says the criterion
is "retained by ruling" and "fixed"; it must state this provenance. Fix: either justify 0.6 from a Y-free source (an a-priori
null or the T1 ceiling) or label 0.6 as Y-derived and report the result's sensitivity to it. Owner decision.

MAJOR 2. Y-independence is audited for named knobs, not for the code that runs P0. The P0 values exist at cc33b37, and they
survive at dev. But the code path changed after cc33b37: caminos_stack.py (218 lines changed), caminos_e0.py (23),
run_e2_stack.py (33), allen_v1_replica.py (26), caminos_analysis.py (5). Commits after Y touch that path, including 89a1272,
263dc7d, 8519540, 3c29ca0, 7a83ac6 (a refusal guard on connection probability), 8aa14d5 and aae2bda. A post-Y change that
alters P0's output makes P0 Y-exposed through code, though no value changed. Fix: the seal needs a bitwise baseline of P0
(one seed, short T) at cc33b37 and at the seal commit, in the style of 8519540's "baseline sha test"; the run must not hit
7a83ac6's refusal. Alternatively list each post-Y change to the path and show it output-neutral for P0.

MINOR 1. The kd1 row says the value first appears at 734fede. The cc33b37 message already reads "kd1 x SST 15/30 queued".
Fix: "named in a queued plan at cc33b37; first result post-Y at 734fede". The P1 exposure stands, because the choice of kd1
used Y.

MINOR 2. Event delivery is verified for the T1 confirmation records only. The lfs and sw records carry no edge_delivery field.
Fix: set the mode explicitly in the seal (JAXFNE_EDGE_DELIVERY or the driver) and record it in the run JSON.

MINOR 3. Falsification is asymmetric. Support needs median r_ab >= 0.6 (section 7); falsification names only r_gm and K1.
A run with r_gm >= 0.6 and r_ab < 0.6 is neither. Fix: add that outcome explicitly.

MINOR 4. Section 1 states "Hamm, 2026-11-25: route R2" as a fact. The word was relayed by jxm-dev's session and was not seen
in this lane. Fix: mark it as relayed, as the 2026-11-25 ruling does.

## Not verified here

- The owner route and the seed and cost choices, beyond the receipts cited.
- The GPU or CPU status of the gate venv (a peer report; the spec's own text agrees).

## Verdict

REVISE. Majors 1 and 2 must be resolved before any seal. Minors 1 to 4 are wording and record fixes. A second review is
needed after the revision; it is not the writer's.

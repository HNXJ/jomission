# Re-review of the R2 spec, draft 2 (6185106), 2026-11-25

Reviewer: main-jomission session. Not the writer. Subject: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md` at 6185106
(branch `_temp_lansa_r2_spec_20261125`, based on dev 54048dc). Draft 1 (6b95a85) was reviewed REVISE in
`docs/receipts/lansa_r2_spec_review_2026-11-25.md`. Verdict: REVISE (one major, four minors).

## Status of the draft 1 points

| Point | Draft 2 | Status |
|---|---|---|
| Major 1: 0.6 threshold provenance | section 7 names 9681fc5 (after Y); 0.5 and 0.7 reported descriptively | addressed; matches 9681fc5 (`R_MIN = 0.6`; "r_gm 0.40 < 0.6") |
| Major 2: P0 code path | section 4 precondition: bitwise P0 at cc33b37 and at the seal commit | addressed in intent; design gap, Major A below |
| Minor: kd1 row | "planned before Y, selected after" | addressed; matches cc33b37 ("kd1 x SST 15/30" queued) and 734fede (diff quotes "Best 3-unit run so far: kd1 + SST->E 15 ms, r_proto 0.376") |
| Minor: event delivery | `RuntimeConfig(...)` quoted; "recorded in every run record" | partly; Minor 1 |
| Minor: r_ab falsification | "or median r_ab < 0.6" added | addressed |
| Minor: owner word | "relayed to other lanes" | addressed |

## Findings

MAJOR A. The bitwise precondition compares two delivery modes, and cannot run as written. `cc33b37` has no delivery switch:
no `JAXFNE_EDGE_DELIVERY` or `edge_delivery` in `scripts/` or `jomission/` at that commit, so its driver runs the default
route. The seal side runs event delivery. The only existing evidence that the two modes agree is commit 1bef838
(2026-11-10): an 8 s event run on 351f6048 bitwise equal to the 0.5.8 run (22 arrays, 248,116 spikes). That is one
configuration, not P0 at cc33b37, and the spec does not cite it. Fix: (a) make the precondition run both sides on the
default route (`JAXFNE_EDGE_DELIVERY` unset), which needs no cross-mode claim; (b) keep the event route as a separate check,
citing 1bef838 and a P0 run; (c) state how the cc33b37 side is built (a git worktree at cc33b37, JaxFNE 0.5.8 venv) and
that its record has no `edge_delivery` field.

MINOR 1. Name the switch. The spec quotes `RuntimeConfig(recurrent_backend="edge_list", edge_delivery="event")`. That object
is built only when `JAXFNE_EDGE_DELIVERY=event` is set (dev `scripts/run_e2_stack.py` lines 40-47); unset gives the default
route. The record writes `"edge_delivery": os.environ.get("JAXFNE_EDGE_DELIVERY") or "default"` (line 122), so the record
matches the runtime. A seal command without the variable would run the default route and record "default". Fix: name the
variable in sections 5 and 7, and state that seal records must show `event`.

MINOR 2. Stale owner item. Section 5 still says "a 0.5.8 GPU environment is an OWNER/environment item". Section 9 says it is
decided, then lists it as agent work. The gate venv `/home/hamm/.venvs/jomission058` is a CPU build (jxm-dev's report); no
GPU venv exists. Fix: one statement: decided; not built; the build is an environment task.

MINOR 3. Keep the caveat on sigma_alt. Section 9 records "sigma_alt 0.25" as decided. The protocol states that 0.25 has no
data basis. Fix: keep "no data basis" beside the decided value.

MINOR 4. Unchecked here. Section 3 says "1.41 is the T1 axis-1 winner". `FINDINGS_2026-11-10.md` lines 130-138 show the sstd
axis and the frozen values (kd 1.41, sst 15, sstd 21 at line 137). I did not trace axis 1. Fix: cite the line or drop the
phrase.

## Checked and correct in draft 2

- R_MIN = 0.6 first set in 9681fc5 (line 38); its message reads "Best run so far (kd1_sst15) fails: r_gm 0.40 < 0.6".
- cc33b37 queues "kd1 x SST 15/30" (its subject line); 734fede's diff contains the quoted "Best 3-unit run so far" line.
- Falsification now includes median r_ab < 0.6.
- Owner choices are marked as relayed from jxm-dev's session, not seen in this lane.

## Verdict

REVISE. Major A must be fixed before the precondition can be run. Minors 1 to 4 are wording and record fixes. A third
review is needed after the revision.

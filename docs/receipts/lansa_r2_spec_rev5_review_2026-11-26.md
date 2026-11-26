# Review of the R2 spec, draft 5 (3b2ceb1), 2026-11-26

Reviewer: main-jomission session. Not the writer. Subject: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md`, the
pre-seal receipt `docs/receipts/lansa_r2_preseal_2026-11-26.md` and the new `manifests/todo.json` entry, all at 3b2ceb1 on
`_temp_lansa_r2_spec_20261125` (based on dev 54048dc). Draft 4 (d2df9d5, with the wording fixes at a4b9a89) was ACCEPTed.
Draft 5 is a design change: scored runs move from the event route to the default route on 0.5.8. Verdict: REVISE, two
required fixes before the seal and three minors. The design change itself is sound.

## Verified

| Claim | Check | Result |
|---|---|---|
| Tag `v0.5.8` (11957afa) has no `edge_delivery` | `git grep edge_delivery v0.5.8` in `/mnt/e/repos/jaxfne` | verified: no match |
| 351f6048 is a descendant of v0.5.8 | `git describe --tags 351f6048` | verified: `v0.5.9-107-g351f6048` |
| The T1 confirmation records are jaxfne 0.5.9 | `t1_confirm_kd1p41_sst15_sstd21_n4000_s4_T12.json` at 21b2697 | verified: `jaxfne_version` 0.5.9 (line 179) |
| The event route ran only on 0.5.9 | the two checks above | verified; so the scored runs cannot use event on the 0.5.8 pin |
| The owner decisions (D1, the §7 rule, the manifest entry) | spec section 1, todo trigger | recorded as relayed from jxm-dev's session (Hamm 2026-11-26); not seen in this lane |
| Cost, 212.7 s and 211.3 s per 1 s, so about 42 min per 12 s run | receipt section 1 | consistent with the T1 scaling; not re-run |

Not re-run here: the EQUAL verdict, the known-negative flip, and the 22 array and 10 field counts. They are the writer's
receipt.

## Required before the seal

R1. The todo entry's authority sentence conflicts with `AGENTS.md`. The entry's `parent_authority` says "this item
authorises the lineage alongside next_authorized_task". `AGENTS.md` line 12 allows execution only of
`next_authorized_task` or of a task the reviewer names. OPEN row 9 says "Hamm chooses the next task; no silent change". The
entry does not change `next_authorized_task`, and that is correct, but the sentence claims an authority that the scope
rule does not give it. Fix: rewrite `parent_authority` to say the entry does not authorise execution, and that execution
needs the owner to set `next_authorized_task` to `SCI-LANSA-GEOM-R2`. That setting is the owner's act. It is not made here.

R2. The EQUAL verdict depends on a script that is not in the repo. The receipt names `_temp_r2_compare.py` in session
scratch. `git ls-tree -r 3b2ceb1` contains no such file. A later reader cannot re-run the check. Fix: commit the comparator
under `scripts/` (not as `_temp_`, because it is seal evidence), and record the sha256 of the comparator and of both run
JSONs in the receipt.

## Minor

M1. Three seeds run in parallel on CPU. Each JAX process uses multiple threads by default. The host rules require BLAS
threads pinned to 1 for grid work, with a serial equivalence check on samples. Fix: state the thread setting in section 5,
and check that one parallel seed equals its serial run.

M2. Section 5 says the route was "decided by the agent 2026-11-26 under the owner's standing rule". The reason is given, and
the choice keeps the 0.5.8 pin, so the rule applies. The sentence should also name the cost of the choice: default route at
about 212 s per 1 s, against the event route's about 181-194 s per 12 s on 0.5.9 for the T1 confirmation. Those runs are not
the same engine, so the cost is not comparable across them.

M3. The section 4 "Result" line cites the receipt. The receipt's run outputs are untracked (`.npz` files are not committed).
Fix: the receipt should list the sha256 of the two `.npz` files it compared, so a later reader can tell which outputs the
verdict covers.

## Verdict

REVISE. The design change is correct, and the pre-seal baseline is a real control: it flips one bit and reports DIFFERENT.
Before the seal, fix R1 (the todo wording; the owner sets `next_authorized_task`) and R2 (commit the comparator). M1 to M3 are
wording and record fixes. Draft 6 needs a review before the seal.

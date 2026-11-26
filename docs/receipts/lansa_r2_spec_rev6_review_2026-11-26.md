# Review of the R2 spec, draft 6 (bc68dbb), 2026-11-26

Reviewer: main-jomission session. Not the writer. Subject: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md`, the pre-seal
receipt `docs/receipts/lansa_r2_preseal_2026-11-26.md`, `scripts/r2_compare_runs.py`, `manifests/todo.json`,
`manifests/current_state.json` and `docs/OPEN.md`, all at bc68dbb on `_temp_lansa_r2_spec_20261125` (remote tip checked
with `git ls-remote`). Re-review of draft 6 after the draft 5 review (REVISE, 8015c5e).

Verdict: REVISE. Draft 5's R1 (manifest authority wording) and R2 (comparator not committed) are fixed in the manifests and
in `scripts/`. Three spec-text items remain. Q1 is the seal gate itself and restates the authority error R1 named.

## Verified

| Claim | Check | Result |
|---|---|---|
| Comparator committed; sha256 `6866a6d0...` | `git show bc68dbb:scripts/r2_compare_runs.py \| sha256sum` | verified |
| `next_authorized_task` = SCI-LANSA-GEOM-R2 | diff vs 3b2ceb1; `python3 -m json.tool` | verified (valid JSON). The owner ruling is relayed; not seen in this lane |
| todo `parent_authority` names `next_authorized_task` as execution authority (R1, todo half) | todo diff | verified |
| OPEN row 9 and OPEN-43 record the pointer | OPEN.md diff | verified |
| Offline project check, clone at bc68dbb, gate venv `jomission058` (jaxfne 0.5.8) | `scripts/project_check.py --offline --no-tests` | every check [ok] except `editable install: jomission -> /mnt/d/jomission/jomission/__init__.py` (FAIL). The gate venv installs jomission from `/mnt/d/jomission`, so a clone cannot pass this check. Result PROJECT_CHECK_FAIL for that reason only. `V2_LOCAL_OPERATION` reads FAIL; this draft does not change it |
| GPU venv `jomission058gpu`; tag `v0.5.8` | `pip show`; `git rev-parse v0.5.8^{commit}` in `/mnt/e/repos/jaxfne` | jaxfne 0.5.8, jax and jaxlib 0.10.2; `11957afa` |
| Driver lines 40-47 (`runtime_kw`) | `git show bc68dbb:scripts/run_e2_stack.py` | verified |
| 24 cores (§5) | `nproc` | 24 |
| T1 device records | `git grep -A2 jax_devices 21b2697` in the cowork repo, `results/laminar_e2` | 127 records, all `"cpu:0"` |
| No GPU result in text at 21b2697 | `git grep -I -i -E 'gpu\|cuda'` | no text match; the hits are base64 image data in `gallery.html` and binary PNGs |
| Code changed a4b9a89..bc68dbb | `git diff --name-only` | only `scripts/r2_compare_runs.py`; driver and engine unchanged |
| 22 arrays, 10 fields, EQUAL, known negative, the `.npz`/`.json` sha256 | run outputs | not reproduced: the outputs are not in the repo or in session scratch. The comparator was read, not re-run. These are the writer's receipt |

## Required before the seal

Q1. Spec §0(b) reads "(b) a manifest entry authorises the lineage". This is R1 in the spec text. AGENTS.md line 12 gives
execution scope to `next_authorized_task`, and the todo entry now says so. Fix §0(b) to: "(b) `manifests/current_state.json`
`next_authorized_task` names SCI-LANSA-GEOM-R2 (owner, 2026-11-26; AGENTS.md scope)". §1 "the manifest entry" should read
"the todo entry `SCI-LANSA-GEOM-R2`".

Q2. Spec §5, device D2 bullet: "8 analysis files carry no device field" is not supported. Measured at 21b2697: 127 run
records carry `jax_devices` (all `cpu:0`); 366 of 493 JSON files in `results/laminar_e2` carry no device field. Fix: state
these counts, or remove the clause. Count: `git ls-tree -r --name-only 21b2697 -- results/laminar_e2 | grep '\.json$'`, then
`git grep -q jax_devices 21b2697 -- <file>` per file. The a4b9a89 apply note already records this figure as not counted.

Q3. Spec §9 item 1 is stale. It asks for "an independent review of draft 5"; the review now needed is of the next draft. §9
also lists "Open:" with a pre-seal checklist, which is not an owner item. Under §0(a) the owner items are all decided
(P0, sigma_alt, depths, venv, D1, §7 rule, manifest entry). Fix: "Open: none" for owner items, and move the checklist to §4.

## Minor

m1. §5 Threads says "every run sets" the thread variables. The §1 baseline ran with BLAS threads at default (the receipt
says so), and §3 shows default and pinned runs EQUAL. Write "every scored and smoke run", and name the baseline's threads.

m2. Receipt §1 says "later commits change only docs and manifests". `scripts/r2_compare_runs.py` is added in that range.
Write "docs, manifests and the comparator script". The driver and engine are unchanged.

m3. Cost (§5). Default route: 211-213 s per 1 s, about 2540 s per 12 s (42 min). Event route on 0.5.9, T1 confirmation:
181-194 s per 12 s. Ratio about 13-14x (DERIVED from those figures). State the ratio as the cost of the choice, and say the
cause (route or engine) is unknown. "Not comparable" alone hides it.

m4. §1 review list names drafts 1-3 only. Add draft 4 (d2df9d5, ACCEPT, `lansa_r2_spec_rev4_review_2026-11-26.md`) and
draft 5 (3b2ceb1, REVISE, `lansa_r2_spec_rev5_review_2026-11-26.md`).

m5. Source of the 2026-11-26 pointer ruling. `current_state.json` says "Hamm, session chat". Name the session the words
came from, as the handoff does for other rulings; this reviewer did not see them.

m6. Comparator, nit. The summary line reads `za['counts']`; a file without `counts` raises KeyError instead of printing
DIFFERENT. Use `.get` with a DIFFERENT result.

## Not done here

- No run, no seal, no edit to the spec branch.
- The comparator was not re-run and the run-output sha256 values were not checked (outputs unreachable). Those values and the
  EQUAL and DIFFERENT verdicts are the writer's receipt.
- The seal must run `project_check.py` to PROJECT_CHECK_PASS in the checkout the gate venv's editable install names. A clone
  cannot pass (above).

## Verdict

REVISE. Q1 to Q3 are wording and count fixes; m1 to m6 are minor. Draft 7 needs a short re-review of its diff before the seal;
ACCEPT requires zero Q items. The design (default route on 0.5.8, device D1, P0, committed comparator, pre-seal EQUAL with
the known negative) is sound.

Tier-0 tests were not run. No spec, manifest, OPEN.md or seal changed in this commit.

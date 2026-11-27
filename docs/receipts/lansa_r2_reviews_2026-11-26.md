# LANSA R2: review record (lineage SCI-LANSA-GEOM-R2)

Each review was read-only. The writer did not review its own work. Verdicts and findings are recorded as the reviewers reported them; the writer's dispositions follow each table.

## Review 1: independent reviewer subagent (non-writer), first pass

Scope: the readout outputs, the G0 cross-check, the per-area estimand, K1, the falsification logic, receipt sections 1 to 5, and the manifest. The commit reviewed is not named in its report; it preceded the viz work.

Verdict: ACCEPT. It reproduced all 18 per-seed medians (G0, G1, G2, G3, G1', K1) and the G1 max-area values from `sc.unit_spectra` and Y directly. It confirmed the manifest (`sha256sum -c`, 38 OK at that time) and the sealed R2 thresholds, bands, sigmas and depths. It found one receipt defect, a stale section 6, which was fixed.

## Review 2: Gemini neon (agy, `gemini-4-neon-high`), read-only

Scope: the R2 open items and general review, at `1b2e19e` (`r2_run` clone).

Verdict: SOUND WITH FIXES. Its findings were checked against the repository; the dispositions are in receipt section 8 (19 items: items 1 to 14 fixed, item 15 recorded as an erratum with the sealed text unedited, items 16 to 19 owner decisions).

## Review 3: independent reviewer subagent (non-writer), second pass, on commit 489ecd3

Scope (the ruling's checklist): exact correspondence between outputs and the sealed protocol; per-condition crossing, sensitivity and falsification decision; known-answer tests and their negative controls; V0 to V3 provenance and figure claims; archive, manifest and receipt completeness; classification of the +0.095 discrepancy and of K2; seal authority for the spec text.

Verdict: REVISE.

| # | severity | finding | disposition in the next commit |
|---|---|---|---|
| 1 | major | Receipt line 3 and atlas call 46899e2 "sealed". The spec says "Nothing here is sealed"; the seal gate calls it a candidate. | Fixed. The receipt defines "spec" and "sealed" usage (header); seal status in section 10 with the evidence and the owner decision. The atlas now says "spec text at 46899e2, whose seal status is in receipt section 10". |
| 2 | major | The section 7 decision logic had no tests. | Fixed. Factored into `sealed_decision()` (scripts/e2_r2_readout.py) and called by the readout. 8 tests cover: support, boundaries (inclusive), r_gm failure, r_ab failure, equal and below K1, all three failing, and the three observed seeds. Mutation check: a threshold of 0.5 instead of 0.6 fails 2 tests in a scratch copy. |
| 3 | minor | The negative control could not fail. | Fixed. Exact known-answer test for the single-unit readout, plus a test that the single-unit and flat readouts differ by more than 0.5 (measured 0.98). |
| 4 | minor | The LFP axis said "PSD (pA^2/Hz)"; the scored cross-spectrum has no 1/fs factor. | Fixed. Relabelled "power (pA^2, unnormalised: no 1/fs factor)". |
| 5 | minor | The smoke npz and json, and the logs, were not archived. | Fixed. Archived in `results/laminar_e2_r2/` with the manifest (section 7). |
| 6 | minor | The cross-check JSON recorded hardcoded `"pass": true` flags. | Fixed. The JSON records the measured absolute differences (`abs_diff_r_gm`, `abs_diff_r_ab`, `abs_diff_contrast_M`, `max_abs_diff_profiles`). |
| 7 | minor | The +0.095 erratum did not say that owner confirmation is pending. | Fixed (receipt section 9). |
| Q4 | (checked) | Identity gate: would it fail on a mismatch? | Verified non-vacuous by the reviewer: an injected spike shift and an injected count change both make it fail. |

Reviewer's classification of the other items (no change needed):
- +0.095 is the mouse-derived L4 boundary (z = (505 - 431)/779), used in `scripts/lansa_t1_anatomy_recount.py`. It enters no R2 simulation, readout, scorer or verdict number. Section 7 contains no depth value, so the estimand and decision rule are unchanged.
- K2 is mandatory under the sealed protocol and is not in the spec section 7 decision rule. Its selection procedure is not defined.
- The seal authority: the text at 46899e2 equals the text accepted in the draft-8 review at c33e340. "Seal" is not recorded as an act. See receipt section 10.

## Review 4: independent reviewer subagent (non-writer), third pass, on commit 63d891c

Scope: review-3 repairs 1 to 7; the ruling checklist (readout against the pre-repair values, section 10 against git, gate). The reviewer changed nothing and ran no pytest; its coverage check was a read-only reimplementation.

Verdict: ACCEPT of repairs 1 to 7. The repository gate FAILS (tier-0 `tests/test_harness_guards.py::test_results_coverage_registered`: two tracked `V0_rule_table.json` files are not in the sealed registry or a lineage record). Not integration-ready.

| # | severity | finding | disposition |
|---|---|---|---|
| 1 | gate | `manifests/lineage_registry.json` has no SCI-LANSA-GEOM-R2 record, so the two tracked V0 table JSON files cannot be named. The reviewer recommends adding the record (naming the table and the `visualization` block) and rerunning the gate; the gate itself must not be weakened. | open; owner decision (creating the record is the owner's step at lineage completion). No change in this commit. |
| 2 | minor | receipt section 11 said "all paths compared"; the pre-repair JSON is not preserved, so this cannot be reproduced. | fixed in this commit: section 11 states the comparison was made before the archive refresh and that the pre-repair files are not preserved. |
| 3 | minor | the scored `.npz` and output JSON are gitignored (`.gitignore:16`); git holds checksums only. | disclosed in receipt section 7; a second off-disk copy is the owner's call. |

Checks the reviewer passed: `sha256sum -c MANIFEST.sha256` in `results/laminar_e2_r2` (65 entries, 0 failures); the repo manifest equals the archive manifest; archive scripts and tests are byte-identical to the clone; receipt section 4 values agree with the JSON; section 10 facts agree with git (46899e2 is an ancestor of 2138fe0 and of `origin/dev` at ec21df7; the seal-gate quotes; run files written 17:04 to 18:40, after 2138fe0).

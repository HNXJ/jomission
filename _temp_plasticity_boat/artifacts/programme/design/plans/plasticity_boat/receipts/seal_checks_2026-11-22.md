# Seal checks, 2026-11-22

Measured on branch `_temp_plasticity_boat` at commit `c1a5b73` (the tip before this receipt was added), in a fresh `git clone --shared` of that branch. Each check is a command and its summary line.

| Check | Command (run in the clone) | Result |
|---|---|---|
| Tier-0 harness (33 files) | `python -m pytest -q <files listed in manifests/test_tiers.json tiers.0.tests>` | `258 passed, 5 warnings in 264.70s`, rc 0. Not under `suite_lock` (the lock file is outside the sandbox); run in the background. |
| DMS acceptance | `cd _temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/code && PYTHONPATH=<a JaxFNE checkout with jaxfne/paradigm.py SailorConfig> python -m pytest -q test_dms_task.py -rs` | `19 passed in 15.38s`, no skips. The JaxFNE checkout used was jaxfne `origin/dev`, which has `SailorConfig`. |
| Project quick check | `python scripts/project_check.py --offline --no-tests` | `PROJECT_CHECK_PASS`; tree clean; next authorized task `SCI-AGSDR-CALIBRATION` (not this programme); gate `V2_LOCAL_OPERATION` FAIL (jomission scientific state, not this programme). |
| Evidence manifest | `receipts/gitignored_evidence_manifest.tsv` verified with `shasum -a 256 -c` run in the evidence folder | 0 non-OK lines out of 123 files, on the Mac that ran the work. |

Not run: `release_artifacts` gate (dist absent; jomission EX-31). Full project check (needs network; the quick check was run).

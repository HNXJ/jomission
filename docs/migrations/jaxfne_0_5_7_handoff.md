# JaxFNE 0.5.7 migration: handover

**State.** The compatibility work is committed and the repository is green under its
declared authority. The engine promotion is withheld on one measured fact: JaxFNE 0.5.6
changed the default native-noise law, which changes the substrate's stochastic input. The
owner picks the engine version; everything else is prepared.

**Read first.** `AGENTS.md`, `docs/HANDOFF_CURRENT.md`, `manifests/current_state.json`,
`manifests/todo.json` (item `JAXFNE-0-5-7-MIGRATION`), then this file and
`docs/migrations/jaxfne_0_5_7_migration_spec.md`.

## 1. Where the work is

| item | value |
|---|---|
| authoritative clone | `E:\jomission` (owner designation, 2026-10-23) |
| branch | `migration/jaxfne-0-5-7`, unpushed |
| `main` | `5b2c861`, untouched |
| commits (promotion path) | `46aa09e` spec · `6b2b9b0` sealed specification · `e9b2367` compatibility deltas, identity re-baseline, sealed record · `27ab1e0` registry, lineage edge, handoff · `0bb2fae` candidate receipt note |
| interpreter | `C:\Python314\python.exe` (CPython 3.14.3), jaxfne 0.4.25, matching the authority |
| editable install | `jomission` resolves to `E:\jomission\jomission` |
| reference clone | `E:\repos\jomission` at `0d39ad5`, clean, 7 behind, not to be modified |
| recovery work | `stash@{0}` `66d578e38f756316e0c9f94cf7a7503ac4049685`, `recovery/atlas-suite` `e9c52fc`; both untouched |
| measurement baselines | `E:\wt\venv-jaxfne-{0425,050,055,056}` (one pinned engine each), `E:\wt\jomission-pre` (worktree at `5b2c861`, the pre-change code) |
| wheels (hash-verified) | `C:\Users\nejath\AppData\Local\Temp\opencode\wheels\jaxfne-0.4.25/0.5.0/0.5.5/0.5.6/0.5.7-py3-none-any.whl` |

pip cannot reach the package index from this machine (DNS failure). Wheels were fetched
with `curl` from `files.pythonhosted.org` and installed with
`pip install --no-deps --no-index <wheel>`. `curl` reaches PyPI; pip does not.

## 2. What is done

| item | receipt |
|---|---|
| retinal area declaration | `jomission/tfne/realize.py:declare_retinal_interface`, used by the ten whole-system arm scripts. JaxFNE 0.5.6 checks at construct that the declared areas are the built columns; the arms built a `Retina` column without declaring its area |
| Gen2 compatibility deltas | `jomission/network/builder.py`: the parameters JaxFNE 0.5.5 refuses are removed and the locality and readout markers move into this module's metadata |
| Gen2 identity re-baseline | `f00e6a5e24f3dab4`, measured identically under 0.4.25 and 0.5.7; live pins updated in `jomission/simulation/state_replacement.py`, `jomission/ablations/rf_rate_factorial.py`, `manifests/rf_rate_factorial_design.json` (self-hash recomputed), `tests/test_rf.py`, `tests/test_area_local_field.py` |
| re-seal | the ten sealed whole-system paths re-registered in `manifests/sealed_artifacts.json` with their new blobs; the specification and the record registered |
| sealed record | `results/jaxfne_0_5_7_migration.json`, verdict `MIGRATION_BLOCKED_ON_NATIVE_NOISE_LAW`, status `FAIL` |
| lineage edge | `JAXFNE-0-5-7-MIGRATION` in `manifests/lineage_registry.json` |
| TODO item | `manifests/todo.json`: `JAXFNE-0-5-7-MIGRATION`, status `BLOCKED` |
| gates | `python scripts/project_check.py` ends `PROJECT_CHECK_PASS` (tier 0: 212 passed); tier 2: 406 passed, 3 skipped, 0 failed; tier-1 subsets green |
| tools added | `scripts/jaxfne_construct_ab.py` (construct invariants per engine), `scripts/jaxfne_probe_ab.py` (probe arrays per engine), `tests/test_migration_jaxfne_057.py` (construct-only regression test, tier 0) |

## 3. The blocking finding

JaxFNE 0.4.25 through 0.5.5 add `0.5 * N(0,1)` per neuron per step. JaxFNE 0.5.6 and 0.5.7
compute the coefficient as `0.5 * sqrt(DT_REF_MS / dt_ms)` with `DT_REF_MS = 0.5`
(`jaxfne/emitters.py`, `_noise_coef`), so at the canonical dt 0.1 ms the per-step
coefficient is 1.1180. The project's `NATIVE_NOISE_DEFAULT` invariant
(`docs/project-sources/SUBSTRATE_PROGRAM.md`) states the per-step law, and every sealed
result was produced with it.

Measured over a 20 ms segment at dt 0.1 ms, seed 0, Gen2 model (`scripts/jaxfne_probe_ab.py`):

| engine | noise law | V_m, sources, spikes against 0.4.25 |
|---|---|---|
| 0.4.25 (authority) | per-step 0.5 | reference |
| 0.5.0 | per-step 0.5 | bit-identical |
| 0.5.5 | per-step 0.5 | differ; the engine re-chains the noise stream per chunk. 335 against 337 spikes |
| 0.5.6 | dt-invariant, 1.1180 per step at dt 0.1 | differ |
| 0.5.7 | dt-invariant, 1.1180 per step at dt 0.1 | differ; 331 against 337 spikes, V_m max abs diff 97.7 |

The construction is invariant in every case: the TFNE model has 2224 neurons, 1 294 396
edges before `enforce` and 246 844 after, the same projection identity set and the same
weight bytes (`sha256 8fccec054edeee9dce09166c622c77862e457acdf7fb2df393060fbf3d0df813`) under
0.4.25 and 0.5.7, and the Gen2 model has 10 666 edges, weight bytes
`sha256 038b5bf1b577f25b343244e36cfdf036e00e96c5f50b0abfb14ad1ba7be4afde` and delays
{20: 10 076, 80: 287, 120: 303} under both.

The historical law cannot be pinned from jomission at the emitter level: the built emitter
params (`IzhikevichParams`) carry no `noise_scale` field in either engine version. The only
route is `runtime_cfg.hdp_params["noise_scale"]`, which the baseline kernel forwards when
HDP is off and which the HDP path reads from the frozen canonical hp (`f327f9d2ad64cc88`).
Declaring it is therefore a substrate-wide re-declaration, at 42 `RuntimeConfig` sites and
with a re-frozen hp hash, which is a lineage of its own.

## 4. The owner's decision

| option | engine | consequence | what it needs |
|---|---|---|---|
| A (recommended) | 0.5.5 | the newest engine that keeps the historical noise law; the seeded realization changes, so no historical result is bit-reproducible | install the wheel, promote, re-run the gates |
| B | 0.5.7 as it is | the substrate's per-step noise at dt 0.1 ms becomes 1.1180; a science change that needs its own authorization and a re-baseline of the noise-dependent lineages | the same, with the change recorded as a limitation |
| C | 0.5.0 | the newest engine that changes no science; the compatibility deltas become unnecessary and are harmless | install the wheel, promote, re-run the gates |
| D | 0.5.7 with the law declared | the historical law is preserved under the latest engine, at the cost of 42 runtime sites, a re-frozen hp hash and an amended `NATIVE_NOISE_DEFAULT` wording | a separate lineage; the current branch does not carry it |

## 5. Applying a decision

For any option, in this order, on the branch `migration/jaxfne-0-5-7`:

1. Install the engine: `python -m pip install --no-deps --no-index <wheels>\jaxfne-<version>-py3-none-any.whl`, then `python -c "import importlib.metadata as m; print(m.version('jaxfne'))"`.
2. Promote the authority: `manifests/current_state.json` (`jaxfne_version` and
   `execution_authority.evidence` -> `results/jaxfne_0_5_7_migration.json`) and
   `pyproject.toml` (`jaxfne==0.4.25` -> `jaxfne==<version>`). The Gen2 identity needs no
   re-baseline: `f00e6a5e24f3dab4` holds under every engine measured.
3. Update the sealed record's `promotion_state` and `verdict`, and re-register its blob in
   `manifests/sealed_artifacts.json` with the sealing commit.
4. Set the TODO item to `DONE` with its receipts, and the lineage edge's verdict to the
   promoted outcome.
5. Refresh the handoff: `python -m jomission.harness.handoff`.
6. Run the gates: `python scripts/project_check.py` (full, no flags), then tier 1 and tier 2
   from `manifests/test_tiers.json`. Any failure or unexplained drift stops the promotion.
7. Commit on the branch, then merge to `main` and push only on the owner's word. The
   push uses `git push git@github.com:HNXJ/jomission.git <branch>:refs/heads/<branch>`.

For option B, the record must state that the noise law changed and that the sealed
0.4.25-era results are not reproducible under the new authority. For option D, this
branch is the wrong place: the noise declaration has to be specified and sealed first.

## 6. Constraints

- No scientific execution. `next_authorized_task` is null and stays null until the owner
  names a task.
- The sealed `results/whole_system_ei_native_state.json` (`8527125785223aa5614a2ae94a232ef7b3b9e520`)
  and the SCI-EI-NATIVE-STATE-1 verdict (`MIXED_STATE_MECHANISM`, qualification `FAIL`) are
  untouched history, produced under JaxFNE 0.4.25.
- `stash@{0}` and `recovery/atlas-suite` stay as they are. They are the rollback evidence
  for the atlas work, and the atlas content is already represented on `main` in a later
  form.
- `E:\repos\jomission` is a reference clone. Do not synchronize, reset or modify it.
- The migration is one lineage with one principal delta. A second change needs a new
  authorization.

## 7. Verify the state

```bash
cd /e/jomission
git log --oneline migration/jaxfne-0-5-7
git status --porcelain --untracked-files=no          # expect empty
git stash list                                       # expect one entry, 66d578e3
python scripts/project_check.py                      # expect PROJECT_CHECK_PASS
python -m pytest -q -p no:cacheprovider tests/test_migration_jaxfne_057.py
python scripts/jaxfne_construct_ab.py --out <tmp>/a.json --model tfne
```

## 8. Open items

| item | state |
|---|---|
| Q8 quarantine | `manifests/q8_021_quarantine.json` remains in force. The live Gen2 hash moved again (`f00e6a5e24f3dab4`), so its stale-identity class still applies; its lift condition (renew the builder freeze, make the `delay_state` assertions array-aware, and have the Q8 suite green) is untouched and needs its own TODO |
| `NATIVE_NOISE_DEFAULT` wording | states the per-step law and ties it to the engine's default. Under 0.5.6+ the wording needs amendment whichever option is taken, and under option D the law has to be declared in jomission |
| documentation | `AGENTS.md` and `docs/HANDOFF_CURRENT.md` named `E:\repos\jomission` as the working clone; both now name `E:\jomission`, and `ENV-STALE-EDITABLE` in `manifests/harness/environment_failures.json` describes the repaired pointer |
| test residue | a tier-2 run writes `.tmp.driveupload/` and `.pages_test_tmp/` into the repository root, and the second one left a read-only directory that failed seven tier-0 tests with `WinError 5`. Neither path is ignored. Both were removed; an ignore rule is the cheap fix |
| `G0` | no gate named `G0` exists in this repository. The live gate is `V2_LOCAL_OPERATION` (FAIL), and the open tasks are `SCI-AGSDR-CALIBRATION` (BLOCKED) and `SCI-PROP-QUALIFIED-1` (LOCKED) |
| pages build | the `.pages_test_tmp` lock is a Windows file-lock artifact, not a project defect; the pages tests pass once the stale directory is removed |

# JaxFNE 0.5.7 migration candidate — receipt

**Branch:** `migration/jaxfne-0-5-7` (unpushed; `main` untouched at `5b2c861`)
**Base SHA:** `5b2c8612716764a401a4f0432a5cc734735ce6cc`
**Candidate SHA:** `5a01add` (implementation), `46aa09e` (specification, committed before the runs)
**Interpreter:** `C:\Python314\python.exe` (CPython 3.14.3), jaxfne `0.5.7`
**Specification:** `docs/migrations/jaxfne_0_5_7_migration_spec.md`
**Authority:** unchanged. `manifests/current_state.json` still names jaxfne 0.4.25;
`next_authorized_task` is null; no scientific execution; no `results/*.json` written.

## Verdict

**The candidate does not pass the gate.** Scope A (the TFNE substrate that produced the sealed
`MIXED_STATE_MECHANISM` result) is migration-clean and graph-identical to jaxfne 0.4.25.
Scope B (the retired V2.1 chain's Gen2 builder) is blocked by engine API deprecations that
reach a probe-semantics parameter. Two further failures are the price of withholding the
authority promotion and the sealing step.

## What was run

| gate | command | result |
|---|---|---|
| construct tests | `pytest tests/test_whole_system_ei_native_state.py tests/test_migration_jaxfne_057.py tests/test_tfne_v2_spec.py` | 14 passed (42 s) |
| tier 0 | the `manifests/test_tiers.json` tier-0 list | 2 failed, 210 passed (108 s) |
| tier 1 | covered by the tier-2 run (same files, no ignores) | — |
| tier 2 | the `manifests/test_tiers.json` tier-2 command | 59 failed, 337 passed, 3 skipped, 10 errors (202 s) |
| full project check | `python scripts/project_check.py` (no flags) | `PROJECT_CHECK_FAIL: jaxfne identity: installed 0.5.7, execution authority 0.4.25` |
| cross-engine A/B | `scripts/jaxfne_construct_ab.py` under 0.4.25 and 0.5.7, then `--compare` | every invariant equal except the config hash |
| engine matrix | one process per engine in a pinned venv (`E:\wt\venv-jaxfne-<v>`) | 0.5.0 fully compatible; 0.5.5+ need the retinal declaration; 0.5.5+ break the Gen2 path |

## Failure classification (tier 2, 69 failures and errors)

| cause | count | attributable to |
|---|---|---|
| `ValueError: inter_column_connectivity(delay_ms_or_status=8.0) is not realized` | 66 | the Gen2 builder's first engine-API refusal (spec section 6) |
| `ValueError: connectivity(within_area='sparse') is not realized` | 1 | the same class of refusal on the C-min column path |
| `AssertionError: jaxfne 0.5.7 != execution authority 0.4.25` | 1 | the withheld authority promotion (spec section 9, item 1) |
| `AssertionError: sealed jomission/tfne/realize.py: working tree differs from sealed blob` | 1 | the candidate modifies 10 sealed paths, which only the sealing step can re-register |

AC5 as declared ("tier 0, the relevant tier-1 subsystems and tier 2 pass, except failures
attributable to the withheld authority promotion or to the enumerated Gen2 obstruction; each
such failure is named") therefore **is not met**: the seal-guard failure is a third
attributable cause that AC5 did not name. The specification was not rewritten after the
runs; widening AC5 is a review decision, not an agent edit.

## The ten sealed paths the candidate modifies

An engine migration is a sealing event by construction: the whole-system arms and the TFNE
realizer are sealed artifacts and must construct under the new engine.

```
jomission/tfne/realize.py
scripts/whole_system_authority_arm.py
scripts/whole_system_authority_arm_v2.py
scripts/whole_system_causal_arm.py
scripts/whole_system_diagnostic.py
scripts/whole_system_diagnostic_3.py
scripts/whole_system_ei_interaction_arm.py
scripts/whole_system_ei_regime_arm.py
scripts/whole_system_isolated_arm.py
scripts/whole_system_oscillator_arm.py
```

(`scripts/whole_system_ei_native_state_arm.py` is not sealed; its spec and result are.)
Promotion must re-register these paths in `manifests/sealed_artifacts.json` with the new
blobs and the sealing commit. Until then the tier-0 seal guard fails by construction.

## Scope A evidence (the sealed-lineage substrate)

The SCI-EI-NATIVE-STATE-1 construction under 0.4.25 and under the candidate engine, same
code, no simulation:

| invariant | 0.4.25 | 0.5.7 | equal |
|---|---|---|---|
| neurons | 2224 | 2224 | yes |
| edges before / after `enforce` | 1 294 396 / 246 844 | 1 294 396 / 246 844 | yes |
| weight sha256 after `enforce` | `8fccec054edeee9dce09166c622c77862e457acdf7fb2df393060fbf3d0df813` | same | yes |
| weight min / max / mean / std | -0.007421537768095732 / 0.5 / 0.01649426536853056 / 0.08329818194522277 | same | yes |
| projection identity set | 26 identities (24 cortical, 2 retinal afferents) | same | yes |
| tau / receptor / delay values | {2.0, 5.0} / {1, 2} / {0} | same | yes |
| intra-retinal edges removed, retinal tonic after | 1 047 552, 0.0 | same | yes |
| config hash | `129dcb4e8f1ad88e` | `78708ca2c3214dd9` | **no (metadata drift)** |

The 246 844 post-`enforce` edge count equals the synapse count recorded in the sealed
`results/whole_system_isolated_pop.json`.

## Scope B status (Gen2 builder under 0.5.5+)

Four refusals were measured and each candidate adaptation was shown graph-neutral against
the 0.4.25 baseline (10 666 edges, weight sha256 `038b5bf1b577f25b…`, delays {20: 10 076,
80: 287, 120: 303}); none is applied in this candidate, because each changes the Gen2 config
hash and the pinned Gen2 identities live in tier-1/2 tests and the Q8 lineage. The fifth
refusal, `field(source_mode=...)`, is probe semantics (what the LFP/CSD proxies are) and
needs a simulation-based comparison and a scientific decision. Details: specification
section 6.

## Artifacts left by the candidate

- `docs/migrations/jaxfne_0_5_7_migration_spec.md` (committed before the runs)
- `scripts/jaxfne_construct_ab.py` (cross-engine construct invariants; `--compare` diffs two runs)
- `tests/test_migration_jaxfne_057.py` (construct-only regression test, registered in tier 0)
- `jomission/tfne/realize.py`: `declare_retinal_interface` and the ten arm call sites
- This receipt

Nothing else in the repository changed: `git status --porcelain results/` is empty, no
sealed result was rewritten, no manifest authority was touched, `next_authorized_task` is
still null, `V2_LOCAL_OPERATION` is still FAIL and the SCI-EI-NATIVE-STATE-1 verdict is still
`MIXED_STATE_MECHANISM` / qualification FAIL.

# JaxFNE 0.5.7 migration — candidate specification

**Status:** candidate. Execution authority remains `jaxfne 0.4.25` (`manifests/current_state.json`).
This document proposes; it does not promote. No scientific execution, no result file, no
sealed artifact and no manifest authority is changed by the candidate branch.

**Owner order (2026-10-23):** 0.4.25 is deprecated; move to the latest JaxFNE. Latest on
PyPI at that date is `jaxfne 0.5.7`.

---

## 1. Engine provenance

| item | value |
|---|---|
| package | `jaxfne` 0.5.7 (PyPI) |
| wheel | `jaxfne-0.5.7-py3-none-any.whl` (965 784 bytes) |
| wheel sha256 | `3f295317221bba4ab0b05d33a951813eeea4d5400ae640d29164bc744b89c1c8` (matches the PyPI digest) |
| url | `https://files.pythonhosted.org/packages/8b/b2/4fb21da813266086e839eb567f146f8f484847cd2fad361623cb153a3307/jaxfne-0.5.7-py3-none-any.whl` |
| install | `python -m pip install --no-deps --no-index <wheel>` into `C:\Python314` (the primary clone's interpreter) |
| interpreter | `C:\Python314\python.exe`, CPython 3.14.3, pip 25.3 |
| dependency constraints | 0.5.7 requires `jax>=0.4.25,<0.12`, `jaxlib>=0.4.25,<0.12`, `numpy>=1.24`, `scipy>=1.10`; installed `jax 0.10.1`, `jaxlib 0.10.1`, `numpy 2.4.6`, `scipy 1.18.0` satisfy them and were not touched |
| index reachability | pip's index is unreachable from this machine (DNS failure); wheels were fetched with `curl` from `pypi.org` / `files.pythonhosted.org` and installed from local files |

### Rollback wheels (fetched, hash-verified, not installed)

| version | sha256 |
|---|---|
| 0.5.7 | `3f295317221bba4ab0b05d33a951813eeea4d5400ae640d29164bc744b89c1c8` |
| 0.5.6 | `5f2f6caa89c0a9fc84bdad8d837679a7318ded1f9adc5eb62f0e50cbf9785af9` |
| 0.5.5 | `496a8a310938334dda490e364bd5c3e945e7bb58ce6cb9d2d893b803ebfdf2ce` |
| 0.5.0 | `1a6cbe75b08f8855421f2792d8557bd39631db95ba75e3f09a0667d9b8ac2233` |
| 0.4.25 | `035103bc81b2aebeb22b7cd73e45e59bba301059ddd0ee8c9d519e42fa3c4c10` |

Rollback of the interpreter is one command: `python -m pip install --no-deps --no-index
<jaxfne-0.4.25 wheel>`. Rollback of the editable pointer is
`python -m pip install --no-deps -e E:\repos\jomission` (the pre-candidate state pointed
there); the candidate points at `E:\jomission`.

---

## 2. Measured engine compatibility matrix

Built with the candidate code (`realize.declare_retinal_interface`), one process per
engine, in throwaway venvs that pin one JaxFNE version each
(`E:\wt\venv-jaxfne-<version>`, `--system-site-packages`).

| engine | TFNE without area declaration | TFNE with area declaration | Gen2 builder |
|---|---|---|---|
| 0.4.25 (sealed authority) | constructs | constructs | constructs |
| 0.5.0 | constructs | constructs | constructs |
| 0.5.5 | **refused** (area validation) | constructs | **refused** (`delay_ms_or_status`) |
| 0.5.6 | **refused** | constructs | **refused** |
| 0.5.7 | **refused** | constructs | **refused** |

Under 0.5.0 the whole repository builds with no code change: the TFNE config hash
(`76779213f71a1cc7`) and the Gen2 config hash (`4574a15519ee8c33`) equal the 0.4.25 values.

---

## 3. The Retina area: decision and rejected alternatives

**Intended semantics.** `architecture.py` declares the retina as a structural input
interface ("no retinal population is constructed in this lineage"); the executed arms build
it as a column of 1024 units so its units can be addressed by id, then `enforce()` removes
the intra-retinal edges and zeroes the retinal tonic. The six cortical areas
(V1_1, V1_2, V4_1, V4_2, FEF, PFC) are the cortical contract and are untouched by any
version of the engine.

**What changed in the engine.** JaxFNE >= 0.5.6 validates at construct that the declared
areas are exactly the built columns (`jaxfne/_construct_core.py:571-577`). The arms built a
`Retina` column without declaring its area, so construct now raises. The check states the
engine's contract: an area is what construct builds.

**Decision (D1, applied).** `realize.declare_retinal_interface` adds the retinal column and
declares its area in the engine's metadata. `Configuration.areas` writes metadata only, so
the graph is unchanged.

*Consequence:* the engine's area set becomes six cortical names plus the retinal input
interface; the retinal interface holds no slot in the cortical hierarchy (the
feedforward/feedback wiring is fixed when the scaffold is built) and is not one of the six
cortical areas; the config hash changes (metadata participates), which is an identity
re-baseline item, not a graph change.

**Rejected alternatives.**

| design | consequence | why rejected |
|---|---|---|
| D2 — do not build the retinal column; realize the retinal drive as schedule or emitter tonic | the input interface leaves the graph: no retinal units, no AMPA afferent edges with per-unit weights and receptive fields. That is a change to the omission paradigm's input, not a compatibility fix | out of scope for a migration; a scientific decision |
| D3 — build the retina as a `population()` instead of a `column()` | the units would leave `column_names` and satisfy the check without an area declaration, but the arms address retinal units through the column's neuron-index space and the engine's source-field addressing expects a column area. Unproven id resolution, for the sole purpose of avoiding one metadata declaration | unproven path, larger change than D1 |
| D4 — relax or monkeypatch the engine's validation | masks a genuine contract violation and makes the candidate non-portable to the next engine | the engine is not ours |

---

## 4. Stash inventory (`stash@{0}`, base `e9c52fc`, kept untouched)

Nothing from the stash is applied. Each delta was compared against `5b2c861` and classified.

| stash delta | class | disposition |
|---|---|---|
| `state_replacement.py`: `DynamicState` 9-field handling (`_fields`, `_replace`), public imports | interface adaptation | **already in HEAD** (HD-STATE-REPLACEMENT-0424, 2026-09-19): HEAD has `DYNAMIC_FIELDS = tuple(DynamicState._fields)` and `_replace`. `jaxfne._pipeline` still exists in 0.5.7 and `run_continuation` is exported publicly, so the import change is optional |
| `network/builder.py`: `_clone_edge_list` preserving compact storage at six rebuild sites | interface adaptation | **superseded**: HEAD materializes edge storage once (`_materialize_edge_storage`, v0.4.22+ migration) before the field-wise seams, so the seams work on per-edge arrays |
| `w4_dashboard.py`, `scripts/b1b2b3_intrinsic.py`, `scripts/w4a_v1_to_v4.py`, `scripts/w4a_vertical_representation.py`: storage preservation in control clones | interface adaptation | **unnecessary**: those scripts clone the edge list of `build_jomission_model`, whose storage is already materialized |
| `manifests/environment.json`, `README.md`, `pyproject.toml` floor, `manifests/rf_rate_factorial_design.json` | provenance / authority records | **not applied**: authority promotion (phase 5), not a candidate change |
| `FROZEN_CONFIG_HASH` 4f9fdeae -> 4574a155, labelled "0.5.0 re-baseline" | identity | **attribution wrong**: 4574a15519ee8c33 is the Gen2 config hash measured under **0.4.25** at HEAD, and the tests already expect it (`tests/test_rf.py`, `tests/test_area_local_field.py`). HEAD's `FROZEN_CONFIG_HASH = 4f9fdeae7428199a` in `state_replacement.py` / `ablations` is stale for a different reason. Not applied: an identity re-baseline needs review |
| `tests/test_hdp_integration.py`: HDP `w` bound -0.1 -> -0.5 | numerics-affecting expectation | **not applied**: under 0.5.7 the unchanged -0.1 bound passes. Applying it would weaken a gate without evidence |
| `tests/test_q8_state_replacement.py`: delay_state truth, 9-field set | test expectation | **not applied**: the file is tier 3 and is not run in this candidate; no evidence under 0.5.7 |
| `jomission/dynamics/hdp.py` GOVERNOR_V1 + `scripts/calibrate_hdp_governor.py`, `scripts/grow_diff_demo.py`, `scripts/waveform_replay.py`, `jomission/growth/*` | new science | **not applied**: unrelated to compatibility |
| untracked `manifests/jaxfne_0_5_0_rebaseline_receipt.json` | 0.5.0 re-baseline receipt, claims "human decision 2026-09-25" | **read, not applied**: it is the only record of the earlier 0.5.0 decision; its config-hash attribution is wrong as recorded above |

---

## 5. Numerical regression plan and result

**Plan.** Build the sealed SCI-EI-NATIVE-STATE-1 construction (the arm's own `build()`, then
its `enforce()`) under JaxFNE 0.4.25 and under the candidate engine, and compare the built
graph. No simulation: construction only. Script: `scripts/jaxfne_construct_ab.py`
(`--out` to write, `--compare a.json b.json` to diff).

**Result (0.4.25 vs 0.5.7, candidate code).** Every invariant is equal except the config
hash:

| invariant | 0.4.25 | 0.5.7 | equal |
|---|---|---|---|
| neurons | 2224 | 2224 | yes |
| edges before `enforce` | 1 294 396 | 1 294 396 | yes |
| edges after `enforce` | 246 844 | 246 844 | yes |
| weight sha256 (after `enforce`) | `8fccec054edeee9dce09166c622c77862e457acdf7fb2df393060fbf3d0df813` | same | yes |
| weight min / max / mean / std | -0.007421537768095732 / 0.5 / 0.01649426536853056 / 0.08329818194522277 | same | yes |
| projection identity set (21 entries with counts) | — | — | yes |
| tau values, receptor indices, delay steps | {2.0, 5.0} / {1, 2} / {0} | same | yes |
| column names | 6 cortical + Retina | same | yes |
| intra-retinal edges removed / retinal tonic after | 1 047 552 / 0.0 | same | yes |
| **config hash** | `129dcb4e8f1ad88e` | `78708ca2c3214dd9` | **no (metadata drift)** |

The 246 844 post-`enforce` edge count equals the synapse count recorded in the sealed
`results/whole_system_isolated_pop.json`, so the candidate reproduces the sealed
construction's size under the new engine.

---

## 6. The Gen2 path under 0.5.5+ — measured, not adapted

`jomission/network/builder.py` (the retired V2.1 chain's substrate) declares engine
parameters that 0.5.5+ no longer realizes. Each was measured against the 0.4.25 baseline
(10 666 edges, weight sha256 `038b5bf1b577f25b343244e36cfdf036e00e96c5f50b0abfb14ad1ba7be4afde`,
delays {20: 10 076, 80: 287, 120: 303}).

| refusal | adaptation tested | effect on the realized graph under 0.4.25 |
|---|---|---|
| `inter_column_connectivity(delay_ms_or_status=8.0)` refused | drop the argument (the delays are applied post-construct by the `_apply_laminar_delays` seam, FF 80 / FB 120 steps) | none: identical edges, weights, delays; config hash 4574a15519ee8c33 -> 76a6cc4c95f3359f |
| `inter_column_connectivity(cell_type_to_cell_type_map=...)` refused | drop the argument | none (the argument was a label; it never entered the hashed metadata) |
| `connectivity(spatial_sigma=..., max_in_degree=...)` refused | move the locality parameters into this module's own metadata and let the `_apply_spatial_locality` seam apply them | none: identical edges, weights, delays |
| `connectivity(within_area='spatial_local')` refused | use the only realized value, `all_to_all_uniform_random` | none: identical edges, weights, delays (`spatial_local` was a label under 0.4.25 as well) |
| `field(source_mode=...)` has no consumer | **not tested** | **probe semantics, not graph**: `source_mode` selects how the source field is computed, i.e. what the LFP/CSD proxies are. Equivalence here needs a simulation-based comparison of probe outputs, which is outside a compatibility candidate and is a scientific decision |

**Why the candidate does not carry these edits.** Four of them are graph-neutral but each
changes the Gen2 config hash, and the pinned Gen2 identities live in tier-1/2 tests
(`tests/test_rf.py`, `tests/test_area_local_field.py`: 4574a15519ee8c33) and in the Q8
lineage's frozen hash. Applying them before the identity re-baseline is reviewed would
break those pins for no gain, because the path still cannot construct under 0.5.7 (the
`field(source_mode=...)` refusal). The route is therefore one reviewed change: adapt the
four graph-neutral parameters, re-baseline the Gen2 identity, and settle `source_mode`
against probe outputs.

---

## 7. Test tiers to run

`manifests/test_tiers.json` maps "JaxFNE upgrade or environment change" to `0 + 1 + 2`, and
tier 3 (scientific execution) only inside an authorizing TODO. This candidate runs tier 0,
the tier-1 subsystems that touch the changed surface, and tier 2. It runs no tier-3 test and
no simulation.

---

## 8. Acceptance criteria (declared before the runs)

- **AC1** The sealed arm construction succeeds under 0.5.7 with the retinal area declared,
  and fails without it with the engine's area validation (the defect is reproducible).
- **AC2** The built TFNE graph is invariant to the engine version: the table in section 5
  holds for 0.4.25 vs 0.5.7, including the weight sha256 and the projection identity set.
- **AC3** The six cortical areas are unchanged; the retinal interface stays non-cortical:
  no hierarchy slot, no intra-retinal edge after `enforce`, zero retinal tonic after
  `enforce`.
- **AC4** The Gen2 obstructions under 0.5.5+ are enumerated, each graph-neutral adaptation
  measured against 0.4.25, and the remaining one named as a probe-semantics decision.
- **AC5** Tier 0, the relevant tier-1 subsystems and tier 2 pass, except failures that are
  attributable to the withheld authority promotion (`jaxfne identity`) or to the enumerated
  Gen2 obstruction; each such failure is named.
- **AC6** Nothing scientific ran: no simulation, no `results/*.json` written, no sealed
  artifact modified, no manifest authority changed, `next_authorized_task` still null.

---

## 9. Proposed authority delta (withheld until review)

1. `manifests/current_state.json`: `jaxfne_version` 0.4.25 -> 0.5.7; `execution_authority.evidence`
   -> a new sealed migration record; `scientific_head` unchanged.
2. `pyproject.toml`: `jaxfne==0.4.25` -> `jaxfne==0.5.7`.
3. A sealed `results/jaxfne_0_5_7_migration.json` recording: wheel provenance, the matrix in
   section 2, the retinal design, the A/B table in section 5, the Gen2 findings, the tier
   results, and the rollback wheels.
4. A `manifests/todo.json` item `JAXFNE-0-5-7-MIGRATION` (status DONE) and the
   `manifests/lineage_registry.json` edge.
5. Documentation: `AGENTS.md` line 17 and `docs/HANDOFF_CURRENT.md` line 139 name
   `E:\repos\jomission` as the working clone; the owner's designation is `E:\jomission`.
   `manifests/harness/environment_failures.json` `ENV-STALE-EDITABLE` still describes the
   stale clone as `E:/jomission` and the repair as installing from `E:/repos/jomission`,
   which is now inverted.
6. The Gen2 identity re-baseline (4574a15519ee8c33 -> the post-adaptation hash) and the
   `field(source_mode=...)` decision — both out of this candidate's scope.

---

## 10. Rollback

| level | action |
|---|---|
| engine | `python -m pip install --no-deps --no-index <0.4.25 wheel>`; the interpreter returns to the sealed authority |
| repository | `git branch -D migration/jaxfne-0-5-7` (the candidate is one branch off `main`; `main` is untouched at `5b2c861`) |
| editable pointer | `python -m pip install --no-deps -e E:\repos\jomission` restores the pre-candidate pointer |
| recovery work | `stash@{0}` (`66d578e38f756316e0c9f94cf7a7503ac4049685`) and `recovery/atlas-suite` (`e9c52fc`) are untouched throughout and remain the rollback evidence for the atlas work |

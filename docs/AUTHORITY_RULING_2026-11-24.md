# Authority ruling 2026-11-24: JaxFNE minimum version

Status: ruling recorded. Additive reconciliation only. Historical records, sealed artifacts and results are unchanged.

## 1. Ruling (verbatim)

> Ruling: JaxFNE >=0.5.8 is the mandatory minimum version, not an exact-version constraint. Rationale: versions before 0.5.8 were unreliable in core functionality; 0.5.8 was the first sufficiently trusted baseline. The requirement sets a minimum reliability floor, not an obligation to execute exclusively on 0.5.8. Application: JaxFNE 0.5.8 = minimum admissible version. JaxFNE 0.5.9 = admissible for E0 and E1b under their recorded seals. Versions below 0.5.8 = inadmissible. Exact-version reproducibility = still required for each run. Scientific acceptance = requires all other experiment-specific gates. Reconciliation: update the authority documents to distinguish the global minimum (>=0.5.8) from experiment-specific version pins; preserve historical records and the original seals. Do not read the ruling as proof that 0.5.9 results reproduce 0.5.8 results; each result must retain its exact dependency version and provenance.

## 2. Admissibility

| JaxFNE version | Category | Scope |
|---|---|---|
| below 0.5.8 | INADMISSIBLE | any run, any family |
| 0.5.8 | ADMISSIBLE (global minimum) | any run; exact version recorded and required per run |
| 0.5.9 | ADMISSIBLE for E0 under docs/laminar_motif/E0_SEAL.md (Amendment 1) and for E1b (families e1bA_*, e1bB_*) under docs/laminar_motif/E1B_SEAL.md | those families only |
| 0.5.9 on any other family, or any version above 0.5.9 | NOT COVERED | no admissibility from this ruling; needs a separate owner ruling |

Each category above is a version floor or family scope only. Exact-version reproducibility is still required per run. Scientific acceptance still needs every experiment-specific gate. This ruling grants neither.

## 3. Reconciliation rule

- The global minimum is `>=0.5.8`. It is recorded in `manifests/current_state.json` as `jaxfne_minimum` = `"0.5.8"`, with `jaxfne_minimum_ruling` pointing to this file.
- Experiment-specific pins are recorded at the experiment's own seal. Where a seal names a version, that version governs that experiment only.
- `jaxfne_version` in `manifests/current_state.json` stays `"0.5.8"`. It is the execution pin read by `scripts/project_check.py` and `tests/test_environment_provenance.py`. It is not changed by this ruling.
- Each run keeps its exact dependency version and provenance (version field, source commit or tag where recorded). A result from one version is not evidence for another version's result.
- Sealed files and results are not edited. Corrections are additive records.

## 4. Matrix

Per-run admissibility under this rule is tabulated in `docs/laminar_motif/PROVENANCE_MATRIX_2026-11-24.md` (human-readable) and `docs/laminar_motif/PROVENANCE_MATRIX_2026-11-24.json` (machine-readable). The matrix classifies runs by version field and governing seal. Its counts describe the state on 2026-11-24 only.

## 5. Not changed here (owner decision required)

The following still enforce the exact pin 0.5.8. They are not changed by this ruling and need an owner decision before they become `>=`:

| Location | Current behaviour |
|---|---|
| `scripts/project_check.py` line 73-75 | `got == want`, installed jaxfne must equal `jaxfne_version` (0.5.8) |
| `tests/test_environment_provenance.py` line 34 | reads `jaxfne_version` and checks equality |
| `pyproject.toml` line 18 | `"jaxfne==0.5.8"` (exact pin) |

## 6. Open items linked to this ruling

- OPEN-37 and OPEN-41 remain open. This ruling does not close them.
- OPEN.md blocker row 8 (JaxFNE version conflict) records the minimum ruling. Admissibility for runs outside E0 and E1b remains open.

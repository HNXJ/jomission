# Authority ruling 2026-11-25: JaxFNE 0.5.9 admissibility for T1, O38 and E1

Status: ruling recorded. Additive to `docs/AUTHORITY_RULING_2026-11-24.md`; that file, sealed artifacts and results are
unchanged.

## 1. Owner word (verbatim) and what it answers

Hamm, 2026-11-25, in session chat: "proceed with your recommendations".

It answers the question put to him in the same session: "Should T1, O38 and E1 records on 0.5.9 stand, be rerun on
0.5.8, or get a new ruling?", whose recommended answer was a new ruling admitting them. The finding behind the question
is `docs/laminar_motif/HANDOFF_LANSA_GEOMETRY_2026-11-25.md` section 4 item 10 (independently reviewed, ACCEPT, at
ee61a8d; wording fixes at e52b950).

## 2. Admissibility added

The 2026-11-24 table row "0.5.9 on any other family ... NOT COVERED" is narrowed by these families:

| Family | Records (per-run `jaxfne_version` = 0.5.9) | Governing record |
|---|---|---|
| T1 axis 3 (sstd) search | 18: `results/laminar_e2/t1_sstd{7p5,10p6,21,30,42,60}_n4000_s{1,2,3}_T8.json`; the `t1_sstd15_*` reference links resolve to 0.5.8 records and are not in this row | `docs/laminar_motif/SEAL_T1.md` |
| T1 confirmation (the T1 FAIL) | 3: `t1_confirm_kd1p41_sst15_sstd21_n4000_s{4,5,6}_T12` | `docs/laminar_motif/SEAL_T1.md` |
| T1 reference | 7: `sw_n4000_dens_k1_kd1_sst15_*` | `docs/laminar_motif/SEAL_T1.md` |
| O38 | 47 run records `o38s_*` on `origin/dev` 264c1e8 | the O38 analysis accepted 2026-11-17 |
| E1 (rho sweep) | 15: `results/laminar_e1/e1_rho*_seed*.json` | `docs/laminar_motif/E1_SEAL.md` |

Counts were read on 2026-11-25 from cowork HEAD 21b2697 (T1, `sw_*`, E1), `origin/dev` 264c1e8 (O38) and
`/mnt/d/jomission` ef0a766 (E1). They describe that state only.

## 3. Conditions (unchanged from 2026-11-24)

- Admission is a version floor and family scope only. It grants no scientific acceptance and changes no verdict: the T1
  FAIL stays FAIL.
- Each run keeps its exact version and provenance. A 0.5.9 result is not evidence for a 0.5.8 result, or the reverse.
- The T1 axis-1 reference level mixes versions: seed 1 resolves to `sw_n4000_dens_k1_kd1_sst15_s1_T8` (0.5.9), seeds 2
  and 3 to `lfs_n4000_s{2,3}_T8` (0.5.8). Both stand. Any statistic pooled over that level must state the mix.
- Versions above 0.5.9, and 0.5.9 on any family not listed here or in the 2026-11-24 ruling, stay NOT COVERED.
- T8 records were not checked and are not covered by this ruling.

## 4. Not changed

- The execution pin: `manifests/current_state.json` `jaxfne_version` = `"0.5.8"`, `scripts/project_check.py` lines
  73-75, `tests/test_environment_provenance.py` line 34, `pyproject.toml` line 18. The gate environment was fixed
  instead (owner's same word, option A'): venv `/home/hamm/.venvs/jomission058` with jaxfne 0.5.8 built from tag
  `v0.5.8` = 11957afa; receipt `docs/receipts/gate_jomission058_2026-11-25.md`.
- OPEN-37 and OPEN-41 stay open; this ruling does not close them.
- Manifests, seals and the provenance matrix are not edited here. Registering this ruling in
  `manifests/current_state.json` and `docs/OPEN.md` is a separate step.

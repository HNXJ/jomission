# OPEN-51 closure: registration and exemption scope (2026-11-19)

Ruling (Hamm 2026-11-19): register E0/E1 with their provenance, keep the historical visualization exemption,
require V0-V3 for future runs, and scope historical exemptions explicitly. Scientific conclusions stay frozen:
E1 INCONCLUSIVE, T1 FAIL, mechanism unresolved. No simulation was run and no criterion changed.

Reconstruction result and its independent verification: `OPEN51_A1_RESULT_2026-11-19.md`. That file is pinned in
the manuscript's source hashes (record D6), so it is unchanged here.

## Changes

- `manifests/lineage_registry.json`: SCI-LAMINAR-MOTIF-1 and SCI-LAMINAR-E1 each gain a `reproduction` block
  (status, protocol, original execution, seeds, jax dependence, outputs, verification, reconstructed figures). The
  `visualization` reason now states the historical exemption; the original reason is kept as `reason_original`.
  Commits, observations, verdicts and artifacts are unchanged.
- `manifests/visualization_contract.json`: `exemption.premise_check`, `exemption.premise_check_limit`, and
  `historical_exemptions`, which lists exactly these two lineages, each at its own commit (ea249f1, 8200fac), with
  ruling, scope and evidence.
- `jomission/harness/validate.py`: an exemption on an executed lineage passes only if `driver_provenance.path` is
  `none` or starts with `none:`, or if the lineage is listed at its own commit. A rerun at another commit, a child
  lineage, or a lineage that runs, omits or malforms a driver declaration fails. Limit: a driver that ran but is
  declared `none:` is not detectable by the validator.
- `tests/test_visualization_contract.py`: premise, scope and registry-exactness tests, run red against the old
  validator first.

## Verification

An independent reviewer checked the registry diff, the provenance facts against their sources, the contract
entries, and attempts to defeat the validator (child lineage, rerun commit, omitted or empty driver). Verdict
ACCEPT-WITH-CHANGES. Its points were all fixed:
- the D6 count was wrong because the pinned note had been edited; the note is restored and these sections moved here;
- the `none` prefix match was too loose; it is now `none` or `none:`;
- a non-object `driver_provenance` crashed the validator; it is now guarded;
- the honor-system limit is now documented.

## Manuscript source hashes

`paper/jomission_lansa/data/sources_sha256.txt` (record D6), checked in this clone on 2026-11-19: 49 PASS, 0
mismatch, 122 UNKNOWN. The 122 are run records outside version control and absent from this clone. They are
neither checked nor passed.

# Review commands

Run from the repository root. `SPEC` and `RESULT` are the lineage record's
`preexecution_spec.commit` and `commit`; `BR` is the lineage branch.

Record validity (schema, commit order, receipts, evidence classes):

```bash
python -c "from jomission.harness import validate as v; import json,sys; r=json.load(open(sys.argv[1])); print(v.validate_lineage(r) or 'clear')" record.json
```

Seal precedes result, and nothing moved in between:

```bash
git merge-base --is-ancestor SPEC RESULT && echo ordered
git log --oneline SPEC..RESULT -- <spec path> <parameter files>
```

The second command must print nothing.

Gate definition used equals the definition at seal time:

```bash
git rev-parse SPEC:manifests/gates/v2_local_operation.json
```

Compare with the `gate_blob` stored in the result JSON.

Branch touches only its own paths:

```bash
git diff --stat main...BR
```

Sealed files unchanged on the branch:

```bash
python -m pytest -q -p no:cacheprovider tests/test_harness_guards.py -k sealed
```

Verdict and status agree:

```bash
python -c "import json; print(json.load(open('results/<lineage>_lineage.json'))['verdict'])"
python -m pytest -q -p no:cacheprovider tests/test_harness_state.py
```

Superseded or quarantined evidence: search `manifests/vocabulary.json`,
`results/*correction*.json`, and `manifests/*quarantine*.json` for each cited
artifact name.

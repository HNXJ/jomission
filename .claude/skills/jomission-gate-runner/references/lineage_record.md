# Lineage record fields (`manifests/lineage_registry.json`)

Validated by `jomission.harness.validate.validate_lineage`.

| field | content |
|---|---|
| `lineage_id` | short id, unique |
| `parent` | parent lineage id or branch point |
| `principal_delta` | the one change |
| `introduces_mechanism` | true when the delta adds a mechanism; then `qualification` is required |
| `preexecution_spec` | `{"path", "commit"}`; commit strictly precedes `commit`; `null` only for unexecuted records |
| `authority` | document or reviewer order that authorized the edge |
| `inputs` | files read |
| `frozen` | everything held fixed |
| `executed` | bool |
| `qualification` | `{configured, realized, executed, effective}` evidence strings |
| `observations` | `[{claim, evidence_class: OBSERVED, receipt, kind?, estimator?}]`; `kind: transient` requires estimator bin_width_ms, aggregation, baseline, sign |
| `derived` | `[{claim, evidence_class: DERIVED, ...}]` |
| `inferred` | `[{claim, evidence_class: INFERRED}]` |
| `verdict` | `{label, status}`; status from `manifests/vocabulary.json` `status_labels` |
| `acceptance` | gate profile or criteria reference |
| `fail_boundary` | where the failure sits |
| `supersedes`, `retired` | ids |
| `artifacts` | result files present at `commit` |
| `tests` | tests that produced or check the result |
| `commit` | result commit |

Evidence classes (`OBSERVED`, `DERIVED`, `INFERRED`, `ASSUMED`, `UNKNOWN`)
describe how a claim is known. Status labels (`PASS`, `FAIL`, `PARTIAL`,
`UNRESOLVED`, ...) describe a gate outcome. Do not mix the two lists.

Skeleton:

```json
{"lineage_id": "", "parent": "", "principal_delta": "", "introduces_mechanism": false,
 "preexecution_spec": {"path": "", "commit": ""}, "authority": "", "inputs": [], "frozen": [],
 "executed": true, "observations": [], "derived": [], "inferred": [],
 "verdict": {"label": "", "status": ""}, "acceptance": "", "fail_boundary": "",
 "supersedes": [], "retired": [], "artifacts": [], "tests": [], "commit": ""}
```

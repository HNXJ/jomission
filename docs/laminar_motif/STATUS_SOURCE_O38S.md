# OPEN-38 status source for the Jomission/LANSA dashboard card and the HNXJ Labyrinth card (owner 2026-11-17)

Ownership. The scientific status source belongs to main-jomission-lansa: this generator and the receipts it reads.
Dashboard and Labyrinth integration belongs to main-vwin or the HNXJ website owner. An integrator runs the generator
and renders its JSON. It never edits receipts, receipt docs or this generator, and the scientific lane never edits
the dashboard.

Command (from a jomission checkout; any Python ≥ 3.10 with the standard library):

    python -B scripts/open38_status.py OUT_JSON [--repo DIR] [--run-tests] [--stale-min 30]

`--run-tests` runs `tests/test_open38_*.py` with that interpreter's pytest. Those tests import jaxfne, so set
`PYTHONPATH` to a verified 351f6048 tree (the export `D:/repos/_work/_temp_jaxfne_351f6048`, or the venv
`D:/venvs/o38_jaxfne_351f6048`). Without `--run-tests` the test block is `UNKNOWN` and gives the static count.

Refresh: on every push to a branch that carries OPEN-38 receipts, and after each completed CI run. Show `source.sha`
and `generated_utc` on the card.

## Status semantics

| Status | Meaning |
|---|---|
| PASS | the stage driver logged DONE and every checker receipt it names reports `"pass": true` |
| FAIL | a checker receipt reports fail, the construction gate failed, or a run exited non-zero |
| RUNNING | a driver segment started, has no DONE or STOP, and its log changed within `--stale-min` minutes |
| BLOCKED | the driver logged STOP, or a phase passed and the next is not launched (`reason` says which) |
| UNKNOWN | a needed receipt is missing or unreadable, a running log went stale, or tests were not run |

A launched simulation is never PASS. A run counts only with its `TAG rc=0` log line, and a stage only with DONE plus
its checker receipts. Owner acceptance and independent review live in the receipt docs (linked under `evidence`);
they are not inferred here.

## Fields (schema `o38s_status/1`)

| Field | Card use |
|---|---|
| `source.sha`, `source.branch`, `source.dirty_tracked`, `generated_utc` | provenance line |
| `project.name`, `project.objective` | Labyrinth card header |
| `current_stage` | dashboard headline |
| `stage1.status`, `stage1.checks`, `stage1.runs_exit0` / `runs_expected` | stage 1 row |
| `stage2.status`, `stage2.reason`, `stage2.phases[]` (mode, status, started, exit0, construction, integrity, last_line_utc), `stage2.runs_exit0` / `runs_expected` | stage 2 row and running attempts |
| `stage2.construction_gate.verdict`, `.family` (alpha, n_tests, per-test level, ruling) | construction-gate verdict |
| `stage2.attempts[]` | earlier stopped attempts |
| `tests.status`, `.passed`, `.failed`, `.defined` | test counts |
| `blockers[]` | blockers |
| `latest_verified_milestone` | Labyrinth "latest verified milestone" |
| `evidence.{spec, stage1_receipt, stage2_receipt, gate_ruling}.path` and `.sha256` | links: `{repo_url}/blob/{sha}/{path}` |

Limitation: a running stage's log stays untracked until the stage closes, so the generator shows RUNNING only on the
machine that runs it. A refresh from a clone shows the state at the last commit.

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

Cross-commit consistency (owner 2026-11-17): a PASS never combines receipts from different source revisions. The
driver-log HEAD, each run's `git_sha` and each integrity receipt's `head` must be one SHA. A receipt written at
another revision counts only with an entry in `results/laminar_e2/o38s_receipt_compat.json` that names the paths
that must be identical to the execution SHA, and `git diff` must confirm them. Identical code is not experimental
equivalence (owner 2026-11-17), so the entry must also name the execution-SHA receipt it re-checks
(`equivalent_to`), and the two must be equal field by field except the declared `ignore_keys`: the same layouts,
seeds, sizes, edge counts, parameters and drive. The comparison is exhaustive and fails closed (owner 2026-11-17):
any extra, missing or changed field outside the declared keys, a malformed receipt, or a declared key found outside
its enumerated slot makes the stage UNKNOWN. Declared keys are never trusted. Each needs a written justification
(`ignore_justification`) and a re-derivation `rule`, and `ignore_keys` without a rule fails. For
`ruling_b_construction`, the only slots allowed to differ are the top-level `family` and `pass`, each gate's `ok`, and
each PV/SOM rule's `interval` and `ok`. Each is re-derived from the frozen receipt:

- `family`: alpha 0.05, `n_tests` equal to the number of PV/SOM rules, and level 1 − 0.05/n_tests.
- Each interval: its centre within 2 edges of the frozen one, and its width equal to the frozen width × z_B/z_0.999
  within 1 % (stage 1 observed 0.69 % at most, from binomial discreteness).
- Each rule's `ok`: recomputed as count in interval and parameters equal.
- Each other gate's `ok`: must equal the frozen value.
- `pass`: the AND of all gates.

The following were added after the independent review of 2026-11-17:

- `identical_paths` must include `jomission` and `scripts/run_e2_stack.py`, and every listed path must exist at both
  SHAs.
- The frozen receipt is itself recomputed: each `ok` and `pass` from its own counts and intervals. Its pairs must be
  exactly arms × seeds, with one PV/SOM rule set across pairs.
- The number of PV/SOM tests must equal the entry's `expected_n_tests` (stage 1: 260).
- Stage 2 is PASS only as smoke PASS followed by rest PASS. A segment left unfinished before a later one is UNKNOWN.

So a construction that failed under the frozen gate becomes PASS only by lying inside the ruling-B interval. Residual
slack: the tolerances can admit a count at most 1 % of a half-width plus 2 edges outside the exact ruling-B bound
(about 0.04 SD at |z| = 4). Otherwise the stage is UNKNOWN. Environment, configuration, seeds
and dependency versions of the runs themselves are covered by each stage's freeze file, pip freeze, environment
check, export-manifest checks and the integrity receipts.
The stage's `consistency` block lists any problem and each verified compatibility.

Public view (owner 2026-11-17): cards render `public_status`. It equals `status` only when every receipt the stage
reads is committed and unmodified at `source.sha`; otherwise it is UNKNOWN. A machine-local RUNNING is never shown
as repository-verified, and `latest_verified_milestone` uses `public_status` only.

A launched simulation is never PASS. A run counts only with its `TAG rc=0` log line, and a stage only with DONE plus
its checker receipts. Owner acceptance and independent review live in the receipt docs (linked under `evidence`);
they are not inferred here.

## Fields (schema `o38s_status/1`)

| Field | Card use |
|---|---|
| `source.sha`, `source.branch`, `source.dirty_tracked`, `generated_utc` | provenance line |
| `project.name`, `project.objective` | Labyrinth card header |
| `current_stage` | dashboard headline |
| `stage1.public_status` (card), `stage1.status` (local), `stage1.checks`, `stage1.consistency`, `stage1.runs_exit0` / `runs_expected` | stage 1 row |
| `stage2.public_status` (card), `stage2.status` (local), `stage2.reason`, `stage2.consistency`, `stage2.phases[]` (mode, status, started, exit0, construction, integrity, last_line_utc), `stage2.runs_exit0` / `runs_expected` | stage 2 row and running attempts |
| `stage2.construction_gate.verdict`, `.family` (alpha, n_tests, per-test level, ruling) | construction-gate verdict |
| `stage2.attempts[]` | earlier stopped attempts |
| `tests.status`, `.passed`, `.failed`, `.defined` | test counts |
| `blockers[]` | blockers |
| `latest_verified_milestone` | Labyrinth "latest verified milestone" |
| `evidence.{spec, stage1_receipt, stage2_receipt, gate_ruling}.path` and `.sha256` | links: `{repo_url}/blob/{sha}/{path}` |

Limitation: a running stage's log stays untracked until the stage closes, so the generator shows RUNNING only on the
machine that runs it. A refresh from a clone shows the state at the last commit.

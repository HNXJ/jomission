# Open stack

Audit of 2026-11-02 at HEAD 9b3d52a, plus two uncommitted doc edits. Each row cites its source. Owner
decisions are listed, not decided here. Fallback home: `artifacts/` does not exist in this repo.

## Counts (observed, `manifests/todo.json`)

17 items: BLOCKED 4 (MIN-02, MIN-04, SCI-AGSDR-CALIBRATION, MIN-05), OPEN 1 (MIN-07), LOCKED 1
(SCI-PROP-QUALIFIED-1), FAIL 1 (SCI-V21-BACKGROUND, final disposition), DONE 3, RETIRED 8.
`next_authorized_task` is null.

Scoreboard 75/100 (`docs/PLAN_MINIMAL_MODELS.md` §4a). Lost points: S-09 5, S-13 5, CK-A 8, CK-B 7.

## Problems

| id | problem | evidence | fix (owner or main model) |
|---|---|---|---|
| P1 | Tier-0 count has four values: 222 (`PLAN_MINIMAL_MODELS.md:152`, at c7758a9), 196 (`AUDIT_MINIMAL_TREE.md`), 213 (`PLAN_NEXT_GOALS.md:8`, project_check log), 212 (`lineage_registry.json:3353`) | receipts audit; log | re-run tier-0 under the suite lock, record one count and its commit |
| P2 | §4a says later commits change only plan and state files | `git diff --stat c7758a9 HEAD -- jomission harness tests`: 4 files changed (`s13_sweep.py`, `smoke.py`, `validate.py`, `test_harness_state.py`) | owner: correct the §4a sentence |
| P3 | PV→VIP is present and declared absent | `connectivity.json:48` rule present; `:57` "declared absent" | text fix after your PV→VIP answer (kept as a bound) |
| P4 | AGENTS.md names the G: path, which is not mounted | `AGENTS.md:17`; `ls G:` absent | owner answered: E: clone. Not yet applied |
| P5 | MIN-05 BLOCKED while its spec is committed | `todo.json:81`; `docs/minimal/ckb_spec.json` at a59e2f7 | status update to the spec-committed state |
| P6 | "effective VIP drive 14.0" cited to `drive_mean_by_class` is not in the sweep receipt | `AUDIT_MINIMAL_TREE.md:126`; sweep receipt has no such field | cite the derived 4.0 + 10, or the receipt that holds it |
| P7 | S-13 lineage named MIN-06 in the decision tree, MIN-07 in todo | `PLAN_DECISION_TREE.md:22`; `todo.json:99` | rename in the tree |
| P8 | MIN-02 `blocked_by` names owner decision 2 open | `todo.json:34`; `PLAN_MINIMAL_MODELS.md:139` and `HANDOFF_CURRENT.md:118` say ratified 2026-10-30 | status update |
| P9 | Owner decision 1 is open in one place and pending in another | `PLAN_MINIMAL_MODELS.md:124` and `HANDOFF_CURRENT.md:124` vs `PLAN_MINIMAL_MODELS.md:182` | status update |
| P10 | "15 rules, 14 cited" in `todo.json:17` | count not confirmed | UNKNOWN until counted |
| P11 | The organization lane "remove 3 DONE and 8 RETIRED" would delete `BLIND-OMISSION` and `V2.2-RECURRENCE`, which `test_preserved_science_todo` requires (`tests/test_harness_state.py:43-50`) | test read; DERIVED, not run | revise the lane: keep the three named items and those two |
| P12 | S-13 spec does not declare noise or dtype. JaxFNE noise default is on (1.118 per step at dt 0.1) and dtype default is float32 | `jaxfne/emitters.py:103-110`, `:3270` (installed 0.5.8) | declare both in the spec before any run |
| P13 | S-13 window: the criterion counts 0 to 1000 ms; my single-cell probe discards 0 to 200 ms and runs noise-free | `jomission/minimal/smoke.py:68-81`; scratch probe | the probe is not the criterion; do not cite its counts as the S-13 result |
| P14 | PV u0 effect at v0 -60 is an exact equilibrium start, not a u0 lever | agent check: `(v*, u*) = (-60, -12)` at I = 4.0 (DERIVED) | do not cite it as a finding |
| P15 | Installed JaxFNE is 0.5.8; the branch is named for 0.5.7 | `jaxfne/_model.py:304`; branch name | DECIDED 2026-11-02 (Hamm): keep jaxfne==0.5.8; no pin move; the branch name stays until the owner renames it |
| P16 | Uncited dense baseline, about 9,900 edges at N = 100 (DERIVED, not run) | `jaxfne/_construct_population.py:661-670`, `:783` | owner decides whether the builder uses it |
| P17 | MIN-07 freezes the BLOCKED MIN-02 rules; the MIN-02 unlock is empty | `todo.json:112`, `:33` | owner decides the unlock |

## Owner decisions queue

| id | question | state |
|---|---|---|
| D1 | S-13 next step: VIP drive above rheobase 22.56, or wait for a ruling on the jaxfne VIP preset | DECIDED 2026-11-02 (Hamm): new spec with VIP drive above 22.56 on pinned 0.5.8; the u0 = -13 change is withdrawn. VIP effective drive 25.0 (Hamm 2026-11-02) |
| D2 | PV drive margin: PV sits on the knife edge at drive 4.0 (rheobase 4.0) | DECIDED 2026-11-02 (Hamm): PV effective drive 5.0 in the new spec; not yet probed, smoke-test before any run |
| D3 | S-13 criterion: gate spike count and a window (discard first 200 ms) | DECIDED 2026-11-02 (Hamm): window 200-1000 ms; each cell at least 1 spike; amendment, FAIL receipts stay |
| D4 | S-09 restatement on Campagnola | owner answered 2026-11-02; not yet applied |
| D5 | LANSA inputs: tracked or local copies | DECIDED 2026-11-02 (Hamm): keep git-ignored local copies with hash checks (`manifests/local_inputs.json`) |
| D6 | Jarness rules: subagent cap (3 in rules, 8 set this session), `artifacts/` name, precedence, gh state, commit policy | cap DECIDED 2026-11-02 (Hamm): Haiku 5.5 up to 8, other models 3; the jarness repo edit is pending, not made here. `artifacts/` name DECIDED 2026-11-02 (Hamm): keep `manifests/` and `docs/`, named in AGENTS.md. Precedence DECIDED 2026-11-02 (Hamm): split by scope; jarness governs agent conduct, AGENTS.md governs project evidence and state. gh state DECIDED 2026-11-02 (Hamm): gh invalid; SSH only, no gh login |
| D7 | Commit policy: commit at wrap-up, or only on request (`machine-and-accounts.md:9`) | DECIDED 2026-11-02 (Hamm): commit only at wrap-up or on request |
| D8 | CK-A naive arm: no exposure. Trial order: random, seed 0 | owner answered 2026-11-02; schedule still pending |
| D9 | VIP and PV drives: one lineage or two (AGENTS.md: one principal delta per edge) | DECIDED 2026-11-02 (Hamm): two lineages in sequence, VIP first (VIP drive 25.0), then PV (drive 5.0); each spec committed before its run |
| D10 | PV_M0_STATES: sealed results cite .npz states that are gone | DECIDED 2026-11-02 (Hamm): accept as unrecoverable; add an additive correction record; nothing regenerated |

## JaxFNE tools that would help the open items

| tool or API | solves | evidence (installed 0.5.8) | cost |
|---|---|---|---|
| `connectivity_summary()`, `connectivity(p_connect=0.0)` | S-11 edge read-back for MIN-02 | `_model.py:694`; `_construct_population.py:646` | low |
| `mean_pairwise_spike_correlation(bin_ms=20)` | the keeper's pairwise r_sc for the drive test | `analysis/metrics.py:56`, `:73` | low; check against the keeper's probe first |
| `Simulation(ablation="disconnected_null")` | a synchrony control for MIN-07 | `_signals.py:82`; `_model_simulate.py:326` | low; one run |
| `cell_params` (no v0 or u0 field) | SST a for MIN-04 item 1 | `_config.py:991`, `:999` | low |
| `ParadigmEvent(is_omission=True)`, `stimulus_schedule` | the omission schedule for MIN-04 | `paradigm.py:23`; `_model.py:364` | medium; overlaps the repo paradigm |
| `Model.tune(optimizer=AGSDROptimizerSpec)` | SCI-AGSDR (blocked) | `_model_tune.py:160`; `optim/core.py:182` | medium; optax not installed |
| `spectrolaminar_psd` | MIN-05 depth PSD | `fields/proxy.py:1047` | proxy only; no PCA or PLSR in the engine |

## Audit provenance

- Inventory (low): done. Counts match `todo.json`.
- Flags (medium): done after relaunch. 13 contradictions in 17 candidates.
- Receipts (high): done. 4 of 6 match; tier-0 count unknown; VIP 14.0 unsourced.
- Writing style (high): done. 11 findings verified. Eight wording fixes applied; two owner calls left.
- Jarness (xhigh): done. Proposals only; no rule file edited.
- Flow (xhigh): done. Five duplicated facts and seven gaps; P7 and P11 come from it.
- JaxFNE tools (max): done. Seven tools above.
- Knife edge (max): done. The script matches the emitter update and reset; P12 to P14 come from it.

## Wrap-up 2026-11-02

Owner decisions taken today are in the queue table (D1 to D10). Open work, in order:

| id | item | next step | blocked by |
|---|---|---|---|
| W1 | VIP lineage spec: drive 25.0 effective, pin 0.5.8, window 200-1000 ms, each cell at least 1 spike | write and commit the spec before any run | none |
| W2 | PV lineage spec: drive 5.0 effective | after the VIP receipt is committed | W1 |
| W3 | `tests/test_minimal_vip_state.py` is untracked and asserts the withdrawn u0 = -13 | leave untracked; replace it with the W1 test, run it against the pre-change build first | W1 |
| W4 | `manifests/todo.json` MIN-07 still names u0 as its principal delta | split into the W1 and W2 items; run `project_check` | W1, W2 |
| W5 | Recorded answers not yet applied: `AGENTS.md:17` G: path to E: clone; S-09 restatement (D4); PV→VIP text in `connectivity.json` (P3); VIP→VIP absence; §4a sentence in `PLAN_MINIMAL_MODELS.md` (P2); MIN-06/MIN-07 naming in `PLAN_DECISION_TREE.md` (P7) | one edit each, then `project_check` | none |
| W6 | Jarness subagent cap: Haiku 5.5 up to 8, other models 3 | edit in the jarness repo, then install; not in `~/.claude` directly | jarness clone |
| W7 | D8 CK-A schedule spec | our spec work; no owner question open | none |
| W8 | Missing PV_M0_STATES: accepted as unrecoverable | add the additive correction record | none |
| W9 | jaxfne has not read #140 or #142 (checked once, 2026-11-02) | read `main` at the next session start, once; do not poll | none |

Pin: `jaxfne==0.5.8` (owner, D-queue). `dee12f9e` exists only in `E:/repos/jaxfne`, on no remote branch.

Tier-0 count (P1, closed for the project venv): 213 passed, 6 warnings, 116.9 s, run 2026-11-03 under the suite lock with `.venv` (jaxfne 0.5.8). The same run through the system Python gave 211 passed and 2 failed; the failing test names were not captured, and the identity and editable-install checks in that run failed for that interpreter (jaxfne 0.4.25; editable install at `E:\jomission`).

# Session: `YYYY-MM-DD_NNN_slug` — Title

> Copy this template to `docs/sessions/YYYY/YYYY-MM-DD_NNN_slug.md` and create the matching JSON sidecar per `SCHEMA.md`. Markdown is authoritative.

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Session ID | `YYYY-MM-DD_NNN_slug` |
| UTC | `YYYY-MM-DDTHH:MM:SSZ` |
| Local | `YYYY-MM-DD HH:MM (TZ)` |
| Branch | `main` |
| HEAD / code SHA | `abc1234` (full: `abc1234...`) |
| Parent model | `manifests/<name>.json` (`config_hash: <hash>`, `hp_hash: <hash>`) or `NONE` |
| Config / hash | `config_hash=<hash>`, `hp_hash=<hash>` |
| Agent roles | `Worker: ... | Reviewer: ...` |
| Env | `Python X.Y | JAX X.Y | jaxfne X.Y | manifests sha256[:16]=...` |

## 2. Goal

- **Research question:** _One sentence — what is being asked?_
- **Acceptance predicate (before results):** _POSITIVE if ... ; else NEGATIVE_INSTANCE / NULL_RESULT / ... with threshold._
- **Out of scope / non-goals:** _..._

## 3. Starting authoritative state

Each claim tagged `OBSERVED` / `DERIVED` / `INFERRED` / `MODEL_ASSUMPTION` / `LITERATURE_PRIOR` / `UNRESOLVED`.

| # | Claim | Tag | Authority (manifest / ledger / commit / paper) |
|---|-------|-----|-----------------------------------------------|
| 3.1 | _e.g., RF energy ratio 184.55_ | `OBSERVED` | `docs/FACTORIAL_V0P2_DESIGN.md:11`, `manifests/...` |
| 3.2 | _e.g., Reward schedule_ | `UNRESOLVED` | `README.md:47` — explicitly `None` |
| 3.3 | | | |

## 4. Work performed

Execution ≠ verification — note which artifacts were verified by tests/gates.

| Worker | Assignment | Action | Artifact | Result |
|--------|------------|--------|----------|--------|
| _role_ | _task_ | _command / edit_ | `path/to/artifact` | `done / blocked / ...` |

## 5. Evidence

| Ref | Path | Kind | Hash / receipt | Notes |
|-----|------|------|----------------|-------|
| E5.1 | `manifests/<name>.json` | `manifest` | `sha256[:16]=...` | _..._ |
| E5.2 | `results/...npy` | `array` | `sha256[:16]=...` | _..._ |
| E5.3 | `results/figures/...png` | `figure` | — | Visualization (see §6) |
| E5.4 | `tests/...` | `test receipt` | `pytest ...` | _..._ |

- **EvidenceRefs:** _list or link_
- **Visualization report:** _link to `results/figures/...` or `docs/...` viz report — **required if figures exist**_

## 6. Observations

Compact quantitative tables only — no interpretation.

| ID | Metric | Value | Units | n | Artifact | Notes |
|----|--------|-------|-------|---|----------|-------|
| Obs 6.1 | _e.g., omission rate diff_ | _0.12_ | _spikes/s_ | _96_ | `E5.2` | _per position p3_ |
| Obs 6.2 | | | | | | |

_Figures:_ _link or `None`_

## 7. Interpretation

Observation ≠ interpretation. Each claim cites supporting observation ID(s) from §6.

| # | Interpretation | Supporting observations | Confidence |
|---|----------------|-------------------------|------------|
| 7.1 | _e.g., No omission spiking at p3_ | `Obs 6.1` | `HIGH` |
| 7.2 | | | |

## 8. Negative / insufficient results

Bounded vocabulary: `NEGATIVE_INSTANCE` | `NULL_RESULT` | `INSUFFICIENT_POWER` | `INCONCLUSIVE` | `RESOURCE_BOUNDARY` | `BLOCKED`; `FALSIFIED` reserved for `L0`–`L5` criterion failures.

| # | Claim | Verdict | Level / threshold | Notes |
|---|-------|---------|-------------------|-------|
| 8.1 | _e.g., T1 omission effect_ | `NEGATIVE_INSTANCE` | `p=0.42 > 0.01` | _..._ |
| 8.2 | | | | |

_If none:_ `No negative/insufficient results — all acceptance predicates passed (see §6).`

## 9. Unexpected findings

| # | Finding | Why unexpected | Follow-up priority |
|---|---------|----------------|-------------------|
| 9.1 | _..._ | _..._ | `high / med / low` |

_If none:_ `None.`

## 10. Tool / workflow friction

| Friction | Cause | Workaround | Permanent repair? |
|----------|-------|------------|-------------------|
| _e.g., OOM on 16-way_ | _memory_ | _serial execution_ | `HPC batch — proposed in §12` |

_If none:_ `No friction encountered.`

## 11. Learned lessons

| Lesson | Trigger | Cause | Repair | Evidence | Scope | Confidence | Persistence |
|--------|---------|-------|--------|----------|-------|------------|-------------|
| _e.g., Energy-matched RF required_ | _184x ratio_ | _uniform vs graded_ | _per-slot scaling_ | `E5.1` | `project` | `HIGH` | `YES` |

_Persistence_ ∈ `YES` | `NO`; `Confidence` ∈ `LOW` | `MED` | `HIGH`. `YES` = propose promotion to harness/project law.

## 12. Harness / tool proposals

Type ∈ `PROJECT_RULE` | `TEST/GATE` | `JOMISSION_TOOL` | `JAXFNE_UPSTREAM` | `DOCUMENTATION` | `VISUALIZATION` | `AGENT_WORKFLOW`

| # | Type | Target | Rationale | Evidence |
|---|------|--------|-----------|----------|
| 12.1 | `TEST/GATE` | _e.g., energy-match gate_ | _..._ | `Obs 6.1` |

_If none:_ `None — no harness change proposed this session.`

## 13. Scientific state transition

| Field | Value |
|-------|-------|
| **Before** | _state before session_ |
| **After** | _state after session_ |
| **Unlocked** | _what is now possible_ |
| **Still gated** | _what remains blocked_ |
| **Invalidated** | _what was disproven / deprecated_ |
| **New uncertainty** | _new UNRESOLVED introduced_ |

## 14. Next action

- **Primary (highest-value):** _one action — what, why, ready predicate (when can it start)._
- **Secondary (optional):** _..._

## 15. Progress score

| Field | Value |
|-------|-------|
| Current | _X.Y_ |
| Previous | _X.Y_ (`null` if first session) |
| Delta | _+Z.Z_ |
| Reason | _verified capability gained/lost — cite tests/gates/manifests, not activity_ |

_Score reflects verified capability (tests/gates/manifests), not activity. Cite evidence._

---

**Checklist before seal:**

- [ ] All 15 sections present and in order
- [ ] §1 `session_id` matches filename stem; timestamps ISO-8601
- [ ] §5 every artifact exists; figures link to visualization
- [ ] §6 tables have units, n, artifact refs; no interpretation
- [ ] §7 each claim cites `Obs 6.x`
- [ ] §8 bounded vocabulary used correctly; `FALSIFIED` only with `L0`–`L5` ref
- [ ] §10 friction normalized (for recurrent detection)
- [ ] §11 `Persistence` and `Confidence` enums correct
- [ ] §15 score delta justified by verified capability
- [ ] JSON sidecar `*.json` created and matches `SCHEMA.md`
- [ ] `python docs/sessions/validate.py <this file>` passes

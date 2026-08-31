# Session Report Schema — Normative

## 1. Authority

- **Markdown is authoritative.** `docs/sessions/YYYY/YYYY-MM-DD_NNN_slug.md` is the source of truth for narrative, observations, and verdicts.
- **JSON sidecar is derived** for retrieval/indexing. Same directory and basename, `.json` extension. On conflict, Markdown wins; validator reports the mismatch.
- **Append-only.** After commit, neither file is edited in place. Corrections are new sessions.
- This document (`SCHEMA.md`) is normative for validation. `validate.py` implements it.

## 2. File Naming

```
docs/sessions/YYYY/YYYY-MM-DD_NNN_slug.md
docs/sessions/YYYY/YYYY-MM-DD_NNN_slug.json
```

- `YYYY-MM-DD` — UTC date of session start.
- `NNN` — zero-padded daily sequence (`001`, `002`, …).
- `slug` — `a-z0-9` + hyphen, `1–48` chars, no spaces.
- Year directory MUST match the date prefix.

Regex: `^(?P<year>\d{4})/(?P<date>\d{4}-\d{2}-\d{2})_(?P<seq>\d{3})_(?P<slug>[a-z0-9-]{1,48})\.md$` relative to `docs/sessions/`.

## 3. Markdown — 15 Required Sections (H2, exact spelling, in order)

The Markdown MUST contain these H2 headings verbatim and in order. Extra H3/H4 inside sections is allowed.

```markdown
## 1. Session identity
## 2. Goal
## 3. Starting authoritative state
## 4. Work performed
## 5. Evidence
## 6. Observations
## 7. Interpretation
## 8. Negative / insufficient results
## 9. Unexpected findings
## 10. Tool / workflow friction
## 11. Learned lessons
## 12. Harness / tool proposals
## 13. Scientific state transition
## 14. Next action
## 15. Progress score
```

Validator checks presence and order via `^##\s+(\d+)\.\s+(.+)\s*$`.

### Per-section field expectations (Markdown tables/text)

**§1 Session identity** — MUST state: `session_id` (matches filename), UTC timestamp (ISO-8601), local timestamp + timezone, branch, `HEAD` / `code_sha` (`[0-9a-f]{7,40}`), `parent_model` + `config_hash`/`hp_hash` (or `NONE` for green-field), agent roles, env (Python, JAX/JaxFNE version, relevant manifest `sha256[:16]`).

**§2 Goal** — MUST state: research question (one sentence), acceptance predicate (pass/fail threshold BEFORE results), out-of-scope / non-goals.

**§3 Starting authoritative state** — MUST tag each claim with exactly one of `OBSERVED` / `DERIVED` / `INFERRED` / `MODEL_ASSUMPTION` / `LITERATURE_PRIOR` / `UNRESOLVED`. Each claim cites authority (manifest, ledger, commit, paper).

**§4 Work performed** — Table `Worker | Assignment | Action | Artifact | Result` with ≥1 row if work occurred. Text MUST note execution ≠ verification where applicable.

**§5 Evidence** — Lists `EvidenceRefs`, `manifests/*.json`, arrays (with `sha256[:16]` or `hash`), test receipts, figure paths, visualization reports, literature refs. Relative paths from repo root.

**§6 Observations** — Compact quantitative tables only (no interpretation). Each row SHOULD have units, n, statistic.

**§7 Interpretation** — MUST be distinguishable from observation; each interpretive claim MUST cite supporting observation ID(s) from §6 (e.g., `Obs 6.2`).

**§8 Negative / insufficient results** — Bounded vocabulary. Each entry tagged `NEGATIVE_INSTANCE` | `NULL_RESULT` | `INSUFFICIENT_POWER` | `INCONCLUSIVE` | `RESOURCE_BOUNDARY` | `BLOCKED`; `FALSIFIED` only with `L0`–`L5` criterion ref.

**§9 Unexpected findings** — List or `None` with justification.

**§10 Tool / workflow friction** — Table `Friction | Cause | Workaround | Permanent repair?` — `None` allowed with explicit `No friction encountered.`

**§11 Learned lessons** — Table `Lesson | Trigger | Cause | Repair | Evidence | Scope | Confidence | Persistence` where `Persistence` ∈ `YES` | `NO`, `Confidence` ∈ `LOW` | `MED` | `HIGH`.

**§12 Harness / tool proposals** — Each proposal typed `PROJECT_RULE` | `TEST/GATE` | `JOMISSION_TOOL` | `JAXFNE_UPSTREAM` | `DOCUMENTATION` | `VISUALIZATION` | `AGENT_WORKFLOW` with target + rationale.

**§13 Scientific state transition** — Six fields: `Before` | `After` | `Unlocked` | `Still gated` | `Invalidated` | `New uncertainty`.

**§14 Next action** — One primary (highest-value) + optional secondary, each with rationale and ready predicate.

**§15 Progress score** — `Current / Previous / Delta / Reason` — score reflects **verified capability** (tests/gates/manifests), not activity. Previous is prior session's Current.

## 4. JSON Sidecar Schema

MIME: `application/json`, UTF-8, object at top level.

### 4.1 Required fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | `string` | YES | Must equal filename stem (`YYYY-MM-DD_NNN_slug`). Pattern `^\d{4}-\d{2}-\d{2}_\d{3}_[a-z0-9-]{1,48}$` |
| `timestamp` | `string` (ISO-8601 UTC) | YES | e.g., `2026-08-29T14:03:00Z` |
| `code_sha` | `string` | YES | `HEAD` SHA (`[0-9a-f]{7,40}`) at session start |
| `parent_model` | `string \| null` | YES | Manifest/ledger ID of parent model, or `null` if none |
| `goal` | `string` | YES | One-sentence research question (mirrors §2) |
| `acceptance` | `string` | YES | Acceptance predicate text (mirrors §2) |
| `workers` | `array<object>` | YES | Each `{role, assignment, artifact?, result?}` — mirrors §4 |
| `artifacts` | `array<object>` | YES | Each `{path, kind, hash?}` where `kind` ∈ `manifest`/`array`/`figure`/`ledger`/`log`/`other`; `path` relative to repo root |
| `observations` | `array<object>` | YES | Each `{id, metric, value, units?, n?, artifact?}` — mirrors §6 |
| `verdicts` | `array<object>` | YES | Each `{claim, verdict, level?}` where `verdict` ∈ `POSITIVE`/`NEGATIVE_INSTANCE`/`NULL_RESULT`/`INSUFFICIENT_POWER`/`INCONCLUSIVE`/`RESOURCE_BOUNDARY`/`BLOCKED`/`FALSIFIED` |
| `lessons` | `array<object>` | YES | Each `{lesson, trigger, cause, repair, evidence, scope, confidence, persistence}` with `confidence` ∈ `LOW`/`MED`/`HIGH`, `persistence` ∈ `YES`/`NO` |
| `frictions` | `array<object>` | YES | Each `{friction, cause, workaround, permanent_repair?}` — mirrors §10 |
| `state_before` | `string` | YES | Free text summary of §3 / §13 Before |
| `state_after` | `string` | YES | Free text summary of §13 After |
| `next_action` | `string` | YES | Primary next action (mirrors §14) |
| `score_before` | `number \| null` | YES | Prior session score, or `null` if first |
| `score_after` | `number` | YES | Current progress score (mirrors §15 Current) |

### 4.2 Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `config_hash` | `string` | `config_hash` (`[0-9a-f]{8,64}`) — if set, validator checks against `manifests/*.json` |
| `hp_hash` | `string` | `hp_hash` for HDP/plasticity config |
| `branch` | `string` | Git branch at session start |
| `secondary_next` | `string` | Optional secondary next action |
| `model_delta` | `string` | Ledger entry ID if model mutated, else `NO_MODEL_DELTA` |
| `visualizations` | `array<string>` | Relative paths to visualization reports/figures |

### 4.3 Type & value rules

- Strings non-empty after trim unless explicitly nullable.
- `score_after` and `score_before` (when not null) are finite numbers; `score_after` ≥ 0.
- Enum fields are **case-sensitive** and must match exactly.
- `artifacts[].path` MUST be repo-relative (no leading `/`, no `..` traversal outside repo).
- At least one of `verdicts[]` or `observations[]` should be non-empty if `score_after > 0` (warning, not error).

## 5. Validation Rules

### 5.1 Structural

- Filename matches convention and year directory.
- Markdown has all 15 H2 sections in order; no missing or duplicate numbered section.
- JSON is valid JSON, has all required fields with correct types/enums.
- `session_id` in JSON equals Markdown filename stem; `timestamp` date prefix matches filename date.

### 5.2 Cross-file authority

- **Markdown vs JSON authority:** On mismatch (`session_id`, `goal`/`acceptance` semantic drift, `code_sha`, `score`), emit `MISMATCH` warning; Markdown is authoritative.
- **Artifact existence:** Every `artifacts[].path` MUST exist on disk OR be a known `manifests/*.json` that exists in `git HEAD` (validator checks `Path.exists()`).
- **Config/hash contradiction:** If `config_hash` (or `hp_hash`) is present in JSON, validator searches `manifests/*.json` for that hash substring. If no manifest contains it and no artifact with that hash exists, emit `CONFIG_HASH_UNRESOLVED` warning. If a manifest contains a *different* `config_hash` for the claimed `parent_model`, emit `CONFIG_HASH_CONTRADICTION` error.
- **Append-only:** If file is git-tracked, `git log --follow --diff-filter=M -- <path>` showing post-seal modification emits `MUTATION_WARNING`.

### 5.3 Frontier generation (INDEX.md)

`validate.py --reindex` regenerates `docs/sessions/INDEX.md` from all validated sessions:

1. Collect all `*.md` with matching `*.json`; skip invalid sessions (error).
2. Sort descending by `session_id`.
3. Build session table, frontier, lessons-awaiting-promotion, recurrent friction sections per `README.md` Index Contract.
4. Flag `HARNESS_REVIEW_REQUIRED` if any normalized friction recurs ≥2 sessions.

## 6. JSON Example (minimal valid)

```json
{
  "session_id": "2026-08-29_001_example-question",
  "timestamp": "2026-08-29T10:00:00Z",
  "code_sha": "f9af396a0235",
  "parent_model": "canonical_full_exposure_seal",
  "config_hash": "4f9fdeae7428199a",
  "branch": "main",
  "goal": "Does X affect Y under condition Z?",
  "acceptance": "POSITIVE if p<0.01 on T1 omission effect at p3, else NEGATIVE_INSTANCE",
  "workers": [{"role": "agent", "assignment": "run probe", "artifact": "results/probe.npy", "result": "done"}],
  "artifacts": [{"path": "manifests/canonical_full_exposure_seal.json", "kind": "manifest", "hash": "378492a24c8b331d"}],
  "observations": [{"id": "Obs 6.1", "metric": "omission rate diff", "value": 0.12, "units": "spikes/s", "n": 96}],
  "verdicts": [{"claim": "T1 omission spiking", "verdict": "NEGATIVE_INSTANCE"}],
  "lessons": [{"lesson": "Energy-matched RF required", "trigger": "184x ratio", "cause": "uniform vs graded", "repair": "per-slot scaling", "evidence": "manifest hash", "scope": "project", "confidence": "HIGH", "persistence": "YES"}],
  "frictions": [{"friction": "OOM on 16-way", "cause": "memory", "workaround": "serial", "permanent_repair": "HPC batch"}],
  "state_before": "Before: model A, gated on energy verification",
  "state_after": "After: model A, energy verified, gated on exposure",
  "next_action": "Run HPC 740-trial exposure probes",
  "score_before": 3.0,
  "score_after": 3.5
}
```

## 7. Change Control

- Changes to this schema require a session report documenting the rationale and a bump to the validator's `SCHEMA_VERSION`.
- Existing sessions are never retroactively required to satisfy new schema additions; validator gates new sessions only (or emits warnings for old sessions under `--all`).

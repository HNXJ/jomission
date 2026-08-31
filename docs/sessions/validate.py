#!/usr/bin/env python3
"""
validate.py — Session report validator

Validates Markdown + JSON sidecar against SCHEMA.md.

Usage:
  python docs/sessions/validate.py docs/sessions/2026/2026-08-29_001_slug.md
  python docs/sessions/validate.py --all
  python docs/sessions/validate.py --reindex [--strict]
  python docs/sessions/validate.py --reindex --strict

Checks:
  - 15 required H2 sections present and in order
  - Filename convention + year directory
  - JSON sidecar required fields, types, enums
  - session_id / timestamp / code_sha consistency
  - Artifact existence (repo-relative paths)
  - config_hash / hp_hash no-contradiction with manifests/*.json
  - Append-only mutation warning (git log)
  - Frontier generation (--reindex) → docs/sessions/INDEX.md

Schema version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"

EXPECTED_SECTIONS = [
    "1. Session identity",
    "2. Goal",
    "3. Starting authoritative state",
    "4. Work performed",
    "5. Evidence",
    "6. Observations",
    "7. Interpretation",
    "8. Negative / insufficient results",
    "9. Unexpected findings",
    "10. Tool / workflow friction",
    "11. Learned lessons",
    "12. Harness / tool proposals",
    "13. Scientific state transition",
    "14. Next action",
    "15. Progress score",
]

# Filename: YYYY/YYYY-MM-DD_NNN_slug.md relative to docs/sessions/
FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})/(?P<date>\d{4}-\d{2}-\d{2})_(?P<seq>\d{3})_(?P<slug>[a-z0-9-]{1,48})\.md$"
)
SESSION_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{3}_[a-z0-9-]{1,48}$")
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
HASH_RE = re.compile(r"^[0-9a-f]{8,64}$")
ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

HEADING_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)\s*$")

VERDICT_ENUM = {
    "POSITIVE",
    "NEGATIVE_INSTANCE",
    "NULL_RESULT",
    "INSUFFICIENT_POWER",
    "INCONCLUSIVE",
    "RESOURCE_BOUNDARY",
    "BLOCKED",
    "FALSIFIED",
}
CONFIDENCE_ENUM = {"LOW", "MED", "HIGH"}
PERSISTENCE_ENUM = {"YES", "NO"}
KIND_ENUM = {"manifest", "array", "figure", "ledger", "log", "other"}

REQUIRED_JSON_FIELDS: dict[str, type | tuple] = {
    "session_id": str,
    "timestamp": str,
    "code_sha": str,
    "parent_model": (str, type(None)),
    "goal": str,
    "acceptance": str,
    "workers": list,
    "artifacts": list,
    "observations": list,
    "verdicts": list,
    "lessons": list,
    "frictions": list,
    "state_before": str,
    "state_after": str,
    "next_action": str,
    "score_before": (int, float, type(None)),
    "score_after": (int, float),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    # validate.py is at docs/sessions/validate.py
    return Path(__file__).resolve().parents[2]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def normalize_friction(s: str) -> str:
    """Normalize friction string for recurrence detection."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Markdown validation
# ---------------------------------------------------------------------------
def validate_markdown(md_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    text = md_path.read_text(encoding="utf-8")
    headings: list[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            num = m.group(1)
            title = m.group(2).strip()
            headings.append(f"{num}. {title}")

    # Check count and order
    if len(headings) < len(EXPECTED_SECTIONS):
        errors.append(
            f"MARKDOWN_SECTIONS_MISSING: found {len(headings)} H2 numbered sections, "
            f"expected {len(EXPECTED_SECTIONS)}. Found: {headings}"
        )
    for i, expected in enumerate(EXPECTED_SECTIONS):
        if i >= len(headings):
            errors.append(f"MARKDOWN_SECTION_MISSING: missing section '{expected}'")
        elif headings[i] != expected:
            errors.append(
                f"MARKDOWN_SECTION_MISMATCH at position {i+1}: "
                f"expected '{expected}', got '{headings[i]}'"
            )

    # Check for duplicate numbered sections
    seen: dict[str, int] = {}
    for h in headings:
        seen[h] = seen.get(h, 0) + 1
    for h, count in seen.items():
        if count > 1:
            errors.append(f"MARKDOWN_SECTION_DUPLICATE: '{h}' appears {count} times")

    # Filename convention
    sessions_dir = root / "docs" / "sessions"
    try:
        rel = md_path.relative_to(sessions_dir).as_posix()
    except ValueError:
        errors.append(f"FILENAME_NOT_UNDER_SESSIONS: {md_path} not under {sessions_dir}")
        rel = None

    if rel is not None:
        m = FILENAME_RE.match(rel)
        if not m:
            errors.append(
                f"FILENAME_CONVENTION: '{rel}' does not match "
                f"YYYY/YYYY-MM-DD_NNN_slug.md (slug: a-z0-9-, 1-48 chars)"
            )
        else:
            year = m.group("year")
            date = m.group("date")
            if not date.startswith(year):
                errors.append(
                    f"FILENAME_YEAR_MISMATCH: directory year '{year}' != date prefix '{date[:4]}'"
                )

    # Append-only mutation warning
    git_mut = run_git(
        ["log", "--follow", "--diff-filter=M", "--oneline", "--", str(md_path.relative_to(root)) if md_path.is_relative_to(root) else str(md_path)],
        cwd=root,
    )
    # Alternative: just check git log count for file modifications after creation
    # If file was modified more than once (created + modified), warn
    if git_mut is not None and git_mut != "":
        lines = [l for l in git_mut.splitlines() if l.strip()]
        if len(lines) >= 1:
            # Check if there are modification commits beyond the creation
            # Use git log --follow --diff-filter=A vs M
            mod_log = run_git(
                ["log", "--follow", "--diff-filter=M", "--format=%H", "--", str(md_path.relative_to(root)) if md_path.is_relative_to(root) else str(md_path)],
                cwd=root,
            )
            if mod_log and mod_log.strip():
                warnings.append(
                    f"MUTATION_WARNING: session file has post-creation modifications "
                    f"(append-only violation?) — commits: {mod_log[:80]}"
                )

    return errors, warnings


# ---------------------------------------------------------------------------
# JSON validation
# ---------------------------------------------------------------------------
def validate_json(json_path: Path, md_path: Path, root: Path) -> tuple[list[str], list[str], dict | None]:
    errors: list[str] = []
    warnings: list[str] = []

    if not json_path.exists():
        errors.append(f"JSON_SIDECAR_MISSING: expected sidecar at {json_path} (same basename as .md, .json ext)")
        return errors, warnings, None

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"JSON_PARSE_ERROR: {exc}")
        return errors, warnings, None

    if not isinstance(data, dict):
        errors.append("JSON_TOP_LEVEL: expected JSON object at top level")
        return errors, warnings, None

    # Required fields
    for field, expected_type in REQUIRED_JSON_FIELDS.items():
        if field not in data:
            errors.append(f"JSON_REQUIRED_FIELD_MISSING: '{field}'")
            continue
        val = data[field]
        # Handle tuple types
        if isinstance(expected_type, tuple):
            if not isinstance(val, expected_type):
                type_names = " | ".join(t.__name__ for t in expected_type)
                errors.append(
                    f"JSON_FIELD_TYPE: '{field}' expected {type_names}, got {type(val).__name__}"
                )
        else:
            if not isinstance(val, expected_type):
                errors.append(
                    f"JSON_FIELD_TYPE: '{field}' expected {expected_type.__name__}, got {type(val).__name__}"
                )
        # Non-empty strings
        if isinstance(val, str) and field in ("session_id", "goal", "acceptance", "state_before", "state_after", "next_action"):
            if not val.strip():
                errors.append(f"JSON_FIELD_EMPTY: '{field}' must be non-empty")

    # session_id pattern
    if "session_id" in data and isinstance(data["session_id"], str):
        if not SESSION_ID_RE.match(data["session_id"]):
            errors.append(f"JSON_SESSION_ID_PATTERN: '{data['session_id']}' does not match YYYY-MM-DD_NNN_slug")
        # Must equal filename stem
        expected_stem = md_path.stem
        if data["session_id"] != expected_stem:
            errors.append(
                f"JSON_SESSION_ID_MISMATCH: JSON session_id '{data['session_id']}' != filename stem '{expected_stem}' — Markdown authoritative, JSON must match"
            )

    # timestamp
    if "timestamp" in data and isinstance(data["timestamp"], str):
        if not ISO8601_RE.match(data["timestamp"]):
            warnings.append(f"JSON_TIMESTAMP_FORMAT: '{data['timestamp']}' should be ISO-8601 UTC (YYYY-MM-DDTHH:MM:SSZ)")
        # Date prefix should match filename date
        if "session_id" in data and isinstance(data["session_id"], str) and len(data["session_id"]) >= 10:
            sid_date = data["session_id"][:10]
            ts_date = data["timestamp"][:10]
            if sid_date != ts_date:
                warnings.append(f"JSON_TIMESTAMP_DATE_MISMATCH: timestamp date '{ts_date}' != session_id date '{sid_date}'")

    # code_sha
    if "code_sha" in data and isinstance(data["code_sha"], str):
        if not SHA_RE.match(data["code_sha"]):
            errors.append(f"JSON_CODE_SHA_PATTERN: '{data['code_sha']}' must be [0-9a-f]{{7,40}}")

    # config_hash / hp_hash if present
    for hf in ("config_hash", "hp_hash"):
        if hf in data and data[hf] is not None:
            if not isinstance(data[hf], str) or not HASH_RE.match(data[hf]):
                errors.append(f"JSON_{hf.upper()}_PATTERN: '{data.get(hf)}' must be [0-9a-f]{{8,64}}")

    # score_after finite + >=0
    if "score_after" in data and isinstance(data["score_after"], (int, float)):
        import math
        if not math.isfinite(data["score_after"]):
            errors.append("JSON_SCORE_AFTER: must be finite")
        elif data["score_after"] < 0:
            errors.append("JSON_SCORE_AFTER: must be >= 0")

    # Artifacts
    if "artifacts" in data and isinstance(data["artifacts"], list):
        for idx, art in enumerate(data["artifacts"]):
            if not isinstance(art, dict):
                errors.append(f"JSON_ARTIFACTS[{idx}]: expected object")
                continue
            if "path" not in art or not isinstance(art["path"], str) or not art["path"].strip():
                errors.append(f"JSON_ARTIFACTS[{idx}].path: required non-empty string")
            else:
                p = art["path"].strip()
                if p.startswith("/") or ".." in Path(p).parts:
                    # Allow .. only if still inside repo after resolve; warn
                    if ".." in p:
                        warnings.append(f"JSON_ARTIFACTS[{idx}].path: contains '..' traversal: '{p}'")
                    if p.startswith("/"):
                        errors.append(f"JSON_ARTIFACTS[{idx}].path: must be repo-relative, not absolute: '{p}'")
                else:
                    full = root / p
                    if not full.exists():
                        # Also check git-tracked manifests that might not be on disk in some envs?
                        warnings.append(f"JSON_ARTIFACT_MISSING: artifacts[{idx}].path '{p}' does not exist on disk")
            if "kind" not in art:
                errors.append(f"JSON_ARTIFACTS[{idx}].kind: required, one of {sorted(KIND_ENUM)}")
            elif art["kind"] not in KIND_ENUM:
                errors.append(f"JSON_ARTIFACTS[{idx}].kind: '{art['kind']}' not in {sorted(KIND_ENUM)}")

    # Observations
    if "observations" in data and isinstance(data["observations"], list):
        for idx, obs in enumerate(data["observations"]):
            if not isinstance(obs, dict):
                errors.append(f"JSON_OBSERVATIONS[{idx}]: expected object")
                continue
            for req in ("id", "metric", "value"):
                if req not in obs:
                    errors.append(f"JSON_OBSERVATIONS[{idx}].{req}: required")

    # Verdicts
    if "verdicts" in data and isinstance(data["verdicts"], list):
        for idx, v in enumerate(data["verdicts"]):
            if not isinstance(v, dict):
                errors.append(f"JSON_VERDICTS[{idx}]: expected object")
                continue
            if "verdict" not in v:
                errors.append(f"JSON_VERDICTS[{idx}].verdict: required")
            elif v["verdict"] not in VERDICT_ENUM:
                errors.append(f"JSON_VERDICTS[{idx}].verdict: '{v['verdict']}' not in {sorted(VERDICT_ENUM)}")

    # Lessons
    if "lessons" in data and isinstance(data["lessons"], list):
        for idx, les in enumerate(data["lessons"]):
            if not isinstance(les, dict):
                errors.append(f"JSON_LESSONS[{idx}]: expected object")
                continue
            for req in ("lesson", "trigger", "cause", "repair", "evidence", "scope", "confidence", "persistence"):
                if req not in les:
                    errors.append(f"JSON_LESSONS[{idx}].{req}: required")
            if "confidence" in les and les["confidence"] not in CONFIDENCE_ENUM:
                errors.append(f"JSON_LESSONS[{idx}].confidence: '{les.get('confidence')}' not in {sorted(CONFIDENCE_ENUM)}")
            if "persistence" in les and les["persistence"] not in PERSISTENCE_ENUM:
                errors.append(f"JSON_LESSONS[{idx}].persistence: '{les.get('persistence')}' not in {sorted(PERSISTENCE_ENUM)}")

    # Frictions
    if "frictions" in data and isinstance(data["frictions"], list):
        for idx, fr in enumerate(data["frictions"]):
            if not isinstance(fr, dict):
                errors.append(f"JSON_FRICTIONS[{idx}]: expected object")
                continue
            if "friction" not in fr:
                errors.append(f"JSON_FRICTIONS[{idx}].friction: required")

    # config_hash contradiction check against manifests
    config_hash = data.get("config_hash")
    if config_hash and isinstance(config_hash, str) and HASH_RE.match(config_hash):
        manifests_dir = root / "manifests"
        if manifests_dir.exists():
            found = False
            contradiction = False
            for mf in manifests_dir.glob("*.json"):
                try:
                    mtext = mf.read_text(encoding="utf-8")
                except Exception:
                    continue
                if config_hash[:8] in mtext or config_hash in mtext:
                    found = True
                    break
                # Also parse JSON and check config_hash fields
                try:
                    mj = json.loads(mtext)
                    # Search recursively for hash values
                    def search(obj: object) -> bool:
                        if isinstance(obj, str) and config_hash in obj:
                            return True
                        if isinstance(obj, dict):
                            return any(search(v) for v in obj.values())
                        if isinstance(obj, list):
                            return any(search(v) for v in obj)
                        return False
                    if search(mj):
                        found = True
                        break
                except Exception:
                    pass
            if not found:
                warnings.append(
                    f"CONFIG_HASH_UNRESOLVED: config_hash '{config_hash}' not found in any manifests/*.json — "
                    f"ensure parent_model/manifest is committed or hash is correct"
                )

    # Markdown vs JSON cross-check (warnings, Markdown authoritative)
    # We do this at higher level after both validations

    return errors, warnings, data


# ---------------------------------------------------------------------------
# Single-session validation
# ---------------------------------------------------------------------------
def validate_session(md_path: Path, root: Path) -> tuple[bool, list[str], list[str], dict | None]:
    md_path = md_path.resolve()
    json_path = md_path.with_suffix(".json")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Markdown checks
    md_errors, md_warnings = validate_markdown(md_path, root)
    all_errors.extend(md_errors)
    all_warnings.extend(md_warnings)

    # JSON checks
    j_errors, j_warnings, data = validate_json(json_path, md_path, root)
    all_errors.extend(j_errors)
    all_warnings.extend(j_warnings)

    # Cross-check Markdown vs JSON (Markdown authoritative)
    if data is not None:
        # Compare session_id already done; also check code_sha drift
        md_text = md_path.read_text(encoding="utf-8")
        # Extract code SHA mention from markdown §1 table if present
        sha_in_md = re.findall(r"[0-9a-f]{7,40}", md_text)
        code_sha = data.get("code_sha", "")
        if code_sha and sha_in_md and code_sha not in sha_in_md:
            all_warnings.append(
                f"MISMATCH_CODE_SHA: JSON code_sha '{code_sha}' not found in Markdown — Markdown authoritative"
            )

    passed = len(all_errors) == 0
    return passed, all_errors, all_warnings, data


# ---------------------------------------------------------------------------
# INDEX generation
# ---------------------------------------------------------------------------
def collect_sessions(root: Path) -> list[tuple[Path, Path, dict]]:
    sessions_dir = root / "docs" / "sessions"
    results: list[tuple[Path, Path, dict]] = []
    for md_path in sorted(sessions_dir.rglob("*.md")):
        # Skip non-session files
        if md_path.name in ("README.md", "SCHEMA.md", "TEMPLATE.md", "INDEX.md"):
            continue
        # Only files under year directories (4-digit)
        try:
            rel = md_path.relative_to(sessions_dir)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        if not re.match(r"^\d{4}$", rel.parts[0]):
            continue
        json_path = md_path.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Only include if session_id looks valid
        if not isinstance(data.get("session_id"), str):
            continue
        results.append((md_path, json_path, data))
    # Sort descending by session_id
    results.sort(key=lambda t: t[2].get("session_id", ""), reverse=True)
    return results


def generate_index(root: Path, strict: bool = False) -> tuple[bool, str]:
    sessions = collect_sessions(root)
    sessions_dir = root / "docs" / "sessions"
    index_path = sessions_dir / "INDEX.md"

    # Gather frictions for recurrence
    friction_map: dict[str, list[dict]] = {}
    for _md, _js, data in sessions:
        for fr in data.get("frictions", []):
            if not isinstance(fr, dict):
                continue
            raw = str(fr.get("friction", "")).strip()
            if not raw or raw.lower() == "none":
                continue
            norm = normalize_friction(raw)
            if not norm:
                continue
            friction_map.setdefault(norm, []).append(
                {"raw": raw, "session": data.get("session_id", "?"), "fr": fr}
            )

    recurrent = {k: v for k, v in friction_map.items() if len(v) >= 2}

    # Lessons awaiting promotion (Persistence=YES)
    lessons_rows: list[dict] = []
    for _md, _js, data in sessions:
        sid = data.get("session_id", "?")
        for les in data.get("lessons", []):
            if not isinstance(les, dict):
                continue
            if les.get("persistence") == "YES":
                # Check if already promoted: heuristic — search AGENTS.md / docs for lesson text
                # For now, include all YES; validator warns if needed
                lessons_rows.append(
                    {
                        "lesson": str(les.get("lesson", ""))[:120],
                        "session": sid,
                        "scope": str(les.get("scope", "")),
                        "confidence": str(les.get("confidence", "")),
                    }
                )

    # Build index content
    lines: list[str] = []
    lines.append("# Session Index — Derived (do not hand-edit)")
    lines.append("")
    lines.append("> **Retrieval rule for future agents:** Read authoritative manifests + `INDEX.md` + relevant session Markdown(s). Mutable current state (latest manifest/ledger/HEAD) overrides historical session narrative. Never treat a session as authority over a manifest.")
    lines.append("")
    lines.append(f"> **Schema version:** `{SCHEMA_VERSION}` · **Generated:** auto via `validate.py --reindex` · **Sessions:** {len(sessions)}")
    lines.append("")
    lines.append("## Session Table")
    lines.append("")
    lines.append("| Session | Date | Parent | Question | Verdict | Model Δ | Key lesson | Next |")
    lines.append("|---------|------|--------|----------|---------|---------|------------|------|")

    if not sessions:
        lines.append("| _No sessions yet — frontier is manifests + ledger._ | — | — | — | — | — | — | — |")
    else:
        for md_path, _js, data in sessions:
            sid = data.get("session_id", md_path.stem)
            # Date from session_id
            date = sid[:10] if len(sid) >= 10 else "?"
            parent = data.get("parent_model") or "—"
            if isinstance(parent, str) and len(parent) > 24:
                parent = parent[:24] + "…"
            question = str(data.get("goal", "—")).replace("|", "\\|").replace("\n", " ")[:80]
            # Verdict: first verdicts[0].verdict or —
            verdict = "—"
            verdicts = data.get("verdicts", [])
            if verdicts and isinstance(verdicts, list) and len(verdicts) > 0:
                v0 = verdicts[0]
                if isinstance(v0, dict):
                    verdict = str(v0.get("verdict", "—"))
            model_delta = str(data.get("model_delta", "NO_MODEL_DELTA") or "NO_MODEL_DELTA").replace("|", "\\|")
            # Key lesson: first Persistence=YES lesson
            key_lesson = "—"
            for les in data.get("lessons", []):
                if isinstance(les, dict) and les.get("persistence") == "YES":
                    key_lesson = str(les.get("lesson", "—")).replace("|", "\\|").replace("\n", " ")[:60]
                    break
            next_action = str(data.get("next_action", "—")).replace("|", "\\|").replace("\n", " ")[:60]
            rel_md = md_path.relative_to(sessions_dir).as_posix()
            # Viz link if visualizations present
            viz = ""
            vizs = data.get("visualizations", [])
            if vizs and isinstance(vizs, list) and len(vizs) > 0:
                viz = f" [Viz]({vizs[0]})"
            # Check artifacts for figure kind
            has_fig = any(isinstance(a, dict) and a.get("kind") == "figure" for a in data.get("artifacts", []))
            sess_cell = f"[{sid}]({rel_md}){viz}" if viz else f"[{sid}]({rel_md})"
            if has_fig and not viz:
                sess_cell += " · Viz: see §5"

            lines.append(f"| {sess_cell} | {date} | {parent} | {question} | {verdict} | {model_delta} | {key_lesson} | {next_action} |")

    lines.append("")
    lines.append("## Current Frontier")
    lines.append("")
    if not sessions:
        lines.append("No sessions yet — frontier is manifests + ledger.")
        lines.append("")
        lines.append("- **Unlocked:** —")
        lines.append("- **Still gated:** See `manifests/*.json` and `gen2_modification_ledger.jsonl`.")
    else:
        latest_data = sessions[0][2]
        lines.append(f"Synthesized from latest session `{latest_data.get('session_id', '?')}` §13.")
        lines.append("")
        # state_before/after are summaries; also emit next_action as frontier hint
        lines.append(f"- **Before:** {str(latest_data.get('state_before', '—')).replace(chr(10), ' ')[:200]}")
        lines.append(f"- **After:** {str(latest_data.get('state_after', '—')).replace(chr(10), ' ')[:200]}")
        lines.append(f"- **Next:** {str(latest_data.get('next_action', '—')).replace(chr(10), ' ')[:200]}")
        # Try to extract Unlocked/Still gated from latest Markdown §13 if available
        latest_md = sessions[0][0]
        try:
            md_text = latest_md.read_text(encoding="utf-8")
            # Extract §13 block
            sec13_match = re.search(r"##\s+13\.\s+Scientific state transition(.*?)(?=##\s+14\.)", md_text, re.DOTALL)
            if sec13_match:
                block = sec13_match.group(1).strip()
                # Keep first few lines
                blines = [l.strip() for l in block.splitlines() if l.strip()][:12]
                if blines:
                    lines.append("")
                    lines.append("<details><summary>Latest §13 excerpt</summary>")
                    lines.append("")
                    for bl in blines:
                        lines.append(bl)
                    lines.append("")
                    lines.append("</details>")
        except Exception:
            pass

    lines.append("")
    lines.append("## Lessons Awaiting Promotion")
    lines.append("")
    lines.append("> Lessons with `Persistence=YES` not yet promoted to `AGENTS.md` / project law / harness docs.")
    lines.append("")
    if not lessons_rows:
        lines.append("_None._")
    else:
        lines.append("| Lesson | Source session | Scope | Confidence | Proposed target |")
        lines.append("|--------|----------------|-------|------------|-----------------|")
        for lr in lessons_rows:
            lesson = lr["lesson"].replace("|", "\\|")
            lines.append(f"| {lesson} | {lr['session']} | {lr['scope']} | {lr['confidence']} | `AGENTS.md` / harness |")

    lines.append("")
    lines.append("## Recurrent Friction")
    lines.append("")
    if not friction_map:
        lines.append("_No friction reported yet._")
    elif not recurrent:
        lines.append("_No recurrent friction (no friction repeated ≥2 sessions)._")
        # Still list all frictions
        lines.append("")
        lines.append("| Friction | Occurrences | Sessions | Last workaround | Permanent repair? |")
        lines.append("|----------|-------------|----------|-----------------|-------------------|")
        for norm, occs in sorted(friction_map.items(), key=lambda kv: len(kv[1]), reverse=True):
            raw = occs[0]["raw"].replace("|", "\\|")
            count = len(occs)
            sess_list = ", ".join(o["session"] for o in occs[:3])
            last_wa = str(occs[0]["fr"].get("workaround", "—")).replace("|", "\\|")[:40]
            repair = str(occs[0]["fr"].get("permanent_repair", "—")).replace("|", "\\|")[:40]
            lines.append(f"| {raw} | {count} | {sess_list} | {last_wa} | {repair} |")
    else:
        lines.append("> ⚠️ `HARNESS_REVIEW_REQUIRED` — friction recurred ≥2 sessions; propose harness/tool fix in next session §12.")
        lines.append("")
        lines.append("| Friction | Occurrences | Sessions | Last workaround | Permanent repair? | Flag |")
        lines.append("|----------|-------------|----------|-----------------|-------------------|------|")
        for norm, occs in sorted(friction_map.items(), key=lambda kv: len(kv[1]), reverse=True):
            raw = occs[0]["raw"].replace("|", "\\|")
            count = len(occs)
            sess_list = ", ".join(o["session"] for o in occs[:3])
            last_wa = str(occs[0]["fr"].get("workaround", "—")).replace("|", "\\|")[:40]
            repair = str(occs[0]["fr"].get("permanent_repair", "—")).replace("|", "\\|")[:40]
            flag = "⚠️ HARNESS_REVIEW_REQUIRED" if count >= 2 else ""
            lines.append(f"| {raw} | {count} | {sess_list} | {last_wa} | {repair} | {flag} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_Generated by `docs/sessions/validate.py` · schema `{SCHEMA_VERSION}` · do not hand-edit — run `python docs/sessions/validate.py --reindex` to regenerate._")
    lines.append("")

    index_path.write_text("\n".join(lines), encoding="utf-8")

    # Strict mode: fail if recurrent friction exists
    if strict and recurrent:
        return False, f"HARNESS_REVIEW_REQUIRED: {len(recurrent)} friction(s) recurred ≥2 sessions — propose fix in §12"

    return True, f"INDEX regenerated at {index_path} ({len(sessions)} sessions)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate session report Markdown + JSON sidecar")
    parser.add_argument("path", nargs="?", help="Path to session Markdown (e.g., docs/sessions/2026/2026-08-29_001_slug.md)")
    parser.add_argument("--all", action="store_true", help="Validate all sessions under docs/sessions/")
    parser.add_argument("--reindex", action="store_true", help="Regenerate docs/sessions/INDEX.md")
    parser.add_argument("--strict", action="store_true", help="With --reindex, fail if recurrent friction ≥2 without §12 proposal")
    args = parser.parse_args()

    root = repo_root()

    # --reindex
    if args.reindex:
        ok, msg = generate_index(root, strict=args.strict)
        print(msg)
        if not ok:
            eprint(f"ERROR: {msg}")
            sys.exit(1)
        # If --reindex alone, exit. If also --all or path, continue
        if not args.all and not args.path:
            sys.exit(0)

    # --all
    if args.all:
        sessions = collect_sessions(root)
        # Also find sessions missing JSON sidecar (report as error)
        sessions_dir = root / "docs" / "sessions"
        all_md = [p for p in sessions_dir.rglob("*.md") if p.name not in ("README.md", "SCHEMA.md", "TEMPLATE.md", "INDEX.md") and len(p.relative_to(sessions_dir).parts) >= 2 and re.match(r"^\d{4}$", p.relative_to(sessions_dir).parts[0])]
        ok_all = True
        for md_path in sorted(all_md):
            json_path = md_path.with_suffix(".json")
            if not json_path.exists():
                # Check if it's a real session file (matches filename convention)
                try:
                    rel = md_path.relative_to(sessions_dir).as_posix()
                    if FILENAME_RE.match(rel):
                        eprint(f"FAIL {rel}: JSON_SIDECAR_MISSING")
                        ok_all = False
                    else:
                        # Not a session file, skip
                        pass
                except ValueError:
                    pass
                continue
            passed, errors, warnings, _data = validate_session(md_path, root)
            rel = md_path.relative_to(root).as_posix() if md_path.is_relative_to(root) else str(md_path)
            for w in warnings:
                print(f"WARN {rel}: {w}")
            if not passed:
                ok_all = False
                for e in errors:
                    eprint(f"FAIL {rel}: {e}")
            else:
                print(f"PASS {rel}")
                if warnings:
                    pass
        if not all_md:
            print("No sessions found.")
        sys.exit(0 if ok_all else 1)

    # Single path
    if args.path:
        md_path = Path(args.path)
        if not md_path.is_absolute():
            # Resolve relative to repo root or cwd
            candidate = root / md_path
            if candidate.exists():
                md_path = candidate
            elif Path(args.path).exists():
                md_path = Path(args.path).resolve()
            else:
                md_path = (Path.cwd() / md_path).resolve()
        if not md_path.exists():
            eprint(f"ERROR: file not found: {md_path}")
            sys.exit(2)
        if md_path.suffix != ".md":
            eprint(f"ERROR: expected .md file, got {md_path.suffix}: {md_path}")
            sys.exit(2)
        passed, errors, warnings, _data = validate_session(md_path, root)
        rel = md_path.relative_to(root).as_posix() if md_path.is_relative_to(root) else str(md_path)
        for w in warnings:
            print(f"WARN {rel}: {w}")
        if not passed:
            for e in errors:
                eprint(f"FAIL {rel}: {e}")
            sys.exit(1)
        else:
            print(f"PASS {rel} — schema {SCHEMA_VERSION}")
            sys.exit(0)

    # No args: show help
    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()

"""Validators for harness manifests: current state, TODO, lineage registry,
sealed-artifact registry, estimator metadata, and project skills.

Each validator returns a list of error strings (empty == valid). Artifact
references are "path" (working tree) or "ref:path" (git object).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ESTIMATOR_FIELDS = ("bin_width_ms", "aggregation", "baseline", "sign")
QUALIFICATION_STAGES = ("configured", "realized", "executed", "effective")
CURRENT_STATE_KEYS = ("scientific_head", "execution_authority", "jaxfne_version", "current_program",
                      "current_gate", "latest_verdict", "open_scientific_question", "next_authorized_task",
                      "locked_gates", "retired_axes", "active_invariants", "required_artifacts", "handoff_path")
TODO_KEYS = ("id", "status", "trigger", "goal", "parent_authority", "principal_delta", "required_inputs",
             "frozen", "procedure", "acceptance", "stop_condition", "artifacts_expected", "downstream_unlock")
LINEAGE_KEYS = ("lineage_id", "parent", "principal_delta", "preexecution_spec", "authority", "inputs", "frozen",
                "executed", "observations", "derived", "inferred", "verdict", "acceptance", "fail_boundary",
                "supersedes", "retired", "artifacts", "tests", "commit")
SKILL_MAX_LINES = 80


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def load(rel):
    return json.load(open(ROOT / rel, encoding="utf-8"))


def vocabulary():
    v = load("manifests/vocabulary.json")
    return v["evidence_classes"], v["status_labels"]


def artifact_exists(ref: str) -> bool:
    if ":" in ref and not re.match(r"^[A-Za-z]:[\\/]", ref):
        rev, path = ref.split(":", 1)
        return _git("cat-file", "-e", f"{rev}:{path}").returncode == 0
    return (ROOT / ref).exists()


def commit_exists(c: str) -> bool:
    return _git("cat-file", "-e", f"{c}^{{commit}}").returncode == 0


def is_ancestor(a: str, b: str) -> bool:
    return _git("merge-base", "--is-ancestor", a, b).returncode == 0


def validate_current_state(state: dict, todo: dict) -> list[str]:
    err = [f"current_state missing {k}" for k in CURRENT_STATE_KEYS if k not in state]
    if err:
        return err
    _, labels = vocabulary()
    head = state["scientific_head"]["main"]
    if not commit_exists(head):
        err.append(f"scientific_head {head} not a commit")
    elif not is_ancestor(head, "HEAD"):
        err.append(f"scientific_head {head} is not an ancestor of HEAD")
    for br, c in state["scientific_head"].get("unmerged_scientific_branches", {}).items():
        r = _git("rev-parse", "--short=7", br)
        if r.returncode != 0 or not r.stdout.strip().startswith(c[:7]):
            err.append(f"branch {br} does not point at {c}")
    for name, v in state["latest_verdict"].items():
        if v["status"] not in labels:
            err.append(f"latest_verdict {name}: status {v['status']} not in vocabulary")
        if not artifact_exists(v["evidence"]):
            err.append(f"latest_verdict {name}: evidence {v['evidence']} missing")
            continue
        err += [f"latest_verdict {name}: {e}" for e in _verdict_semantics(v, state["latest_verdict"])]
    if state["current_gate"]["status"] not in labels:
        err.append("current_gate status not in vocabulary")
    if not artifact_exists(state["current_gate"]["definition"]):
        err.append("current_gate definition missing")
    for ref in state["required_artifacts"]:
        if not artifact_exists(ref):
            err.append(f"required artifact missing: {ref}")
    if not artifact_exists(state["handoff_path"]):
        err.append("handoff_path missing")
    items = {i["id"]: i for i in todo["items"]}
    nxt = items.get(state["next_authorized_task"])
    if nxt is None or nxt["status"] not in ("OPEN", "OPEN_AFTER_HARNESS"):
        err.append(f"next_authorized_task {state['next_authorized_task']} is not an OPEN TODO item")
    locked_ids = {i["id"].replace("-", "_").replace(".", "_").upper() for i in todo["items"] if i["status"] == "LOCKED"}
    for g in state["locked_gates"]:
        if g.replace("-", "_").replace(".", "_").upper() not in locked_ids:
            err.append(f"locked gate {g} has no LOCKED TODO item")
    for u in state.get("open_uncertainties", []):
        r = u["route"]
        if not (r in items or r == "record only" or r.split(" ")[0] in state["retired_axes"]):
            err.append(f"open uncertainty routed to unknown {r}")
    src = (ROOT / "docs" / "project-sources" / "SUBSTRATE_PROGRAM.md")
    text = src.read_text(encoding="utf-8") if src.exists() else ""
    for axis in state["retired_axes"]:
        if f"`{axis}`" not in text:
            err.append(f"retired axis {axis} not defined in docs/project-sources/SUBSTRATE_PROGRAM.md")
    for inv in state["active_invariants"]:
        if f"`{inv}`" not in text:
            err.append(f"invariant {inv} not defined in docs/project-sources/SUBSTRATE_PROGRAM.md")
    return err


def read_artifact(ref: str) -> str:
    if ":" in ref and not re.match(r"^[A-Za-z]:[\\/]", ref):
        return _git("show", ref).stdout
    return (ROOT / ref).read_text(encoding="utf-8")


def _verdict_semantics(entry: dict, all_entries: dict) -> list[str]:
    """Result-vs-test semantics: a gate status comes from the evidence verdict, not pytest."""
    scope = entry.get("scope")
    if scope == "component":
        return [] if entry.get("component_of") in all_entries else ["component entry without known component_of"]
    if scope != "gate":
        return ["scope must be gate or component"]
    try:
        label = json.loads(read_artifact(entry["evidence"])).get("verdict")
    except (json.JSONDecodeError, AttributeError):
        return ["gate evidence is not a JSON object with a verdict"]
    if not isinstance(label, str):
        return ["gate evidence has no verdict field"]
    ov = entry.get("review_override")
    if ov:
        missing = [k for k in ("evidence_reads", "held_status", "reason", "authority", "lift") if not ov.get(k)]
        if missing:
            return [f"review_override missing {missing}"]
        if ov["evidence_reads"] != label:
            return [f"review_override.evidence_reads {ov['evidence_reads']} != evidence verdict {label}"]
        return [] if ov["held_status"] == entry["status"] else ["review_override.held_status != status"]
    return [] if label.endswith("_" + entry["status"]) else [f"status {entry['status']} contradicts evidence verdict {label}"]


def validate_todo(todo: dict) -> list[str]:
    err = []
    statuses = set(todo["statuses"])
    ids = [i.get("id") for i in todo["items"]]
    if len(ids) != len(set(ids)):
        err.append("duplicate TODO ids")
    known = set(ids)
    for it in todo["items"]:
        for k in TODO_KEYS:
            if k not in it:
                err.append(f"{it.get('id')}: missing {k}")
        if it.get("status") not in statuses:
            err.append(f"{it.get('id')}: bad status {it.get('status')}")
        for d in it.get("downstream_unlock", []):
            if d not in known:
                err.append(f"{it['id']}: downstream {d} unknown")
        if it.get("status") in ("OPEN", "OPEN_AFTER_HARNESS", "BLOCKED"):
            for ref in it.get("required_inputs", []):
                if "/" in ref and not artifact_exists(ref.split(" ")[0]):
                    err.append(f"{it['id']}: required input missing {ref}")
    return err


def validate_estimator(stat: dict) -> list[str]:
    est = stat.get("estimator") or {}
    return [f"transient statistic missing estimator.{f}" for f in ESTIMATOR_FIELDS
            if est.get(f) in (None, "")]


def validate_lineage(rec: dict) -> list[str]:
    classes, labels = vocabulary()
    lid = rec.get("lineage_id", "?")
    err = [f"{lid}: missing {k}" for k in LINEAGE_KEYS if k not in rec]
    if err:
        return err
    spec = rec["preexecution_spec"]
    if spec is not None:
        if not (commit_exists(spec["commit"]) and commit_exists(rec["commit"])):
            err.append(f"{lid}: pre-execution or result commit missing")
        elif spec["commit"] == rec["commit"] or not is_ancestor(spec["commit"], rec["commit"]):
            err.append(f"{lid}: pre-execution spec {spec['commit']} does not strictly precede {rec['commit']}")
        elif not artifact_exists(f"{spec['commit']}:{spec['path']}"):
            err.append(f"{lid}: spec {spec['path']} absent at {spec['commit']}")
        for p in rec["artifacts"]:
            if not artifact_exists(f"{rec['commit']}:{p}"):
                err.append(f"{lid}: artifact {p} absent at {rec['commit']}")
    elif rec["executed"]:
        err.append(f"{lid}: executed lineage without pre-execution spec")
    for key, want in (("observations", "OBSERVED"), ("derived", "DERIVED"), ("inferred", "INFERRED")):
        for o in rec[key]:
            if o.get("evidence_class") not in classes:
                err.append(f"{lid}: {key} entry with unknown evidence class {o.get('evidence_class')}")
            elif o["evidence_class"] != want:
                err.append(f"{lid}: {key} entry labelled {o['evidence_class']} (expected {want})")
            if want == "OBSERVED" and not (o.get("receipt") and artifact_exists(f"{rec['commit']}:{o['receipt']}")):
                err.append(f"{lid}: OBSERVED claim without resolvable receipt: {o.get('claim')}")
            if o.get("kind") == "transient":
                err += [f"{lid}: {e}" for e in validate_estimator(o)]
    if rec["verdict"].get("status") not in labels:
        err.append(f"{lid}: verdict status {rec['verdict'].get('status')} not in vocabulary")
    q = rec.get("qualification")
    if q is None and rec.get("introduces_mechanism") and rec["executed"]:
        err.append(f"{lid}: executed mechanism lineage without configured/realized/executed/effective qualification")
    if q is not None:
        err += [f"{lid}: qualification missing stage {s}" for s in QUALIFICATION_STAGES if s not in q]
    return err


def validate_sealed_registry(reg: dict) -> list[str]:
    err = []
    arts = reg["artifacts"]
    batch = subprocess.run(["git", "cat-file", "--batch-check=%(objectname)"], cwd=ROOT, capture_output=True,
                           text=True, input="".join(f"{e['sealed_in']}:{e['path']}\n" for e in arts)).stdout.splitlines()
    if len(batch) != len(arts):
        return ["sealed registry: git cat-file batch failed"]
    for e, at in zip(arts, batch):
        if at.strip() != e["blob"]:
            err.append(f"sealed {e['path']}: registry blob != {e['sealed_in']}:{e['path']}")
    present = [e for e in reg["artifacts"] if (ROOT / e["path"]).is_file()]
    err += [f"sealed {e['path']}: missing from working tree" for e in reg["artifacts"] if e not in present]
    wt = subprocess.run(["git", "hash-object", "--stdin-paths"], cwd=ROOT, capture_output=True, text=True,
                        input="\n".join(e["path"] for e in present) + "\n").stdout.split()
    if len(wt) != len(present):
        return err + ["sealed registry: hash-object failed"]
    err += [f"sealed {e['path']}: working tree differs from sealed blob" for e, h in zip(present, wt) if h != e["blob"]]
    return err


def validate_skills(skills_dir: Path | None = None) -> list[str]:
    err = []
    skills_dir = skills_dir or ROOT / ".claude" / "skills"
    sha = re.compile(r"\b[0-9a-f]{7,40}\b")
    state_tokens = re.compile(r"V2_LOCAL_OPERATION_(PASS|FAIL|UNRESOLVED)|INTRINSIC_HETEROGENEITY_|low_sm\dp\d")
    for sk in sorted(p for p in skills_dir.glob("*") if p.is_dir()):
        f = sk / "SKILL.md"
        if not f.exists():
            err.append(f"{sk.name}: SKILL.md missing")
            continue
        text = f.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        if not m:
            err.append(f"{sk.name}: frontmatter missing")
            continue
        fm = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
        if fm.get("name", "").strip() != sk.name:
            err.append(f"{sk.name}: frontmatter name mismatch")
        if len(fm.get("description", "").strip()) < 40:
            err.append(f"{sk.name}: description too short to trigger")
        if len(text.splitlines()) > SKILL_MAX_LINES:
            err.append(f"{sk.name}: SKILL.md over {SKILL_MAX_LINES} lines")
        if sha.search(text) or state_tokens.search(text):
            err.append(f"{sk.name}: SKILL.md contains project state (commit id or verdict)")
        for ref in re.findall(r"\(references/([^)]+)\)", text):
            if not (sk / "references" / ref).exists():
                err.append(f"{sk.name}: missing reference {ref}")
    return err

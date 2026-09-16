"""Render the state block of docs/HANDOFF_CURRENT.md from the manifests.

The block between the markers is generated, so the handoff cannot drift from
manifests/current_state.json and manifests/todo.json:

    python -m jomission.harness.handoff          # rewrite the block in place
    python -m jomission.harness.handoff --check  # exit 1 if stale
"""

from __future__ import annotations

import sys

from jomission.harness.validate import ROOT, load

BEGIN = "<!-- generated:state:begin (python -m jomission.harness.handoff) -->"
END = "<!-- generated:state:end -->"
HANDOFF = ROOT / "docs" / "HANDOFF_CURRENT.md"


def render() -> str:
    s, todo = load("manifests/current_state.json"), load("manifests/todo.json")
    items = {i["id"]: i for i in todo["items"]}
    lineages = load("manifests/lineage_registry.json")["lineages"]
    head = s["scientific_head"]
    out = [BEGIN, "", "### Authority", "",
           f"- scientific head: main `{head['main']}`"
           + "".join(f"; unmerged `{b}` @ `{c}`" for b, c in head.get("unmerged_scientific_branches", {}).items()),
           f"- execution: {s['execution_authority']['engine']} {s['jaxfne_version']} "
           f"(`{s['execution_authority']['evidence']}`)",
           f"- program: {s['current_program']}",
           "", "### Current state", "",
           f"- gate `{s['current_gate']['id']}` = **{s['current_gate']['status']}** "
           f"(definition `{s['current_gate']['definition']}`)",
           f"- stopped because: {s['stopped_because']}", ""]
    for name, v in s["latest_verdict"].items():
        extra = ""
        if v.get("review_override"):
            extra = f" (evidence reads {v['review_override']['evidence_reads']}; held until {v['review_override']['lift']})"
        elif v.get("component_of"):
            extra = f" (component of {v['component_of']})"
        out.append(f"- {name}: **{v['status']}**{extra} — `{v['evidence']}`")
    out += ["", "### Latest evidence (observed, with receipts)", ""]
    for rec in lineages[-3:]:
        for o in rec["observations"]:
            out.append(f"- {rec['lineage_id']}: {o['claim']} — `{rec['commit']}:{o['receipt']}`")
    nxt = items[s["next_authorized_task"]]
    out += ["", "### Next authorized task", "",
            f"- `{nxt['id']}` ({nxt['status']}): {nxt['goal']}",
            f"- stop: {nxt['stop_condition']}"]
    after = [i for i in todo["items"] if i["status"] == "OPEN_AFTER_HARNESS" and i["id"] != nxt["id"]]
    if after:
        out.append("- then: " + "; ".join(f"`{i['id']}`: {i['goal']}" for i in after))
    out += ["", "### Locked gates", "", "- " + ", ".join(f"`{g}`" for g in s["locked_gates"]),
            "", "### Critical invariants", "",
            "- " + ", ".join(f"`{i}`" for i in s["active_invariants"])
            + " — definitions in `docs/project-sources/SUBSTRATE_PROGRAM.md`",
            "", "### Open uncertainties", ""]
    out += [f"- {u['item']} → {u['route']}" for u in s.get("open_uncertainties", [])]
    out += ["", END]
    return "\n".join(out)


def splice(text: str, block: str) -> str:
    i, j = text.index(BEGIN), text.index(END) + len(END)
    return text[:i] + block + text[j:]


def is_current() -> bool:
    text = HANDOFF.read_text(encoding="utf-8")
    return BEGIN in text and END in text and splice(text, render()) == text


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if is_current() else 1)
    text = HANDOFF.read_text(encoding="utf-8")
    HANDOFF.write_text(splice(text, render()), encoding="utf-8", newline="\n")

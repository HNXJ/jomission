"""Sort manifests/todo.json items by status without reformatting the file.

Order: the next authorized task first, then OPEN, OPEN_AFTER_HARNESS, BLOCKED, LOCKED, DONE, RETIRED,
SUPERSEDED, VOID; items keep their relative order inside a status. Item text is moved verbatim, so
the diff is a reordering only. Usage: python scripts/sort_todo.py [--check]
Exit 0 when the file is sorted (or was sorted), 1 under --check when it is not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "manifests/todo.json"
ORDER = ["OPEN", "OPEN_AFTER_HARNESS", "BLOCKED", "LOCKED", "DONE", "RETIRED", "SUPERSEDED", "VOID"]


def split_items(text: str):
    """Return (head, [(start, end, item)], tail) for the top-level "items" array, by raw spans."""
    key = text.index('"items"')
    pos = text.index("[", key) + 1
    dec = json.JSONDecoder()
    spans = []
    while True:
        while text[pos] in " \r\n\t,":
            pos += 1
        if text[pos] == "]":
            break
        item, end = dec.raw_decode(text, pos)
        spans.append((pos, end, item))
        pos = end
    return text[: spans[0][0]], spans, text[spans[-1][1]:]


def main(argv) -> int:
    raw = PATH.read_bytes()
    text = raw.decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    head, spans, tail = split_items(text)
    state = json.loads((ROOT / "manifests/current_state.json").read_text(encoding="utf-8"))
    nxt = state["next_authorized_task"]

    def rank(s):
        item = s[2]
        return (0 if item["id"] == nxt else 1, ORDER.index(item["status"]))

    new = sorted(spans, key=rank)  # stable
    if [s[2]["id"] for s in new] == [s[2]["id"] for s in spans]:
        print("todo already sorted")
        return 0
    if "--check" in argv:
        print("todo NOT sorted")
        return 1
    sep = text[spans[0][1] : spans[1][0]]  # separator text between two items, kept as is
    body = sep.join(text[s[0] : s[1]] for s in new)
    out = head + body + tail
    assert json.loads(out)["items"] and sorted(i["id"] for i in json.loads(out)["items"]) == sorted(s[2]["id"] for s in spans)
    PATH.write_bytes(out.encode("utf-8"))
    print(f"sorted {len(new)} items; first: {[s[2]['id'] for s in new[:4]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

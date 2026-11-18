"""OPEN-51: strict HLO comparison. Strips only metadata={...}, stack_frame_id and the module name; drops the
file/function/frame tables; renames %ids by first appearance (keeps operand identity and order). Everything else,
including attributes, backend_config, constants, shapes and layouts, is compared. Prints `identical True|False`.
Adapted from the independent reviewer's probe of 2026-11-18. Usage: python -B hlodiff_strict.py A.hlo.txt B.hlo.txt
"""
import difflib
import re
import sys
from pathlib import Path

ID = re.compile(r"%[A-Za-z_][\w\-.]*")


def canon(p):
    ids = {}

    def rn(m):
        return ids.setdefault(m.group(0), f"%v{len(ids)}")

    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*\d+ [\"{]", line):
            continue
        line = re.sub(r",? ?metadata=\{[^}]*\}", "", line)
        line = re.sub(r",? ?stack_frame_id=\d+", "", line)
        line = re.sub(r"^HloModule \S+", "HloModule X", line)
        out.append(ID.sub(rn, line.rstrip()))
    return out


a, b = canon(sys.argv[1]), canon(sys.argv[2])
d = [x for x in difflib.unified_diff(a, b, lineterm="", n=0) if x[:1] in "+-" and not x.startswith(("---", "+++"))]
print(f"lines {len(a)} vs {len(b)}; identical {a == b}; diff lines {len(d)}")
for x in d[:20]:
    print(x[:250])

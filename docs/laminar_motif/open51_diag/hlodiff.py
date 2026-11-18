"""OPEN-51 diagnosis: compare two HLO dumps on instructions only (metadata, stack frames, ids normalized)."""
import difflib
import re
import sys
from pathlib import Path


def instrs(p):
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if " = " not in line and not line.lstrip().startswith(("ROOT", "ENTRY", "%", "}")) and "{" not in line:
            continue
        if re.match(r"^\s*\d+ [\"{]", line):  # file/function/frame tables
            continue
        line = re.sub(r",? ?metadata=\{[^}]*\}", "", line)
        line = re.sub(r",? ?stack_frame_id=\d+", "", line)
        line = re.sub(r"^HloModule \S+", "HloModule X", line)
        line = re.sub(r"%([A-Za-z_][\w\-]*?)\.\d+(\.\d+)*", r"%\1", line)   # %add.873 -> %add
        line = re.sub(r"\b(param_\d+)\.\d+", r"\1", line)
        line = re.sub(r"\b(fused_computation|region|wrapped_\w+|fusion|body|cond)(\.\d+)+", r"\1", line)
        out.append(line.rstrip())
    return out


a, b = instrs(sys.argv[1]), instrs(sys.argv[2])
print(f"instruction lines {len(a)} vs {len(b)}; identical {a == b}")
d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if not l.startswith(("---", "+++"))]
print(f"diff lines {sum(1 for l in d if l[:1] in '+-')}")
for l in d:
    print(l[:220])

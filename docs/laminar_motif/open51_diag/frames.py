"""OPEN-51 diagnosis: extract (file, function, line) stack frames from an HLO dump; compare two dumps by function+line."""
import re
import sys
from pathlib import Path


def frames(p):
    t = Path(p).read_text(encoding="utf-8")
    files, funcs, locs = {}, {}, set()
    sec = None
    for line in t.splitlines():
        s = line.strip()
        if s.startswith("FileNames"):
            sec = "f"; continue
        if s.startswith("FunctionNames"):
            sec = "n"; continue
        if s.startswith("FileLocations"):
            sec = "l"; continue
        if s.startswith("StackFrames") or s.startswith("ENTRY") or s.startswith("%") or s.startswith("HloModule"):
            sec = None if not s.startswith("StackFrames") else "s"
            continue
        m = re.match(r'^(\d+) "(.*)"$', s)
        if m and sec == "f":
            files[m[1]] = m[2].replace("\\\\", "/").split("/")[-1]
        elif m and sec == "n":
            funcs[m[1]] = m[2]
        m = re.match(r"^\d+ \{file_name_id=(\d+) function_name_id=(\d+) line=(\d+)", s)
        if m and sec == "l":
            locs.add((files.get(m[1], m[1]), funcs.get(m[2], m[2]), int(m[3])))
    return locs


a, b = frames(sys.argv[1]), frames(sys.argv[2])
ja = {x for x in a if x[0] in ("emitters_adex.py", "_model_simulate.py")}
jb = {x for x in b if x[0] in ("emitters_adex.py", "_model_simulate.py")}
print(f"jaxfne frames: {len(ja)} vs {len(jb)}; shared {len(ja & jb)}")
print("only in first :", sorted(ja - jb))
print("only in second:", sorted(jb - ja))

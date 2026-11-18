"""OPEN-51 read-only: compile candidate JaxFNE sources and compare marshalled code with the side-install .pyc bodies.

Usage: python -B _temp_o51_marshal.py SITE REPO REV [REV ...]   (REV 'export:<dir>' compares a plain directory)
"""
import marshal, subprocess, sys
from pathlib import Path

SITE = Path(sys.argv[1]); REPO = sys.argv[2]
print(f"python {sys.version.split()[0]}")
pyc = {}
for p in SITE.rglob("*.pyc"):
    b = p.read_bytes()
    rel = (p.parent.parent.relative_to(SITE) / (p.name.split(".")[0] + ".py")).as_posix()
    pyc[rel] = (b[16:], marshal.loads(b[16:]).co_filename)


def source(rev, rel):
    if rev.startswith("export:"):
        return (Path(rev[7:]) / rel).read_bytes()
    return subprocess.run(["git", "-C", REPO, "show", f"{rev}:{rel}"], capture_output=True, check=True).stdout


def eq(a, b):
    """Structural code equality; frozenset constants compared as sets (their marshal order follows the hash seed)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, frozenset):
        return a == b
    if isinstance(a, tuple):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    if hasattr(a, "co_code"):
        return all(eq(getattr(a, k), getattr(b, k)) for k in
                   ("co_code", "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_name",
                    "co_qualname", "co_flags", "co_argcount", "co_kwonlyargcount", "co_posonlyargcount",
                    "co_firstlineno", "co_linetable", "co_exceptiontable"))
    return a == b


for rev in sys.argv[3:]:
    same, struct_same, diff = 0, 0, []
    for rel, (body, fn) in sorted(pyc.items()):
        code = compile(source(rev, rel), fn, "exec", dont_inherit=True)
        if marshal.dumps(code) == body:
            same += 1
        elif eq(code, marshal.loads(body)):
            struct_same += 1
        else:
            diff.append(rel)
    print(f"{rev[:40]}: byte-identical {same}, structurally identical {struct_same}, differing {len(diff)} of "
          f"{len(pyc)}: {diff[:6]}{' ...' if len(diff) > 6 else ''}")

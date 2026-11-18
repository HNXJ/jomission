"""OPEN-51 diagnosis: for every jit_scan cache entry, report mtime, whether it holds a YNN fusion, and the jaxfne
source path embedded in it (identifies the environment that compiled it)."""
import re
import sys
from datetime import datetime
from pathlib import Path

from compression import zstd

rows = []
for d in sys.argv[1:]:
    for p in Path(d).glob("jit_scan-*-cache"):
        b = zstd.decompress(p.read_bytes())
        ynn = b"ynn_fusion" in b or b"ynn" in b
        src = re.search(rb"([A-Za-z]:\\[ -~]*?\\jaxfne)\\_model_simulate\.py", b)
        rows.append((datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), ynn,
                     src[1].decode() if src else "-", p.parent.name, p.name[9:21]))
rows.sort()
for r in rows:
    print(*r, sep=" | ")
from collections import Counter
c = Counter((r[0][:10], r[1], r[2]) for r in rows)
print("\nby day, ynn, jaxfne path:")
for k, v in sorted(c.items()):
    print(f"  {k[0]} ynn={k[1]} {k[2]}: {v}")

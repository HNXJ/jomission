"""OPEN-51 diagnosis (read-only): decode JAX persistent-cache entries and extract compile metadata strings.

Usage: python -B _temp_o51_cache.py ENTRY [ENTRY ...]
"""
import re
import sys
import zlib
from pathlib import Path


def decode(raw):
    for name, fn in (("zlib", zlib.decompress),):
        try:
            return name, fn(raw)
        except Exception:
            pass
    try:
        from compression import zstd
        return "zstd", zstd.decompress(raw)
    except Exception as e:
        return f"undecoded ({e.__class__.__name__})", raw


PAT = re.compile(rb"[ -~]{6,}")
KEEP = re.compile(rb"(?i)(jax|xla|llvm|cpu|avx|sse|fma|target|triple|version|thread|fast|denorm|precision|x86)")
for p in sys.argv[1:]:
    raw = Path(p).read_bytes()
    how, body = decode(raw)
    strs = [s for s in PAT.findall(body) if KEEP.search(s)]
    print(f"== {Path(p).name[:40]} raw {len(raw)} {how} {len(body)}")
    seen = set()
    for s in strs:
        t = s[:160]
        if t not in seen:
            seen.add(t)
            print("  ", t.decode("ascii", "replace"))
        if len(seen) > 60:
            break

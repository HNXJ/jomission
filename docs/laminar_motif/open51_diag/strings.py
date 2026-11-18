"""OPEN-51 diagnosis: version and flag strings in decompressed cache entries; diff the string sets of two entries."""
import re
import sys
from pathlib import Path

from compression import zstd


def strings(p):
    b = zstd.decompress(Path(p).read_bytes())
    return {s.decode("ascii") for s in re.findall(rb"[ -~]{5,}", b)}


a, b = strings(sys.argv[1]), strings(sys.argv[2])
ver = re.compile(r"(?i)(\b0\.\d+\.\d+|jaxlib|xla_|version|llvm|fast|contract|fma|denormal|opt)")
print("version/flag-like strings only in first:")
for s in sorted(x for x in a - b if ver.search(x))[:40]:
    print("  ", s[:200])
print("only in second:")
for s in sorted(x for x in b - a if ver.search(x))[:40]:
    print("  ", s[:200])
print("shared, version-like:")
for s in sorted(x for x in a & b if re.search(r"\b0\.\d+\.\d+|jaxlib|llvm", x))[:20]:
    print("  ", s[:200])

"""OPEN-51 diagnosis: compare the machine-code objects embedded in serialized XLA:CPU executables (cache-entry format).

Finds COFF/ELF object images by magic, hashes each object's .text-like sections when parseable, else the whole
object, and reports which objects are byte-identical between entries. Usage: python -B objcode.py NAME=ENTRY ...
"""
import hashlib
import re
import struct
import sys
from pathlib import Path

from compression import zstd


def coff_text_sections(b, off):
    """Return [(name, sha256)] of executable sections of the COFF object starting at off (x86-64, 0x8664)."""
    nsec = struct.unpack_from("<H", b, off + 2)[0]
    opt = struct.unpack_from("<H", b, off + 16)[0]
    out = []
    for i in range(nsec):
        h = off + 20 + opt + 40 * i
        name = b[h:h + 8].rstrip(b"\0").decode("ascii", "replace")
        size, ptr = struct.unpack_from("<II", b, h + 16)
        flags = struct.unpack_from("<I", b, h + 36)[0]
        if flags & 0x20000000 and size:  # IMAGE_SCN_MEM_EXECUTE
            out.append((name, hashlib.sha256(b[off + ptr:off + ptr + size]).hexdigest()[:16], size))
    return out


res = {}
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    b = zstd.decompress(Path(path).read_bytes())[4:]
    elf = [m.start() for m in re.finditer(rb"\x7fELF", b)]
    coff = [m.start() for m in re.finditer(rb"\x64\x86..", b, re.S)
            if 0 < struct.unpack_from("<H", b, m.start() + 2)[0] < 200
            and struct.unpack_from("<H", b, m.start() + 16)[0] == 0]
    secs = []
    for o in coff:
        try:
            secs += coff_text_sections(b, o)
        except struct.error:
            pass
    res[name] = sorted(set(secs))
    print(f"{name}: {len(b)} bytes; ELF magics {len(elf)}; COFF headers {len(coff)}; exec sections {len(res[name])}")
names = list(res)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, c = set(res[names[i]]), set(res[names[j]])
        print(f"{names[i]} vs {names[j]}: exec sections identical {len(a & c)}/{len(a)} vs {len(c)}; "
              f"all equal {a == c}")
        if len(a & c) >= len(a) - 3:
            print("   only first:", sorted(a - c), "\n   only second:", sorted(c - a))

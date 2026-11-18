"""OPEN-51 diagnosis (read-only, no execution): deserialize JAX cache executables and dump optimized HLO.

Usage: python -B _temp_o51_hlo.py OUT_DIR NAME=ENTRY [NAME=ENTRY ...]
Writes OUT_DIR/NAME.hlo.txt (raw) and NAME.norm.txt (metadata stripped) and prints a summary.
"""
import re
import sys
from pathlib import Path

import jax
from compression import zstd
from jax._src import xla_bridge

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
backend = xla_bridge.get_backend("cpu")
devs = jax.devices("cpu")[:1]
print("jax", jax.__version__, "| jaxlib", jax.lib.__version__ if hasattr(jax.lib, "__version__") else "?")
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    body = zstd.decompress(Path(path).read_bytes())
    ser, ctime = body[4:], int.from_bytes(body[:4], "big")
    try:
        try:
            exe = backend.deserialize_executable(ser, devs, None)
        except TypeError:
            exe = backend.deserialize_executable(ser, None)
        mods = exe.hlo_modules()
        text = "\n".join(m.to_string() for m in mods)
    except Exception as e:
        print(f"{name}: compile_time {ctime}; DESERIALIZE FAILED: {e.__class__.__name__}: {str(e)[:300]}")
        continue
    (out / f"{name}.hlo.txt").write_text(text, encoding="utf-8")
    norm = re.sub(r', metadata=\{[^}]*\}', "", text)
    norm = re.sub(r'(?m)^HloModule [^,]*', "HloModule X", norm)
    (out / f"{name}.norm.txt").write_text(norm, encoding="utf-8")
    print(f"{name}: compile_time {ctime}; modules {len(mods)}; hlo lines {text.count(chr(10))}; norm chars {len(norm)}")

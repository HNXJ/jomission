"""Finiteness, key and shape check of the scored R2 npz outputs. Read-only. Usage: script.py DIR"""
import hashlib
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
for s in (7, 8, 9):
    p = D / f"r2_p0_n4000_s{s}_T12.npz"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    z = np.load(p, allow_pickle=False)
    print(f"seed {s} sha256 {h}")
    print(f"  keys {sorted(z.files)}")
    n_nonfinite = 0
    n_checked = 0
    for k in z.files:
        a = z[k]
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating):
            n_checked += 1
            bad = int((~np.isfinite(a)).sum())
            n_nonfinite += bad
            if bad:
                print(f"  NONFINITE {k} shape {a.shape} count {bad}")
        print(f"  {k}: dtype {a.dtype} shape {a.shape}")
    print(f"  float arrays checked {n_checked}, nonfinite values {n_nonfinite}")
    if "counts" in z.files:
        c = z["counts"]
        print(f"  counts total {int(np.asarray(c).sum())}")

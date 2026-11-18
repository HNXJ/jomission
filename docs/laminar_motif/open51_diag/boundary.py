"""OPEN-51 diagnosis (existing outputs only): divergence boundary and growth of the count difference."""
import sys

import numpy as np

new = np.load(sys.argv[1]).astype(np.int64)
orig = np.load(sys.argv[2]).astype(np.int64)
rast = np.load(sys.argv[3])
d = new - orig
bad = np.nonzero(np.any(d != 0, axis=1))[0]
f = int(bad[0])
print(f"first differing step {f} ({f * 0.1:.1f} ms); pops differing there {np.nonzero(d[f])[0].tolist()}; diff {d[f].tolist()}")
for s in range(f - 3, f + 12):
    print(f"  step {s}: new {new[s].tolist()} orig {orig[s].tolist()}")
step, cell = rast["step"], rast["cell"]
print("cells spiking in R1 at the first differing step:", cell[step == f].tolist())
print("differing steps within +k steps of the first:")
for k in (1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000):
    print(f"  {k:5d} steps ({k * 0.1:6.1f} ms): {int(np.sum(np.any(d[f:f + k] != 0, axis=1)))} differ")
w = 1000  # 100 ms windows
for i in range(0, 20):
    a, b = f + i * w, f + (i + 1) * w
    if b > len(d):
        break
    print(f"  window {a * 0.1:7.1f}-{b * 0.1:7.1f} ms: frac steps differ {np.mean(np.any(d[a:b] != 0, axis=1)):.3f}, "
          f"sum|diff| {int(np.abs(d[a:b]).sum())}")

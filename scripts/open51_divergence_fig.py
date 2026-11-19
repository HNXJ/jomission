"""OPEN-51 figure from verified outputs only (no simulation): per-step E/PV/SOM count difference from the original
E0 seed-1 counts, for the jax 0.10.1 re-run (R1, NOT REPRODUCED) and the jax 0.10.2 A1 re-run (R1 under A1).

Usage: python -B open51_divergence_fig.py ORIGINAL_COUNTS R1_0101_COUNTS A1_R1_COUNTS OUT_PNG
Writes OUT_PNG and OUT_PNG with suffix .json (the plotted bin values and the inputs' array sha256).
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT_MS, BIN = 0.1, 100  # 100 steps = 10 ms bins
POPS = ("E", "PV", "SOM")


def sha(a):
    return hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()


orig, r101, a1 = (np.load(p) for p in sys.argv[1:4])
out = Path(sys.argv[4])
assert orig.shape == r101.shape == a1.shape == (120000, 3), "inputs must be (120000, 3) counts"
d101 = np.abs(r101.astype(np.int64) - orig)
d102 = np.abs(a1.astype(np.int64) - orig)
assert d101.sum() > 0, "known positive: the 0.10.1 re-run must differ"
nb = orig.shape[0] // BIN
t = (np.arange(nb) + 0.5) * BIN * DT_MS / 1000
b101 = d101[: nb * BIN].reshape(nb, BIN, 3).sum(1)
b102 = d102[: nb * BIN].reshape(nb, BIN, 3).sum(1)
first = int(np.nonzero(np.any(d101 != 0, axis=1))[0][0])

fig, ax = plt.subplots(2, 1, figsize=(6.8, 4.2), dpi=150, sharex=False, constrained_layout=True)
for i, (p, c) in enumerate(zip(POPS, ("#2b6cb0", "#c53030", "#38a169"))):
    ax[0].plot(t, b101[:, i], lw=0.7, color=c, label=f"{p}, jax 0.10.1 (R1)")
ax[0].plot(t, b102.sum(1), lw=1.2, color="k", ls="--", label="all, jax 0.10.2 (A1 R1)")
ax[0].set_ylabel("|count diff| per 10 ms")
ax[0].set_xlabel("time (s)")
ax[0].legend(fontsize=7, frameon=False, ncol=2)
ax[0].set_title("E0 seed 1: spike-count difference from the original", fontsize=9)
z = slice(first - 50, first + 250)
ts = np.arange(orig.shape[0])[z] * DT_MS
ax[1].step(ts, (r101.astype(np.int64) - orig)[z, 0], where="mid", lw=0.8, color="#2b6cb0", label="E, 0.10.1 minus original")
ax[1].step(ts, (a1.astype(np.int64) - orig)[z, 0], where="mid", lw=1.0, color="k", ls="--", label="E, 0.10.2 minus original")
ax[1].axvline(first * DT_MS, color="0.5", lw=0.6)
ax[1].text(first * DT_MS, ax[1].get_ylim()[1] * 0.8, f" first difference {first * DT_MS:.1f} ms", fontsize=7)
ax[1].set_xlabel("time (ms)")
ax[1].set_ylabel("E count diff per step")
ax[1].legend(fontsize=7, frameon=False)
fig.savefig(out)
meta = {"inputs_array_sha256": {"original": sha(orig), "r1_jax0101": sha(r101), "a1_r1_jax0102": sha(a1)},
        "first_differing_step_0101": first, "total_abs_diff_0101": int(d101.sum()), "total_abs_diff_0102": int(d102.sum()),
        "bin_steps": BIN, "dt_ms": DT_MS}
out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(meta))

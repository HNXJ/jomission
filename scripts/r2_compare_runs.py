"""Bitwise comparison of two run_e2_stack.py outputs (GEOMETRY_R2_SPEC_2026-11-25.md section 4, objects (a) and (b)).

(a) the .npz: equal key sets, then per array equal dtype, shape and tobytes() (the compressed container is not compared);
(b) the run JSON network fields in FIELDS, equal as parsed JSON.

    python scripts/r2_compare_runs.py A_PREFIX B_PREFIX     (paths without .npz/.json)

Prints one line per array and field, then EQUAL or DIFFERENT. Exit 0 equal, 1 different.
"""

from __future__ import annotations

import json
import sys

import numpy as np

FIELDS = ["sizes", "edges", "p_local", "p_inter", "rho", "depth_rel_L4", "tau_sst_e_ms", "g_columns",
          "g_reversal_mV", "g_tau_ms"]


def main(a: str, b: str) -> int:
    za, zb = np.load(a + ".npz"), np.load(b + ".npz")
    ok = sorted(za.files) == sorted(zb.files)
    print(f"keys {'equal' if ok else 'DIFFER'} ({len(za.files)} vs {len(zb.files)})")
    for k in sorted(set(za.files) & set(zb.files)):
        x, y = za[k], zb[k]
        eq = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        ok &= eq
        print(f"npz {k}: {'equal' if eq else 'DIFFER'} {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
    with open(a + ".json", encoding="utf-8") as fa, open(b + ".json", encoding="utf-8") as fb:
        ja, jb = json.load(fa), json.load(fb)
    for f in FIELDS:
        eq = f in ja and f in jb and ja[f] == jb[f]
        ok &= eq
        print(f"json {f}: {'equal' if eq else 'DIFFER'}")
    spikes = int(za["counts"].sum()) if "counts" in za.files else None
    print(f"spikes_a {spikes} devices {ja.get('jax_devices')} {jb.get('jax_devices')} "
          f"jaxfne {ja.get('jaxfne_version')} {jb.get('jaxfne_version')} "
          f"edge_delivery {ja.get('edge_delivery')} {jb.get('edge_delivery')}")
    print("EQUAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))

"""Edge-identity receipt for the jaxfne engine migration.

Usage:
  python scripts/jaxfne_edge_identity.py dump OUT.npz
      Build the canonical model (build_jomission_model, seed 0) under the engine
      on sys.path and save every EdgeList array field to OUT.npz. Also prints
      the canonical config_hash.
  python scripts/jaxfne_edge_identity.py compare A.npz B.npz
      Compare two dumps field by field (exact equality). Exit 1 on any mismatch.

Baseline: run `dump` on the pre-migration commit (d3eaa4f) with that tree first on
PYTHONPATH, and the migrated tree second. Keep the working directory out of sys.path
for the baseline, otherwise the editable install of the working tree is imported.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np


def dump(out: str) -> None:
    import jaxfne
    from jomission.network.builder import build_jomission_model
    from jaxfne.io import config_hash  # noqa: F401  (engine identity)
    import jomission.network.builder as b

    m = build_jomission_model(seed=0)
    el = m.params["edge_list"]
    arrays = {}
    for f in dataclasses.fields(el):
        v = getattr(el, f.name)
        arrays[f.name] = np.asarray(v) if hasattr(v, "shape") else np.asarray(str(v))
    np.savez(out, **arrays)
    print("engine", getattr(jaxfne, "__version__", "?"), "n_edges", int(el.pre.shape[0]))
    print("jomission", b.__file__)


def compare(a: str, b: str) -> int:
    x, y = np.load(a), np.load(b)
    if sorted(x.files) != sorted(y.files):
        print("field sets differ")
        return 1
    bad = [k for k in sorted(x.files) if x[k].shape != y[k].shape or not np.array_equal(x[k], y[k])]
    print(f"fields compared {len(x.files)}; mismatched {bad or 'none'}; n_edges {x['pre'].shape[0]} vs {y['pre'].shape[0]}")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)

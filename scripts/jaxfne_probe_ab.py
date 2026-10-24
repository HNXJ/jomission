"""Cross-engine probe comparison for the JaxFNE 0.5.7 migration (GEN2-PROBE check).

A declared compatibility check, not a scientific result: one short segment of the Gen2
model, no result file written. It compares the realized field parameters and the probe
arrays between two engine versions and/or two code states, so an engine migration can be
shown not to change the observables.

    python scripts/jaxfne_probe_ab.py --code-root <checkout> --out <path>
    python scripts/jaxfne_probe_ab.py --compare <a.json> <b.json>

--code-root selects which checkout's `jomission` package is imported (the editable
install is ignored). Run it with the interpreter pinned to each engine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def _sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def _probe_payload(sig) -> dict:
    """Every public array on the simulation result, hashed."""
    out = {}
    for name in sorted(dir(sig)):
        if name.startswith("_"):
            continue
        value = getattr(sig, name)
        if isinstance(value, np.ndarray) or hasattr(value, "shape"):
            try:
                arr = np.asarray(value)
            except Exception:  # noqa: BLE001 - not an array-like
                continue
            if arr.dtype.kind in "fiu":
                out[name] = {"sha256": _sha(arr), "shape": list(arr.shape), "dtype": str(arr.dtype)}
        field = getattr(value, "field", None)
        if field is not None and not isinstance(value, np.ndarray):
            for sub in sorted(dir(field)):
                if sub.startswith("_"):
                    continue
                sub_value = getattr(field, sub)
                if hasattr(sub_value, "shape"):
                    try:
                        arr = np.asarray(sub_value)
                    except Exception:  # noqa: BLE001
                        continue
                    if arr.dtype.kind in "fiu":
                        out[f"field.{sub}"] = {
                            "sha256": _sha(arr),
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--code-root", default=str(HERE), help="checkout whose jomission package is used"
    )
    ap.add_argument("--out", help="write the probe payload JSON here")
    ap.add_argument("--duration-ms", type=float, default=20.0)
    ap.add_argument("--dt-ms", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument(
        "--npz", help="also write the raw probe arrays here (for element-wise comparison)"
    )
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        b = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        unequal = []
        for section in ("probes", "field_metadata"):
            left = a.get(section) or {}
            right = b.get(section) or {}
            for key in sorted(set(left) | set(right)):
                lv, rv = left.get(key), right.get(key)
                same = lv == rv
                if section == "probes" and lv and rv:
                    same = lv.get("sha256") == rv.get("sha256")
                if not same:
                    unequal.append(f"{section}.{key}")
        print(
            json.dumps(
                {
                    "a": {"jaxfne": a.get("jaxfne_version"), "code_root": a.get("code_root")},
                    "b": {"jaxfne": b.get("jaxfne_version"), "code_root": b.get("code_root")},
                    "unequal": unequal,
                },
                indent=2,
            )
        )
        print("UNEQUAL:", ", ".join(unequal) if unequal else "none")
        return 0

    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))
    for name in [m for m in list(sys.modules) if m == "jomission" or m.startswith("jomission.")]:
        del sys.modules[name]

    import jaxfne as jtfne

    from jomission.network.builder import build_jomission_model
    from jomission.simulation.runtime import simulation_for_model
    from jaxfne.io import config_hash

    model = build_jomission_model(n_per_area=100, seed=args.seed, dt_ms=args.dt_ms)
    sim = simulation_for_model(
        model, duration_ms=args.duration_ms, dt_ms=args.dt_ms, seed=args.seed
    )
    sig = jtfne.simulate(model, sim)

    payload = {
        "jaxfne_version": importlib.metadata.version("jaxfne"),
        "code_root": str(code_root),
        "duration_ms": args.duration_ms,
        "dt_ms": args.dt_ms,
        "seed": args.seed,
        "config_hash": config_hash(model.cfg),
        "field_metadata": {
            k: v
            for k, v in (model.static or {}).items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        },
        "probes": _probe_payload(sig),
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    if args.npz:
        raw = {}
        for name in sorted(dir(sig)):
            if name.startswith("_"):
                continue
            value = getattr(sig, name)
            if hasattr(value, "shape"):
                try:
                    arr = np.asarray(value)
                except Exception:  # noqa: BLE001
                    continue
                if arr.dtype.kind in "fiu":
                    raw[name] = arr
        np.savez_compressed(args.npz, **raw)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

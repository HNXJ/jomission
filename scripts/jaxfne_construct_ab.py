"""Construct-level invariants for the JaxFNE 0.5.7 migration candidate.

Read-only with respect to the repository: models are built in memory and one JSON is
written to --out. No simulation, no result file, no manifest.

    python scripts/jaxfne_construct_ab.py --out <path> [--model tfne|gen2|both]
    python scripts/jaxfne_construct_ab.py --compare <a.json> <b.json>

The compare mode reports, per invariant, whether the two engine runs agree, so an
engine migration can be shown not to change the built graph without running a
simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Invariants compared across engines. Each is a value or a list of values; equality
#: is exact unless the invariant name ends in "_float".
EXACT_INVARIANTS = (
    "column_names",
    "metadata_areas",
    "n_neurons",
    "n_edges",
    "n_edges_after_enforce",
    "receptor_index_values",
    "tau_ms_values",
    "delay_steps_values",
    "storage_modes",
    "projection_edge_counts",
    "intra_retinal_edges_removed",
    "retinal_tonic_after",
)


def _sha256_array(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _stats(arr) -> dict:
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "sha256": _sha256_array(arr),
    }


def _hp_hash() -> str:
    import jaxfne.hdp_network as hdp

    hp = hdp.v1_pfc_aaab_hdp_params()
    return hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]


def _edge_invariants(model) -> dict:
    from jaxfne import emitters as em

    el = model.params["edge_list"]
    tau = em.resolve_edge_tau_ms(el, el.weight.dtype)
    delay = em.resolve_edge_delay_steps(el)
    receptor = em.resolve_receptor_index(el)
    return {
        "n_edges": int(np.asarray(el.pre).size),
        "weight": _stats(el.weight),
        "receptor_index_values": sorted({int(v) for v in np.asarray(receptor).ravel().tolist()}),
        "tau_ms_values": sorted({round(float(v), 9) for v in np.asarray(tau).ravel().tolist()}),
        "delay_steps_values": sorted({int(v) for v in np.asarray(delay).ravel().tolist()}),
        "storage_modes": {
            "tau_storage": getattr(el, "tau_storage", None),
            "delay_storage": getattr(el, "delay_storage", None),
            "weight_storage": getattr(el, "weight_storage", None),
            "receptor_index_storage": getattr(el, "receptor_index_storage", None),
            "uniform_delay_steps": int(getattr(el, "uniform_delay_steps", 0) or 0),
        },
    }


def _base_invariants(model) -> dict:
    from jaxfne.io import config_hash

    table = model.neuron_table()
    out = {
        "config_hash": config_hash(model.cfg),
        "hp_hash": _hp_hash(),
        "column_names": [str(c) for c in (model.cfg.metadata.get("column_names") or [])],
        "metadata_areas": [str(a) for a in (model.cfg.metadata.get("areas") or [])],
        "n_neurons": int(len(table)),
        "n_areas_in_table": int(len({str(r["area"]) for r in table})),
    }
    out.update(_edge_invariants(model))
    return out


def tfne_invariants() -> dict:
    """The SCI-EI-NATIVE-STATE-1 construction path, as the sealed arm builds it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import whole_system_ei_native_state_arm as arm  # noqa: E402

    model, _normal, _rf_decl, _tonic = arm.build()
    index, (area, layer, cls) = arm.realize.index_map(model)
    from jomission.tfne import realize  # noqa: E402

    counts, n_total = realize.realized_projections(model, area, layer, cls)
    out = _base_invariants(model)
    out["projection_edge_counts"] = {f"{s} -> {t}": int(n) for (s, t), n in sorted(counts.items())}
    out["n_edges"] = int(n_total)

    enforced, enforcement = arm.enforce(model, index)
    out["n_edges_after_enforce"] = int(np.asarray(enforced.params["edge_list"].pre).size)
    out["intra_retinal_edges_removed"] = int(enforcement["intra_retinal_edges_removed"])
    out["retinal_tonic_after"] = float(enforcement["retinal_tonic_after"])
    out["weight_after_enforce"] = _stats(enforced.params["edge_list"].weight)
    return out


def gen2_invariants() -> dict:
    """The Gen2 construction path (jomission.network.builder)."""
    from jomission.network.builder import build_jomission_model

    model = build_jomission_model(n_per_area=100, seed=0, dt_ms=0.1)
    out = _base_invariants(model)
    out["projection_edge_counts"] = None
    return out


def compare(a: dict, b: dict) -> dict:
    report = {}
    for model_key in sorted(set(a) | set(b)):
        if model_key in ("engine", "jaxfne_version", "python_version", "generated_by"):
            continue
        left = a.get(model_key) or {}
        right = b.get(model_key) or {}
        rows = {}
        for name in sorted(set(left) | set(right)):
            lv, rv = left.get(name), right.get(name)
            if name in EXACT_INVARIANTS:
                rows[name] = {"equal": lv == rv, "a": lv, "b": rv}
            elif name.endswith("weight") or name == "weight":
                rows[name] = {
                    "equal": lv == rv,
                    "a": lv,
                    "b": rv,
                    "max_abs_diff": (
                        float(
                            max(
                                abs(float(lv["min"]) - float(rv["min"])),
                                abs(float(lv["max"]) - float(rv["max"])),
                                abs(float(lv["mean"]) - float(rv["mean"])),
                            )
                        )
                        if lv and rv and "min" in lv and "min" in rv
                        else None
                    ),
                }
            else:
                rows[name] = {"equal": lv == rv, "a": lv, "b": rv}
        report[model_key] = rows
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write the invariants JSON here")
    ap.add_argument("--model", default="both", choices=["tfne", "gen2", "both"])
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        b = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        report = compare(a, b)
        print(json.dumps(report, indent=2, sort_keys=True))
        unequal = [
            f"{model}.{name}"
            for model, rows in report.items()
            for name, row in rows.items()
            if not row["equal"]
        ]
        print("UNEQUAL:", ", ".join(unequal) if unequal else "none")
        return 0

    payload = {
        "engine": "jaxfne",
        "jaxfne_version": importlib.metadata.version("jaxfne"),
        "python_version": sys.version.split()[0],
        "generated_by": "scripts/jaxfne_construct_ab.py",
    }
    if args.model in ("tfne", "both"):
        payload["tfne"] = tfne_invariants()
    if args.model in ("gen2", "both"):
        payload["gen2"] = gen2_invariants()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

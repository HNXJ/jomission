"""Render the VISUALIZATION_CONTRACT stages for a lineage.

V0 is rendered from the construction a sealed driver performs, imported rather than
reimplemented, so the schematic is a receipt of the circuit that actually runs and not of a
second description of it.

  python scripts/render_visualization_contract.py --stage V0 --lineage SCI-WS-OSC-1
  python scripts/render_visualization_contract.py --stage V1 --lineage SCI-WS-OSC-1 \
      --spikes run.npz --dt-ms 1.0

Writes under results/viz/<lineage_id>/ per the manifest's path convention.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jomission.harness import validate as V  # noqa: E402
from jomission.visualization import contract as VC  # noqa: E402

DEFAULT_DRIVER = "scripts/whole_system_causal_arm.py"
TITLE = "x : {V1²X[lat]} O[fffb] {V4²X[lat]} O[fffb] {FEF X PFC} : y"


def load_driver(path: str):
    spec = importlib.util.spec_from_file_location("_viz_driver", ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_from_driver(path: str):
    """Use the driver's own build() and enforce(), so V0 shows the circuit that runs."""
    drv = load_driver(path)
    from jomission.tfne import realize
    model, _normal, rf_decl, _tonic = drv.build()
    index, _ = realize.index_map(model)
    model, enforcement = drv.enforce(model, index)
    index, _ = realize.index_map(model)
    return model, index, rf_decl, enforcement, drv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["V0", "V1", "V2"])
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--driver", default=DEFAULT_DRIVER)
    ap.add_argument("--spikes", help="npz holding a (n_time, n_neurons) spike array")
    ap.add_argument("--spikes-key", default="spikes")
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--t0-ms", type=float, default=0.0)
    ap.add_argument("--window-ms", type=float, default=1000.0)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--out", help="override the contract path")
    args = ap.parse_args()

    contract = V.visualization_contract()
    paths = VC.stage_paths(args.lineage, contract)
    out = args.out or paths[args.stage]

    if args.stage == "V0":
        model, index, rf_decl, enforcement, _ = build_from_driver(args.driver)
        n_units = sum(v["n_units"] for v in rf_decl.values())
        info = VC.schematic(
            model, out, title=TITLE, theme=args.theme,
            x_label=f"Retina\n{len(rf_decl)} RFs / {n_units} edges",
            y_label="FEF.L6.E\nv(t)")
        info["driver"] = args.driver
        info["enforcement"] = enforcement
    else:
        if not args.spikes:
            ap.error("--spikes is required for V1 and V2")
        z = np.load(args.spikes)
        spikes = z[args.spikes_key]
        model, _index, _rf, _enf, _drv = build_from_driver(args.driver)
        stage = contract["stages"][args.stage]
        info = VC.raster(
            model, spikes, out, dt_ms=args.dt_ms, t0_ms=args.t0_ms,
            window_ms=args.window_ms, theme=args.theme,
            title=f"{args.lineage} — {stage['name']} — {stage['artifact']}",
            subtitle=args.subtitle or stage["purpose"])

    info["stage"] = args.stage
    info["lineage_id"] = args.lineage
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def build_from_driver(path: str, *, g=None, ee_cut=False, isolate=False):
    """Use the driver's own build() and enforce(), so V0 shows the circuit that runs.

    ``g`` and ``ee_cut`` carry the realization the arm actually compiles. Without them the
    schematic shows the inherited weights, which is a different circuit from the one that
    runs whenever an arm realizes an authority or ablates a projection.
    """
    drv = load_driver(path)
    from jomission.tfne import o_authority, realize
    model, _normal, rf_decl, _tonic = drv.build()
    index, (area, layer, cls) = realize.index_map(model)
    model, enforcement = drv.enforce(model, index)
    realization = {}
    if g is not None:
        model, authority = o_authority.realize_equal_g(model, index, area, layer, cls, g)
        realization["o_authority"] = {"g": g, "pass": authority["acceptance"]["pass"]}
    if ee_cut:
        model, ee = drv.cut_local_ee(model, area, cls)
        realization["ee_cut"] = ee["realized"]
    if isolate:
        model, iso = drv.isolate(model)
        realization["isolation"] = iso["realized"]
    index, _ = realize.index_map(model)
    return model, index, rf_decl, {**enforcement, **realization}, drv


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
    ap.add_argument("--g", type=float, default=None,
                    help="realize EQUAL_G at this authority before rendering")
    ap.add_argument("--ee-cut", action="store_true",
                    help="apply the driver's local E->E ablation before rendering")
    ap.add_argument("--isolate", action="store_true",
                    help="apply the driver's complete synaptic isolation before rendering")
    ap.add_argument("--arm", default="", help="arm name, appended to the title")
    args = ap.parse_args()

    contract = V.visualization_contract()
    paths = VC.stage_paths(args.lineage, contract)
    out = args.out or paths[args.stage]

    if args.stage == "V0":
        model, index, rf_decl, enforcement, _ = build_from_driver(
            args.driver, g=args.g, ee_cut=args.ee_cut, isolate=args.isolate)
        n_units = sum(v["n_units"] for v in rf_decl.values())
        # The x arrow is structural decoration in the portable renderer and stays solid
        # whatever the weights are, so the realized retinal weight goes in the label. Without
        # it an isolated construction shows a live input path it does not have.
        import numpy as _np
        _el = model.params["edge_list"]
        _w = _np.asarray(_el.weight)
        _pre = _np.asarray(_el.pre)
        _r0 = index["Retina.L4.E"][0] if "Retina.L4.E" in index else None
        _ret = _w[_pre >= _r0].sum() if _r0 is not None else float("nan")
        info = VC.schematic(
            model, out, title=TITLE + (f"   —   {args.arm}" if args.arm else ""),
            theme=args.theme,
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

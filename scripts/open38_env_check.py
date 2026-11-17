"""OPEN-38 stage-2 environment record and minimal dependency smoke (owner 2026-11-17), run by the driver before any run.

Usage: python -B scripts/open38_env_check.py OUT_JSON

Records the interpreter, the JaxFNE version, file and source identity, and the dependency versions. The smoke
imports jomission's CAMINOS stack (which imports jaxfne), computes one jax operation, and resolves every stage-2
layout. Each layout must give the spec §3 rho per unit, N_PV > 0 and N_SOM > 0 in every unit, and a realized ratio
within one cell. No simulation runs.
Exit 0 only if jaxfne loads 0.5.9 from the verified export and every smoke item passes.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import open38_smoke_check as sc  # noqa: E402
import open38_stage1_integrity as integ  # noqa: E402

ENGINE_COMMIT = "351f6048c2f80b32dcaf27d47e59c3c22507eef6"  # identity of the export: SMOKE_O38S_2026-11-16.md
DEPS = ("jax", "jaxlib", "numpy", "scipy", "matplotlib", "ml_dtypes", "opt_einsum")
RHO2 = {  # spec §3, S / M / D
    "v1": (1.688, 0.558, 1.584), "mt": (1.280, 0.599, 1.014), "mst": (1.220, 0.615, 1.103),
    "teo": (1.409, 0.695, 1.153), "chc": (1.345, 0.595, 1.178), "shape": (1.095, 0.485, 0.942),
}


def main(out: str) -> int:
    import jax.numpy as jnp
    import jaxfne
    from jomission.laminar import caminos_stack as cs

    jf = str(jaxfne.__file__).replace("\\", "/")
    engine = {"version": jaxfne.__version__, "file": jf, "export": sc.ENGINE_EXPORT, "source_commit": ENGINE_COMMIT,
              "ok": jaxfne.__version__ == sc.ENGINE_VERSION and jf.lower().startswith(sc.ENGINE_EXPORT.lower() + "/")}
    smoke = {"jax_op": {"ok": float(jnp.arange(4.0).sum()) == 6.0}}
    for arm, lay in integ.LAYOUT2.items():
        sizes, rho, _, _ = cs.layout(lay)
        want = dict(zip("SMD", RHO2[arm]))
        ratio = sc.ratio_ok(sizes, rho)
        ok = (all(abs(rho[u] - want[u]) < 1e-12 for u in "SMD")
              and all(sizes[u]["PV"] > 0 and sizes[u]["SOM"] > 0 for u in "SMD")
              and all(v["ok"] for v in ratio.values()))
        smoke[arm] = {"layout": lay, "rho": rho, "sizes": sizes, "ok": ok}
    rep = {"python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
           "flags": {"dont_write_bytecode": sys.dont_write_bytecode},
           "engine": engine, "deps": {p: md.version(p) for p in DEPS}, "smoke": smoke}
    rep["pass"] = engine["ok"] and all(v["ok"] for v in smoke.values())
    Path(out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"pass": rep["pass"], "jaxfne": engine["version"], "failed": [k for k, v in smoke.items() if not v["ok"]]}))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))

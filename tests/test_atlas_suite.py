"""Atlas suite: 6 fixed panels for any model, incl. N=1 (observed receipts)."""

import json
import pathlib

import jaxfne as J

from jomission.visualization.atlas_suite import PANELS, build_atlas

FIXED = [p[0] for p in PANELS]


def _build(kind: str, tmp: pathlib.Path):
    if kind == "single":
        cfg = J.suite2_single_neuron_config(seed=7, duration_ms=200.0, dt_ms=0.1)
    else:
        cfg = J.suite2_net1_config(seed=7, n=10, duration_ms=200.0, dt_ms=0.1)
    model = J.construct(cfg)
    out = tmp / kind
    manifest = build_atlas(model, out_dir=str(out), duration_ms=200.0, dt_ms=0.1)
    return model, out, manifest


def test_atlas_single_neuron(tmp_path):
    _, out, manifest = _build("single", tmp_path)
    assert manifest["n_neurons"] == 1
    for f in FIXED:
        p = out / f
        assert p.exists(), f"missing fixed panel {f}"
        assert p.stat().st_size > 0
    assert (out / "index.html").exists()
    assert (out / "manifest.json").exists()
    disk = json.loads((out / "manifest.json").read_text())
    assert disk["n_neurons"] == 1 and len(disk["panels"]) >= 6


def test_atlas_small_network(tmp_path):
    _, out, manifest = _build("net10", tmp_path)
    assert manifest["n_neurons"] == 10
    for f in FIXED:
        assert (out / f).exists(), f"missing fixed panel {f}"

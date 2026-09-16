# Atlas suite (`jaxfne.vis.atlas_suite`)

One standard 6-panel visualization atlas for **any** model, from a single
neuron to a multi-area hierarchy. Same panels, same provenance card, same
`manifest.json` — every time.

## Quickstart

```python
import jaxfne as J
from jaxfne.vis.atlas_suite import build_atlas

cfg = J.suite2_single_neuron_config(seed=7, duration_ms=500.0, dt_ms=0.1)
model = J.construct(cfg)
build_atlas(model, out_dir="outputs/atlas")  # runs a default simulation
```

Or bring your own signals:

```python
sim = J.Simulation(duration_ms=500.0, dt_ms=0.1, seed=0)
signals = J.simulate(model, sim)
build_atlas(model, signals, out_dir="outputs/atlas")
```

Open `outputs/atlas/index.html`.

## Minimal contract

`model.neuron_table()` / `model.edge_table()` / `model.summary()` plus
`signals.time_ms / V_m / spikes` (optional `sources` / `field`). No RF
lattice, plasticity log, or qualification file required.

## Panels

| File | Source | Evidence |
|------|--------|----------|
| `network_3d.html` | `vis.canonical.plot_network_3d` | OBSERVED |
| `connectivity.html` | `vis.canonical.plot_connectivity` | OBSERVED |
| `raster.html` | `vis.canonical.plot_raster` | OBSERVED |
| `traces.html` | `vis.canonical.plot_membrane_potentials` | OBSERVED |
| `spectral.html` | `vis.canonical.plot_psd` (+ spectrogram when long enough) | DERIVED |
| `operating_point.html` | rates/silence per cell type from spike counts | DERIVED |
| `field.html` (optional) | `vis.canonical.plot_lfp` / `plot_csd` | DERIVED |

No fixed panel is skipped: failures degrade to a labeled placeholder recorded
in `manifest.json`. Short runs get PSD-only spectral; a missing field skips
only the optional file.

## Statement discipline

Amplitudes are `relative_proxy_readout`; fields are proxy readouts, never
calibrated measurements. Cards are labeled OBSERVED (direct structure/arrays)
vs DERIVED (aggregates/transforms) per the relative-quantity grammar.

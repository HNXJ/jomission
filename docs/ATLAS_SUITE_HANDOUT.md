# Atlas Suite Handout — one atlas for any model (N ≥ 1)

The visualization atlas is a **standard suite**: the same 6 panels for every
model, from a single neuron to a multi-area hierarchy. If your model exposes
the jaxfne public surface, you already have everything the suite needs.

## 1. Minimal contract (you already satisfy this)

| Need | Public API | Used for |
|------|-----------|----------|
| neuron list | `model.neuron_table()` → `[{neuron_id, area, layer, cell_type, x, y, z}]` | network 3D, rates by cell type |
| edge list | `model.edge_table()` → `[{...}]` (may be `[]`) | connectivity |
| counts + hash | `model.summary()` → `{n_units, config_hash, ...}` | provenance, operating point |
| run | `signals.time_ms / V_m / spikes` `(steps × N)` | raster, traces, spectral, rates |
| optional | `signals.sources`, `signals.field` (may be `None`/empty) | field panel only |

No RF lattice, no plasticity logs, no qualification JSON required. Those are
jomission extras that map onto generic panels (see §4).

## 2. Quickstart (3 lines)

```python
import jaxfne as J
from jomission.visualization.atlas_suite import build_atlas  # jaxfne: from jaxfne.vis.atlas_suite import build_atlas

cfg = J.suite2_single_neuron_config(seed=7, duration_ms=500.0, dt_ms=0.1)
model = J.construct(cfg)
manifest = build_atlas(model, out_dir="docs/_static/atlas")  # simulates a default run
```

Bring your own run instead:

```python
sim = J.Simulation(duration_ms=500.0, dt_ms=0.1, seed=0)
signals = J.simulate(model, sim)
manifest = build_atlas(model, signals, out_dir="docs/_static/atlas")
```

Open `docs/_static/atlas/index.html` — 6 panels + `manifest.json`.

## 3. The 6 fixed panels

| File | Builder | Evidence | Minimum data |
|------|---------|----------|--------------|
| `network_3d.html` | `jaxfne.vis.canonical.plot_network_3d` | OBSERVED | `neuron_table()` (1 row is enough) |
| `connectivity.html` | `canonical.plot_connectivity` | OBSERVED | `edge_table()` (empty → empty-matrix card) |
| `raster.html` | `canonical.plot_raster` | OBSERVED | `signals.spikes` |
| `traces.html` | `canonical.plot_membrane_potentials` | OBSERVED | `signals.V_m` |
| `spectral.html` | `canonical.plot_psd` (+ spectrogram when long enough) | DERIVED | `signals.V_m`, ≥ ~200 ms preferred |
| `operating_point.html` | rates/silence per cell type from spike counts | DERIVED | `spikes` + `cell_type` column |
| `field.html` (optional) | `canonical.plot_lfp`/`plot_csd` | DERIVED | non-empty `signals.field` only |

Rules: **no fixed panel is ever skipped.** Failures degrade to a labeled
placeholder card (reason recorded in `manifest.json`). Short runs get PSD-only
spectral. Missing field skips only the optional file.

## 4. Mapping jomission extras → generic panels

| Jomission gallery figure | Generic atlas equivalent | Extra needed |
|--------------------------|--------------------------|--------------|
| network_3d | panel 1 (direct) | — |
| visual field / RF tiling | panel 2 area×layer×cell table; pass RF dict for the blob overlay | optional RF operator |
| raster_population | panels 3 + 4 (direct) | — |
| spectral_response | panel 5 (direct) | — |
| plasticity_trajectory | panel 6 + objective history if tuned | optional `TuneResult` |
| b1_b2_b3_dashboard | panel 6 (rates/silence/rheobase gaps) | optional thresholds dict |

## 5. Statement discipline

- Amplitude is always `relative_proxy_readout`; field panels are proxy_readout.
- OBSERVED = plotted directly from realized structure / simulated arrays.
- DERIVED = aggregates/transforms (PSD, rates, matrices). Labeled on every card.
- Every file carries `config_hash, N, edges, steps, dt, jaxfne version`.

## 6. Standard-suite proposal for jaxfne (drop-in ready)

`docs/jaxfne_atlas_suite_dropin/` contains verbatim-ready files:

- `atlas_suite.py` → copy to `jaxfne/jaxfne/vis/atlas_suite.py`; add
  `from .atlas_suite import build_atlas, PANELS` to `jaxfne/vis/__init__.py`.
- `atlas_suite_docs.md` → copy to `jaxfne/docs/guides/atlas_suite.md`.
- `mkdocs_nav_snippet.yml` → merge under `Guides:` in `jaxfne/mkdocs.yml`:
  `- Atlas suite: guides/atlas_suite.md`.

No new dependencies (plotly + numpy only, both already `viz` extras).

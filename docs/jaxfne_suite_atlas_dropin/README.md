# jaxfne_suite_atlas

An interactive Plotly atlas for any jaxfne circuit. One self-contained file, no dependency
beyond `numpy` and `plotly`; `jaxfne` is imported lazily and only when `signals` is not supplied.

```python
from jaxfne_suite_atlas import build_suite_atlas

build_suite_atlas(model, out_dir="atlas")                    # simulates if signals is None
build_suite_atlas(model, signals, out_dir="atlas",           # or pass a run you already have
                  H_trace=H, w_trace=w, dt_ms=0.1)
```

Writes `index.html`, `manifest.json`, and one standalone HTML per panel.

## Panels

| key | panel | evidence | reads |
|---|---|---|---|
| `network_3d` | 3D explorer, areas along x, layers in depth, edges by projection type | OBSERVED | `neuron_table()`, `edge_list` |
| `raster` | spike raster ordered area → layer → class, over per-class population rate | OBSERVED | `signals.spikes` |
| `membrane` | sampled V(t) per class with pooled voltage occupancy | OBSERVED | `signals.V_m` |
| `field` | LFP and CSD proxies as depth × time heatmaps, plus stacked LFP by depth | DERIVED | `signals.field.lfp_proxy`, `.csd_proxy`, `.contact_depths` |
| `connectivity` | signed weight sums: area × area, layer × layer, class × class | OBSERVED | `edge_list.pre/post/weight` |
| `plasticity` | H per class with 5–95th band, and \|w\|/\|w₀\| quantiles | OBSERVED | `H_trace`, `w_trace` (optional) |

## Design rules it follows

**Nothing is configured that can be read.** Areas, layers and cell classes come from
`model.neuron_table()`. There is no area list, no layer order, no `area_x` dict. Layer ordering
is derived from the first digit in the name, so `L1 L2 L2/3 L23 L4 L5 L6` all sort superficial
to deep. Known names (`V1 V4 FEF PFC`, `E PV SST VIP`) keep fixed colors so figures stay
recognisable across circuits; unknown names draw from a palette in a stable order.

**A missing input is stated, never hidden.** A panel whose input is absent renders a labelled
placeholder saying which input was missing and is marked `UNAVAILABLE` in the index and
manifest. It is not dropped. An exception during build is caught, reported on the panel, and
marked `ERROR` — one broken panel never costs you the other five.

**Every panel carries a provenance card** naming the exact attribute it read, the array shapes,
and any estimator applied. Evidence badges distinguish `OBSERVED` (read from the model or the
run) from `DERIVED` (computed here — population rates, the field proxies).

**Captions say what the figure is not.** The connectivity cells are sums of *signed* weight, so
a balanced block reads near zero and that is not the same as an absent projection — the caption
says so. The field panel says the proxies are computed through `field.kernel_matrix` and are not
recorded potentials. The plasticity panel reports a flat trace as flat rather than as regulation.

**Decimated before it reaches the browser.** Traces and violins are subsampled, and the
decimation factor is printed on the provenance card. Shipping every sample is what makes a page
unopenable: 10,000 steps × 828 units is 8.3M points per violin and a 69 MB file, for a
distribution indistinguishable from a 20,000-point sample.

## Verified on

| circuit | areas | layers | units | edges | result |
|---|---|---|---|---|---|
| jomission whole-system TFNE | V1_1 V1_2 V4_1 V4_2 FEF PFC | L1–L6 | 1200 | 245,820 | 6/6 `AVAILABLE` |
| generic 3-column | A1 MT LIP | L2/3 L4 L5 | 360 | 42,840 | 5/6 `AVAILABLE`, `plasticity` correctly `UNAVAILABLE` (no H trace passed) |

## Known upstream issue this works around

`jaxfne.vis.build_atlas` with its default simulation raises
`ValueError: INVALID_TIME_GRID (non_uniform)` for any `duration_ms` above roughly 50 ms:
`jaxfne.simulate` accumulates `time_ms` in float32, so over 5000 steps the spacing drifts
(`diff min 0.099975586, max 0.1000061`, 14 distinct values) and `classify_dt_ms` rejects its own
output. This module never reads `signals.time_ms`; it derives time from `dt_ms` and the step
count, so it is unaffected. The underlying float32 accumulation is still worth fixing.

## Style source

The visual standard is the jomission Plotly reference gallery (`jomission/visualization/theme.py`,
commit `57d8682`): `#0d1117` paper, `#161b22` plot, Inter, and the provenance-card HTML wrapper.
That theme is inlined here so this file stands alone.

# Handout: jaxfne model-inspection packet

Proposed upstream addition to `jaxfne.vis`: a circuit schematic and a hierarchical
raster, so a user can see what they constructed and what it immediately does before
beginning analysis.

Drop-in files: `docs/jaxfne_vis_dropin/`.
Source of truth while unmerged: `jomission/visualization/jaxfne_vis.py`.

## Why

Jomission's WS-PROP-1 built a causal propagation estimator and ran it over six matched
13 s simulations. It returned exactly `0.000000` in all six areas — including the area
the stimulus drives directly — because the observable it was specified over is a 1 s
spike count, and that count is conserved at exactly 21.0 per excitatory cell in every
area and condition. The stimulus moved spike timing, not spike count. Twelve of the
driven area's groups visibly differed the whole time.

A raster shows that in one glance, with no estimator involved. The capability belongs
upstream because the failure mode is not specific to Jomission: any scalar gate can be
blind to structure the trajectories plainly show.

## API

```python
jaxfne.vis.network_hspice(model, *, x=None, y=None, theme="light", stages=None,
                          title=None, path=None, ...)
jaxfne.vis.network_raster(model, signals, *, window_ms=None, theme="light",
                          dt_ms=1.0, areas=None, path=None, ...)
```

Both return a dict describing the figure, suitable for a run record.

**Name collision.** `jaxfne.vis.raster` already exists as a public function
(`jaxfne.vis.rasters.raster`, signature `(signals, **kwargs)`). The proposed name
`raster(model, signals, ...)` would shadow it. This packet therefore ships
`network_raster`, pairing with `network_hspice`. Adopting the shorter name upstream is
a breaking change that needs a deprecation path and is not a decision this packet
makes.

## Design rule

> theme changes presentation only, never information or semantics.

`theme="dark"` is one argument, not a second renderer. Enforced, not documented:
`test_theme_changes_presentation_only` asserts the returned description is identical
across themes and that the rendered bytes differ.

## Scope boundary

| | |
|---|---|
| **JaxFNE** | visualization capability |
| **Jomission** | when visualization is mandatory |

The V0 → V1 → V2 → V3 contract stays in Jomission
(`manifests/visualization_contract.json`). Nothing in the packet mentions stages,
lineages or contracts.

## Acceptance — run and passing

| Requirement | Evidence |
|---|---|
| six-area cortical model | `case1_six_area_hspice_{light,dark}.png`, real construction, 246,844 edges, 26 block projections |
| unrelated three-area model, different layer names | `case2_three_area_*`, areas Cochlea/A1/Belt, layers supra/granular/infra, classes Exc/Inh/Mod |
| tiny / single-area model | `case3_single_area_*` (one area, two layers), `case3_n1_*` (N=1, zero projections) |
| `theme="light"` and `theme="dark"` | all 14 gallery figures, both themes |
| no hardcoded V1/V4/FEF/PFC | `test_module_hardcodes_no_area_or_layer_names` greps the source |
| no six-layer assumption | `test_layers_sort_naturally_not_lexically`, layers `top/mid/bot` and `L2/3`, `L5` |
| no TFNE dependency, no jomission import | `test_module_imports_nothing_from_jomission_or_tfne` parses the module's imports |
| empty projections | `test_hspice_renders_with_no_projections_at_all`, `test_a_model_with_no_projections_is_one_column` |
| deterministic rendering | `test_rendering_is_deterministic` compares PNG bytes across two calls |

23 tests, all passing (`jomission/tests/test_jaxfne_vis.py`).

Stage inference was checked against the six-area model **without** being told the
hierarchy: from the projection graph alone it recovers
`[V1_1, V1_2] -> [V4_1, V4_2] -> [FEF, PFC]`.

## Defects found by the acceptance cases, fixed before handoff

These are recorded because they are exactly what a Jomission-only test would have
missed:

1. Cell classes outside the palette all rendered the same grey. A per-name hash was
   tried and produced a collision (`Fast` and `Slow` shared a colour), silently merging
   two populations in the reader's eye. Colours are now assigned in one pass over the
   model's class list, so distinctness is structural.
2. Stage inference put a feedback-connected area in the input column. A reciprocal pair
   blocked depth propagation entirely. Replaced with majority-direction-by-edge-count
   plus longest path.
3. Layer names longer than `L1` were overdrawn by the class chips. The gutter is now
   sized from the longest layer name.
4. Both feedforward and feedback abbreviated to `FE` on arrow labels, making the two
   indistinguishable in text. Now `FF` / `FB` / `LAT`.

## Not done

- `build_atlas` already exists upstream and is unchanged by this packet. It has **not**
  been exercised against a TFNE-realized model; the earlier atlas drop-in was written
  against a different builder.
- The schematic crowds where many projections share one gap: twelve arrows cross
  between the V4 and frontal stages of the six-area model. Legible, not elegant.
- No interactive or Plotly variant. These are static PNG renderers.

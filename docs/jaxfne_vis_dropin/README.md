# jaxfne drop-in: model inspection (`network_hspice`, `network_raster`)

Verbatim-ready files for the jaxfne repo:

1. `network_inspect.py` → `jaxfne/jaxfne/vis/network_inspect.py`
2. Add to `jaxfne/jaxfne/vis/__init__.py`:
   `from .network_inspect import network_hspice, network_raster, describe, Theme, THEMES`
   (+ `__all__` entries).
3. `test_network_inspect.py` → `jaxfne/tests/test_network_inspect.py`
4. `model_inspection.md` → `jaxfne/docs/guides/model_inspection.md`
5. `mkdocs_nav_snippet.yml` → merge the one line under `Guides:` in `mkdocs.yml`
6. `gallery/` → `jaxfne/docs/_static/model_inspection/`

Source of truth while unmerged: `jomission/jomission/visualization/jaxfne_vis.py`
(handout: `jomission/docs/JAXFNE_VIS_HANDOUT.md`). Verified on a six-area cortical
model, a three-area model with unrelated layer and cell-class names, a single-area
model and an N=1 model, in both themes, by `jomission/tests/test_jaxfne_vis.py`.

## Name collision to resolve before merging

`jaxfne.vis.raster` is **already a public function** (`jaxfne.vis.rasters.raster`,
signature `(signals, **kwargs)`). The renderer here is therefore named
`network_raster`, pairing with `network_hspice`, rather than `raster`. If upstream
prefers `raster(model, signals, ...)` as the public name, that is a breaking change
to the existing function and needs a deprecation path — it is not a rename this
packet can make unilaterally.

## What it is

Two renderers answering the two questions a user has before any analysis:

| Call | Question |
|---|---|
| `network_hspice(model)` | what did I actually construct? |
| `network_raster(model, signals)` | what does it immediately do? |

Both take `theme="light" \| "dark" \| Theme`. The design rule is that **theme changes
presentation only, never information or semantics**, and it is enforced rather than
documented: every renderer returns a description of what it drew, and
`test_theme_changes_presentation_only` asserts that description is byte-identical
across themes while the rendered pixels differ.

## Deliberately not included

The Jomission visualization contract (V0 → V1 → V2 → V3, when the figures are
*mandatory*) stays in Jomission. JaxFNE supplies the capability; a downstream project
decides when it is required. Nothing in `network_inspect.py` refers to stages,
lineages or contracts.

## Genericity

No hardcoded area or layer names, no six-layer assumption, no TFNE dependency, no
jomission import — the last two are asserted by tests that parse the module's own
imports and grep its source.

- Areas, layers and cell classes are whatever `neuron_table()` declares.
- Layers sort naturally, so `L2` precedes `L10`.
- Cell classes outside the palette get distinct colours assigned in one pass over the
  model's class list, so two unnamed populations can never share a colour.
- Which receptor index means inhibition is a caller argument
  (`inhibitory_receptors=(1,)`), not an assumption.
- Stage order is inferred from the projection graph: for each area pair the direction
  carrying more edges is forward, an exact tie leaves both at the same stage, and the
  column is the longest path over the result. Pass `stages=` to assert an order the
  topology does not show.

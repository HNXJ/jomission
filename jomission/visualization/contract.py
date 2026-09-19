"""VISUALIZATION_CONTRACT: the jomission side of the four required figures.

The contract is manifests/visualization_contract.json. The *rendering* lives in
jomission.visualization.jaxfne_vis, which is portable and proposed for upstream jaxfne
(docs/JAXFNE_VIS_HANDOUT.md). This module holds only what is jomission's own: which
figures are mandatory, where they go, and the TFNE annotations the generic renderer takes
as free metadata.

    jaxfne  = visualization capability
    jomission = when visualization is mandatory

  V0  schematic(...)       construct -> block schematic, before simulation
  V1  raster(...)          the native regime, first 1000 ms, before interpretation
  V2  raster(...)          the same figure after the authorized work
  V3  the atlas, built by jomission.visualization.jaxfne_suite_atlas

There is deliberately no second renderer here. A figure drawn by code that drifted from
the upstream one would be a receipt for a circuit nobody runs.
"""

from __future__ import annotations

import pathlib
from typing import Any, Sequence

import numpy as np

from jomission.visualization import jaxfne_vis as _vis

# The TFNE layer identities behind each channel. The generic renderer reads direction from
# the projection graph and knows nothing of these; they are used only to annotate.
RELATION = {("L2", "L4"): "ff", ("L3", "L4"): "ff", ("L6", "L1"): "fb", ("L3", "L3"): "lat"}
TFNE_TITLE = "x : {V1²X[lat]} O[fffb] {V4²X[lat]} O[fffb] {FEF X PFC} : y"
CORTICAL_STAGES = [["V1_1", "V1_2"], ["V4_1", "V4_2"], ["FEF", "PFC"]]
CORTICAL_AREAS = [a for col in CORTICAL_STAGES for a in col]


def stage_paths(lineage_id: str, contract: dict) -> dict[str, str]:
    """The four contract paths for one lineage, under the manifest's path convention."""
    root = contract["path_convention"].replace("<lineage_id>", lineage_id)
    return {s: root.replace("<artifact>", contract["stages"][s]["artifact"])
            for s in contract["order"]}


def schematic(model: Any, out_path: str | pathlib.Path, *, title: str = TFNE_TITLE,
              x_label: str = "input", y_label: str = "output",
              stages: Sequence[Sequence[str]] | None = None, theme: str = "light",
              **kw) -> dict:
    """V0. Delegates to the portable renderer; supplies jomission's own annotations."""
    return _vis.network_hspice(model, x=x_label, y=y_label, title=title,
                               stages=stages if stages is not None else CORTICAL_STAGES,
                               theme=theme, path=out_path, **kw)


def raster(model: Any, spikes: np.ndarray, out_path: str | pathlib.Path, *, title: str,
           dt_ms: float = 1.0, t0_ms: float = 0.0, window_ms: float = 1000.0,
           areas: Sequence[str] | None = None, subtitle: str = "", theme: str = "light",
           **kw) -> dict:
    """V1 and V2. The same call for both, which is what makes them comparable."""
    return _vis.network_raster(model, spikes, window_ms=(t0_ms, t0_ms + window_ms),
                               dt_ms=dt_ms,
                               areas=list(areas) if areas is not None else CORTICAL_AREAS,
                               title=title, subtitle=subtitle, theme=theme, path=out_path,
                               **kw)

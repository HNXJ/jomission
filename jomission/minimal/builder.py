"""Build the minimal typed column: a Configuration and a JaxFNE Model for any neuron count N.

Connectivity comes from docs/minimal/connectivity.json (one rule per cell-type pair, each with
its citation). Neurons are Izhikevich through the JaxFNE emitter; HDP is declared with the
JaxFNE default parameters (jaxfne/hdp_network.py DEFAULT_HDP), not the detached homeostatic
configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jaxfne as jtfne
from jaxfne import Configuration, RuntimeConfig
from jaxfne.hdp_network import DEFAULT_HDP

REPO_ROOT = Path(__file__).resolve().parents[2]
CONNECTIVITY_PATH = REPO_ROOT / "docs" / "minimal" / "connectivity.json"
CELL_TYPES: tuple[str, ...] = ("E", "PV", "SST", "VIP")


def load_connectivity(path: Path = CONNECTIVITY_PATH) -> dict:
    """Read the connectivity table; the bytes are hashed into the model metadata."""
    return json.loads(Path(path).read_bytes())


def cell_counts(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Integer counts per cell type that sum to n. Inhibitory counts are rounded; E takes the rest."""
    if n < 2:
        raise ValueError(f"n must be at least 2; got {n}")
    inhibitory = {ct: round(n * fractions[ct]) for ct in ("PV", "SST", "VIP")}
    counts = {"E": n - sum(inhibitory.values()), **inhibitory}
    if counts["E"] < 1 or any(c < 1 for c in counts.values()):
        raise ValueError(f"n={n} leaves an empty cell type: {counts}")
    return {ct: counts[ct] for ct in CELL_TYPES}


def build_minimal_network(n: int = 100, seed: int = 0, connectivity: dict | None = None) -> Configuration:
    """Declarative Configuration for the minimal typed column with n neurons."""
    table = connectivity if connectivity is not None else load_connectivity()
    area, layer = table["area"], table["layer"]
    fractions = {ct: table["cell_types"][ct]["fraction"] for ct in CELL_TYPES}
    counts = cell_counts(n, fractions)

    cfg = Configuration().column(area, layers=[layer], n=n)
    cfg = cfg.area_layer_cell_types(area, {layer: {ct: counts[ct] / n for ct in CELL_TYPES}})
    for rule in table["rules"]:
        cfg = cfg.connections(
            name=rule["name"],
            source={"area": area, "layer": layer, "cell_type": rule["pre"]},
            target={"area": area, "layer": layer, "cell_type": rule["post"]},
            probability=rule["probability"],
            weight=rule["weight"],
            sign=rule["sign"],
        )
    cfg = cfg.emitter(family="izhikevich")
    cfg = cfg.hdp(**DEFAULT_HDP)
    cfg = cfg.field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",
        gauge="mean_zero",
    )
    cfg = cfg.probe(n_contacts=4)
    cfg = cfg.runtime(recurrent_backend="edge_list")
    cfg = cfg.update_metadata(
        model="minimal-typed-column",
        n_neurons=int(n),
        cell_counts=counts,
        seed=int(seed),
        connectivity_sha256=hashlib.sha256(Path(CONNECTIVITY_PATH).read_bytes()).hexdigest(),
        hdp_source="jaxfne.hdp_network.DEFAULT_HDP",
    )
    return cfg


def build_minimal_model(**kwargs) -> jtfne.Model:
    """Construct the minimal model. Keyword arguments as in build_minimal_network."""
    return jtfne.construct(build_minimal_network(**kwargs))


def minimal_runtime(model: jtfne.Model, recurrent_backend: str = "edge_list") -> RuntimeConfig:
    """RuntimeConfig that carries the model's HDP declaration into simulate()."""
    meta = model.cfg.metadata
    return RuntimeConfig(
        recurrent_backend=recurrent_backend,
        enable_hdp=bool(meta["enable_hdp"]),
        hdp_params=dict(meta["hdp_params"]),
    )

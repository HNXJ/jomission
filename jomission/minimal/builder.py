"""Build the minimal typed column: a Configuration and a JaxFNE Model for any neuron count N.

Connectivity comes from docs/minimal/connectivity.json (one rule per cell-type pair, each with
its citation). Neurons are Izhikevich through the JaxFNE emitter; HDP is declared with the
JaxFNE default parameters (jaxfne/hdp_network.py DEFAULT_HDP), not the detached homeostatic
configuration.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne import Configuration, RuntimeConfig
from jaxfne.hdp_network import DEFAULT_HDP

REPO_ROOT = Path(__file__).resolve().parents[2]
CONNECTIVITY_PATH = REPO_ROOT / "docs" / "minimal" / "connectivity.json"
CELL_TYPES: tuple[str, ...] = ("E", "PV", "SST", "VIP")
# Checklist A item 1: canonical SST adaptation a = 0.02. JaxFNE 0.5.8's SST preset is 0.05;
# the deviation is applied after construction (apply_sst_adaptation), not in JaxFNE.
SST_ADAPTATION_A: float = 0.02


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


def apply_sst_adaptation(model: jtfne.Model, a_value: float = SST_ADAPTATION_A) -> jtfne.Model:
    """Set the SST recovery time scale a on the built parameters (checklist A item 1).

    JaxFNE 0.5.8 reads Izhikevich values only from its global table, so this patches the built
    emitter parameters for SST neurons and leaves every other class at the table value.
    """
    params = model.params["emitter"]
    is_sst = jnp.asarray([label == "SST" for label in params.labels])
    a = jnp.where(is_sst, jnp.asarray(a_value, dtype=params.a.dtype), params.a)
    model.params["emitter"] = dataclasses.replace(params, a=a)
    return model


def build_minimal_model(**kwargs) -> jtfne.Model:
    """Construct the minimal model with SST a = SST_ADAPTATION_A. Keyword arguments as in build_minimal_network."""
    return apply_sst_adaptation(jtfne.construct(build_minimal_network(**kwargs)))


def minimal_runtime(model: jtfne.Model, recurrent_backend: str = "edge_list") -> RuntimeConfig:
    """RuntimeConfig that carries the model's HDP declaration into simulate()."""
    meta = model.cfg.metadata
    return RuntimeConfig(
        recurrent_backend=recurrent_backend,
        enable_hdp=bool(meta["enable_hdp"]),
        hdp_params=dict(meta["hdp_params"]),
    )

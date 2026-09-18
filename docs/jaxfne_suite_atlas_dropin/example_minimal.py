"""Smallest end-to-end example: build a circuit, simulate it, render the atlas."""

import jaxfne as J

from jaxfne_suite_atlas import build_suite_atlas

cfg = J.Configuration()
for area in ["A1", "MT", "LIP"]:
    cfg = cfg.add_column(name=area, layers=["L2/3", "L4", "L5"], n=120)
cfg = (cfg.runtime(seed=1, duration_ms=1000.0, dt_ms=0.1, dtype="float32")
          .set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
          .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
model = J.construct(cfg)

signals = J.simulate(model, J.Simulation(duration_ms=300.0, dt_ms=0.1, seed=1))
manifest = build_suite_atlas(model, signals, out_dir="atlas", dt_ms=0.1, seed=1,
                             title="jaxfne suite atlas - A1/MT/LIP")
for panel in manifest["panels"]:
    print(f"{panel['key']:<14} {panel['status']:<12} {panel['bytes']:>9,} bytes")

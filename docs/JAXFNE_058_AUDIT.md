# jaxfne 0.5.8 refusal audit (jomission migration)

Method: each refusal below was reproduced by building the canonical model
(`build_jomission_model`, seed 0) or the test that exercises it, under jaxfne 0.5.8.
Source lines are from the installed wheel. Rules are from the jaxfne repo at `11957af`.
No defect in jaxfne was found. No issue was filed.

| Declared argument (0.4.25) | 0.5.8 behaviour | Rule and source | jomission change |
|---|---|---|---|
| `inter_column_connectivity(delay_ms_or_status=8.0 / 12.0)` | refused unless `None` (`_config.py:1718`) | Stored-but-unconsumed parameters are defects (P-014, `artifacts/AGENTS.md:37`) | `None`; delay applied by `edge_list_with_delay_ms` (existing shim) |
| `inter_column_connectivity(cell_type_to_cell_type_map=identity)` | refused unless `None`; default is identity E,PV,SST,VIP | docstring (`_config.py`, `cell_type_to_cell_type_map`) | `None` |
| `connectivity(within_area="spatial_local")` | only `all_to_all_uniform_random` realized (`_config.py` `connectivity()`) | 0.4.25 never consumed `spatial_local` | `all_to_all_uniform_random`; locality in `update_metadata(within_locality=…)`, applied by existing pruning shim |
| `connectivity(spatial_sigma=…)` | refused; `connectivity()` reads only its listed keys | `_config.py` `connectivity()` docstring | moved to `within_locality` (see above) |
| `field(source_mode=…)` | refused; field keys are domain, conductivity, boundary, gauge, kind, solver | `_config.py` field keys; stored-but-unconsumed (P-014) | `field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")` |
| `areas` metadata differs from realized columns | refused (`_construct_core.py:573-575`) | metadata must equal realized columns | Retina declared in `areas` (arm script) |
| `Simulation.seed` ≠ `runtime.seed` (multi-trial) | refused (`_signals.py:137-141`) | one seed per simulation | `dataclasses.replace(runtime, seed=seed+i)` in `exposure.py` |
| `max_in_degree` rule without `mechanism=` | refused (`configuration_grammar.md:134-135`) | mechanism-aware path changes per-edge tau (`configuration_grammar.md:111-121`) | not used; no local rule declared |

Open note: jaxfne's own guide says local sparse connectivity is a `connections(max_in_degree=…)` rule with a mechanism, plus `connectivity(p_connect=0.0)`. jomission keeps the engine's uniform baseline plus its own pruning, which reproduces the 0.4.25 edge list exactly. This choice is recorded in `results/jaxfne_058_migration.json`.

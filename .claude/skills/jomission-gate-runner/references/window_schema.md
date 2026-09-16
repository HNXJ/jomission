# Window summary schema (input to `jomission.harness.gates.evaluate`)

One dict per analysis window, in window order:

| key | type | profile | meaning |
|---|---|---|---|
| `rates` | `{class: Hz}` | both | class-mean rate in the window |
| `isi_fraction_pooled` | float | historical | fraction of evaluable neurons (all classes) with ISI-CV in band |
| `isi_fraction` | `{class: float or None}` | prospective | same, per class; `None` when no neuron is evaluable (fails the check) |
| `rate_cv` | `{class: float}` | both | std/mean of per-neuron rates across the class, silent neurons included |
| `sync` | float | both | fraction of steps with more than half of E spiking |

`drift` is a separate `{class: float}`: (max - min) / mean of class rate across
the analysis windows.

Classes: `E`, `PV`, `SST`, `VIP`. Rates in Hz, CVs dimensionless.

Sealed V2.1-layout cell JSONs convert with `gates.windows_from_v21_result(res)`
(historical keys only). Class-resolved ISI for a sealed cell comes from its
class-ISI record (`per_window_class[w][class]["frac_in"]`).

Minimal prospective example:

```python
from jomission.harness import gates

windows = [{"rates": {"E": 8.1, "PV": 20.3, "SST": 6.0, "VIP": 12.2},
            "isi_fraction": {"E": 0.84, "PV": 0.97, "SST": 0.88, "VIP": 0.81},
            "rate_cv": {"E": 0.41}, "sync": 0.002}] * 3
checks = gates.evaluate("prospective_v2", windows, {"E": 0.05, "PV": 0.04, "SST": 0.1, "VIP": 0.07})
```

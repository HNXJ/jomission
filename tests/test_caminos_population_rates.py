"""Known-answer check of the population_rates call in caminos_analysis.analyze_run (OPEN-61).

The call must use the JaxFNE 0.5.8 signature population_rates(counts, dt, sizes, pops, discard_s):
dt in seconds, discard_s in seconds. Synthetic counts make every rate exact. The first 2 s carry a
sentinel count that would dominate any rate if the discard were wrong or in the wrong unit.
"""

import numpy as np

from jomission.laminar import caminos_analysis as an

DT = 1e-4  # s
DISCARD_STEPS = 20000  # 2 s
KEPT_STEPS = 30000  # 3 s
SIZES = {"E": 8000, "PV": 750, "SOM": 750}


def _counts():
    rng = np.random.default_rng(0)
    n = DISCARD_STEPS + KEPT_STEPS
    arr = np.empty((n, 3), dtype=np.int32)
    arr[:DISCARD_STEPS] = 1000  # sentinel: must be discarded
    arr[DISCARD_STEPS:, 0] = rng.integers(0, 5, KEPT_STEPS)  # E, varying so the spectral step is defined
    arr[DISCARD_STEPS:, 1] = 5  # PV
    arr[DISCARD_STEPS:, 2] = 3  # SOM
    return arr


def test_analyze_run_population_rates_known_answer():
    arr = _counts()
    out = an.analyze_run(arr, DT, SIZES)
    kept = arr[DISCARD_STEPS:]
    dur = KEPT_STEPS * DT  # 3.0 s, kept window in seconds
    expected = {p: kept[:, i].sum() / dur / SIZES[p] for i, p in enumerate(("E", "PV", "SOM"))}
    assert out["spikes_kept"] == {p: int(kept[:, i].sum()) for i, p in enumerate(("E", "PV", "SOM"))}
    assert out["spikes_kept"]["PV"] == 5 * KEPT_STEPS and out["spikes_kept"]["SOM"] == 3 * KEPT_STEPS
    for p, rate in expected.items():
        assert abs(out["rates_Hz"][p] - rate) < 1e-9 * rate, (p, out["rates_Hz"][p], rate)
    assert abs(out["rates_Hz"]["PV"] - 5 / DT / SIZES["PV"]) < 1e-9 * out["rates_Hz"]["PV"]  # 66.67 Hz

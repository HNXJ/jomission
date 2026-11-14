"""Numeric equivalence check for population_rates across JaxFNE 0.5.8 and 0.5.9.

Usage:
  poprates_equiv.py run <out.json>      # run with whichever jaxfne is first on sys.path
  poprates_equiv.py compare <a.json> <b.json>

The run mode records which jaxfne file was imported, so the JSON names its engine.
"""
import json
import sys
import warnings

import numpy as np

POPS = ("E", "PV", "SOM")
SIZES = {"E": 8000, "PV": 750, "SOM": 750}
DT = 1e-4          # s, same as the caminos run (dt = 0.0001)
DISCARD_S = 2.0    # s, same as caminos DISCARD_S
N_STEPS = 120000   # 12 s


def make_counts(seed):
    rng = np.random.default_rng(seed)
    # per-step spike counts per population, Poisson-like, int32 [steps, 3]
    lam = np.array([0.9, 0.3, 0.25])
    return rng.poisson(lam, size=(N_STEPS, len(POPS))).astype(np.int32)


def run(out_path):
    import jaxfne.analysis as an
    from jaxfne.analysis import population_rates

    rec = {"jaxfne_file": an.__file__, "cases": {}}
    for seed in (0, 1, 2):
        counts = make_counts(seed)
        case = {}
        # seconds form: the one the patch calls
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            case["sec"] = population_rates(counts, dt=DT, sizes=SIZES, pops=POPS, discard_s=DISCARD_S)
            case["sec_warnings"] = sorted({str(x.category.__name__) for x in w})
        # ms form: only exists on 0.5.9 (TypeError on 0.5.8)
        try:
            case["ms"] = population_rates(counts, sizes=SIZES, pops=POPS,
                                          dt_ms=DT * 1000.0, discard_ms=DISCARD_S * 1000.0)
        except TypeError as e:
            case["ms"] = "TypeError: " + str(e)
        # edge: discard longer than the run must raise on both engines
        try:
            population_rates(counts[:1000], dt=DT, sizes=SIZES, pops=POPS, discard_s=DISCARD_S)
            case["short_run"] = "no error"
        except ValueError as e:
            case["short_run"] = "ValueError"
        rec["cases"][str(seed)] = case
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    print("engine:", rec["jaxfne_file"])
    print("wrote:", out_path)


def compare(a_path, b_path):
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    print("A:", a["jaxfne_file"])
    print("B:", b["jaxfne_file"])
    n_checked = 0
    worst = 0.0
    for seed in sorted(a["cases"]):
        ca, cb = a["cases"][seed], b["cases"][seed]
        print(f"seed {seed}: A sec={ca['sec']}  warnings={ca['sec_warnings']}")
        print(f"seed {seed}: B sec={cb['sec']}  warnings={cb['sec_warnings']}")
        print(f"seed {seed}: B ms={cb['ms']}")
        print(f"seed {seed}: short_run A={ca['short_run']} B={cb['short_run']}")
        for pop in POPS:
            x, y = ca["sec"][pop], cb["sec"][pop]
            worst = max(worst, abs(x - y) / max(abs(x), 1e-300))
            n_checked += 1
        if isinstance(cb["ms"], dict):
            for pop in POPS:
                x, y = cb["sec"][pop], cb["ms"][pop]
                worst = max(worst, abs(x - y) / max(abs(x), 1e-300))
                n_checked += 1
    print(f"values compared: {n_checked}; worst relative difference: {worst:.3e}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)

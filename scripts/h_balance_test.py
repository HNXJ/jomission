"""Balance H rule vs baseline on the canonical 400-neuron network (spec: docs/designs/H_BALANCE_TEST_spec.md).

Usage:
  python scripts/h_balance_test.py control    # null wiring, determinism, engagement (200 ms)
  python scripts/h_balance_test.py run        # 2 conditions x 2 seeds x 2 arms (2000 ms)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import jaxfne as jt  # noqa: E402
from jaxfne import RuntimeConfig, Simulation  # noqa: E402

import agsdr_local_sweep as A  # noqa: E402
import h_active_test as T  # noqa: E402  (metrics, h_stats, build, CONDITIONS, PD, DT, DUR)
import jomission.dynamics.activity_balance as ab  # noqa: E402

SPEC = REPO / "docs/designs/H_BALANCE_TEST_spec.md"
OUT = REPO / "docs/receipts/h_balance_test"


def hdp_params(kappa: float | None = None) -> dict:
    rp = {} if kappa is None else {"kappa": kappa}
    return {"hdp_rule": ab.RULE_NAME, "hdp_rule_params": rp}


def run(model, seed: int, hdp: dict | None, dur: float = T.DUR):
    rt = (RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hdp)
          if hdp is not None else RuntimeConfig(recurrent_backend="edge_list"))
    sim = Simulation(duration_ms=dur, dt_ms=T.DT, seed=seed, runtime=rt,
                     poisson_drive=dict(T.PD, seed=seed + 7919))
    sig = jt.simulate(model, sim)
    spikes = np.asarray(sig.spikes)
    diag = model.last_hdp_diagnostics() if hdp is not None else None
    return spikes, diag


def b_stats(diag) -> dict:
    H = np.asarray(diag["H_final"], dtype=np.float64)
    b = np.asarray(diag["b_final"], dtype=np.float64)
    return {"b_mean": float(b.mean()), "b_sd": float(b.std()), "b_min": float(b.min()),
            "b_max": float(b.max()),
            "max_abs_b_minus_kI_Hm1": float(np.max(np.abs(b - ab.DEFAULT_PARAMS["kappa_I"] * (H - 1.0))))}


def control() -> int:
    m = T.build(T.BASE_TONIC)
    cells = np.array([r["cell_type"] for r in m.neuron_table()])
    b1, _ = run(m, 0, None, 200.0)
    b2, _ = run(m, 0, None, 200.0)
    nul, _ = run(m, 0, hdp_params(0.0), 200.0)
    full_b, _ = run(m, 0, None)
    full_h, diag = run(m, 0, hdp_params())
    rates = full_h.sum(axis=0) / (T.DUR / 1000.0)
    hs = T.h_stats(diag, rates, cells)
    res = {"baseline_deterministic": bool(np.array_equal(b1, b2)),
           "null_kappa0_equals_baseline": bool(np.array_equal(b1, nul)),
           "engaged_differs_from_baseline": bool(not np.array_equal(full_b, full_h)),
           "H_sd": hs["H_sd"]}
    res["control_pass"] = bool(res["baseline_deterministic"] and res["null_kappa0_equals_baseline"]
                               and res["engaged_differs_from_baseline"] and res["H_sd"] > 0)
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for cname, tonic in T.CONDITIONS.items():
        m = T.build(tonic)
        cells = np.array([r["cell_type"] for r in m.neuron_table()])
        for seed in (0, 1):
            key = f"{cname}_seed{seed}"
            bs, _ = run(m, seed, None)
            hs_sp, diag = run(m, seed, hdp_params())
            rates_h = hs_sp.sum(axis=0) / (T.DUR / 1000.0)
            base_m, bal_m = T.metrics(bs, cells, T.DUR), T.metrics(hs_sp, cells, T.DUR)
            rec = {"condition": cname, "tonic_drive_scale": tonic, "seed": seed,
                   "rule_params": dict(ab.DEFAULT_PARAMS),
                   "baseline": base_m, "balance": bal_m,
                   "H_state": T.h_stats(diag, rates_h, cells), "b_state": b_stats(diag),
                   "arms_identical": bool(np.array_equal(bs, hs_sp))}
            runs.append(key)
            (OUT / f"{key}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n",
                                             encoding="utf-8")
            print(key, "bar_r base %.2f bal %.2f | rho %.3f %.3f | fano %.2f %.2f | H mean %.3f sd %.3f"
                  % (base_m["bar_r_Hz"], bal_m["bar_r_Hz"], base_m["mean_rho"], bal_m["mean_rho"],
                     base_m["pop_synchrony_fano_10ms"], bal_m["pop_synchrony_fano_10ms"],
                     rec["H_state"]["H_mean"], rec["H_state"]["H_sd"]), flush=True)
    summary = {"spec": "docs/designs/H_BALANCE_TEST_spec.md", "spec_sha256": A.sha256_of(SPEC),
               "script_sha256": A.sha256_of(Path(__file__).resolve()),
               "rule_sha256": A.sha256_of(Path(ab.__file__).resolve()), "runs": sorted(runs)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print("wrote", OUT / "summary.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run"])
    a = ap.parse_args(argv)
    return control() if a.cmd == "control" else run_all()


if __name__ == "__main__":
    raise SystemExit(main())

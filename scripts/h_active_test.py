"""Active H configuration vs baseline on the canonical 400-neuron network (spec: docs/designs/H_ACTIVE_TEST_spec.md).

Usage:
  python scripts/h_active_test.py control    # H engagement and baseline determinism (short)
  python scripts/h_active_test.py run        # 2 conditions x 2 seeds x 2 arms (2000 ms)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import jaxfne as jt  # noqa: E402
from jaxfne import RuntimeConfig, Simulation  # noqa: E402
from jaxfne.hdp_network import DEFAULT_HDP_DESYNC  # noqa: E402

from jomission.network.builder import build_jomission_model  # noqa: E402

import agsdr_local_sweep as A  # noqa: E402

SPEC = REPO / "docs/designs/H_ACTIVE_TEST_spec.md"
OUT = REPO / "docs/receipts/h_active_test"
DT = 0.1
DUR = 2000.0
CLASSES = ("E", "PV", "SST", "VIP")
BASE_TONIC = 0.6  # builder default tonic_drive_scale
CONDITIONS = {"A_normal": BASE_TONIC, "B_stress_x1p5": BASE_TONIC * 1.5}
PD = {"rate_hz": 2988.0, "amplitude": 2.997, "target": "all"}


def run(model, seed: int, hdp: bool, dur: float = DUR):
    rt = (RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                        hdp_params=dict(DEFAULT_HDP_DESYNC))
          if hdp else RuntimeConfig(recurrent_backend="edge_list"))
    sim = Simulation(duration_ms=dur, dt_ms=DT, seed=seed, runtime=rt,
                     poisson_drive=dict(PD, seed=seed + 7919))
    sig = jt.simulate(model, sim)
    spikes = np.asarray(sig.spikes)
    diag = model.last_hdp_diagnostics() if hdp else None
    return spikes, diag


def metrics(spikes, cells, dur: float) -> dict:
    S = np.asarray(spikes, dtype=np.float64)
    n_steps, n = S.shape
    rates = S.sum(axis=0) / (dur / 1000.0)
    out = {"bar_r_Hz": float(rates.mean()), "rate_sd_Hz": float(rates.std()),
           "class_mean_Hz": {c: float(rates[cells == c].mean()) for c in CLASSES}}
    n_bins = round(dur / 100.0)
    B = S.reshape(n_bins, n_steps // n_bins, n).sum(axis=1)
    rng = np.random.default_rng(A.RHO_PAIR_SEED + 0)
    pi = rng.integers(0, n, A.RHO_PAIRS)
    pj = rng.integers(0, n - 1, A.RHO_PAIRS)
    pj = pj + (pj >= pi)
    Bc = B - B.mean(axis=0)
    sd = np.sqrt((Bc ** 2).sum(axis=0))
    den = sd[pi] * sd[pj]
    num = (Bc[:, pi] * Bc[:, pj]).sum(axis=0)
    ok = den > 0
    out["mean_rho"] = float((num[ok] / den[ok]).mean()) if ok.any() else math.nan
    pop = S.sum(axis=1).reshape(-1, round(10.0 / DT)).sum(axis=1)
    out["pop_synchrony_fano_10ms"] = float(pop.var() / pop.mean()) if pop.mean() > 0 else math.nan
    return out


def h_stats(diag, rates, cells) -> dict:
    H = np.asarray(diag["H_final"], dtype=np.float64)
    out = {"H_mean": float(H.mean()), "H_sd": float(H.std()), "H_min": float(H.min()),
           "H_max": float(H.max()),
           "frac_outside_0p8_1p2": float(np.mean((H < 0.8) | (H > 1.2))),
           "corr_H_rate": float(np.corrcoef(H, rates)[0, 1]) if H.std() > 0 and rates.std() > 0 else math.nan,
           "H_by_class": {c: {"mean": float(H[cells == c].mean()), "sd": float(H[cells == c].std())}
                          for c in CLASSES},
           "H_by_rate_bin": {}}
    for name, mask in {"<1Hz": rates < 1.0, "1to10Hz": (rates >= 1.0) & (rates <= 10.0),
                       ">10Hz": rates > 10.0}.items():
        out["H_by_rate_bin"][name] = {"n": int(mask.sum()),
                                      "H_mean": float(H[mask].mean()) if mask.any() else math.nan}
    return out


def build(tonic: float):
    return build_jomission_model(n_per_area=100, seed=0, tonic_drive_scale=tonic)


def control() -> int:
    m = build(BASE_TONIC)
    cells = np.array([r["cell_type"] for r in m.neuron_table()])
    b1, _ = run(m, 0, False, dur=200.0)
    b2, _ = run(m, 0, False, dur=200.0)
    h, diag = run(m, 0, True, dur=200.0)
    rates = np.asarray(h, dtype=np.float64).sum(axis=0) / 0.2
    hs = h_stats(diag, rates, cells)
    res = {"baseline_deterministic": bool(np.array_equal(b1, b2)),
           "H_arm_spikes": int(np.asarray(h).sum()), "baseline_spikes": int(np.asarray(b1).sum()),
           "arms_identical": bool(np.array_equal(b1, h)),
           "H_engaged_H_sd": hs["H_sd"], "H_mean": hs["H_mean"]}
    res["control_pass"] = bool(res["baseline_deterministic"] and not res["arms_identical"]
                               and res["H_engaged_H_sd"] > 0)
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for cname, tonic in CONDITIONS.items():
        m = build(tonic)
        cells = np.array([r["cell_type"] for r in m.neuron_table()])
        for seed in (0, 1):
            key = f"{cname}_seed{seed}"
            bs, _ = run(m, seed, False)
            hs_sp, diag = run(m, seed, True)
            rates_h = np.asarray(hs_sp, dtype=np.float64).sum(axis=0) / (DUR / 1000.0)
            rec = {"condition": cname, "tonic_drive_scale": tonic, "seed": seed,
                   "baseline": metrics(bs, cells, DUR),
                   "H": metrics(hs_sp, cells, DUR),
                   "H_state": h_stats(diag, rates_h, cells),
                   "arms_identical": bool(np.array_equal(bs, hs_sp))}
            results[key] = rec
            (OUT / f"{key}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n",
                                             encoding="utf-8")
            print(key, "bar_r base %.2f H %.2f | rho base %.3f H %.3f | H mean %.3f sd %.3f"
                  % (rec["baseline"]["bar_r_Hz"], rec["H"]["bar_r_Hz"], rec["baseline"]["mean_rho"],
                     rec["H"]["mean_rho"], rec["H_state"]["H_mean"], rec["H_state"]["H_sd"]), flush=True)
    summary = {"spec": "docs/designs/H_ACTIVE_TEST_spec.md",
               "spec_sha256": A.sha256_of(SPEC),
               "script_sha256": A.sha256_of(Path(__file__).resolve()),
               "runs": sorted(results)}
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

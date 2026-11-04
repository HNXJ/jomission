"""SCI-AGSDR-CALIBRATION J_CL1C rerun (sealed spec: docs/designs/SCI-AGSDR-J_CL1C_spec.md).

For each stage-0 candidate: build with its own gains and builder args (runner functions),
run a 1000 ms baseline, then three 1000 ms perturbations (PV x1.5; SST x1.5; VIP x1.5 and +2.0),
same seeds. Computes class-rate deltas, delta_rho, shortfalls, violations and J_CL1C.
Writes new receipts only.

Usage:
  python scripts/agsdr_jcl1c_rerun.py smoke AGS_S0_06   # controls 1 and 2, then one candidate
  python scripts/agsdr_jcl1c_rerun.py run               # all 24
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402

SPEC = A.REPO / "docs/designs/SCI-AGSDR-J_CL1C_spec.md"
OUT = A.REPO / "docs/receipts/agsdr_jcl1c_rerun"
RECEIPTS = A.RECEIPT_DIR
DURATION_MS = 1000.0
BIN_MS = 100.0
RHO_PAIRS = A.RHO_PAIRS
CLASSES = ("E", "PV", "SST", "VIP")
DRIVE_FACTOR = 1.5
VIP_ADDITIVE = 2.0


def _receipt(cid: str) -> dict:
    return json.loads((RECEIPTS / f"candidate_{cid}_seed0.json").read_text(encoding="utf-8"))


def build(cid: str):
    rec = _receipt(cid)
    cand = {"candidate_id": cid, "values": rec["values"], "builder_args": rec["builder_args"]}
    return A.build_candidate_model(cand, 0), rec


def drive_array(model) -> np.ndarray:
    d = getattr(model.params["emitter"], "drive", None)
    if d is None:
        raise RuntimeError("emitter drive array not found; stop (spec: missing input)")
    return np.asarray(d, dtype=np.float64).copy()


def simulate(model, rec: dict, drive: np.ndarray | None):
    import jax.numpy as jnp
    import jaxfne as jtfne
    from jaxfne import RuntimeConfig, Simulation

    m = model if drive is None else model.with_emitter_parameters(
        drive_per_neuron=jnp.asarray(drive))
    args = rec["builder_args"]
    sim = Simulation(
        duration_ms=DURATION_MS, dt_ms=A.DT_MS, seed=0,
        runtime=RuntimeConfig(recurrent_backend="edge_list"),
        poisson_drive={"rate_hz": args["background_poisson_rate_hz"],
                       "amplitude": args["background_poisson_amplitude"],
                       "target": "all", "seed": 0 + 7919},
    )
    sig = jtfne.simulate(m, sim)
    return np.asarray(sig.spikes)


def class_rates(spikes, cells) -> dict:
    rates = np.asarray(spikes, dtype=np.float64).sum(axis=0) / (DURATION_MS / 1000.0)
    return {c: float(rates[cells == c].mean()) for c in CLASSES if (cells == c).any()}


def rho_mean(spikes) -> float:
    S = np.asarray(spikes, dtype=np.float64)
    n_steps, n = S.shape
    n_bins = round(DURATION_MS / BIN_MS)
    B = S.reshape(n_bins, n_steps // n_bins, n).sum(axis=1)
    rng = np.random.default_rng(A.RHO_PAIR_SEED + 0)
    pi = rng.integers(0, n, RHO_PAIRS)
    pj = rng.integers(0, n - 1, RHO_PAIRS)
    pj = pj + (pj >= pi)
    Bc = B - B.mean(axis=0)
    sd = np.sqrt((Bc ** 2).sum(axis=0))
    den = sd[pi] * sd[pj]
    num = (Bc[:, pi] * Bc[:, pj]).sum(axis=0)
    ok = den > 0
    return float((num[ok] / den[ok]).mean()) if ok.any() else math.nan


def shortfall_ge(x: float, t: float) -> float:
    """Requirement x >= t (t > 0): clip((t - x) / t, 0, 1)."""
    return min(1.0, max(0.0, (t - x) / t))


def shortfall_le(x: float, t: float) -> float:
    """Requirement x <= t (t < 0): clip((x - t) / |t|, 0, 1)."""
    return min(1.0, max(0.0, (x - t) / abs(t)))


def violations(d: dict) -> dict:
    pv = (shortfall_ge(d["delta_r_PV"], 20.0) + shortfall_le(d["delta_rho"], -0.05)) / 2.0
    sst = (shortfall_ge(d["SST_delta_r_SST"], 8.0) + shortfall_le(d["SST_delta_r_E"], -0.5)) / 2.0
    vip = (shortfall_ge(d["VIP_delta_r_VIP"], 1.5) + shortfall_le(d["VIP_delta_r_SST"], -1.0)) / 2.0
    return {"PV": pv, "SST": sst, "VIP": vip, "J_CL1C": (pv + sst + vip) / 3.0}


def target_only_ok(base: np.ndarray, pert: np.ndarray, cells: np.ndarray, target: str) -> bool:
    diff = np.flatnonzero(~np.isclose(base, pert, rtol=0.0, atol=0.0))
    return bool(diff.size > 0 and np.all(cells[diff] == target))


def run_candidate(cid: str) -> dict:
    model, rec = build(cid)
    cells = np.array([r["cell_type"] for r in model.neuron_table()])
    base_drive = drive_array(model)
    base_sp = simulate(model, rec, None)
    base_rates = class_rates(base_sp, cells)
    base_rho = rho_mean(base_sp)

    perts = {}
    checks = {}
    for target in ("PV", "SST", "VIP"):
        drive = base_drive.copy()
        idx = cells == target
        drive[idx] *= DRIVE_FACTOR
        if target == "VIP":
            drive[idx] += VIP_ADDITIVE
        checks[target] = target_only_ok(base_drive, drive, cells, target)
        sp = simulate(model, rec, drive)
        perts[target] = {"rates": class_rates(sp, cells), "rho": rho_mean(sp)}

    d = {
        "delta_r_PV": perts["PV"]["rates"]["PV"] - base_rates["PV"],
        "delta_rho": perts["PV"]["rho"] - base_rho,
        "SST_delta_r_SST": perts["SST"]["rates"]["SST"] - base_rates["SST"],
        "SST_delta_r_E": perts["SST"]["rates"]["E"] - base_rates["E"],
        "VIP_delta_r_VIP": perts["VIP"]["rates"]["VIP"] - base_rates["VIP"],
        "VIP_delta_r_SST": perts["VIP"]["rates"]["SST"] - base_rates["SST"],
    }
    v = violations(d)
    bonuses = {
        "PV_delta_rho_le_m0p10": d["delta_rho"] <= -0.10,
        "PV_delta_r_E_lt_0": (perts["PV"]["rates"]["E"] - base_rates["E"]) < 0,
        "SST_delta_rho_le_m0p03": (perts["SST"]["rho"] - base_rho) <= -0.03,
        "VIP_delta_r_E_ge_0p1": (perts["VIP"]["rates"]["E"] - base_rates["E"]) >= 0.1,
    }
    return {
        "candidate_id": cid,
        "baseline": {"rates_Hz": base_rates, "rho": base_rho},
        "perturbed": perts,
        "deltas": d,
        "violations": v,
        "J_CL1C": v["J_CL1C"],
        "bonuses_reported_not_in_J": bonuses,
        "target_only_check": checks,
        "pass": bool(all(checks.values())),
    }


def smoke(cid: str) -> int:
    model, rec = build(cid)
    cells = np.array([r["cell_type"] for r in model.neuron_table()])
    base = drive_array(model)
    s1 = simulate(model, rec, None)
    s2 = simulate(model, rec, None)
    det = bool(np.array_equal(s1, s2))
    zero = bool(np.array_equal(simulate(model, rec, base), s1))
    print(json.dumps({"control_1_determinism_identical": det,
                      "control_2_zero_perturbation_identical": zero}, indent=1))
    if not (det and zero):
        print("STOP: control failed")
        return 1
    r = run_candidate(cid)
    print(json.dumps({"candidate_id": cid, "pass": r["pass"], "J_CL1C": r["J_CL1C"],
                      "violations": r["violations"], "deltas": r["deltas"]}, indent=1))
    return 0 if r["pass"] else 1


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(24):
        cid = f"AGS_S0_{i:02d}"
        r = run_candidate(cid)
        (OUT / f"candidate_{cid}_jcl1c.json").write_text(
            json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        results.append(r)
        print(cid, "pass" if r["pass"] else "FAIL", "J_CL1C %.4f" % r["J_CL1C"], flush=True)
        if not r["pass"]:
            print("STOP: target-only control failed", flush=True)
            break
    all_pass = len(results) == 24 and all(r["pass"] for r in results)
    summary = {
        "schema": "agsdr_jcl1c_rerun_summary_v1",
        "spec": "docs/designs/SCI-AGSDR-J_CL1C_spec.md",
        "spec_sha256": A.sha256_of(SPEC),
        "rerun_script_sha256": A.sha256_of(Path(__file__).resolve()),
        "runner_sha256": A.sha256_of(A.THIS_FILE),
        "controls_all_pass": all_pass,
        "candidates_run": len(results),
        "candidates": {r["candidate_id"]: {"J_CL1C": r["J_CL1C"], "violations": r["violations"]}
                       for r in results},
        "selection": "NOT RUN: J_CL1C alone does not select; the six-term rule and OPEN-26 apply",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print("controls_all_pass", all_pass, "wrote", OUT / "summary.json")
    return 0 if all_pass else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("candidate")
    sub.add_parser("run")
    a = ap.parse_args(argv)
    return smoke(a.candidate) if a.cmd == "smoke" else run_all()


if __name__ == "__main__":
    raise SystemExit(main())

"""VIP-drive existence probe on the two H05-feasible AGSDR candidates (spec: docs/designs/SCI-VIP-DRIVE-PROBE_spec.md).

Usage:
  python scripts/vip_drive_probe.py control            # C2 (only VIP drive changes), C3 (determinism, 200 ms)
  python scripts/vip_drive_probe.py run CAND SEED      # one (candidate, seed): all f values; writes one receipt
  python scripts/vip_drive_probe.py summary            # controls C1, hypotheses, verdict from the receipts
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402

SPEC = REPO / "docs/designs/SCI-VIP-DRIVE-PROBE_spec.md"
OUT = REPO / "docs/receipts/vip_drive_probe"
CANDIDATES = ("AGS_S0_06", "AGS_S0_20")
SEEDS = (0, 1)
F_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
BANDS = {"PV": (12.0, 25.0), "VIP": (8.0, 15.0), "SST": (6.0, 12.0)}
DRIVE_CLIP = (1.0, 8.0)  # builder clip bounds for emitter drive (builder._apply_pv_drive_boost)


def load_candidate(cid: str) -> dict:
    r = json.loads((A.RECEIPT_DIR / f"candidate_{cid}_seed0.json").read_text(encoding="utf-8"))
    return {"candidate_id": cid, "values": r["values"], "builder_args": r["builder_args"],
            "motif_gains_effective": r["motif_gains_effective"], "mu_total": A.mu_of(r["values"])}


def scale_vip_drive(model, f: float):
    """Multiply the emitter drive of VIP cells by f, clip to builder bounds; other cells untouched."""
    import jax.numpy as jnp

    em = model.params["emitter"]
    d0 = np.asarray(em.drive, dtype=np.float64)
    cells = np.array([r["cell_type"] for r in model.neuron_table()])
    d = d0.copy()
    vip = cells == "VIP"
    d[vip] = np.clip(d0[vip] * f, *DRIVE_CLIP)
    new_em = replace(em, drive=jnp.asarray(d, dtype=jnp.float32))
    params = dict(model.params)
    params["emitter"] = new_em
    return replace(model, params=params), d0, d, cells


def simulate(model, cand: dict, seed: int, dur: float = A.DURATION_MS):
    import jaxfne as jtfne
    from jaxfne import RuntimeConfig, Simulation

    args = cand["builder_args"]
    sim = Simulation(duration_ms=dur, dt_ms=A.DT_MS, seed=seed,
                     runtime=RuntimeConfig(recurrent_backend="edge_list"),
                     poisson_drive={"rate_hz": args["background_poisson_rate_hz"],
                                    "amplitude": args["background_poisson_amplitude"],
                                    "target": "all", "seed": seed + 7919})
    sig = jtfne.simulate(model, sim)
    return np.asarray(sig.spikes), np.asarray(sig.V_m)


def viable(c: dict) -> dict:
    out = {k: bool(lo <= c[k] <= hi) for k, (lo, hi) in BANDS.items()}
    out["E_lt_PV"] = bool(c["E"] < c["PV"])
    out["all"] = all(out.values())
    return out


def run_one(cid: str, seed: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = load_candidate(cid)
    base = A.build_candidate_model(cand, seed)
    rows = []
    for f in F_GRID:
        model, d0, d, cells = scale_vip_drive(base, f)
        spikes, v_m = simulate(model, cand, seed)
        res = A.score_run(model, spikes, v_m, cand, seed)
        c = res["metrics"]["class_mean_Hz"]
        rows.append({"f": f, "class_mean_Hz": c, "viable": viable(c), "hard_pass": bool(res["hard_pass"]),
                     "failed_hard_gates": res["failed"], "bar_r": res["metrics"]["bar_r_Hz"],
                     "vip_drive_mean": float(d[cells == "VIP"].mean()),
                     "vip_drive_clipped_frac": float(np.mean(d[cells == "VIP"] >= DRIVE_CLIP[1]))})
        print(cid, seed, f, {k: round(v, 2) for k, v in c.items()}, rows[-1]["viable"]["all"], flush=True)
    rec = {"candidate_id": cid, "seed": seed, "f_grid": list(F_GRID), "rows": rows,
           "spec_sha256": A.sha256_of(SPEC), "script_sha256": A.sha256_of(Path(__file__).resolve())}
    (OUT / f"{cid}_seed{seed}.json").write_text(json.dumps(A._jsonable(rec), indent=1, allow_nan=False) + "\n",
                                                encoding="utf-8")
    return 0


def control() -> int:
    cand = load_candidate(CANDIDATES[0])
    base = A.build_candidate_model(cand, 0)
    model, d0, d, cells = scale_vip_drive(base, 2.0)
    non = cells != "VIP"
    c2 = bool(np.array_equal(d0[non], d[non]) and np.allclose(d[~non], np.clip(d0[~non] * 2.0, *DRIVE_CLIP)))
    s1, _ = simulate(model, cand, 0, dur=200.0)
    s2, _ = simulate(model, cand, 0, dur=200.0)
    s0, _ = simulate(base, cand, 0, dur=200.0)
    res = {"C2_only_vip_drive_changes": c2, "C3_deterministic": bool(np.array_equal(s1, s2)),
           "engaged_differs_from_f1": bool(not np.array_equal(s0, s1)),
           "vip_drive_f1_mean": float(d0[~non].mean()), "nonvip_drive_mean": float(d0[non].mean())}
    res["control_pass"] = bool(c2 and res["C3_deterministic"] and res["engaged_differs_from_f1"])
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def summary() -> int:
    recs = {(r["candidate_id"], r["seed"]): r for r in
            (json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("AGS_*_seed*.json")))}
    c1 = {}
    for cid in CANDIDATES:
        stored = json.loads((A.RECEIPT_DIR / f"candidate_{cid}_seed0.json").read_text(encoding="utf-8"))
        got = recs[(cid, 0)]["rows"][0]
        c1[cid] = bool(got["class_mean_Hz"] == stored["metrics"]["class_mean_Hz"])
    keys = sorted(recs)
    vip = {f"{k[0]}_s{k[1]}": [r["class_mean_Hz"]["VIP"] for r in recs[k]["rows"]] for k in keys}
    sst = {f"{k[0]}_s{k[1]}": [r["class_mean_Hz"]["SST"] for r in recs[k]["rows"]] for k in keys}
    hv1 = all(all(b > a for a, b in zip(v, v[1:])) for v in vip.values())
    hv2 = all(v[-1] < v[0] for v in sst.values())
    exist = {}
    for cid in CANDIDATES:
        fs = [f for i, f in enumerate(F_GRID) if all(recs[(cid, s)]["rows"][i]["viable"]["all"] for s in SEEDS)]
        exist[cid] = fs
    hv3 = any(exist.values())
    vip_in_band = any(r["viable"]["VIP"] for k in keys for r in recs[k]["rows"])
    hv4 = {cid: {f: all(recs[(cid, s)]["rows"][F_GRID.index(f)]["hard_pass"] for s in SEEDS) for f in fs}
           for cid, fs in exist.items()}
    if not all(c1.values()):
        verdict = "CONTROL_FAILED_NO_INTERPRETATION"
    elif hv3:
        verdict = "VIABLE_BRANCH_ON_VIP_DRIVE_AXIS"
    elif vip_in_band:
        verdict = "VIP_RECRUITED_BUT_OTHER_BANDS_FAIL"
    else:
        verdict = "VIP_NOT_RECRUITED_BY_DRIVE_IN_GRID"
    out = {"spec": "docs/designs/SCI-VIP-DRIVE-PROBE_spec.md", "spec_sha256": A.sha256_of(SPEC),
           "script_sha256": A.sha256_of(Path(__file__).resolve()),
           "C1_f1_reproduces_stage0_seed0": c1, "HV1_vip_monotone_in_f": hv1, "HV2_sst_lower_at_fmax": hv2,
           "HV3_viable_f_both_seeds": exist, "HV3": hv3, "HV4_hard_pass_at_viable_f": hv4,
           "vip_series": vip, "sst_series": sst, "verdict": verdict}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run", "summary"])
    ap.add_argument("cand", nargs="?")
    ap.add_argument("seed", nargs="?", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "control":
        return control()
    if a.cmd == "run":
        return run_one(a.cand, a.seed)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())

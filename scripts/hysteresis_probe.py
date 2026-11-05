"""Hysteresis probe: up and down parameter sweeps with network state carried (spec: docs/designs/SCI-PV-HYSTERESIS_spec.md).

Usage:
  python scripts/hysteresis_probe.py control          # C1 continuation determinism, C2 differs from fresh, C3 scaling scope
  python scripts/hysteresis_probe.py run SEED         # both axes, all arms, one seed; writes one receipt
  python scripts/hysteresis_probe.py summary          # null control, hypotheses, verdicts
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
import vip_drive_probe as P  # noqa: E402

SPEC = REPO / "docs/designs/SCI-PV-HYSTERESIS_spec.md"
OUT = REPO / "docs/receipts/hysteresis_probe"
CAND = "AGS_S0_06"
SEEDS = (0, 1, 2)
SEG_MS = 2000.0
SCORE_MS = 1000.0  # last SCORE_MS of each segment are scored
CLASSES = ("E", "PV", "SST", "VIP")
AXES = {
    "V": {"grid": (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0), "fixed": {"g": 1.0}, "repeat_at": 2.5},
    "G": {"grid": (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25), "fixed": {"f": 2.25}, "repeat_at": 0.875},
}
N_REPEAT = 4
NULL_POISSON_OFFSET = 1000
TOL_FLOOR_HZ = 1.0


def scaled_model(base, f: float, g: float):
    """VIP drive x f, then every drive x g; clip to builder bounds [1, 8]."""
    import jax.numpy as jnp

    em = base.params["emitter"]
    d0 = np.asarray(em.drive, dtype=np.float64)
    cells = np.array([r["cell_type"] for r in base.neuron_table()])
    d = d0.copy()
    d[cells == "VIP"] *= f
    d = np.clip(d * g, *P.DRIVE_CLIP)
    params = dict(base.params)
    params["emitter"] = replace(em, drive=jnp.asarray(d, dtype=jnp.float32))
    clipped = float(np.mean((d == P.DRIVE_CLIP[0]) | (d == P.DRIVE_CLIP[1])))
    return replace(base, params=params), d0, d, cells, clipped


def make_sim(cand: dict, seed: int, poisson_seed: int):
    from jaxfne import RuntimeConfig, Simulation

    args = cand["builder_args"]
    return Simulation(duration_ms=SEG_MS, dt_ms=A.DT_MS, seed=seed,
                      runtime=RuntimeConfig(recurrent_backend="edge_list"),
                      poisson_drive={"rate_hz": args["background_poisson_rate_hz"],
                                     "amplitude": args["background_poisson_amplitude"],
                                     "target": "all", "seed": poisson_seed})


def class_rates(spikes, cells) -> dict:
    S = np.asarray(spikes, dtype=np.float64)
    k = round(SCORE_MS / A.DT_MS)
    r = S[-k:].sum(axis=0) / (SCORE_MS / 1000.0)
    return {c: float(r[cells == c].mean()) for c in CLASSES}


def sweep(base, cand, seed: int, axis: str, values, poisson_seed: int) -> list[dict]:
    """Warm-up segment at values[0], then one segment per value, state carried throughout."""
    import jaxfne as jtfne

    fixed = AXES[axis]["fixed"]
    state, rows = None, []
    for i, v in enumerate([values[0]] + list(values)):
        f = v if axis == "V" else fixed["f"]
        g = v if axis == "G" else fixed["g"]
        model, _, _, cells, clipped = scaled_model(base, f, g)
        sim = make_sim(cand, seed, poisson_seed)
        sig, state = jtfne.simulate(model, sim, continuation=state, return_state=True)
        if i == 0:
            continue
        rows.append({"value": v, "class_rate_Hz": class_rates(sig.spikes, cells), "drive_clipped_frac": clipped})
    return rows


def run_one(seed: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = P.load_candidate(CAND)
    base = A.build_candidate_model(cand, seed)
    rec = {"candidate_id": CAND, "seed": seed, "seg_ms": SEG_MS, "score_ms": SCORE_MS, "axes": {}}
    ps = seed + 7919
    for axis, spec in AXES.items():
        grid = spec["grid"]
        rec["axes"][axis] = {
            "up": sweep(base, cand, seed, axis, grid, ps),
            "down": sweep(base, cand, seed, axis, tuple(reversed(grid)), ps),
            "null_up": sweep(base, cand, seed, axis, grid, ps + NULL_POISSON_OFFSET),
            "repeat": sweep(base, cand, seed, axis, (spec["repeat_at"],) * N_REPEAT, ps),
        }
        print(seed, axis, "done", flush=True)
    rec["spec_sha256"] = A.sha256_of(SPEC)
    rec["script_sha256"] = A.sha256_of(Path(__file__).resolve())
    (OUT / f"{CAND}_seed{seed}.json").write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return 0


def tolerance(rep: list[dict]) -> dict:
    out = {}
    for c in CLASSES:
        x = [r["class_rate_Hz"][c] for r in rep]
        out[c] = max(TOL_FLOOR_HZ, 3.0 * max(abs(b - a) for a, b in zip(x, x[1:])))
    return out


def differs(a: list[dict], b: list[dict], tol: dict) -> dict:
    """Persistent differences: class c differs (|a - b| > tol_c) with the same sign at two adjacent grid values."""
    bm = {r["value"]: r["class_rate_Hz"] for r in b}
    out = {}
    for c in CLASSES:
        d = [r["class_rate_Hz"][c] - bm[r["value"]][c] for r in a]
        vals = [r["value"] for r in a]
        for i in range(len(d) - 1):
            if abs(d[i]) > tol[c] and abs(d[i + 1]) > tol[c] and np.sign(d[i]) == np.sign(d[i + 1]):
                out.setdefault(c, []).append([vals[i], vals[i + 1], round(d[i], 3), round(d[i + 1], 3)])
    return out


def summary() -> int:
    recs = {r["seed"]: r for r in (json.loads(p.read_text(encoding="utf-8"))
                                   for p in sorted(OUT.glob(f"{CAND}_seed*.json")))}
    out = {"spec": "docs/designs/SCI-PV-HYSTERESIS_spec.md", "spec_sha256": A.sha256_of(SPEC),
           "script_sha256": A.sha256_of(Path(__file__).resolve()), "seeds": sorted(recs), "axes": {}}
    for axis in AXES:
        per = {}
        for s, r in recs.items():
            ax = r["axes"][axis]
            tol = tolerance(ax["repeat"])
            per[s] = {"tol_Hz": tol, "up_vs_down": differs(ax["up"], ax["down"], tol),
                      "up_vs_null_up": differs(ax["up"], ax["null_up"], tol)}
        null_ok = all(not v["up_vs_null_up"] for v in per.values())
        hyst = all(bool(v["up_vs_down"]) for v in per.values()) and len(per) == len(SEEDS)
        verdict = ("NULL_CONTROL_FAILED_NO_INTERPRETATION" if not null_ok
                   else "HYSTERESIS_PRESENT" if hyst else "NO_HYSTERESIS_DETECTED")
        out["axes"][axis] = {"per_seed": per, "null_control_pass": null_ok, "hysteresis_all_seeds": hyst,
                             "verdict": verdict}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({a: {k: v[k] for k in ("null_control_pass", "hysteresis_all_seeds", "verdict")}
                      for a, v in out["axes"].items()}, indent=1))
    return 0


def control() -> int:
    import jaxfne as jtfne

    cand = P.load_candidate(CAND)
    base = A.build_candidate_model(cand, 0)
    short = lambda ps: replace(make_sim(cand, 0, ps), duration_ms=200.0)  # noqa: E731
    _, st = jtfne.simulate(base, short(7919), return_state=True)
    m2, d0, d, cells, _ = scaled_model(base, 3.0, 1.0)
    a, _ = jtfne.simulate(m2, short(7919), continuation=st, return_state=True)
    b, _ = jtfne.simulate(m2, short(7919), continuation=st, return_state=True)
    fresh = jtfne.simulate(m2, short(7919))
    m3, d0b, d3, _, _ = scaled_model(base, 1.0, 0.75)
    non = cells != "VIP"
    res = {"C1_continuation_deterministic": bool(np.array_equal(np.asarray(a.spikes), np.asarray(b.spikes))),
           "C2_continuation_differs_from_fresh": bool(not np.array_equal(np.asarray(a.spikes), np.asarray(fresh.spikes))),
           "C3_f_scales_only_vip": bool(np.array_equal(d0[non], d[non])),
           "C3_g_scales_all": bool(np.allclose(d3, np.clip(d0b * 0.75, *P.DRIVE_CLIP)))}
    res["control_pass"] = all(res.values())
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run", "summary"])
    ap.add_argument("seed", nargs="?", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "control":
        return control()
    if a.cmd == "run":
        return run_one(a.seed)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())

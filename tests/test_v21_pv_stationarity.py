"""SCI-V21-PV-STATIONARITY: 60 s PV stationarity diagnostic at the frozen candidate lambda 0.65625.

Spec (committed and pushed before the procedure runs): results/v21_pv_stationarity_spec.json.
Model and path as in tests/test_v21_coupled_background.py (tonic, _v21 imported, not copied).
Three initial states: fresh (PRNGKey(11), step 0); end state of a 20 s fresh run at lambda 0;
end state of a 20 s fresh run at lambda 1. Each runs 12 x 5 s chunks at lambda 0.65625.
Chunk 0 is settling; windows 1-11 are the trajectory; windows 9-11 are the qualification region.
Tier 3: writes results/v21_pv_stationarity.json; the label is data.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

from jomission.harness import operation
from jomission.harness.drive import check_additive_schedule

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_pv_stationarity_spec.json"
OUT = ROOT / "results" / "v21_pv_stationarity.json"
SEALED = ROOT / "results" / "v21_coupled_background.json"
CLASSES = ("E", "PV", "SST", "VIP")
LAMBDA_C = 0.65625
N_CHUNK = 12
PRE_CHUNKS = 4
LATE = (9, 10, 11)
REPRO_TOL_HZ = 0.5
TOL_ABS_HZ, TOL_REL = 1.0, 0.10
SEED = 11
IC_NAMES = ("fresh", "lambda_0_end_state", "lambda_1_end_state")


def _blob(rel):
    return subprocess.run(["git", "hash-object", str(ROOT / rel)], capture_output=True, text=True).stdout.strip()


def _cb():
    spec = importlib.util.spec_from_file_location("_v21_cb", ROOT / "tests" / "test_v21_coupled_background.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spec_pins_match():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for group in ("code", "data"):
        for rel, pin in spec["frozen"][group].items():
            assert _blob(rel) == pin.split()[0], rel
    p = spec["procedure_constants"]
    assert (LAMBDA_C, N_CHUNK, PRE_CHUNKS, list(LATE), REPRO_TOL_HZ, TOL_ABS_HZ, TOL_REL, SEED) == (
        p["lambda"], p["n_chunk"], p["pre_run_chunks"], p["late_windows"], p["reproduction_tol_hz"],
        p["tolerance_abs_hz"], p["tolerance_rel"], p["engine_seed"])


def reversal(series, tol):
    """True if some i < j < k has series[j] exceeding both neighbours, or below both, by more than tol."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    for j in range(1, n - 1):
        lo_before, hi_before = x[:j].min(), x[:j].max()
        lo_after, hi_after = x[j + 1:].min(), x[j + 1:].max()
        if (x[j] - lo_before > tol and x[j] - lo_after > tol) or (hi_before - x[j] > tol and hi_after - x[j] > tol):
            return True
    return False


class _Sim:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        import jaxfne as jtfne
        self.cb = _cb()
        self.v21 = self.cb._v21()
        self.jax, self.jnp, self.jtfne = jax, jnp, jtfne
        self.mid, self.high = self.cb._vectors()

    def run(self, lam, n_chunk, init_state=None, summarize_from=1):
        vec = self.cb.tonic(lam, self.mid, self.high)
        model, step_fn, cls = self.v21.build_plant(vec)
        n = len(cls)
        sched = self.jnp.zeros((self.v21.STEPS_CHUNK, n), dtype=model.params["emitter"].v0.dtype)
        check_additive_schedule(np.asarray(model.params["emitter"].drive, dtype=np.float64), np.zeros((1, n)), cls)
        st = init_state if init_state is not None else self.jtfne.ContinuationState(
            dynamic=self.jtfne.dynamic_state_from_model(model), prng_key=self.jax.random.PRNGKey(SEED),
            step_index=0, delay_state=None)
        summaries = []
        for k in range(n_chunk):
            st, out = self.jtfne.run_continuation(step_fn, st, sched)
            self.jax.block_until_ready(out[0])
            if k >= summarize_from:
                spikes = np.asarray(out[1], dtype=np.float64)
                assert np.isfinite(spikes).all(), f"non-finite spikes lambda {lam} chunk {k}"
                summaries.append(operation.window_summary(spikes > 0.5, cls))
        return vec, summaries, st


def classify(trajs, repro_ok):
    """trajs: {ic: {"late_stationary", "late_bands", "late_mean_hz", "pv_windows_hz"}}; returns (label, reason)."""
    if not repro_ok:
        return "PV_STATIONARITY_UNRESOLVED", "first 20 s do not reproduce the sealed coupled-background runs"
    ref = trajs["fresh"]["late_mean_hz"]
    same = all(abs(trajs[a]["late_mean_hz"][c] - trajs[b]["late_mean_hz"][c]) <= max(TOL_ABS_HZ, TOL_REL * ref[c])
               for a in IC_NAMES for b in IC_NAMES if a < b for c in CLASSES)
    if all(t["late_stationary"] for t in trajs.values()):
        if not same:
            return "PV_MULTISTABILITY", "every trajectory is late-stationary but late class means differ beyond tolerance"
        if all(t["late_bands"] for t in trajs.values()):
            return "PV_STATIONARITY_PASS", "all three trajectories converge to one stationary, band-valid late regime"
        return "PV_STATIONARITY_UNRESOLVED", "one shared stationary late regime, but it fails the class-rate bands (case outside the reviewer's four labels)"
    moving = [ic for ic, t in trajs.items() if not t["late_stationary"]]
    switching = [ic for ic in moving if reversal(trajs[ic]["pv_windows_hz"],
                                                 max(TOL_ABS_HZ, TOL_REL * float(np.mean(trajs[ic]["pv_windows_hz"]))))]
    if switching:
        return "PV_NONSTATIONARY", f"late drift exceeded and PV reverses direction over windows 1-11: {switching}"
    return "PV_STATIONARITY_UNRESOLVED", f"late drift exceeded without a PV reversal (monotone; convergence not excluded): {moving}"


def test_v21_pv_stationarity():
    sim = _Sim()
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    ref20 = {"fresh": next(e for e in sealed["evaluations"] if e["lambda"] == LAMBDA_C)["window_rates_hz"],
             "lambda_0_end_state": sealed["hysteresis"]["from_lambda_0_end_state"]["run"]["window_rates_hz"],
             "lambda_1_end_state": sealed["hysteresis"]["from_lambda_1_end_state"]["run"]["window_rates_hz"]}
    _, _, st0 = sim.run(0.0, PRE_CHUNKS, summarize_from=PRE_CHUNKS)
    _, _, st1 = sim.run(1.0, PRE_CHUNKS, summarize_from=PRE_CHUNKS)
    inits = {"fresh": None, "lambda_0_end_state": st0, "lambda_1_end_state": st1}
    res = {"spec": "results/v21_pv_stationarity_spec.json", "lambda": LAMBDA_C, "trajectories": {}}
    trajs, repro = {}, {}
    for ic in IC_NAMES:
        vec, summaries, _ = sim.run(LAMBDA_C, N_CHUNK, init_state=inits[ic])
        wr = [w["rates"] for w in summaries]  # windows 1..11
        late = [summaries[k - 1] for k in LATE]
        ev = operation.evaluate(late)
        d = max(abs(wr[k][c] - ref20[ic][k][c]) for k in range(3) for c in CLASSES)
        repro[ic] = {"max_abs_diff_hz": d, "within_tol": d <= REPRO_TOL_HZ}
        trajs[ic] = {"late_stationary": bool(ev["checks"]["drift"]), "late_bands": bool(ev["checks"]["class_rate_bands"]),
                     "late_drift": ev["drift"],
                     "late_mean_hz": {c: float(np.mean([w["rates"][c] for w in late])) for c in CLASSES},
                     "pv_windows_hz": [w["PV"] for w in wr]}
        res["trajectories"][ic] = dict(trajs[ic], tonic=vec, window_rates_hz=wr,
                                       reported_not_gating={"late_checks": {k: v for k, v in ev["checks"].items()
                                                                            if k not in ("drift", "class_rate_bands")},
                                                            "active_fraction": [w["active_fraction"] for w in summaries],
                                                            "sync": [w["sync"] for w in summaries]},
                                       profile=ev["profile"], gate_blob=ev["gate_blob"])
    res["reproduction_first_20s"] = repro
    label, reason = classify(trajs, all(v["within_tol"] for v in repro.values()))
    res["late_pairwise_diff_hz"] = {f"{a}|{b}": {c: trajs[a]["late_mean_hz"][c] - trajs[b]["late_mean_hz"][c] for c in CLASSES}
                                    for a in IC_NAMES for b in IC_NAMES if a < b}
    res["verdict"] = label
    res["reason"] = reason
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(label, reason)


def test_classify_known_cases():
    flat = {"late_stationary": True, "late_bands": True, "late_mean_hz": dict.fromkeys(CLASSES, 15.0),
            "pv_windows_hz": [15.0] * 11}
    base = {ic: dict(flat) for ic in IC_NAMES}
    assert classify(base, True)[0] == "PV_STATIONARITY_" + "PASS"
    assert classify(base, False)[0] == "PV_STATIONARITY_UNRESOLVED"
    multi = dict(base, lambda_0_end_state=dict(flat, late_mean_hz=dict(flat["late_mean_hz"], PV=3.0)))
    assert classify(multi, True)[0] == "PV_MULTISTABILITY"
    switch = dict(base, fresh=dict(flat, late_stationary=False, pv_windows_hz=[16, 16, 10, 16, 2, 15, 15, 15, 15, 15, 15]))
    assert classify(switch, True)[0] == "PV_NONSTATIONARY"
    decay = dict(base, fresh=dict(flat, late_stationary=False, pv_windows_hz=[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]))
    assert classify(decay, True)[0] == "PV_STATIONARITY_UNRESOLVED"

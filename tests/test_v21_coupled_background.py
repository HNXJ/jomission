"""SCI-V21-COUPLED-BACKGROUND: one-dimensional coupled tonic path from V_MID to V_HIGH.

Spec (committed and pushed before the procedure runs): results/v21_coupled_background_spec.json.
Model: the frozen V2.1 static column (build_plant loaded by path from tests/test_v21_operation.py),
engine PRNGKey(11), native noise, all-zero schedule. Only coordinate: lambda in
I(lambda) = (1 - lambda) I_MID + lambda I_HIGH, all four classes together.
Criterion: prospective_v2 class_rate_bands in every analysis window. Every other check is recorded,
not gating. Tier 3: writes results/v21_coupled_background.json; the driver passes when the
procedure ran and the label is data.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

from jomission.harness import gates, operation
from jomission.harness.drive import check_additive_schedule

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_coupled_background_spec.json"
OUT = ROOT / "results" / "v21_coupled_background.json"
CLASSES = ("E", "PV", "SST", "VIP")
BRACKET = (0.0, 0.5, 1.0)
RESOLUTION = 0.0625
MAX_BISECTIONS = 12
MARGIN_MIN = 0.125
REPRO_TOL_HZ = 0.5
HYST_ABS_HZ, HYST_REL = 1.0, 0.10
SEED = 11
CRITERION = "class_rate_bands"


def _blob(rel):
    return subprocess.run(["git", "hash-object", str(ROOT / rel)], capture_output=True, text=True).stdout.strip()


def _v21():
    spec = importlib.util.spec_from_file_location("_v21_frozen", ROOT / "tests" / "test_v21_operation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import jomission
    assert Path(jomission.__file__).resolve().is_relative_to(ROOT), jomission.__file__
    return mod


def _vectors():
    v = json.loads((ROOT / "results" / "v21_vectors.json").read_text(encoding="utf-8"))["vectors"]
    return v["V_MID"], v["V_HIGH"]


def tonic(lam, mid, high):
    return {c: (1.0 - lam) * float(mid[c]) + lam * float(high[c]) for c in CLASSES}


def test_spec_pins_match():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for group in ("code", "data"):
        for rel, pin in spec["frozen"][group].items():
            assert _blob(rel) == pin.split()[0], rel
    p = spec["procedure_constants"]
    assert (list(BRACKET), RESOLUTION, MAX_BISECTIONS, MARGIN_MIN, REPRO_TOL_HZ, HYST_ABS_HZ, HYST_REL, SEED) == (
        p["bracket"], p["resolution"], p["max_bisections"], p["margin_min"], p["reproduction_tol_hz"],
        p["hysteresis_abs_hz"], p["hysteresis_rel"], p["engine_seed"])


class _Runner:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        import jaxfne as jtfne
        self.jax, self.jnp, self.jtfne, self.v21 = jax, jnp, jtfne, _v21()
        self.mid, self.high = _vectors()
        gate = gates.load_gate()
        self.n_chunk = int(round(gate["windows"]["run_s"] / gate["windows"]["chunk_s"]))
        self.windows = set(gate["windows"]["analysis_chunks"])

    def run(self, lam, init_state=None):
        """20 s at I(lambda). init_state None: fresh model state, PRNGKey(11), step 0; else continue from it."""
        vec = tonic(lam, self.mid, self.high)
        model, step_fn, cls = self.v21.build_plant(vec)
        n = len(cls)
        dtype = model.params["emitter"].v0.dtype
        sched = self.jnp.zeros((self.v21.STEPS_CHUNK, n), dtype=dtype)
        drive = np.asarray(model.params["emitter"].drive, dtype=np.float64)
        additivity = check_additive_schedule(drive, np.zeros((1, n)), cls)
        if init_state is None:
            st = self.jtfne.ContinuationState(dynamic=self.jtfne.dynamic_state_from_model(model),
                                              prng_key=self.jax.random.PRNGKey(SEED), step_index=0, delay_state=None)
        else:
            st = init_state
        summaries = []
        for k in range(self.n_chunk):
            st, out = self.jtfne.run_continuation(step_fn, st, sched)
            self.jax.block_until_ready(out[0])
            if k in self.windows:
                spikes = np.asarray(out[1], dtype=np.float64)
                assert np.isfinite(spikes).all(), f"non-finite spikes lambda {lam} chunk {k}"
                summaries.append(operation.window_summary(spikes > 0.5, cls))
        rec = operation.evaluate(summaries)
        mean_rates = {c: float(np.mean([w["rates"][c] for w in summaries])) for c in CLASSES}
        return {"lambda": lam, "tonic": vec, "init": "fresh" if init_state is None else "continued",
                "passes": bool(rec["checks"][CRITERION]), "window_rates_hz": [w["rates"] for w in summaries],
                "mean_rates_hz": mean_rates,
                "reported_not_gating": {"checks": {k: v for k, v in rec["checks"].items() if k != CRITERION},
                                        "drift": rec["drift"],
                                        "active_fraction": [w["active_fraction"] for w in summaries],
                                        "isi_fraction": [w["isi_fraction"] for w in summaries],
                                        "rate_cv": [w["rate_cv"] for w in summaries],
                                        "sync": [w["sync"] for w in summaries]},
                "profile": rec["profile"], "gate_blob": rec["gate_blob"], "drive_additivity": additivity}, st


def _runs(pts):
    """Sorted evaluated lambdas and pass flags (fresh runs only)."""
    lams = sorted(pts)
    return lams, [pts[x]["passes"] for x in lams]


def _classify(res, pts, runner, end_states):
    lams, flags = _runs(pts)
    # single-interval check: pass flags change at most twice and passing points are consecutive
    passing = [x for x, f in zip(lams, flags) if f]
    idx = [lams.index(x) for x in passing]
    if idx and idx != list(range(idx[0], idx[-1] + 1)):
        return "COUPLED_BACKGROUND_UNRESOLVED", "evaluated passing lambdas do not form one consecutive run (non-monotone)"
    interior = [x for x in passing if 0.0 < x < 1.0]
    if not interior:
        return "COUPLED_BACKGROUND_FAIL", "no evaluated interior lambda satisfies all class-rate bands"
    lo, hi = passing[0], passing[-1]
    lam_c = 0.5 * (lo + hi)
    fails = [x for x, f in zip(lams, flags) if not f]
    below = [x for x in fails if x < lam_c]
    above = [x for x in fails if x > lam_c]
    margin = min(([lam_c - max(below)] if below else []) + ([min(above) - lam_c] if above else []) or [1.0])
    if lam_c not in pts:
        pts[lam_c], _ = runner.run(lam_c)
        pts[lam_c]["role"] = "candidate"
    cand = pts[lam_c]
    res["candidate"] = {"lambda": lam_c, "tonic": cand["tonic"], "passing_run": [lo, hi], "margin": margin,
                        "frozen": False, "note": "reported candidate; the reviewer freezes"}
    if not cand["passes"]:
        return "COUPLED_BACKGROUND_UNRESOLVED", "candidate midpoint of the passing run fails (non-monotone)"
    lams, flags = _runs(pts)
    if [lams.index(x) for x in lams if pts[x]["passes"]] != list(range(lams.index(lo), lams.index(hi) + 1)):
        return "COUPLED_BACKGROUND_UNRESOLVED", "evaluated passing lambdas do not form one consecutive run (non-monotone)"
    if margin < MARGIN_MIN:
        return "COUPLED_BACKGROUND_UNRESOLVED", f"passing run too narrow: candidate margin {margin} < {MARGIN_MIN}"
    hyst = {}
    ok = True
    for tag, lam_end in (("from_lambda_0_end_state", 0.0), ("from_lambda_1_end_state", 1.0)):
        r, _ = runner.run(lam_c, init_state=end_states[lam_end])
        diffs = {c: r["mean_rates_hz"][c] - cand["mean_rates_hz"][c] for c in CLASSES}
        within = all(abs(diffs[c]) <= max(HYST_ABS_HZ, HYST_REL * cand["mean_rates_hz"][c]) for c in CLASSES)
        hyst[tag] = {"run": r, "rate_diff_vs_fresh_hz": diffs, "within_tolerance": within,
                     "same_pass": r["passes"] == cand["passes"]}
        ok = ok and within and r["passes"] == cand["passes"]
    res["hysteresis"] = hyst
    if not ok:
        return "COUPLED_BACKGROUND_UNRESOLVED", "state dependence at the candidate: initial-state runs disagree"
    return "COUPLED_BACKGROUND_PASS", f"interior passing run [{lo}, {hi}], candidate lambda {lam_c}, margin {margin}"


def test_v21_coupled_background():
    runner = _Runner()
    pts, end_states = {}, {}
    res = {"spec": "results/v21_coupled_background_spec.json", "criterion": f"prospective_v2 {CRITERION}, every window"}
    for lam in BRACKET:
        pts[lam], st = runner.run(lam)
        pts[lam]["role"] = "bracket"
        end_states[lam] = st
    sealed = {0.0: "results/v21_mid.json", 1.0: "results/v21_high.json"}
    repro = {}
    for lam, rel in sealed.items():
        old = json.loads((ROOT / rel).read_text(encoding="utf-8"))["rates"]
        d = max(abs(pts[lam]["window_rates_hz"][k][c] - old[k][c]) for k in range(3) for c in CLASSES)
        repro[str(lam)] = {"sealed": rel, "max_abs_diff_hz": d, "within_tol": d <= REPRO_TOL_HZ}
    res["endpoint_reproduction"] = repro
    label = reason = None
    if not all(v["within_tol"] for v in repro.values()):
        label, reason = "COUPLED_BACKGROUND_UNRESOLVED", "endpoint rates do not reproduce the sealed V2.1 runs"
    else:
        n_bis = 0
        while True:
            lams, flags = _runs(pts)
            pair = next(((a, b) for a, b, fa, fb in zip(lams, lams[1:], flags, flags[1:])
                         if fa != fb and b - a > RESOLUTION), None)
            if pair is None:
                break
            if n_bis >= MAX_BISECTIONS:
                label, reason = "COUPLED_BACKGROUND_UNRESOLVED", "bisection budget exhausted"
                break
            m = 0.5 * (pair[0] + pair[1])
            pts[m], _ = runner.run(m)
            pts[m]["role"] = "bisection"
            n_bis += 1
        if label is None:
            label, reason = _classify(res, pts, runner, end_states)
    lams, _ = _runs(pts)
    res["evaluations"] = [pts[x] for x in lams]
    res["transitions"] = [{"between": [a, b], "mean_rate_jump_hz": {c: pts[b]["mean_rates_hz"][c] - pts[a]["mean_rates_hz"][c]
                                                                    for c in CLASSES}}
                          for a, b in zip(lams, lams[1:]) if pts[a]["passes"] != pts[b]["passes"]]
    res["verdict"] = label
    res["reason"] = reason
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(label, reason)

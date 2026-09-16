"""V2.1b: frozen V2.1 battery + private mean-controlled shot-noise drive.

Parent: main@40fb739. Sealed bracket: results/v21b_vectors.json (committed
before execution). Single delta vs V2.1: a per-neuron additive drive
schedule x_i(t) = s_i(t) - mu_c, where s_i is an exponentially filtered
private Poisson event train (tau_s = plant excitatory receptor tau) whose
stationary mean equals the class tonic mu_c. The emitter tonic, cells,
topology, efficacies, baseline kernel (HDP detached), engine PRNGKey(11)
and its internal noise stream (noise_scale default 0.5, recorded in the
seal) are the V2.1 plant as executed.

Gates are the V2.1 gates verbatim: run_regime is loaded from the frozen
tests/test_v21_operation.py and called unchanged. Its jtfne handle is
proxied only to add the shot chunk to the all-zero schedule; its result
write is redirected from results/v21_<cell>.json to results/v21b_<cell>.json.

Realization gate (configured != realized): per class, realized shot mean
over the analysis windows within MEAN_TOL of mu_c and realized sigma/mu
within SM_TOL of the discrete-time target. Unrealized cell -> UNRESOLVED,
never PASS. Output-rate shift vs V2.1 is recorded, not gated or tuned.

Verdict: V2_LOCAL_OPERATION_{PASS,FAIL,UNRESOLVED}; passing subset reported,
no cell or regime selected.
"""

import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "results" / "v21b_vectors.json"
V21_BATTERY = ROOT / "tests" / "test_v21_operation.py"
CLASSES = ("E", "PV", "SST", "VIP")


def _load_seal():
    return json.load(open(VECTORS))


def _load_v21():
    """Frozen V2.1 battery module, loaded by path (never edited)."""
    spec = importlib.util.spec_from_file_location("_v21_frozen", V21_BATTERY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import jomission
    # Editable install points at a stale clone; refuse to run against it.
    assert Path(jomission.__file__).resolve().is_relative_to(ROOT), jomission.__file__
    return mod


def shot_params(mu, lam_hz, tau_ms, dt_ms):
    """Discrete-time exact mean control.

    s[t] = d*s[t-1] + A*k[t], k[t] ~ Poisson(p), p = lam*dt, d = exp(-dt/tau).
    E[s] = A*p/(1-d) = mu  ->  A = mu*(1-d)/p.
    Var[s] = A^2*p/(1-d^2)  ->  sigma/mu = sqrt((1-d)^2 / (p*(1-d^2))).
    mu: (n,) native current units; returns A (n,), p, d, sigma/mu target.
    """
    p = lam_hz * dt_ms / 1000.0
    d = math.exp(-dt_ms / tau_ms)
    A = np.asarray(mu, dtype=np.float64) * (1.0 - d) / p
    sm = math.sqrt((1.0 - d) ** 2 / (p * (1.0 - d * d)))
    return A, p, d, sm


def shot_chunks(mu, lam_hz, tau_ms, dt_ms, seed, n_chunk, steps, stats_from=1):
    """Yield float32 schedule chunks (steps, n) = s(t) - mu; private per neuron.

    Computed in float64 (PCG64(seed) Poisson counts, i.i.d. over neurons and
    steps), cast to float32 at the boundary. s starts at its stationary mean.
    Accumulates float64 realization statistics over chunks >= stats_from
    into the returned dict (filled as chunks are consumed).
    """
    mu = np.asarray(mu, dtype=np.float64)
    A, p, d, _ = shot_params(mu, lam_hz, tau_ms, dt_ms)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n = mu.shape[0]
    s = mu.copy()
    stats = {"sum": np.zeros(n), "sumsq": np.zeros(n), "count": 0,
             "cast_err_max": 0.0, "chunks": 0}

    def gen():
        nonlocal s
        for c in range(n_chunk):
            k = rng.poisson(p, size=(steps, n)).astype(np.float64)
            out = np.empty((steps, n), dtype=np.float64)
            for t in range(steps):
                s = s * d + A * k[t]
                out[t] = s
            if c >= stats_from:
                stats["sum"] += out.sum(axis=0)
                stats["sumsq"] += (out * out).sum(axis=0)
                stats["count"] += steps
            x64 = out - mu
            x32 = x64.astype(np.float32)
            stats["cast_err_max"] = max(stats["cast_err_max"],
                                        float(np.abs(x32.astype(np.float64) - x64).max()))
            stats["chunks"] += 1
            yield x32

    return gen(), stats


def realization(stats, mu, cls, sm_target):
    """Per-class realized mean ratio and sigma/mu (float64, analysis windows)."""
    m = stats["sum"] / stats["count"]
    var = stats["sumsq"] / stats["count"] - m * m
    out = {}
    for c in CLASSES:
        sel = cls == c
        mu_c = float(mu[sel][0])
        out[c] = {"mu": mu_c,
                  "mean_ratio": float(m[sel].mean() / mu_c),
                  "sigma_over_mu": float(np.sqrt(np.maximum(var[sel], 0.0)).mean() / mu_c),
                  "sigma_over_mu_target": sm_target}
    return out


class _ShotJaxfne:
    """Forwards to jaxfne; run_continuation adds the next shot chunk."""

    def __init__(self, base, chunks, shape):
        self._base, self._chunks, self._shape = base, chunks, shape
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._base, name)

    def run_continuation(self, step_fn, state, sched):
        import jax.numpy as jnp
        assert tuple(sched.shape) == self._shape, sched.shape
        assert not bool(jnp.any(sched != 0)), "V2.1 schedule expected all-zero"
        x = next(self._chunks)
        self.calls += 1
        return self._base.run_continuation(step_fn, state, sched + jnp.asarray(x, dtype=sched.dtype))


def _redirect_open(name):
    src = f"results/v21_{name}.json"
    dst = ROOT / "results" / f"v21b_{name}.json"

    def _open(path, mode="r", *a, **k):
        if str(path) == src:
            assert "w" in mode
            return open(dst, mode, *a, **k)
        return open(path, mode, *a, **k)
    return _open, dst


def _vectors_blob():
    return subprocess.run(["git", "hash-object", str(VECTORS)], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def run_cell(regime, cell):
    seal = _load_seal()
    fz = seal["frozen"]
    sh = seal["shot"]
    cfg = seal["bracket"]["cells"][f"{regime}_{cell}"]
    v21 = _load_v21()
    assert v21.DT_MS == fz["dt_ms"] and v21.SEED == fz["engine_prng_seed"]
    assert v21.WINDOWS == tuple(fz["analysis_windows"]) and v21.N_CHUNK == fz["n_chunk"]
    vec = json.load(open(ROOT / "results" / "v21_vectors.json"))["vectors"][cfg["tonic_vector"]]
    assert vec == seal["tonic_vectors_copy"][cfg["tonic_vector"]]

    model, _, cls = v21.build_plant(vec)  # class labels only; plant rebuilt inside run_regime
    n = len(cls)
    mu = np.array([vec[c] for c in cls], dtype=np.float64)
    A, p, d, sm = shot_params(mu, cfg["lambda_hz"], sh["tau_ms"], fz["dt_ms"])
    for c in CLASSES:
        assert math.isclose(float(A[cls == c][0]), cfg["amplitude"][c], rel_tol=1e-9)
    assert math.isclose(sm, cfg["sigma_over_mu_discrete"], rel_tol=1e-9)

    chunks, stats = shot_chunks(mu, cfg["lambda_hz"], sh["tau_ms"], fz["dt_ms"],
                                cfg["shot_seed"], v21.N_CHUNK, v21.STEPS_CHUNK,
                                stats_from=min(v21.WINDOWS))
    proxy = _ShotJaxfne(v21.jtfne, chunks, (v21.STEPS_CHUNK, n))
    v21.jtfne = proxy
    name = f"{regime}_{cell}"
    v21.open, dst = _redirect_open(name)

    res = v21.run_regime(name, vec)
    assert proxy.calls == v21.N_CHUNK and stats["chunks"] == v21.N_CHUNK

    real = realization(stats, mu, cls, sm)
    tol = seal["realization_gate"]
    real_ok = all(abs(r["mean_ratio"] - 1.0) <= tol["mean_tol"]
                  and abs(r["sigma_over_mu"] / sm - 1.0) <= tol["sigma_over_mu_tol"]
                  for r in real.values())
    base = json.load(open(ROOT / "results" / f"v21_{regime}.json"))
    ratio = [{c: (r[c] / b[c] if b[c] > 0 else None) for c in CLASSES}
             for r, b in zip(res["rates"], base["rates"])]
    res.update({
        "cell": cell, "regime": regime,
        "shot": {"lambda_hz": cfg["lambda_hz"], "tau_ms": sh["tau_ms"],
                 "sigma_over_mu_discrete": sm, "shot_seed": cfg["shot_seed"],
                 "amplitude": cfg["amplitude"], "p_event_per_step": p},
        "realized": real, "realization_ok": bool(real_ok),
        "cast_float64_to_float32_max_abs": stats["cast_err_max"],
        "rate_ratio_vs_v21_recorded_not_gated": ratio,
        "pass_v21_gates": res["pass"],
        "cell_pass": bool(res["pass"] and real_ok),
        "vectors_blob": _vectors_blob(),
    })
    json.dump(res, open(dst, "w"), indent=2)
    print(name, "cell_pass" if res["cell_pass"] else "no",
          "realized" if real_ok else "UNREALIZED", flush=True)
    return res


# ---- pre-execution: generator validation against analytic moments ----

@pytest.mark.parametrize("sm_target", [2.0, 1.0, 0.5])
def test_shot_generator_moments(sm_target):
    dt, tau = 0.1, 2.0
    lam = 1000.0 / (2.0 * sm_target ** 2 * tau)
    mu = np.full(400, 3.0)
    chunks, st = shot_chunks(mu, lam, tau, dt, seed=123, n_chunk=2, steps=40000, stats_from=1)
    x = [c for c in chunks]
    _, _, _, sm = shot_params(mu, lam, tau, dt)
    m = st["sum"] / st["count"]
    sd = np.sqrt(st["sumsq"] / st["count"] - m * m)
    assert abs(m.mean() / 3.0 - 1.0) < 0.01
    assert abs(sd.mean() / 3.0 / sm - 1.0) < 0.03
    assert abs(sm / sm_target - 1.0) < 0.06  # discrete vs continuous sigma/mu
    # schedule is zero-mean; neurons are private (near-zero cross-correlation)
    assert abs(x[1].astype(np.float64).mean()) < 0.01 * 3.0
    cc = np.corrcoef(x[1][:, :50].T)
    assert np.abs(cc[np.triu_indices(50, 1)]).mean() < 0.02
    # deterministic for a fixed seed
    c2, _ = shot_chunks(mu, lam, tau, dt, seed=123, n_chunk=1, steps=1000)
    c3, _ = shot_chunks(mu, lam, tau, dt, seed=123, n_chunk=1, steps=1000)
    assert np.array_equal(next(c2), next(c3))


V21_BATTERY_BLOB = "7749863506ea2fdca611191edbcd89b434b67edf"  # f7e168a, as executed


def test_v21b_wiring_without_plant():
    """Frozen battery unchanged; its hooks exist; proxy adds exactly one chunk."""
    blob = subprocess.run(["git", "hash-object", str(V21_BATTERY)], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert blob == V21_BATTERY_BLOB
    src = V21_BATTERY.read_text()
    assert src.count("jtfne.run_continuation(step_fn, st, sched)") == 1
    assert src.count('open(f"results/v21_{name}.json", "w")') == 1
    assert src.count("sched = jnp.zeros((STEPS_CHUNK, n), dtype=dtype)") == 1

    import jax.numpy as jnp
    seen = []

    class _Base:
        def run_continuation(self, step_fn, state, sched):
            seen.append(np.asarray(sched))
            return state, None

        other = "forwarded"

    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    proxy = _ShotJaxfne(_Base(), iter([x, x + 1]), (3, 2))
    zero = jnp.zeros((3, 2), dtype=jnp.float32)
    proxy.run_continuation(None, "s", zero)
    proxy.run_continuation(None, "s", zero)
    assert proxy.other == "forwarded" and proxy.calls == 2
    assert np.array_equal(seen[0], x) and np.array_equal(seen[1], x + 1)
    with pytest.raises(AssertionError):
        proxy.run_continuation(None, "s", zero + 1)

    op, dst = _redirect_open("probe_cell")
    assert dst == ROOT / "results" / "v21b_probe_cell.json"
    with op(__file__) as fh:  # non-matching paths pass through read-only
        assert fh.readline()


def test_v21b_seal_consistent():
    """Sealed bracket equals its declared derivation; tonic copy equals V2.1."""
    seal = _load_seal()
    fz, sh = seal["frozen"], seal["shot"]
    assert json.load(open(ROOT / "results" / "v21_vectors.json"))["vectors"] == seal["tonic_vectors_copy"]
    assert sh["tau_ms"] == 2.0 and fz["dt_ms"] == 0.1
    cells = seal["bracket"]["cells"]
    assert len(cells) == len(seal["bracket"]["regimes"]) * len(seal["bracket"]["sigma_over_mu_targets"])
    seeds = [c["shot_seed"] for c in cells.values()]
    assert len(set(seeds)) == len(seeds)
    for key, c in cells.items():
        lam = 1000.0 / (2.0 * c["sigma_over_mu_target"] ** 2 * sh["tau_ms"])
        assert math.isclose(lam, c["lambda_hz"], rel_tol=1e-12)
        assert c["lambda_hz"] < seal["bracket"]["excluded"]["lambda_hz_upper_exclusive"]
        vec = seal["tonic_vectors_copy"][c["tonic_vector"]]
        for cl in CLASSES:
            A, _, _, sm = shot_params([vec[cl]], c["lambda_hz"], sh["tau_ms"], fz["dt_ms"])
            assert math.isclose(float(A[0]), c["amplitude"][cl], rel_tol=1e-9)
            assert math.isclose(sm, c["sigma_over_mu_discrete"], rel_tol=1e-9)


# ---- execution (slow): 3 regimes x 3 sigma/mu cells ----

def _cells(regime):
    return [k.split("_", 1)[1] for k in _load_seal()["bracket"]["cells"] if k.startswith(regime + "_")]


@pytest.mark.parametrize("cell", ["sm2p0", "sm1p0", "sm0p5"])
def test_v21b_low(cell):
    assert cell in _cells("low")
    run_cell("low", cell)


@pytest.mark.parametrize("cell", ["sm2p0", "sm1p0", "sm0p5"])
def test_v21b_mid(cell):
    assert cell in _cells("mid")
    run_cell("mid", cell)


@pytest.mark.parametrize("cell", ["sm2p0", "sm1p0", "sm0p5"])
def test_v21b_high(cell):
    assert cell in _cells("high")
    run_cell("high", cell)


def test_v21b_verdict():
    seal = _load_seal()
    keys = list(seal["bracket"]["cells"])
    regs = {}
    for k in keys:
        f = ROOT / "results" / f"v21b_{k}.json"
        regs[k] = json.load(open(f)) if f.exists() else None
    passing = [k for k, r in regs.items() if r and r["cell_pass"]]
    complete = all(r is not None and "rates" in r for r in regs.values())
    realized = complete and all(r["realization_ok"] for r in regs.values())
    if passing:
        verdict = "V2_LOCAL_OPERATION_PASS"
    elif complete and realized:
        verdict = "V2_LOCAL_OPERATION_FAIL"
    else:
        verdict = "V2_LOCAL_OPERATION_UNRESOLVED"
    per_cell = {k: (None if r is None else {
        "failed_gates": [g for g, v in r["checks"].items() if not v],
        "realization_ok": r["realization_ok"],
        "rates_mean": {c: round(sum(w[c] for w in r["rates"]) / 3, 3) for c in CLASSES},
        "isi_cv_frac_in": [w["frac_in"] for w in r["cv"]],
        "E_rate_CV": [w["E_rate_CV"] for w in r["cv"]],
        "sync_max": max(r["sync"]),
        "drift_max": round(max(r["drift"].values()), 4),
    }) for k, r in regs.items()}
    blobs = {r["vectors_blob"] for r in regs.values() if r}
    lin = {"parent": "main@40fb739",
           "sealed_bracket": "results/v21b_vectors.json",
           "vectors_blob_at_execution": sorted(blobs),
           "vectors_blob_now": _vectors_blob(),
           "passing_subset": passing,
           "per_cell": per_cell,
           "selection": "none performed",
           "verdict": verdict,
           "next_authorized_action": ("STOP; V2.2 locked until review"
                                      if verdict == "V2_LOCAL_OPERATION_PASS"
                                      else "STOP; no retuning")}
    json.dump(lin, open(ROOT / "results" / "v21b_lineage.json", "w"), indent=2)
    print("V2.1b verdict:", verdict, passing)
    assert len(blobs) <= 1 and (not blobs or blobs == {lin["vectors_blob_now"]})
    assert verdict in ("V2_LOCAL_OPERATION_PASS", "V2_LOCAL_OPERATION_FAIL",
                       "V2_LOCAL_OPERATION_UNRESOLVED")

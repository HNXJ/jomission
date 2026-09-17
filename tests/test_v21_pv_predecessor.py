"""SCI-V21-PV-PREDECESSOR: instrumented matched rerun at lambda 0.65625.

Spec (committed and pushed before the procedure runs): results/v21_pv_predecessor_spec.json.
Instrumentation by retention: the engine already returns per-step v, spikes and per-neuron
sources; this driver keeps them. No engine argument differs from the sealed runs, so runs A and B
must reproduce results/v21_pv_stationarity.json exactly.

Runs: A fresh (collapses), B lambda-1 end state (stays active), C the lambda-1 state carrying the
fresh key stream (state of B, noise of A). Tier 3: writes results/v21_pv_predecessor.json; 1 ms
traces go to the session scratch directory, not the repository.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from jomission.harness import operation
from jomission.harness.drive import check_additive_schedule

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_pv_predecessor_spec.json"
OUT = ROOT / "results" / "v21_pv_predecessor.json"
SEALED = ROOT / "results" / "v21_pv_stationarity.json"
CLASSES = ("E", "PV", "SST", "VIP")
LAMBDA_C = 0.65625
N_CHUNK, PRE_CHUNKS = 12, 4
BIN_STEPS = 10           # 1 ms means at dt 0.1 ms
DT_MS = 0.1
SEED = 11
REPRO_TOL_HZ = 0.5
U_TOL = 1e-3
WARMUP_MS = 50           # reconstructed synaptic traces start from zero
BASELINE_S = 5.0
K_SIGMA = 5.0
SUSTAIN_MS = 1000
PRECEDE_MS = 1000
TOL_ABS_HZ, TOL_REL = 1.0, 0.10
TAU_BY_CLASS = {"E": 2.0, "PV": 5.0, "SST": 5.0, "VIP": 5.0}
RUNS = ("A_collapse", "B_active", "C_control")


def _scratch():
    base = os.environ.get("CLAUDE_SCRATCH") or str(Path.home() / "AppData/Local/Temp/claude/pv_predecessor")
    p = Path(base) / "pv_predecessor"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _blob(rel):
    return subprocess.run(["git", "hash-object", str(ROOT / rel)], capture_output=True, text=True).stdout.strip()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spec_pins_match():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for group in ("code", "data"):
        for rel, pin in spec["frozen"][group].items():
            assert _blob(rel) == pin.split()[0], rel
    assert spec["frozen"]["lambda"] == LAMBDA_C
    c = spec["procedure_constants"]
    assert (LAMBDA_C, N_CHUNK, PRE_CHUNKS, BIN_STEPS, SEED, REPRO_TOL_HZ, U_TOL, WARMUP_MS,
            BASELINE_S, K_SIGMA, SUSTAIN_MS, PRECEDE_MS, TOL_ABS_HZ, TOL_REL) == (
        c["lambda"], c["n_chunk"], c["pre_run_chunks"], c["bin_steps"], c["engine_seed"],
        c["reproduction_tol_hz"], c["u_tol"], c["warmup_ms"], c["baseline_s"], c["k_sigma"],
        c["sustain_ms"], c["precede_ms"], c["tolerance_abs_hz"], c["tolerance_rel"])


def bin_mean(x, k=BIN_STEPS):
    """(steps, n) -> (steps // k, n) means. steps must be a multiple of k."""
    s, n = x.shape
    return x[: s // k * k].reshape(s // k, k, n).mean(axis=1)


def deviation_time_ms(trace, dt_ms, baseline_s=BASELINE_S, k=K_SIGMA, sustain_ms=SUSTAIN_MS):
    """First time (ms) the trace leaves its own baseline band by k sigma and stays out for sustain_ms."""
    t = np.asarray(trace, dtype=np.float64)
    nb = int(baseline_s * 1000.0 / dt_ms)
    if t.size <= nb:
        return None
    mu, sd = float(t[:nb].mean()), float(t[:nb].std())
    if sd == 0.0:
        sd = 1e-12
    out = np.abs(t - mu) > k * sd
    need = int(sustain_ms / dt_ms)
    if out.size < need:
        return None
    run = np.convolve(out.astype(np.int32), np.ones(need, dtype=np.int32), mode="valid")
    hit = np.flatnonzero(run == need)
    return float(hit[0] * dt_ms) if hit.size else None


class _Sim:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        import jaxfne as jtfne
        self.cb = _load("tests/test_v21_coupled_background.py", "_v21_cb")
        self.v21 = self.cb._v21()
        self.jax, self.jnp, self.jtfne = jax, jnp, jtfne
        self.mid, self.high = self.cb._vectors()
        self.vec = self.cb.tonic(LAMBDA_C, self.mid, self.high)
        self.model, self.step_fn, self.cls = self.v21.build_plant(self.vec)
        self.n = len(self.cls)
        self.pv = np.flatnonzero(self.cls == "PV")
        el = self.model.params["edge_list"]
        self.pre, self.post = np.asarray(el.pre), np.asarray(el.post)
        self.w = np.asarray(el.weight, dtype=np.float64)
        self.drive = np.asarray(self.model.params["emitter"].drive, dtype=np.float64)
        em = self.model.params["emitter"]
        self.a = np.asarray(em.a, dtype=np.float64)[self.pv]
        self.b = np.asarray(em.b, dtype=np.float64)[self.pv]
        self.d = np.asarray(em.d, dtype=np.float64)[self.pv]
        check_additive_schedule(self.drive, np.zeros((1, self.n)), self.cls)
        self.W = {}                      # class -> (n_pre, n_PV) weight matrix
        pv_col = {j: k for k, j in enumerate(self.pv)}
        for c in CLASSES:
            m = (self.cls[self.pre] == c) & (self.cls[self.post] == "PV")
            W = np.zeros((self.n, self.pv.size))
            for e in np.flatnonzero(m):
                W[self.pre[e], pv_col[self.post[e]]] += self.w[e]
            self.W[c] = W

    def fresh_state(self):
        return self.jtfne.ContinuationState(dynamic=self.jtfne.dynamic_state_from_model(self.model),
                                            prng_key=self.jax.random.PRNGKey(SEED), step_index=0, delay_state=None)

    def lambda1_end_state(self):
        model, step_fn, _ = self.v21.build_plant(self.cb.tonic(1.0, self.mid, self.high))
        sched = self.jnp.zeros((self.v21.STEPS_CHUNK, self.n), dtype=model.params["emitter"].v0.dtype)
        st = self.jtfne.ContinuationState(dynamic=self.jtfne.dynamic_state_from_model(model),
                                          prng_key=self.jax.random.PRNGKey(SEED), step_index=0, delay_state=None)
        for _ in range(PRE_CHUNKS):
            st, out = self.jtfne.run_continuation(step_fn, st, sched)
            self.jax.block_until_ready(out[0])
        return st

    def run(self, state):
        """12 x 5 s at lambda_c, retaining v, spikes and sources. Returns traces and validations."""
        sched = self.jnp.zeros((self.v21.STEPS_CHUNK, self.n), dtype=self.model.params["emitter"].v0.dtype)
        st = state
        v_bins, src_bins, u_bins, syn_bins = [], [], [], {c: [] for c in CLASSES}
        pv_spike_steps, summaries, u_checks = [], [], []
        syn_carry = {c: np.zeros(self.n) for c in CLASSES}
        step0 = 0
        for k in range(N_CHUNK):
            u_start = np.asarray(st.dynamic.u, dtype=np.float64)[self.pv]
            v_start = np.asarray(st.dynamic.v, dtype=np.float64)[self.pv]
            st, out = self.jtfne.run_continuation(self.step_fn, st, sched)
            self.jax.block_until_ready(out[0])
            spikes = np.asarray(out[1]) > 0.5
            v_pv = np.asarray(out[0])[:, self.pv].astype(np.float64)
            src_pv = np.asarray(out[2])[:, self.pv].astype(np.float64)
            steps = v_pv.shape[0]
            s_pv = spikes[:, self.pv]
            # u reconstruction in engine order, then chunk-boundary validation against the carry
            v_prev = np.vstack([v_start[None, :], v_pv[:-1]])
            alpha = DT_MS * self.a
            drive_u = alpha * self.b * v_prev + self.d * s_pv
            u = np.empty_like(v_pv)
            for j in range(self.pv.size):
                u[:, j] = lfilter([1.0], [1.0, -(1.0 - alpha[j])], drive_u[:, j], zi=[u_start[j] * (1.0 - alpha[j])])[0]
            u_checks.append(float(np.max(np.abs(u[-1] - np.asarray(st.dynamic.u, dtype=np.float64)[self.pv]))))
            # per-source-class synaptic input to each PV cell, reconstructed from presynaptic spikes
            for c in CLASSES:
                decay = float(np.exp(-DT_MS / TAU_BY_CLASS[c]))
                sel = self.cls == c
                syn, _ = lfilter([1.0], [1.0, -decay], spikes[:, sel].astype(np.float64), axis=0,
                                 zi=(syn_carry[c][sel] * decay)[None, :])
                syn_carry[c][sel] = syn[-1]
                syn_bins[c].append(bin_mean(syn @ self.W[c][sel]))
            v_bins.append(bin_mean(v_pv))
            u_bins.append(bin_mean(u))
            src_bins.append(bin_mean(src_pv))
            t, i = np.nonzero(s_pv)
            pv_spike_steps.append(np.stack([t + step0, i], axis=1).astype(np.int32))
            if k >= 1:
                summaries.append(operation.window_summary(spikes, self.cls))
            step0 += steps
        return {"v": np.vstack(v_bins), "u": np.vstack(u_bins), "sources": np.vstack(src_bins),
                "syn": {c: np.vstack(syn_bins[c]) for c in CLASSES},
                "pv_spikes": np.vstack(pv_spike_steps), "summaries": summaries,
                "u_max_boundary_err": max(u_checks)}


def _rates(summaries):
    return [{c: w["rates"][c] for c in CLASSES} for w in summaries]


def _channels(res, sim):
    """Population-mean 1 ms channels, warm-up trimmed for reconstructed synaptic input."""
    wu = WARMUP_MS
    ch = {"v_PV": res["v"].mean(axis=1), "u_PV": res["u"].mean(axis=1),
          "sources_engine": res["sources"].mean(axis=1)}
    for c in CLASSES:
        x = res["syn"][c].mean(axis=1).copy()
        x[:wu] = np.nan
        ch[f"input_{c}_to_PV"] = x
    recon = sum(res["syn"][c].mean(axis=1) for c in CLASSES)
    ch["residual_sources_minus_synaptic"] = ch["sources_engine"] - recon
    ch["residual_minus_synaptic_and_tonic"] = ch["residual_sources_minus_synaptic"] - sim.vec["PV"]
    return ch


def _pv_rate_per_s(res, n_pv):
    t = res["pv_spikes"][:, 0]
    n_s = int(N_CHUNK * 5)
    counts = np.bincount(t // int(1000.0 / DT_MS), minlength=n_s)[:n_s]
    return counts / float(n_pv)


def test_v21_pv_predecessor():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    sim = _Sim()
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))["trajectories"]
    st1 = sim.lambda1_end_state()
    inits = {"A_collapse": sim.fresh_state(),
             "B_active": st1,
             "C_control": st1._replace(prng_key=sim.jax.random.PRNGKey(SEED), step_index=0)}
    scratch = _scratch()
    res, rec = {}, {"spec": "results/v21_pv_predecessor_spec.json", "lambda": LAMBDA_C,
                    "scratch_dir": str(scratch), "runs": {}}
    for name in RUNS:
        r = sim.run(inits[name])
        res[name] = r
        np.savez_compressed(scratch / f"{name}.npz", v=r["v"].astype(np.float32), u=r["u"].astype(np.float32),
                            sources=r["sources"].astype(np.float32), pv_spikes=r["pv_spikes"],
                            **{f"syn_{c}": r["syn"][c].astype(np.float32) for c in CLASSES})
        rec["runs"][name] = {"window_rates_hz": _rates(r["summaries"]),
                             "u_max_boundary_err": r["u_max_boundary_err"],
                             "pv_spike_count": int(r["pv_spikes"].shape[0]),
                             "state_hash": hashlib.sha256(np.asarray(inits[name].dynamic.v).tobytes()).hexdigest()[:16]}

    # validation 1: reproduction of the sealed trajectories (A and B only)
    repro = {}
    for name, key in (("A_collapse", "fresh"), ("B_active", "lambda_1_end_state")):
        old = sealed[key]["window_rates_hz"]
        new = rec["runs"][name]["window_rates_hz"]
        d = max(abs(new[k][c] - old[k][c]) for k in range(len(old)) for c in CLASSES)
        repro[name] = {"max_abs_diff_hz": d, "within_tol": d <= REPRO_TOL_HZ}
    rec["reproduction"] = repro

    # validation 2 and 3: u reconstruction and the source split
    rec["u_reconstruction"] = {n: {"max_boundary_err": rec["runs"][n]["u_max_boundary_err"],
                                   "valid": rec["runs"][n]["u_max_boundary_err"] <= U_TOL} for n in RUNS}
    chans = {n: _channels(res[n], sim) for n in RUNS}
    split = {}
    for n in RUNS:
        r1 = chans[n]["residual_sources_minus_synaptic"][WARMUP_MS:]
        r2 = chans[n]["residual_minus_synaptic_and_tonic"][WARMUP_MS:]
        recon_scale = float(np.nanmean(np.abs(sum(res[n]["syn"][c].mean(axis=1) for c in CLASSES)[WARMUP_MS:])))
        split[n] = {"residual_synaptic_only": {"mean": float(r1.mean()), "std": float(r1.std())},
                    "residual_minus_tonic": {"mean": float(r2.mean()), "std": float(r2.std())},
                    "reconstructed_input_scale": recon_scale,
                    "closer_hypothesis": "sources_includes_tonic" if abs(r2.mean()) < abs(r1.mean()) else "sources_synaptic_only"}
    rec["source_split"] = split

    # P1 fate attribution
    late = {n: float(np.mean([w["rates"]["PV"] for w in res[n]["summaries"][-3:]])) for n in RUNS}
    tol = max(TOL_ABS_HZ, TOL_REL * late["B_active"])
    like_b = abs(late["C_control"] - late["B_active"]) <= tol
    like_a = abs(late["C_control"] - late["A_collapse"]) <= max(TOL_ABS_HZ, TOL_REL * max(late["A_collapse"], 1.0))
    fate = ("initial_state" if like_b and not like_a else
            "noise_realization" if like_a and not like_b else "unresolved")
    rec["P1_fate"] = {"late_pv_hz": late, "tolerance_hz": tol, "C_like_B": like_b, "C_like_A": like_a,
                      "attribution": fate}

    # P2 predecessor ordering inside each collapsing run
    order = {}
    for n in RUNS:
        pv_s = _pv_rate_per_s(res[n], sim.pv.size)
        rate_dev_s = deviation_time_ms(pv_s, 1000.0, baseline_s=BASELINE_S, sustain_ms=SUSTAIN_MS)
        dev = {k: deviation_time_ms(v, 1.0) for k, v in chans[n].items()}
        collapsing = late[n] < 1.0
        precede = {k: t for k, t in dev.items()
                   if t is not None and rate_dev_s is not None and t <= rate_dev_s - PRECEDE_MS}
        order[n] = {"pv_rate_deviation_ms": rate_dev_s, "channel_deviation_ms": dev,
                    "preceding_channels": precede, "collapsing": collapsing,
                    "earliest_preceding": min(precede, key=precede.get) if precede else None}
    rec["P2_order"] = order

    # classification
    if not all(v["within_tol"] for v in repro.values()):
        label, reason = "INSTRUMENTATION_PERTURBED_EXECUTION", "run A or B did not reproduce the sealed trajectory"
    else:
        collapsing = [n for n in RUNS if order[n]["collapsing"]]
        cands = {n: order[n]["earliest_preceding"] for n in collapsing}
        first = {n: order[n]["preceding_channels"] for n in collapsing}
        agreed = set.intersection(*[set(f) for f in first.values()]) if first else set()
        earliest = None
        if agreed:
            earliest = min(agreed, key=lambda k: max(first[n][k] for n in collapsing))
        if earliest is None:
            label, reason = "UNRESOLVED", f"no recorded channel precedes the PV-rate separation in every collapsing run ({cands})"
        elif earliest.startswith("input_"):
            label, reason = "PV_RECURRENT_STATE", f"earliest preceding channel {earliest} (recurrent synaptic input)"
        elif earliest.startswith("residual") or earliest == "sources_engine":
            label, reason = "PV_INPUT_STATE", f"earliest preceding channel {earliest}"
        else:
            label, reason = "PV_INTRINSIC_STATE", f"earliest preceding channel {earliest}"
        if not all(rec["u_reconstruction"][n]["valid"] for n in RUNS) and earliest in (None, "u_PV"):
            label, reason = "MISSING_OBSERVABLE", "the u channel failed its reconstruction validation"
    rec["verdict"], rec["reason"] = label, reason
    rec["forbidden_respected"] = spec["forbidden"]
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(label, reason)

"""SCI-V21-GATE-REPAIR: prospective V2 operation battery through the canonical harness.

No plant or network simulation. Checks that jomission.harness.operation:
1. reproduces the frozen V2.1 battery estimators and checks (run_regime executed on
   synthetic rasters through a fake engine; frozen file untouched);
2. reproduces the frozen class-resolved ISI statistic of the V2.1b ISI lineage;
3. answers known cases (regular, silent, above-band);
4. enforces amendment 1 prospectively: full class rate bands and per-class ISI;
5. drives chunks only through the drive-additivity guard.
"""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jomission.harness import gates, operation
from jomission.harness.drive import DriveAdditivityError

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"E": 30, "PV": 10, "SST": 10, "VIP": 10}
DT_MS, STEPS = 0.1, 50000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cls():
    return np.concatenate([np.full(n, c) for c, n in COUNTS.items()])


def _poisson(rng, rate_hz, n):
    return rng.random((STEPS, n)) < rate_hz * DT_MS / 1000.0


def _regular(period_steps, n, phase=0):
    r = np.zeros((STEPS, n), dtype=bool)
    r[phase::period_steps, :] = True
    return r


def _synthetic_chunks(seed=7, sync=False):
    """4 chunks (steps, 60): heterogeneous Poisson E with 3 silent cells, Poisson PV, regular SST, bursty VIP.

    sync=True adds all-E co-spiking on every 67th step (~1.5% of steps) so the sync guard fails.
    """
    rng = np.random.default_rng(seed)
    e_rates = np.linspace(2.0, 25.0, COUNTS["E"])
    e_rates[:3] = 0.0
    chunks = []
    for k in range(4):
        e = np.column_stack([_poisson(rng, r * (1.0 + 0.03 * k), 1)[:, 0] for r in e_rates])
        pv = _poisson(rng, 20.0, COUNTS["PV"])
        sst = _regular(1000, COUNTS["SST"], phase=k)
        vip = np.zeros((STEPS, COUNTS["VIP"]), dtype=bool)
        for i in range(COUNTS["VIP"]):
            starts = np.flatnonzero(rng.random(STEPS) < 2.0 * DT_MS / 1000.0)
            for s in starts:
                vip[s:s + 200:40, i] = True
        if sync:
            e[::67, :] = True
        chunks.append(np.hstack([e, pv, sst, vip]).astype(np.float32))
    return chunks


class _FakeEngine:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.calls = 0

    def run_continuation(self, step_fn, state, sched):
        self.calls += 1
        return state, (None, next(self.chunks))


# ---------------- 1. frozen V2.1 battery as oracle ----------------

@pytest.mark.parametrize("seed,sync", [(7, False), (13, True)])
def test_matches_frozen_v21_battery_estimators(tmp_path, seed, sync):
    v21 = _load("_v21_frozen_oracle", "tests/test_v21_operation.py")
    assert (v21.DT_MS, v21.STEPS_CHUNK, v21.WINDOWS) == (DT_MS, STEPS, (1, 2, 3))
    cls = _cls()
    chunks = _synthetic_chunks(seed, sync)
    for ch in chunks[1:]:                    # per-neuron counts and CVs identical to the frozen estimator
        f_nsp, f_cvs = v21.window_isis(ch)
        nsp, cv = operation.isi_cvs(ch, DT_MS, 4)
        assert np.array_equal(nsp, f_nsp)
        assert np.array_equal(np.sort(cv[~np.isnan(cv)]), np.sort(f_cvs))
    written = {}

    def _open(path, mode="r", *a, **k):
        assert path == "results/v21_oracle.json" and mode == "w"
        written["path"] = tmp_path / "v21_oracle.json"
        return open(written["path"], mode, *a, **k)

    fake_model = SimpleNamespace(params={"emitter": SimpleNamespace(v0=np.zeros(1, dtype=np.float32))})
    v21.build_plant = lambda vec: (fake_model, None, cls)
    v21.jtfne = SimpleNamespace(ContinuationState=lambda **kw: SimpleNamespace(**kw),
                                dynamic_state_from_model=lambda m: None,
                                run_continuation=_FakeEngine(chunks).run_continuation)
    v21.jax = SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda s: s), block_until_ready=lambda x: x)
    v21.jnp = np
    v21.pathway_currents = lambda model, w: {}
    v21.open = _open
    frozen = v21.run_regime("oracle", {"E": 0.0})
    assert json.load(open(written["path"]))["checks"] == frozen["checks"]

    windows = [operation.window_summary(chunks[k], cls, "historical_v21_battery") for k in (1, 2, 3)]
    for w, fr, fcv, fs in zip(windows, frozen["rates"], frozen["cv"], frozen["sync"]):
        for c in COUNTS:
            assert w["rates"][c] == pytest.approx(fr[c], rel=1e-9, abs=1e-9)
        assert sum(w["cv_evaluable"].values()) == fcv["cv_evaluable"]
        assert abs(w["isi_fraction_pooled"] - fcv["frac_in"]) <= 5e-4          # frozen stores 3 decimals
        assert abs(w["rate_cv"]["E"] - fcv["E_rate_CV"]) <= 5e-4
        assert abs(w["rate_cv"]["E"] - 0.3) > 1e-3                              # away from the rounding edge
        assert abs(w["sync"] - fs) <= 5e-6
    rec = operation.evaluate(windows, "historical_v21_battery")
    for c in COUNTS:
        assert rec["drift"][c] == pytest.approx(frozen["drift"][c], rel=1e-9, abs=1e-12)
    assert rec["checks"] == frozen["checks"]
    assert rec["checks"]["sync"] is (not sync)


# ---------------- 2. frozen class-resolved ISI statistic ----------------

def test_matches_frozen_class_isi_statistic():
    isi = _load("_v21b_isi_frozen", "tests/test_v21b_isi_classes.py")
    cls = _cls()
    raster = _synthetic_chunks(seed=11)[2]
    nsp, cv_all = operation.isi_cvs(raster, DT_MS, 4)
    frozen = isi._class_stats(nsp, cv_all, cls)
    w = operation.window_summary(raster, cls)
    for c in COUNTS:
        assert w["isi_fraction"][c] == frozen[c]["frac_in"]
        assert w["cv_evaluable"][c] == frozen[c]["cv_evaluable"]


# ---------------- 3. known answers ----------------

def test_known_answers():
    cls = _cls()
    r = np.zeros((STEPS, 60), dtype=bool)
    r[::1000, :] = True                      # every 100 ms: 10 Hz, CV 0
    r[:, cls == "VIP"] = False               # VIP silent
    w = operation.window_summary(r, cls)
    assert w["rates"]["E"] == pytest.approx(10.0)
    assert w["isi_fraction"]["E"] == 0.0 and w["isi_fraction"]["VIP"] is None
    assert w["silent_fraction"]["VIP"] == 1.0 and w["rate_cv"]["E"] == pytest.approx(0.0, abs=1e-12)
    assert w["active_fraction"]["VIP"] == 0.0 and w["active_fraction"]["E"] == 1.0
    rng = np.random.default_rng(3)
    p = _poisson(rng, 40.0, 60)
    _, cv = operation.isi_cvs(p, DT_MS, 4)
    assert abs(np.nanmean(cv) - 1.0) < 0.05


def test_estimator_edges():
    """Band edges inclusive, >= min_spikes evaluable, sync counts strictly more than half of E."""
    cls = _cls()
    r = np.zeros((STEPS, 60), dtype=bool)

    def spikes(col, isis, start=100):
        idx = start + np.concatenate([[0], np.cumsum(isis)])
        r[idx, col] = True

    spikes(0, [3, 3, 9, 9])          # CV exactly 0.5
    spikes(1, [3, 3, 3, 3, 48])      # CV exactly 1.5
    spikes(2, [3, 3, 3, 3, 49])      # CV just above 1.5
    spikes(3, [5, 5, 5])             # exactly 4 spikes, CV 0
    spikes(4, [5, 5])                # 3 spikes: not evaluable
    _, cv = operation.isi_cvs(r, DT_MS, 4)
    assert cv[0] == 0.5 and cv[1] == 1.5 and cv[2] > 1.5 and cv[3] == 0.0 and np.isnan(cv[4])
    w = operation.window_summary(r, cls)
    assert w["cv_evaluable"]["E"] == 4 and w["isi_fraction"]["E"] == 0.5
    assert w["active_fraction"]["E"] == 4 / int((cls == "E").sum())
    v21 = _load("_v21_frozen_edges", "tests/test_v21_operation.py")
    _, f_cvs = v21.window_isis(r.astype(np.float32))
    assert sorted(f_cvs) == sorted(cv[~np.isnan(cv)])
    q = np.zeros((STEPS, 60), dtype=bool)
    q[1000:1100, 10:25] = True       # 15 of 30 E: exactly half, not counted
    q[2000:2010, 10:26] = True       # 16 of 30 E: counted
    assert operation.window_summary(q, cls)["sync"] == 10 / STEPS


# ---------------- 4. amendment 1 enforced prospectively ----------------

def test_prospective_bands_and_class_isi():
    spec = gates.load_gate()["profiles"]["prospective_v2"]["checks"]
    assert spec["class_rate_bands"]["rate_hz"] == {"E": [2.0, 30.0], "PV": [1.0, 80.0], "SST": [1.0, 80.0], "VIP": [1.0, 80.0]}
    assert spec["isi_cv_by_class"]["classes"] == ["E", "PV", "SST", "VIP"]
    base = {"rates": {"E": 10.0, "PV": 20.0, "SST": 8.0, "VIP": 12.0}, "rate_cv": {"E": 0.4}, "sync": 0.0,
            "isi_fraction": {"E": 0.9, "PV": 0.9, "SST": 0.9, "VIP": 0.9}, "isi_fraction_pooled": 0.9,
            "active_fraction": {"E": 1.0, "PV": 1.0, "SST": 1.0, "VIP": 1.0}}
    ok = operation.evaluate([base] * 3)
    assert ok["verdict"] == "V2_LOCAL_OPERATION_PASS"
    vip_high = copy.deepcopy(base)
    vip_high["rates"]["VIP"] = 130.0
    hist = gates.evaluate("historical_v21_battery", [vip_high] * 3, operation.drift([vip_high] * 3))
    pro = operation.evaluate([vip_high] * 3)
    assert hist["classes_active"] is True and pro["checks"]["class_rate_bands"] is False
    e_irregular_only_pooled = copy.deepcopy(base)
    e_irregular_only_pooled["isi_fraction"]["E"] = 0.7
    assert gates.evaluate("historical_v21_battery", [e_irregular_only_pooled] * 3, operation.drift([base] * 3))["isi_cv"]
    assert operation.evaluate([e_irregular_only_pooled] * 3)["checks"]["isi_cv_by_class"] is False
    assert operation.evaluate([base] * 3)["gate_blob"] == subprocess.run(
        ["git", "hash-object", str(ROOT / "manifests/gates/v2_local_operation.json")],
        capture_output=True, text=True).stdout.strip()


def test_amendment_2_active_fraction_gate():
    """Silencing cells cannot buy the ISI check: f_active >= 0.95 per class, every window, inclusive."""
    spec = gates.load_gate()["profiles"]["prospective_v2"]["checks"]
    assert spec["active_fraction_by_class"] == {"kind": "active_fraction_min_by_class",
                                                "classes": ["E", "PV", "SST", "VIP"], "min_fraction": 0.95}
    cls = np.array(["E"] * 20 + ["PV"] * 20 + ["SST"] * 20 + ["VIP"] * 20)
    rng = np.random.default_rng(5)
    r = _poisson(rng, 10.0, 80).astype(bool)
    r[:, 0] = False                                  # 1 of 20 E silent: 0.95, passes (inclusive)
    w = operation.window_summary(r, cls)
    assert w["active_fraction"]["E"] == 0.95 and w["isi_fraction"]["E"] is not None
    r2 = r.copy()
    r2[:, 1] = False                                 # 2 of 20 silent: 0.90
    w2 = operation.window_summary(r2, cls)
    assert w2["active_fraction"]["E"] == 0.9 and w2["isi_fraction"]["E"] == w["isi_fraction"]["E"]
    assert gates.evaluate("prospective_v2", [w] * 3, operation.drift([w] * 3))["active_fraction_by_class"] is True
    assert gates.evaluate("prospective_v2", [w, w2, w], operation.drift([w] * 3))["active_fraction_by_class"] is False
    hist = gates.load_gate()["profiles"]["historical_v21_battery"]["checks"]
    assert not any(c["kind"] == "active_fraction_min_by_class" for c in hist.values())


def test_amendment_vip_values_match_sealed_cells():
    listed = {"low_sm2p0": 130.2, "mid_sm2p0": 139.5, "high_sm2p0": 159.0, "high_sm1p0": 86.1}
    above = {}
    for r in ("low", "mid", "high"):
        for s in ("sm2p0", "sm1p0", "sm0p5"):
            res = json.load(open(ROOT / "results" / f"v21b_{r}_{s}.json"))
            top = max(w["VIP"] for w in res["rates"])
            if top > 80.0:
                above[f"{r}_{s}"] = round(top, 1)
    assert above == listed


# ---------------- 5. driver ----------------

def test_run_battery_wiring_and_additivity_guard():
    cls = _cls()
    chunks = _synthetic_chunks(seed=5)
    drive = np.full(60, 3.0)
    zero = [np.zeros((10, 60)) for _ in range(4)]
    eng = _FakeEngine(chunks)
    rec = operation.run_battery(eng, None, "s0", cls, drive, zero, expected_schedule_mean={c: 0.0 for c in COUNTS})
    assert eng.calls == 4 and len(rec["windows"]) == 3 and len(rec["drive_additivity"]) == 4
    assert rec["profile"] == "prospective_v2" and rec["verdict"].startswith("V2_LOCAL_OPERATION_")
    expected = [operation.window_summary(chunks[k] > 0.5, cls) for k in (1, 2, 3)]
    assert rec["windows"] == expected
    eng2 = _FakeEngine(chunks)
    with pytest.raises(DriveAdditivityError):
        operation.run_battery(eng2, None, "s0", cls, drive, [np.tile(drive, (10, 1))] * 4)
    assert eng2.calls == 0

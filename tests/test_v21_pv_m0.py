"""SCI-V21-PV-M0: typed decomposition of the two initial states that carry opposite PV fates.

Spec (committed and pushed before the procedure runs): results/v21_pv_m0_spec.json.

h_B(0) was never persisted, so it is regenerated deterministically from the sealed 20 s lambda-1
pre-run and accepted only if it reproduces the sealed hash of dynamic.v AND a 5 s continuation
from it reproduces run B's first sealed window rate within 0.5 Hz. Nothing is transplanted here:
M0 reports which typed blocks differ, not which one causes the fate.

Tier 3: steps the engine and writes results/v21_pv_m0.json.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from jomission.harness import operation
from jomission.harness.drive import check_additive_schedule

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_pv_m0_spec.json"
OUT = ROOT / "results" / "v21_pv_m0.json"
SEALED = ROOT / "results" / "v21_pv_predecessor.json"

LAMBDA_C = 0.65625
PRE_CHUNKS = 4
TRAJECTORY_CHUNKS = 1
REPRO_TOL_HZ = 0.5
SEED = 11
A_HASH = "83ce0d0c3fc44a71"
B_HASH = "8704423170e6ae32"
CLASSES = ("E", "PV", "SST", "VIP")


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def v_hash(dynamic):
    return hashlib.sha256(np.asarray(dynamic.v).tobytes()).hexdigest()[:16]


def leaves(dynamic):
    """Every DynamicState field, flattened through its own structure, as leaf path -> array."""
    out = {}

    def walk(prefix, obj):
        if obj is None:
            out[prefix] = None
        elif hasattr(obj, "_fields"):
            for f in obj._fields:
                walk(f"{prefix}.{f}", getattr(obj, f))
        elif isinstance(obj, dict):
            for k in sorted(obj):
                walk(f"{prefix}.{k}", obj[k])
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(f"{prefix}.{i}", v)
        else:
            out[prefix] = np.asarray(obj)

    for f in dynamic._fields:
        walk(f, getattr(dynamic, f))
    return out


def compare(a, b, cls):
    """Typed-block comparison: presence, shape, identity and the size of the difference."""
    rows = {}
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if x is None and y is None:
            rows[key] = {"present": False, "note": "absent in both states"}
            continue
        if x is None or y is None:
            rows[key] = {"present": True, "note": "present in only one state",
                         "in_A": x is not None, "in_B": y is not None}
            continue
        same_shape = x.shape == y.shape
        row = {"present": True, "shape": list(x.shape), "dtype": str(x.dtype), "same_shape": same_shape}
        if not same_shape:
            rows[key] = row
            continue
        xf, yf = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        diff = np.abs(xf - yf)
        row.update({"identical": bool(np.array_equal(xf, yf)),
                    "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                    "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                    "fraction_differing": float((diff > 0).mean()) if diff.size else 0.0})
        if x.shape and x.shape[0] == cls.size:
            row["by_class"] = {
                c: {"max_abs_diff": float(diff[cls == c].max()), "mean_abs_diff": float(diff[cls == c].mean()),
                    "fraction_differing": float((diff[cls == c] > 0).mean())}
                for c in CLASSES}
        rows[key] = row
    return rows


def window_rates(sim, state, n_chunk):
    """Continue at lambda_c and summarize with the canonical estimator."""
    sched = sim.jnp.zeros((sim.v21.STEPS_CHUNK, sim.n), dtype=sim.model.params["emitter"].v0.dtype)
    check_additive_schedule(sim.drive, np.zeros((1, sim.n)), sim.cls)
    st, rates = state, []
    for _ in range(n_chunk):
        st, out = sim.jtfne.run_continuation(sim.step_fn, st, sched)
        raster = np.asarray(out[1])
        rates.append({c: operation.window_summary(raster, sim.cls)["rates"][c] for c in CLASSES})
    return rates


def test_v21_pv_m0():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    pred = _load("tests/test_v21_pv_predecessor.py", "_v21_pred")
    assert (pred.PRE_CHUNKS, pred.LAMBDA_C, pred.SEED) == (PRE_CHUNKS, LAMBDA_C, SEED)
    sim = pred._Sim()
    cls = np.asarray(sim.cls)

    a0, b0 = sim.fresh_state(), sim.lambda1_end_state()
    rec = {"spec": "results/v21_pv_m0_spec.json", "lambda": LAMBDA_C,
           "regeneration": {"A_hash": v_hash(a0.dynamic), "B_hash": v_hash(b0.dynamic),
                            "A_expected": A_HASH, "B_expected": B_HASH}}
    rec["regeneration"]["hashes_match"] = bool(rec["regeneration"]["A_hash"] == A_HASH
                                               and rec["regeneration"]["B_hash"] == B_HASH)

    got = window_rates(sim, b0, TRAJECTORY_CHUNKS)[0]
    want = sealed["runs"]["B_active"]["window_rates_hz"][0]
    dev = {c: float(got[c] - want[c]) for c in CLASSES}
    rec["trajectory_match"] = {"window_1_rates_hz": {c: float(got[c]) for c in CLASSES},
                               "sealed": want, "deviation_hz": dev,
                               "within_tol": bool(max(abs(v) for v in dev.values()) <= REPRO_TOL_HZ)}

    if not (rec["regeneration"]["hashes_match"] and rec["trajectory_match"]["within_tol"]):
        rec["verdict"] = "STATE_REGENERATION_" + "FAIL"
        rec["reason"] = "a regenerated state failed its v-hash or the 5 s trajectory match; no decomposition is reported"
        OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        raise AssertionError(rec["reason"])

    la, lb = leaves(a0.dynamic), leaves(b0.dynamic)
    rec["blocks"] = compare(la, lb, cls)
    rec["context_excluded"] = {
        "prng_key": {"A": np.asarray(a0.prng_key).tolist(), "B": np.asarray(b0.prng_key).tolist()},
        "step_index": {"A": int(a0.step_index), "B": int(b0.step_index)},
        "reason": "P1 showed swapping the noise stream does not swap fate, so these are context, not causal candidates"}

    differing = [k for k, r in rec["blocks"].items()
                 if r.get("present") and r.get("identical") is False]
    absent = [k for k, r in rec["blocks"].items() if not r.get("present")]
    rec["differing_blocks"] = differing
    rec["absent_blocks"] = absent
    pv = [k for k in differing if rec["blocks"][k].get("by_class", {}).get("PV", {}).get("fraction_differing", 0.0) > 0]
    rec["justified_transplant_arms"] = {
        "order": spec["search_order_for_a_later_lineage"],
        "blocks_differing_in_PV": pv,
        "note": "M0 reports which blocks differ; no arm is executed and none is claimed causal"}

    scratch = Path(sealed["scratch_dir"]) / "m0_states"
    scratch.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, lv in (("h_A0", la), ("h_B0", lb)):
        f = scratch / f"{name}.npz"
        np.savez_compressed(f, **{k.replace(".", "__"): v for k, v in lv.items() if v is not None})
        saved[name] = {"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    rec["saved_states"] = saved

    rec["verdict"] = "STATE_DECOMPOSITION_" + "PASS"
    rec["reason"] = (f"both states regenerate with the sealed v-hash and the 5 s trajectory match holds; "
                     f"{len(differing)} typed blocks differ, {len(pv)} of them in the PV population")
    rec["forbidden_respected"] = spec["forbidden"]
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(rec["verdict"], rec["reason"])
    print("differing blocks:", differing)


def test_spec_constants_match():
    c = json.loads(SPEC.read_text(encoding="utf-8"))["procedure_constants"]
    assert (LAMBDA_C, PRE_CHUNKS, TRAJECTORY_CHUNKS, REPRO_TOL_HZ, SEED, A_HASH, B_HASH) == (
        c["lambda"], c["pre_chunks"], c["trajectory_chunks"], c["repro_tol_hz"], c["engine_seed"],
        c["a_hash"], c["b_hash"])

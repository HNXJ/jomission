"""SCI-V21-PV-M0-R2: the M0 decomposition with the corrected trajectory-match window.

Spec: results/v21_pv_m0_r2_spec.json. Additive correction of SCI-V21-PV-M0, whose executed
STATE_REGENERATION_FAIL label stands: both states regenerated with exactly the sealed v-hashes,
but the check continued one chunk and compared it against a sealed window that is the sealed
driver's *second* chunk (it summarizes only k >= 1). Here two chunks are continued and chunk
index 1 is compared. Nothing else differs — the tolerance, states, build path, decomposition and
classification table are unchanged, and the M0 driver is left verbatim as the record of what ran.

Tier 3: steps the engine and writes results/v21_pv_m0_r2.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_pv_m0_r2_spec.json"
OUT = ROOT / "results" / "v21_pv_m0_r2.json"
SEALED = ROOT / "results" / "v21_pv_predecessor.json"

TRAJECTORY_CHUNKS = 2
COMPARE_CHUNK_INDEX = 1
CLASSES = ("E", "PV", "SST", "VIP")


def _load_m0():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_v21_m0", ROOT / "tests" / "test_v21_pv_m0.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spec_records_the_correction():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    c = spec["procedure_constants"]
    assert (c["trajectory_chunks"], c["compare_chunk_index"]) == (TRAJECTORY_CHUNKS, COMPARE_CHUNK_INDEX)
    assert spec["correction"]["of"].startswith("SCI-V21-PV-M0")
    m0 = _load_m0()
    assert (c["lambda"], c["repro_tol_hz"], c["a_hash"], c["b_hash"]) == (
        m0.LAMBDA_C, m0.REPRO_TOL_HZ, m0.A_HASH, m0.B_HASH)


def test_v21_pv_m0_r2():
    m0 = _load_m0()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    pred = m0._load("tests/test_v21_pv_predecessor.py", "_v21_pred")
    sim = pred._Sim()
    cls = np.asarray(sim.cls)

    a0, b0 = sim.fresh_state(), sim.lambda1_end_state()
    rec = {"spec": "results/v21_pv_m0_r2_spec.json", "lambda": m0.LAMBDA_C,
           "corrects": "results/v21_pv_m0.json (STATE_REGENERATION_FAIL, checking-driver window off by one chunk)",
           "regeneration": {"A_hash": m0.v_hash(a0.dynamic), "B_hash": m0.v_hash(b0.dynamic),
                            "A_expected": m0.A_HASH, "B_expected": m0.B_HASH}}
    rec["regeneration"]["hashes_match"] = bool(rec["regeneration"]["A_hash"] == m0.A_HASH
                                               and rec["regeneration"]["B_hash"] == m0.B_HASH)

    rates = m0.window_rates(sim, b0, TRAJECTORY_CHUNKS)
    got = rates[COMPARE_CHUNK_INDEX]
    want = sealed["runs"]["B_active"]["window_rates_hz"][0]
    dev = {c: float(got[c] - want[c]) for c in CLASSES}
    rec["trajectory_match"] = {"chunks_run": TRAJECTORY_CHUNKS, "compared_chunk_index": COMPARE_CHUNK_INDEX,
                               "all_chunk_rates_hz": [{c: float(r[c]) for c in CLASSES} for r in rates],
                               "sealed": want, "deviation_hz": dev,
                               "within_tol": bool(max(abs(v) for v in dev.values()) <= m0.REPRO_TOL_HZ)}

    if not (rec["regeneration"]["hashes_match"] and rec["trajectory_match"]["within_tol"]):
        rec["verdict"] = "STATE_REGENERATION_" + "FAIL"
        rec["reason"] = "a regenerated state failed its v-hash or the corrected trajectory match"
        OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        raise AssertionError(rec["reason"])

    la, lb = m0.leaves(a0.dynamic), m0.leaves(b0.dynamic)
    rec["blocks"] = m0.compare(la, lb, cls)
    rec["context_excluded"] = {
        "prng_key": {"A": np.asarray(a0.prng_key).tolist(), "B": np.asarray(b0.prng_key).tolist()},
        "step_index": {"A": int(a0.step_index), "B": int(b0.step_index)},
        "reason": "P1 showed swapping the noise stream does not swap fate, so these are context, not causal candidates"}

    differing = [k for k, r in rec["blocks"].items() if r.get("present") and r.get("identical") is False]
    rec["differing_blocks"] = differing
    rec["absent_blocks"] = [k for k, r in rec["blocks"].items() if not r.get("present")]
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
    rec["reason"] = (f"both states regenerate with the sealed v-hash and the corrected trajectory match holds; "
                     f"{len(differing)} typed blocks differ, {len(pv)} of them in the PV population")
    rec["forbidden_respected"] = spec["forbidden"]
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(rec["verdict"], rec["reason"])
    print("differing blocks:", differing)

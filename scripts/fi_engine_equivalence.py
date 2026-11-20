"""f-I engine equivalence: NumPy f-I assay (O1) vs JaxFNE 0.5.8 dense Izhikevich kernel (E).

Criteria (binding, not edited here): docs/receipts/fi_engine_equivalence_criteria.md, declared 2026-11-15.
O1: scripts/fi_single_neuron.py::firing_rates as committed. O2 (scalar float64 oracle) lives in
tests/test_fi_engine_equivalence.py and is loaded by path. E: jaxfne.emitters.simulate_eig_izhikevich.
Amendment 2026-11-15 (informational, no verdict effect): G4 rheobase-adjacent grid, G5 spike timing,
G6 f-I summaries, provenance of the loaded modules.
Writes docs/receipts/fi_engine_equivalence_v1.json only. Reads nothing under results/.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne
from jaxfne.emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS as PRESETS
from jaxfne.emitters import IzhikevichParams, simulate_eig_izhikevich

ROOT = Path(__file__).resolve().parents[1]
import fi_single_neuron as fsn  # noqa: E402  (scripts/ is on sys.path when run by path or imported by tests)

FSN_EXPECTED = ROOT / "scripts" / "fi_single_neuron.py"
if Path(fsn.__file__).resolve() != FSN_EXPECTED.resolve():
    raise RuntimeError(f"fi_single_neuron resolved to {fsn.__file__}, expected {FSN_EXPECTED}")

CRITERIA = ROOT / "docs" / "receipts" / "fi_engine_equivalence_criteria.md"
ORACLE_TEST = ROOT / "tests" / "test_fi_engine_equivalence.py"
RECEIPT = ROOT / "docs" / "receipts" / "fi_engine_equivalence_v1.json"

DT_MS = fsn.DT_MS
DURATION_MS = fsn.DURATION_MS
SEC = DURATION_MS / 1000.0
N_STEPS = int(round(DURATION_MS / DT_MS))
EXTRA = fsn.EXTRA
V_TH = 30.0
ENTRY_POINT = "jaxfne.emitters.simulate_eig_izhikevich"
G0_IDX = list(range(0, EXTRA.size, 8))
REFINE_POINTS = 26
REFINE_STEP = 0.01
REFINE_BELOW = 0.25

SETS = (
    {"id": "E_engine_preset", "cls": "E", "b_override": None},
    {"id": "PV_engine_preset", "cls": "PV", "b_override": None},
    {"id": "SST_engine_preset", "cls": "SST", "b_override": None},
    {"id": "VIP_engine_preset", "cls": "VIP", "b_override": None},
    {"id": "SST_jomission_b021", "cls": "SST", "b_override": fsn.JOMISSION_B["SST"]},
    {"id": "VIP_jomission_b020", "cls": "VIP", "b_override": fsn.JOMISSION_B["VIP"]},
)


def params_of(spec):
    p = dict(PRESETS[spec["cls"]])
    if spec["b_override"] is not None:
        p["b"] = float(spec["b_override"])
    return {"cls": spec["cls"], "a": float(p["a"]), "b": float(p["b"]), "c": float(p["c"]),
            "d": float(p["d"]), "tonic": float(p["drive"])}


def o1_counts(p, extra):
    """O1: committed firing_rates. Returns integer spike counts over 2000 ms and the rates."""
    rates = fsn.firing_rates(p["a"], p["b"], p["c"], p["d"], p["tonic"], extra)
    counts_f = rates * SEC
    counts = np.rint(counts_f).astype(np.int64)
    if not np.array_equal(counts.astype(float), counts_f):
        raise RuntimeError("O1 count is not an integer")
    return counts, rates


def o1_traced(p, extra):
    """O1 replica that also records spike steps. Same operations as fsn.firing_rates, same order."""
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    I = p["tonic"] + extra
    v = np.full(extra.size, c, dtype=float)
    u = b * v
    times = [[] for _ in range(extra.size)]
    for t in range(N_STEPS):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = a * (b * v - u)
        v = v + DT_MS * dv
        u = u + DT_MS * du
        fired = v >= V_TH
        for j in np.flatnonzero(fired):
            times[j].append(t)
        v = np.where(fired, c, v)
        u = np.where(fired, u + d, u)
    return [np.asarray(x, dtype=np.int64) for x in times]


@contextmanager
def x64_enabled():
    prev = bool(jax.config.read("jax_enable_x64"))
    jaxfne.enable_x64()
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def engine_run(p, extra, *, v0_mode="explicit", dtype="float64"):
    """E: simulate_eig_izhikevich, one uncoupled neuron per extra value (W = 0, zero noise).

    The lanes share one call; with W = 0 each lane sees only its own state and drive
    (tests/test_fi_engine_equivalence.py checks lane = single-neuron realization).
    v0_mode 'explicit': v0 = c, u0 = b*c. 'engine_default': v0 = -65, u0 = b*(-65), the engine's
    izhikevich_params_from_labels initialization.
    """
    n = extra.size
    if v0_mode == "explicit":
        v0, u0 = p["c"], p["b"] * p["c"]
    elif v0_mode == "engine_default":
        v0, u0 = -65.0, p["b"] * -65.0
    else:
        raise ValueError(v0_mode)
    with x64_enabled():
        def full(x):
            return jnp.full((n,), float(x), dtype=jnp.float64)

        params = IzhikevichParams(
            a=full(p["a"]), b=full(p["b"]), c=full(p["c"]), d=full(p["d"]),
            drive=jnp.asarray(p["tonic"] + extra, dtype=jnp.float64),
            sign=jnp.zeros((n,), dtype=jnp.float64),
            W=jnp.zeros((n, n), dtype=jnp.float64),
            v0=full(v0), u0=full(u0),
            source_scale=jnp.asarray(1.0, dtype=jnp.float64),
            labels=(p["cls"],) * n,
        )
        _, spk, _ = simulate_eig_izhikevich(params, N_STEPS, DT_MS, jax.random.PRNGKey(0),
                                            dtype=dtype, noise_scale=0.0)
        fired = np.asarray(spk) > 0.5
    steps = [np.flatnonzero(fired[:, j]).astype(np.int64) for j in range(n)]
    return np.array([s.size for s in steps], dtype=np.int64), steps


def diff_stats(o1, e, extra):
    """The one comparison used for every count gate (E minus O1, exact integers)."""
    o1 = np.asarray(o1, dtype=np.int64)
    e = np.asarray(e, dtype=np.int64)
    d = e - o1
    mask = d != 0
    out = {"points": int(d.size), "mismatch_points": int(mask.sum()),
           "max_abs_count_diff": int(np.max(np.abs(d))) if d.size else 0,
           "fraction_abs_diff_le_1": float(np.mean(np.abs(d) <= 1)) if d.size else None,
           "first_mismatch": None, "mismatch_extra_range": None}
    if mask.any():
        i = int(np.flatnonzero(mask)[0])
        out["first_mismatch"] = {"extra": float(extra[i]), "O1": int(o1[i]), "E": int(e[i])}
        ex = extra[mask]
        out["mismatch_extra_range"] = [float(ex.min()), float(ex.max())]
    return out


def timing_stats(o1_steps, e_steps, o1_c, e_c):
    """G5: spike-time vectors (steps) for points where both O1 and E have a spike."""
    both = np.flatnonzero((o1_c > 0) & (e_c > 0))
    one_sided = int(np.sum((o1_c > 0) != (e_c > 0)))
    any_diff = len_diff = first_diff = 0
    max_first = max_step = 0
    for j in both:
        s1, s2 = o1_steps[j], e_steps[j]
        if s1.size != s2.size:
            len_diff += 1
            any_diff += 1
        elif not np.array_equal(s1, s2):
            any_diff += 1
            max_step = max(max_step, int(np.max(np.abs(s1 - s2))))
        if int(s1[0]) != int(s2[0]):
            first_diff += 1
            max_first = max(max_first, abs(int(s1[0]) - int(s2[0])))
    return {"points_both_positive": int(both.size), "points_one_side_silent": one_sided,
            "points_any_spike_time_diff": any_diff, "points_spike_count_length_diff": len_diff,
            "points_first_spike_diff": first_diff, "max_abs_first_spike_diff_steps": int(max_first),
            "max_abs_spike_step_diff_steps": int(max_step)}


def summary(rates):
    s = fsn.summarize(rates)
    return {"rheobase_extra": s["rheobase_extra"],
            "gain_hz_per_unit_above_rheobase": s["gain_hz_per_unit_above_rheobase"]}


def fi_diff(o1_sum, e_sum):
    out = {}
    for k in ("rheobase_extra", "gain_hz_per_unit_above_rheobase"):
        a, b = o1_sum[k], e_sum[k]
        out[k] = None if a is None or b is None else float(b) - float(a)
    return out


def decide(g0_mismatch_points, neg_mismatch_points, g1_mismatch_points):
    """Criteria verdict rules, unchanged."""
    if g0_mismatch_points != 0 or neg_mismatch_points == 0:
        return "INVALID"
    return "EQUIVALENT" if g1_mismatch_points == 0 else "NOT_EQUIVALENT"


def compute_all(oracle):
    """Run G0 to G6 and the negative control. `oracle(a, b, c, d, I, v0, u0, n_steps, dt_ms)`
    returns the list of spike steps of O2."""
    rows = []
    e_by_set = []
    for spec in SETS:
        p = params_of(spec)
        o1_c, o1_rates = o1_counts(p, EXTRA)
        o1_steps = o1_traced(p, EXTRA)
        replica = np.array([s.size for s in o1_steps], dtype=np.int64)
        if not np.array_equal(replica, o1_c):
            raise RuntimeError(f"O1 replica counts differ from firing_rates for {spec['id']}")

        o2_c = np.array([len(oracle(p["a"], p["b"], p["c"], p["d"], p["tonic"] + float(EXTRA[i]),
                                    p["c"], p["b"] * p["c"], N_STEPS, DT_MS)) for i in G0_IDX],
                        dtype=np.int64)
        g0 = diff_stats(o1_c[G0_IDX], o2_c, EXTRA[G0_IDX])

        e_c, e_steps = engine_run(p, EXTRA)
        e_by_set.append(e_c)
        g1 = diff_stats(o1_c, e_c, EXTRA)

        e2_c, _ = engine_run(p, EXTRA, v0_mode="engine_default")
        g2 = diff_stats(o1_c, e2_c, EXTRA)
        e3_c, _ = engine_run(p, EXTRA, dtype="float32")
        g3 = diff_stats(o1_c, e3_c, EXTRA)

        rheo = summary(o1_rates)["rheobase_extra"]
        if rheo is None:
            raise RuntimeError(f"no O1 spike on the grid for {spec['id']}")
        refined = np.round(rheo - REFINE_BELOW + REFINE_STEP * np.arange(REFINE_POINTS), 4)
        g4_o1, _ = o1_counts(p, refined)
        g4_e, _ = engine_run(p, refined)
        g4 = diff_stats(g4_o1, g4_e, refined)
        g4["extra"] = [float(x) for x in refined]
        g4["O1_counts"] = [int(x) for x in g4_o1]
        g4["E_counts"] = [int(x) for x in g4_e]

        g5 = timing_stats(o1_steps, e_steps, o1_c, e_c)

        o1_sum = summary(o1_rates)
        e_sum = summary(e_c / SEC)
        g6 = {"O1": o1_sum, "E": e_sum, "E_minus_O1": fi_diff(o1_sum, e_sum)}

        rows.append({
            "id": spec["id"], "cls": spec["cls"], "params": p,
            "O1_counts": [int(x) for x in o1_c], "E_counts": [int(x) for x in e_c],
            "G0": g0 | {"extra_points": [float(EXTRA[i]) for i in G0_IDX]},
            "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6,
        })

    # negative control: O1 with d + 0.5 against E of the first set, same comparison code as G1
    p0 = params_of(SETS[0])
    neg_c, _ = o1_counts(dict(p0, d=p0["d"] + 0.5), EXTRA)
    neg = diff_stats(neg_c, e_by_set[0], EXTRA)
    neg["set"] = SETS[0]["id"]
    neg["d_perturbed"] = p0["d"] + 0.5

    g0_total = sum(r["G0"]["mismatch_points"] for r in rows)
    g0_points = sum(r["G0"]["points"] for r in rows)
    g1_total = sum(r["G1"]["mismatch_points"] for r in rows)
    verdict = decide(g0_total, neg["mismatch_points"], g1_total)

    ranges = [r["G1"]["mismatch_extra_range"] for r in rows if r["G1"]["mismatch_extra_range"]]
    first = next(((r["id"], r["G1"]["first_mismatch"]) for r in rows if r["G1"]["first_mismatch"]), None)
    discrepancies = []
    if g1_total == 0:
        for r in rows:
            if r["G4"]["mismatch_points"]:
                discrepancies.append({"set": r["id"], "gate": "G4", "mismatch_points": r["G4"]["mismatch_points"]})
            if r["G5"]["points_any_spike_time_diff"]:
                discrepancies.append({"set": r["id"], "gate": "G5",
                                      "points_any_spike_time_diff": r["G5"]["points_any_spike_time_diff"]})

    return {
        "totals": {
            "G0_points": g0_points, "G0_mismatch_points": g0_total,
            "G1_points": sum(len(r["E_counts"]) for r in rows), "G1_mismatch_points": g1_total,
            "G1_first_mismatch_overall": None if first is None else {"set": first[0], **first[1]},
            "G1_mismatch_extra_range_overall": None if not ranges else
                [min(x[0] for x in ranges), max(x[1] for x in ranges)],
            "negative_control": neg,
            "verdict": verdict,
        },
        "discrepancies": discrepancies,
        "sets": rows,
    }


def provenance():
    jx = sys.modules["jaxfne"]
    fs = sys.modules["fi_single_neuron"]
    fs_path = Path(fs.__file__).resolve()
    try:
        fs_rel = str(fs_path.relative_to(ROOT))
    except ValueError:
        fs_rel = None
    return {
        "jaxfne_file": str(Path(jx.__file__).resolve()),
        "jaxfne_emitters_file": str(Path(sys.modules["jaxfne.emitters"].__file__).resolve()),
        "jaxfne_version_metadata": importlib.metadata.version("jaxfne"),
        "fi_single_neuron_file": str(fs_path),
        "fi_single_neuron_relative_to_worktree": fs_rel,
        "jax_version": jax.__version__,
        "python": sys.version.split()[0],
        "criteria_sha256": hashlib.sha256(CRITERIA.read_bytes()).hexdigest(),
    }


def load_oracle():
    spec = importlib.util.spec_from_file_location("_fi_engine_oracle", ORACLE_TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.oracle_spike_steps


def build_receipt(oracle):
    res = compute_all(oracle)
    out = {
        "schema": "jomission.fi_engine_equivalence.v1",
        "status": "unsealed receipt; orchestrator rescoring after independent review",
        "criteria": "docs/receipts/fi_engine_equivalence_criteria.md",
        "script": "scripts/fi_engine_equivalence.py",
        "oracle_O2": "tests/test_fi_engine_equivalence.py::oracle_spike_steps",
        "engine": {
            "package": "jaxfne",
            "version_authority": "0.5.8 (results/jaxfne_058_migration.json)",
            "entry_point": ENTRY_POINT,
            "realization": "one call per parameter set, 81 (or 26) uncoupled lanes, W = 0, noise_scale = 0.0, "
                           "drive = tonic + extra, no drive_schedule; lane = single-neuron checked in tests",
            "key": "PRNGKey(0), unused at zero noise",
            "n_steps": N_STEPS, "dt_ms": DT_MS, "duration_ms": DURATION_MS,
        },
        "dtype": {"G1_G2_G4_G5_G6": "float64 (jax_enable_x64 set inside the call, restored after)",
                  "G3": "float32 (same initialization, engine casts)"},
        "v0_u0": {"G1_explicit": "v0 = c, u0 = b*c (O1 initialization)",
                  "G2_engine_default": "v0 = -65.0, u0 = b*(-65.0) (izhikevich_params_from_labels); differs from c for VIP"},
        "grid": {"extra": [float(x) for x in EXTRA], "points": int(EXTRA.size)},
        "G4_grid": {"rheobase_extra_minus": REFINE_BELOW, "step": REFINE_STEP, "points": REFINE_POINTS},
        "G0_grid_indices": G0_IDX,
        "verdict": res["totals"]["verdict"],
        "totals": res["totals"],
        "discrepancies": res["discrepancies"],
        "sets": res["sets"],
        "provenance": provenance(),
    }
    return out


def main():
    oracle = load_oracle()
    out = build_receipt(oracle)
    RECEIPT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    t = out["totals"]
    for r in out["sets"]:
        print(f"{r['id']}: G0 mism {r['G0']['mismatch_points']}/{r['G0']['points']}; "
              f"G1 mism {r['G1']['mismatch_points']}/81 max|d| {r['G1']['max_abs_count_diff']}; "
              f"G2 mism {r['G2']['mismatch_points']}; G3 mism {r['G3']['mismatch_points']}; "
              f"G4 mism {r['G4']['mismatch_points']}/{REFINE_POINTS}; "
              f"G5 time-diff {r['G5']['points_any_spike_time_diff']}")
    print(f"negative control ({t['negative_control']['set']}, d+0.5): mismatch points "
          f"{t['negative_control']['mismatch_points']}/81")
    print(f"discrepancies: {len(out['discrepancies'])}")
    print(f"VERDICT: {t['verdict']}")
    print("wrote", RECEIPT)


if __name__ == "__main__":
    main()

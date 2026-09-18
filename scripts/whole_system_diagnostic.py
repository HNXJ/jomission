"""WS-DIAG-1: the first whole-system diagnostic, executed exactly as sealed in b55fd5e.

    python scripts/whole_system_diagnostic.py

Realize x : A : y, apply one localized deterministic retinal perturbation, and observe. No
stabilization, no tuning, no parameter scan: gates are evaluated in the sealed order and the
lineage stops at the first load-bearing failure.

Writes results/whole_system_diagnostic.json and results/whole_system_diagnostic_y.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxfne as jtfne
import numpy as np

from jomission.harness.drive import check_additive_schedule
from jomission.tfne import architecture, nf, realize, retina

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = "results/whole_system_diagnostic_spec.json"


def configure(spec_path=DEFAULT_SPEC):
    """Load the sealed spec. Every timing constant comes from it; none is written here."""
    global SPEC, DT, SETTLE_MS, STIM_MS, POST_MS, TOTAL_MS, CHUNK_MS, STEPS_PER_MS
    SPEC = json.loads((ROOT / spec_path).read_text(encoding="utf-8"))
    DT = SPEC["frozen"]["dt_ms"]
    SETTLE_MS = SPEC["duration"]["settle_ms"]
    STIM_MS = SPEC["duration"]["stimulus_ms"]
    POST_MS = SPEC["duration"]["post_stimulus_ms"]
    TOTAL_MS = SPEC["duration"]["total_ms"]
    CHUNK_MS = SPEC["duration"]["chunk_ms"]
    STEPS_PER_MS = int(round(1.0 / DT))
    return SPEC


configure()
OUT = ROOT / "results" / "whole_system_diagnostic.json"
OUT_Y = ROOT / "results" / "whole_system_diagnostic_y.npz"


def build():
    """Scaffold + retina + declared projections, then enforce what an input interface must not have."""
    chain, rules, external, _ = architecture.build()
    normal = nf.normalize(chain, rules, external)
    cfg, _ = realize.scaffold(normal)
    cfg = (cfg.add_column(retina.AREA, [retina.LAYER], retina.N)
              .area_layer_cell_types(retina.AREA, {retina.LAYER: {retina.CLASS: 1.0}}))
    cfg = realize.declare_mechanisms(cfg, normal)
    cfg = realize.declare_projections(cfg, normal)

    # The retina must exist before its receptive fields can be addressed by neuron id, so the
    # cortical model is constructed once to resolve the index map, then again with the retinal
    # projections declared against those ids. Nothing is wired by hand: every retinal edge is a
    # declared connections() rule compiled by the engine.
    probe = realize.construct(cfg)
    index, _ = realize.index_map(probe)
    r0 = index[f"{retina.AREA}.{retina.LAYER}.{retina.CLASS}"][0]
    tonic_e = float(np.asarray(probe.params["emitter"].drive)[index["V1_1.L4.E"][0]])

    rf_decl = {}
    for side, path in retina.HEMIFIELD_TARGET.items():
        area = realize.area_name(path)
        t0, _, n_cells = index[f"{area}.L4.E"]
        for cell, units in enumerate(retina.receptive_fields(side, n_cells)):
            w = retina.edge_weight(len(units), tonic_e)
            cfg = cfg.connections(
                name=f"retina_{side}_{cell:02d}",
                source={"area": retina.AREA, "ids": [r0 + u for u in units]},
                target={"area": area, "layer": "L4", "cell_type": "E", "ids": [t0 + cell]},
                probability=1.0, weight=w, mechanism=retina.MECHANISM)
            rf_decl[f"{side}.{cell}"] = {"n_units": len(units), "weight": w,
                                         "summed_weight": w * len(units), "target_index": t0 + cell}
    model = realize.construct(cfg)
    return model, normal, rf_decl, tonic_e


def enforce(model, index):
    """Remove what the column builder supplies but an input interface must not have.

    Retina^1024 declares no connectivity and Retina is not a CTX instance, so it inherits no local
    connectivity to keep; and an input interface must be silent absent stimulus. Both are removed
    here and both are verified in the receipt rather than trusted.
    """
    import dataclasses

    r0, r1, _ = index[f"{retina.AREA}.{retina.LAYER}.{retina.CLASS}"]
    el = model.params["edge_list"]
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    intra = ((pre >= r0) & (pre < r1)) & ((post >= r0) & (post < r1))
    keep = np.flatnonzero(~intra)
    fields = {}
    for f in dataclasses.fields(el):
        val = getattr(el, f.name)
        if isinstance(val, (jnp.ndarray, np.ndarray)) and np.asarray(val).shape[:1] == pre.shape:
            fields[f.name] = jnp.asarray(np.asarray(val)[keep])
    edges = dataclasses.replace(el, **fields)

    em = model.params["emitter"]
    tonic = np.asarray(em.drive).copy()
    tonic_before = float(tonic[r0:r1].mean())
    tonic[r0:r1] = 0.0
    emitter = dataclasses.replace(em, drive=jnp.asarray(tonic))

    params = dict(model.params)
    params["edge_list"] = edges
    params["emitter"] = emitter
    return dataclasses.replace(model, params=params), {
     "intra_retinal_edges_removed": int(intra.sum()),
     "edges_before": int(pre.size), "edges_after": int(keep.size),
     "retinal_tonic_before": tonic_before,
     "retinal_tonic_after": float(np.asarray(emitter.drive)[r0:r1].mean())}


def groups_of(index):
    """Every populated area x layer x class, in a stable order."""
    return sorted(index)


def schedule_chunk(t0_ms, n_steps, n_neurons, lit_idx, dtype):
    """Zero everywhere except lit retinal units inside the stimulation interval.

    Cortical neurons receive no schedule at all, so their executed input is their emitter tonic
    alone: I_executed = I_emitter + I_schedule with no tonic duplicated anywhere.
    """
    t = t0_ms + np.arange(n_steps) * DT
    on = (t >= SETTLE_MS) & (t < SETTLE_MS + STIM_MS)
    sched = np.zeros((n_steps, n_neurons), dtype=dtype)
    if on.any() and lit_idx.size:
        sched[np.ix_(np.flatnonzero(on), lit_idx)] = retina.STIMULUS_DRIVE
    return sched


def summarize(v, u, spikes, current, H, members, n_ms):
    """Per-group, per-1 ms-bin summaries. Reductions are float64 over float32 traces."""
    out = {}
    for name, idx in members.items():
        n = idx.size
        sp = np.asarray(spikes[:, idx], dtype=np.float64).reshape(n_ms, STEPS_PER_MS, n)
        vv = np.asarray(v[:, idx], dtype=np.float64).reshape(n_ms, STEPS_PER_MS, n)
        uu = np.asarray(u[:, idx], dtype=np.float64).reshape(n_ms, STEPS_PER_MS, n)
        cc = np.asarray(current[:, idx], dtype=np.float64).reshape(n_ms, STEPS_PER_MS, n)
        hh = np.asarray(H[:, idx], dtype=np.float64).reshape(n_ms, STEPS_PER_MS, n)
        per_bin = sp.sum(axis=1)                      # (n_ms, n) spikes per neuron per bin
        out[name] = {
         "rate_hz": per_bin.sum(axis=1) / (n * 1e-3),
         "f_active": (per_bin > 0).mean(axis=1),
         "v_mean": vv.mean(axis=(1, 2)), "v_std": vv.std(axis=(1, 2)),
         "u_mean": uu.mean(axis=(1, 2)), "u_std": uu.std(axis=(1, 2)),
         "current_mean": cc.mean(axis=(1, 2)), "H_mean": hh.mean(axis=(1, 2)),
        }
    return out


def window(series, t0_ms, t1_ms):
    """A named analysis window. Raises rather than silently returning a short or empty slice."""
    a, b = int(round(t0_ms)), int(round(t1_ms))
    if a < 0 or b > series.size or b <= a:
        raise ValueError(f"window [{a}, {b}) ms is not inside a series of {series.size} ms")
    return series[a:b]


def main(out=None, out_y=None, spec_name=DEFAULT_SPEC, spec_commit="b55fd5e"):
    # Resolve against the repo root so a relative --out lands beside the sealed artifacts rather
    # than wherever the process happens to be running.
    out = (ROOT / out) if out else OUT
    out_y = (ROOT / out_y) if out_y else OUT_Y
    for path in (out, out_y):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("") if path.suffix == ".json" else path.touch()
    model, normal, rf_decl, tonic_e = build()
    index, (area, layer, cls) = realize.index_map(model)
    model, enforcement = enforce(model, index)

    rec = {"spec": spec_name, "spec_commit": spec_commit,
           "lineage": "WS-DIAG-1, first whole-system diagnostic",
           "kernel": "baseline", "hdp": False,
           "scope": ("this verdict is scoped to the sealed 5 s horizon. A 5 s diagnostic can establish an "
                     "immediate failure; it cannot establish long-term stability"),
           "enforcement": enforcement,
           "retina": retina.declaration(tonic_e=tonic_e),
           "receptive_fields": rf_decl}

    # Realization re-verification: the cortical graph must be exactly the one R4 sealed.
    realized, n_edges = realize.realized_projections(model, area, layer, cls)
    expected = realize.expected_projections(normal)
    cortical = {k: v for k, v in realized.items() if not k[0].startswith(retina.AREA)}
    retinal = {k: v for k, v in realized.items() if k[0].startswith(retina.AREA)}
    rec["realization_recheck"] = {
     "cortical_identities": len(cortical), "expected": len(expected),
     "missing": [list(k) for k in sorted(expected) if k not in cortical],
     "extra": [list(k) for k in sorted(cortical) if k not in expected],
     "retinal_identities": len(retinal),
     "retinal_edges": int(sum(retinal.values())),
     "total_edges": n_edges,
     "intra_retinal_edges": 0 if enforcement["intra_retinal_edges_removed"] else None}

    n_neurons = int(model.params["emitter"].n_neurons)
    r0 = index[f"{retina.AREA}.{retina.LAYER}.{retina.CLASS}"][0]
    lit_idx = np.asarray([r0 + u for u in retina.spot_units()], dtype=np.int64)
    y0, y1, y_n = index["FEF.L6.E"]

    members = {g: np.arange(index[g][0], index[g][1]) for g in groups_of(index)}

    step_fn, state = jtfne.compile_step_fn(
        model, dt_ms=DT, kernel="baseline", record_weight_trace=False,
        record_current_trace=True, record_u_trace=True)
    state = jtfne.ContinuationState(dynamic=state.dynamic, prng_key=jax.random.PRNGKey(0),
                                    step_index=0, delay_state=state.delay_state)

    # Drive additivity, checked before execution on a stimulated chunk.
    probe_sched = schedule_chunk(SETTLE_MS, STEPS_PER_MS * 10, n_neurons, lit_idx, np.float32)
    rec["drive_check"] = check_additive_schedule(
        np.asarray(model.params["emitter"].drive), probe_sched,
        np.array([f"{a}.{c}" for a, c in zip(area, cls)]))

    n_chunks = int(round(TOTAL_MS / CHUNK_MS))
    chunk_steps = int(round(CHUNK_MS / DT))
    chunk_ms = int(round(CHUNK_MS))
    series = {g: {k: [] for k in ("rate_hz", "f_active", "v_mean", "v_std", "u_mean", "u_std",
                                  "current_mean", "H_mean")} for g in members}
    y_trace = np.zeros((int(round(TOTAL_MS / DT)), y_n), dtype=np.float32)
    plastic = []
    nonfinite = {}

    for c in range(n_chunks):
        t0 = c * CHUNK_MS
        sched = jnp.asarray(schedule_chunk(t0, chunk_steps, n_neurons, lit_idx, np.float32))
        state, outputs = jtfne.run_continuation(step_fn, state, sched)
        # record_weight_trace=False drops the per-edge weight slot, so the optional current and
        # u traces sit at 4 and 5, not 5 and 6. Verified against the state below.
        assert len(outputs) == 6, f"unexpected output arity {len(outputs)}"
        v, spikes, H, current, u = (np.asarray(outputs[0]), np.asarray(outputs[1]),
                                    np.asarray(outputs[3]), np.asarray(outputs[4]), np.asarray(outputs[5]))
        assert np.allclose(v[-1], np.asarray(state.dynamic.v), atol=1e-4), "v trace is not v"
        assert np.allclose(u[-1], np.asarray(state.dynamic.u), atol=1e-4), "u trace is not u"
        for label, arr in (("v", v), ("u", u), ("current", current)):
            bad = int((~np.isfinite(arr)).sum())
            if bad:
                nonfinite[label] = nonfinite.get(label, 0) + bad
        y_trace[c * chunk_steps:(c + 1) * chunk_steps] = v[:, y0:y1]
        for g, s in summarize(v, u, spikes, current, H, members, chunk_ms).items():
            for k, arr in s.items():
                series[g][k].append(arr)
        w = np.asarray(state.dynamic.w)
        theta = np.asarray(state.dynamic.theta_S)
        plastic.append({"t_ms": t0 + CHUNK_MS,
                        "w_mean": float(w.mean()), "w_std": float(w.std()),
                        "w_min": float(w.min()), "w_max": float(w.max()),
                        "theta_mean": float(theta.mean()), "theta_std": float(theta.std()),
                        "H_mean": float(np.asarray(state.dynamic.H).mean())})

    series = {g: {k: np.concatenate(v) for k, v in s.items()} for g, s in series.items()}
    np.savez_compressed(out_y, v=y_trace, dt_ms=DT, neuron_index=np.arange(y0, y1),
                        settle_ms=SETTLE_MS, stimulus_ms=STIM_MS, post_stimulus_ms=POST_MS)

    # ---- gates, in the sealed order -------------------------------------------------
    gates = {}
    gates["execution"] = {"nonfinite": nonfinite, "pass": not nonfinite,
                          "checked": "v, u and synaptic current at every step of every chunk"}

    e_groups = {g: s for g, s in series.items() if g.endswith(".E") and not g.startswith(retina.AREA)}
    areas = sorted({g.split(".")[0] for g in e_groups})

    def area_e_rate(a):
        """Area E population rate: spike-count weighted across that area's E layers."""
        parts = [(series[g]["rate_hz"], members[g].size) for g in e_groups if g.split(".")[0] == a]
        tot = sum(n for _, n in parts)
        return sum(r * n for r, n in parts) / tot

    e_rate = {a: area_e_rate(a) for a in areas}
    final = {a: window(r, TOTAL_MS - 1000.0, TOTAL_MS) for a, r in e_rate.items()}
    low = {a: float(r.mean()) for a, r in final.items() if float(r.mean()) < 0.1}
    gates["collapse"] = {"final_1000ms_E_rate_hz": {a: round(float(r.mean()), 4) for a, r in final.items()},
                         "below_0.1_hz": low, "pass": not low}

    hot = {}
    for g, s in series.items():
        run = np.convolve((s["rate_hz"] > 200.0).astype(int), np.ones(100, dtype=int), mode="valid")
        sat = np.convolve((s["f_active"] >= 1.0).astype(int), np.ones(100, dtype=int), mode="valid")
        if (run == 100).any() or (sat == 100).any():
            hot[g] = {"max_rate_hz": float(s["rate_hz"].max()), "max_f_active": float(s["f_active"].max()),
                      "sustained_over_200hz": bool((run == 100).any()),
                      "sustained_f_active_1": bool((sat == 100).any())}
    gates["runaway"] = {"violations": hot, "pass": not hot,
                        "max_rate_hz_overall": round(float(max(s["rate_hz"].max() for s in series.values())), 3)}

    cv = {}
    for a in areas:
        r = window(e_rate[a], TOTAL_MS - 1000.0, TOTAL_MS)
        mu = float(r.mean())
        cv[a] = float(r.std() / mu) if mu > 0 else float("inf")
    gates["synchrony"] = {"cv_final_1000ms": {a: round(v, 4) for a, v in cv.items()},
                          "threshold": 3.0, "violations": {a: v for a, v in cv.items() if v > 3.0},
                          "pass": all(v <= 3.0 for v in cv.values()),
                          "estimator": "coefficient of variation of the 1 ms-binned E population rate"}

    drift = {}
    t_lo, t_hi = TOTAL_MS - 1500.0, TOTAL_MS
    for a in areas:
        r = window(e_rate[a], t_lo, t_hi)
        if r.size != int(round(t_hi - t_lo)):
            raise ValueError(f"drift window for {a} holds {r.size} ms, need {int(round(t_hi - t_lo))}")
        binned = r.reshape(-1, 100).mean(axis=1)          # 100 ms bins
        t = np.arange(binned.size) * 0.1                  # seconds
        slope = float(np.polyfit(t, binned, 1)[0])
        half = binned.size // 2
        s1 = float(np.polyfit(t[:half], binned[:half], 1)[0])
        s2 = float(np.polyfit(t[half:], binned[half:], 1)[0])
        sustained = (abs(s1) > 0.5 and abs(s2) > 0.5 and np.sign(s1) == np.sign(s2))
        drift[a] = {"slope_hz_per_s": round(slope, 4), "first_half": round(s1, 4),
                    "second_half": round(s2, 4), "sustained": bool(sustained),
                    "counts_as_drift": bool(abs(slope) > 0.5 and sustained)}
    gates["drift"] = {"tolerance_hz_per_s": 0.5, "window_ms": [t_lo, t_hi], "by_area": drift,
                      "pass": not any(d["counts_as_drift"] for d in drift.values()),
                      "rule": "a slope counts only if both halves exceed tolerance with the same sign"}

    base = {a: float(window(e_rate[a], SETTLE_MS - 1000.0, SETTLE_MS).mean()) for a in areas}
    stim = {a: float(window(e_rate[a], SETTLE_MS, SETTLE_MS + STIM_MS).mean()) for a in areas}
    resp = {}
    for a in areas:
        band = max(1.0, 0.10 * abs(base[a]))
        resp[a] = {"baseline_hz": round(base[a], 4), "stimulus_hz": round(stim[a], 4),
                   "delta_hz": round(stim[a] - base[a], 4), "band_hz": round(band, 4),
                   "responded": bool(abs(stim[a] - base[a]) > band)}
    levels = {"V1_driven": resp["V1_1"]["responded"],
              "V4": resp["V4_1"]["responded"] or resp["V4_2"]["responded"],
              "frontal": resp["FEF"]["responded"] or resp["PFC"]["responded"]}
    gates["propagation"] = {"by_area": resp, "levels": levels, "pass": all(levels.values()),
                            "lateral_V1_2": {**resp["V1_2"],
                                             "note": ("the spot lit no retinal unit projecting to V1.2, so any "
                                                      "change here is transfer through X[lat] or the V4 level "
                                                      "and is reported separately, not counted as propagation")},
                            "band_authority": "sealed V2.1 convention max(1 Hz, 10%)"}

    # Retinal receipt: did the interface actually drive anything?
    rg = series[f"{retina.AREA}.{retina.LAYER}.{retina.CLASS}"]
    lit_share = lit_idx.size / float(retina.N)
    rec["retinal_response"] = {
     "n_lit": int(lit_idx.size), "lit_fraction": round(lit_share, 5),
     "population_rate_baseline_hz": round(float(window(rg["rate_hz"], SETTLE_MS - 1000.0, SETTLE_MS).mean()), 4),
     "population_rate_stimulus_hz": round(float(window(rg["rate_hz"], SETTLE_MS, SETTLE_MS + STIM_MS).mean()), 4),
     "population_rate_post_hz": round(float(window(rg["rate_hz"], SETTLE_MS + STIM_MS, TOTAL_MS).mean()), 4),
     "implied_lit_unit_rate_hz": round(float(window(rg["rate_hz"], SETTLE_MS, SETTLE_MS + STIM_MS).mean()) / lit_share, 3),
     "note": ("the retina is silent absent stimulus by construction, so the whole population rate during "
              "stimulation is carried by the lit units alone")}

    rec["plastic_state"] = {
     "by_chunk": plastic,
     "w_changed": bool(abs(plastic[-1]["w_mean"] - plastic[0]["w_mean"]) > 1e-12
                       or abs(plastic[-1]["w_std"] - plastic[0]["w_std"]) > 1e-12),
     "theta_changed": bool(abs(plastic[-1]["theta_mean"] - plastic[0]["theta_mean"]) > 1e-12),
     "sampling": ("w, theta_S and H state are sampled at 500 ms chunk boundaries, not per ms: the per-edge "
                  "weight trace would be 50000 steps x 246844 edges x 4 bytes = 4.9 GB. H is additionally "
                  "recorded per 1 ms as a per-neuron trace"),
     "interpretation": ("under kernel=baseline these are controls. Static values confirm no hidden plastic "
                        "stabilization occurred; they are not evidence that stabilization is working")}

    y = y_trace
    stim_slice = slice(int(SETTLE_MS / DT), int((SETTLE_MS + STIM_MS) / DT))
    base_slice = slice(int((SETTLE_MS - 1000.0) / DT), int(SETTLE_MS / DT))
    rec["y_FEF_L6_E"] = {
     "address": "FEF.L6.E.v(t)", "n_neurons": int(y_n), "dt_ms": DT,
     "artifact": str(out_y.relative_to(ROOT)).replace("\\", "/"),
     "retained": "population-resolved v_i(t); the mean is derived from it, never recorded in its place",
     "v_mean_baseline": round(float(y[base_slice].mean()), 4),
     "v_mean_stimulus": round(float(y[stim_slice].mean()), 4),
     "per_neuron_mean_baseline": [round(float(x), 3) for x in y[base_slice].mean(axis=0)],
     "per_neuron_mean_stimulus": [round(float(x), 3) for x in y[stim_slice].mean(axis=0)],
     "across_neuron_spread_baseline": round(float(y[base_slice].mean(axis=0).std()), 4),
     "across_neuron_spread_stimulus": round(float(y[stim_slice].mean(axis=0).std()), 4)}

    rec["series"] = {g: {k: [round(float(x), 5) for x in v[::10]] for k, v in s.items()}
                     for g, s in series.items()}
    rec["series_note"] = "1 ms summaries decimated by 10 for the record; gates were evaluated at full 1 ms resolution"

    order = SPEC["gates"]["order"]
    fail_label = {"execution": "WHOLE_SYSTEM_EXECUTION_FAIL", "collapse": "WHOLE_SYSTEM_COLLAPSE",
                  "runaway": "WHOLE_SYSTEM_RUNAWAY", "synchrony": "WHOLE_SYSTEM_SYNCHRONY_FAIL",
                  "drift": "WHOLE_SYSTEM_DRIFT", "propagation": "WHOLE_SYSTEM_PROPAGATION_FAIL"}
    first = next((g for g in order if not gates[g]["pass"]), None)
    rec["gates"] = gates
    rec["gate_order"] = order
    rec["verdict"] = fail_label[first] if first else "WHOLE_SYSTEM_DIAGNOSTIC_" + "PASS"
    rec["fail_boundary"] = (f"first load-bearing failure: {first}" if first else
                            "none: every gate passed in the sealed order")
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")

    print(rec["verdict"], "|", rec["fail_boundary"])
    print("  enforcement:", enforcement)
    print("  realization recheck: cortical", rec["realization_recheck"]["cortical_identities"],
          "missing", rec["realization_recheck"]["missing"], "extra", rec["realization_recheck"]["extra"],
          "| retinal edges", rec["realization_recheck"]["retinal_edges"])
    print("  retina:", rec["retinal_response"]["population_rate_baseline_hz"], "->",
          rec["retinal_response"]["population_rate_stimulus_hz"], "Hz (lit-unit",
          rec["retinal_response"]["implied_lit_unit_rate_hz"], "Hz)")
    for g in order:
        print(f"  {g}: {'pass' if gates[g]['pass'] else 'FAIL'}")
    print("  E rate baseline -> stimulus by area:")
    for a in sorted(resp):
        print(f"    {a}: {resp[a]['baseline_hz']} -> {resp[a]['stimulus_hz']} "
              f"(delta {resp[a]['delta_hz']}, band {resp[a]['band_hz']}, responded {resp[a]['responded']})")
    print("  y FEF.L6.E v_mean:", rec["y_FEF_L6_E"]["v_mean_baseline"], "->",
          rec["y_FEF_L6_E"]["v_mean_stimulus"])
    print("  w changed:", rec["plastic_state"]["w_changed"], "| theta changed:",
          rec["plastic_state"]["theta_changed"])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-y", default=None)
    ap.add_argument("--spec-commit", default="b55fd5e")
    a = ap.parse_args()
    configure(a.spec)
    main(Path(a.out) if a.out else None, Path(a.out_y) if a.out_y else None, a.spec, a.spec_commit)

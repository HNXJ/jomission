"""SCI-EI-NATIVE-STATE-1 retention re-run: record native per-cell state, change nothing else.

Forked from scripts/whole_system_ei_regime_arm.py, sealed in c98c85d, rather
than edited in place. The sealed source is unchanged.

Additive changes, recording only:
  retention     per-neuron v, u, total current and spikes at engine dt for one
                representative cell per class (first neuron of V1_1.L4.<class>,
                deterministic and stated in the receipt), plus full-population
                1 ms spike trains. The run loop already materializes these
                arrays per chunk; this fork additionally accumulates them.
  record        a retention receipt block (representative identities, array
                shapes, construction-identity checks). No verdict here.

Construction, regime, tonic, dt, seed, duration, kernel, delay, stimulus,
chunking, gates and the run loop are the intact arm's, untouched.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxfne as jtfne
import numpy as np

from jomission.harness.drive import check_additive_schedule
from jomission.qualification.b1_targets import B1_CLASS_TARGETS
from jomission.tfne import architecture, nf, o_authority, realize, retina

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = "results/whole_system_authority_bracket_spec.json"


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


def arm_tag(g):
    """Stable filename tag for an arm: 0.0 -> g000, 0.05 -> g050, 0.5 -> g500."""
    return "g%03d" % round(g * 1000)


def cell_tag(g, stimulus, seed):
    """Name one cell: the arm, the stimulus and the seed. Intact only."""
    if stimulus not in ("on", "off"):
        raise ValueError(f"stimulus must be 'on' or 'off', not {stimulus!r}")
    return f"intact_{arm_tag(g)}_{stimulus}_s{int(seed)}"


def outputs_for(g, stimulus="on", seed=0):
    tag = cell_tag(g, stimulus, seed)
    return (
        ROOT / "results" / f"whole_system_ei_native_state_{tag}.json",
        ROOT / "results" / f"whole_system_ei_native_state_{tag}_y.npz",
    )


def build():
    """Scaffold + retina + declared projections, then enforce what an input interface must not have."""
    chain, rules, external, _ = architecture.build()
    normal = nf.normalize(chain, rules, external)
    cfg, _ = realize.scaffold(normal)
    cfg = cfg.add_column(retina.AREA, [retina.LAYER], retina.N).area_layer_cell_types(
        retina.AREA, {retina.LAYER: {retina.CLASS: 1.0}}
    )
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
                probability=1.0,
                weight=w,
                mechanism=retina.MECHANISM,
            )
            rf_decl[f"{side}.{cell}"] = {
                "n_units": len(units),
                "weight": w,
                "summed_weight": w * len(units),
                "target_index": t0 + cell,
            }
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
        "edges_before": int(pre.size),
        "edges_after": int(keep.size),
        "retinal_tonic_before": tonic_before,
        "retinal_tonic_after": float(np.asarray(emitter.drive)[r0:r1].mean()),
    }


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


def summarize(v, u, spikes, current, H, members, n_ms, A=None):
    """Per-group, per-1 ms-bin summaries. Reductions are float64 over float32 traces."""
    out = {}

    def binned(arr, idx, n):
        return np.asarray(arr[:, idx], dtype=np.float64).reshape(n_ms, STEPS_PER_MS, n)

    for name, idx in members.items():
        n = idx.size
        sp = binned(spikes, idx, n)
        vv = binned(v, idx, n)
        hh = binned(H, idx, n)
        per_bin = sp.sum(axis=1)  # (n_ms, n) spikes per neuron per bin
        row = {
            "rate_hz": per_bin.sum(axis=1) / (n * 1e-3),
            "f_active": (per_bin > 0).mean(axis=1),
            "v_mean": vv.mean(axis=(1, 2)),
            "v_std": vv.std(axis=(1, 2)),
            "H_mean": hh.mean(axis=(1, 2)),
        }
        if u is not None:
            uu = binned(u, idx, n)
            row["u_mean"], row["u_std"] = uu.mean(axis=(1, 2)), uu.std(axis=(1, 2))
        if current is not None:
            row["current_mean"] = binned(current, idx, n).mean(axis=(1, 2))
        if A is not None:
            row["A_mean"] = binned(A, idx, n).mean(axis=(1, 2))
        out[name] = row
    return out


def window(series, t0_ms, t1_ms):
    """A named analysis window. Raises rather than silently returning a short or empty slice."""
    a, b = int(round(t0_ms)), int(round(t1_ms))
    if a < 0 or b > series.size or b <= a:
        raise ValueError(f"window [{a}, {b}) ms is not inside a series of {series.size} ms")
    return series[a:b]


def main(
    g_auth,
    stimulus="on",
    seed=0,
    out=None,
    out_y=None,
    spec_name=DEFAULT_SPEC,
    spec_commit=None,
):
    # Resolve against the repo root so a relative --out lands beside the sealed artifacts rather
    # than wherever the process happens to be running.
    default_out, default_out_y = outputs_for(g_auth, stimulus, seed)
    out = (ROOT / out) if out else default_out
    out_y = (ROOT / out_y) if out_y else default_out_y
    started = time.time()
    for path in (out, out_y):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("") if path.suffix == ".json" else path.touch()
    model, normal, rf_decl, tonic_e = build()
    index, (area, layer, cls) = realize.index_map(model)
    model, enforcement = enforce(model, index)

    # EQUAL_G: authority is the specification, weight is derived from it. Applied after
    # enforcement so the retinal edges it removes are already gone, and before compilation so
    # the kernel never sees the inherited weights.
    model, authority = o_authority.realize_equal_g(model, index, area, layer, cls, g_auth)
    if not authority["acceptance"]["pass"]:
        raise ValueError(
            f"EQUAL_G realization failed its own acceptance at g={g_auth}: "
            f"max|K_long/K_local - g| = {authority['acceptance']['max_abs_error']}"
        )

    rec = {
        "spec": spec_name,
        "spec_commit": spec_commit,
        "g": g_auth,
        "arm": "intact",
        "g_tag": arm_tag(g_auth),
        "cell": cell_tag(g_auth, stimulus, seed),
        "stimulus": stimulus,
        "seed": int(seed),
        "pairing": (
            "ON and OFF of one pair share this seed and every other setting, so their "
            "difference is the stimulus alone"
        ),
        "o_authority": authority,
        "lineage": SPEC.get("lineage_id", "WS-DIAG-1"),
        "kernel": "baseline",
        "hdp": False,
        "kernel_config": "frozen block of results/whole_system_ei_regime_spec.json (kernel baseline, hdp false)",
        "scope": (
            "this diagnosis is scoped to the sealed 13 s horizon at g = 0 with the intact "
            "construction. It attributes the observed E-I rate regime; it does not establish "
            "what the coupled system does"
        ),
        "enforcement": enforcement,
        "retina": retina.declaration(tonic_e=tonic_e),
        "receptive_fields": rf_decl,
    }

    # Realization re-verification: the cortical graph must be exactly the one R4 sealed.
    realized, n_edges = realize.realized_projections(model, area, layer, cls)
    expected = realize.expected_projections(normal)
    cortical = {k: v for k, v in realized.items() if not k[0].startswith(retina.AREA)}
    retinal = {k: v for k, v in realized.items() if k[0].startswith(retina.AREA)}
    rec["realization_recheck"] = {
        "cortical_identities": len(cortical),
        "expected": len(expected),
        "missing": [list(k) for k in sorted(expected) if k not in cortical],
        "extra": [list(k) for k in sorted(cortical) if k not in expected],
        "retinal_identities": len(retinal),
        "retinal_edges": int(sum(retinal.values())),
        "total_edges": n_edges,
        "intra_retinal_edges": 0 if enforcement["intra_retinal_edges_removed"] else None,
    }

    n_neurons = int(model.params["emitter"].n_neurons)
    r0 = index[f"{retina.AREA}.{retina.LAYER}.{retina.CLASS}"][0]
    if stimulus == "on":
        lit_idx = np.asarray([r0 + u for u in retina.spot_units()], dtype=np.int64)
    elif stimulus == "off":
        # No unit is lit for the whole run. schedule_chunk writes nothing, so the retinal
        # schedule is identically zero while tonic, interface and every other input are
        # untouched. This is the paired control for the ON run at the same seed.
        lit_idx = np.zeros((0,), dtype=np.int64)
    else:
        raise ValueError(f"stimulus must be 'on' or 'off', not {stimulus!r}")
    y0, y1, y_n = index["FEF.L6.E"]

    members = {g: np.arange(index[g][0], index[g][1]) for g in groups_of(index)}

    # Representative cells: first neuron of V1_1.L4.<class> per class.
    # Deterministic, stated here and in the receipt; any fixed choice satisfies
    # the spec's "at least one representative cell per class".
    rep_cells = {}
    for c in ("E", "PV", "SST", "VIP"):
        g = f"V1_1.L4.{c}"
        if g not in members or members[g].size == 0:
            raise ValueError(f"representative group {g} missing or empty")
        rep_cells[c] = int(members[g][0])
    rep_idx = np.array([rep_cells[c] for c in ("E", "PV", "SST", "VIP")], dtype=np.int64)

    # Kernel config comes from the sealed spec's frozen block (kernel baseline,
    # hdp false), which carries no top-level kernel block. Values below are that
    # block restated, not new choices.
    kspec = {"kernel": "baseline"}
    kkw = dict(kspec.get("hdp_params", {}))
    rule_params = dict(kspec.get("hdp_rule_params", {}))
    if kspec.get("hdp_rule_module"):
        # A rule registered by this repo, not one the engine ships. Register it before compiling
        # and record the name the module itself returns rather than the name the spec assumed.
        import importlib

        mod = importlib.import_module(kspec["hdp_rule_module"])
        registered = mod.ensure_registered()
        if registered != kspec["hdp_rule"]:
            raise ValueError(
                f"spec names hdp_rule {kspec['hdp_rule']!r} but "
                f"{kspec['hdp_rule_module']} registered {registered!r}"
            )
        if kspec.get("inject_tonic_drive"):
            # compile_step_fn does not pass params.drive to a registered rule.
            rule_params.update(mod.tonic_params(model))
        rec["hdp_rule_registration"] = {
            "module": kspec["hdp_rule_module"],
            "registered_name": registered,
            "tonic_injected": bool(kspec.get("inject_tonic_drive")),
            "rule_param_keys": sorted(rule_params),
        }
    if kspec.get("hdp_rule"):
        kkw["hdp_rule"] = kspec["hdp_rule"]
    if rule_params:
        kkw["hdp_rule_params"] = rule_params
    step_fn, state = jtfne.compile_step_fn(
        model,
        dt_ms=DT,
        kernel=kspec["kernel"],
        record_weight_trace=False,
        record_current_trace=True,
        record_u_trace=True,
        **kkw,
    )
    state = jtfne.ContinuationState(
        dynamic=state.dynamic,
        prng_key=jax.random.PRNGKey(int(seed)),
        step_index=0,
        delay_state=state.delay_state,
    )

    # Drive additivity, checked before execution on a stimulated chunk.
    probe_sched = schedule_chunk(SETTLE_MS, STEPS_PER_MS * 10, n_neurons, lit_idx, np.float32)
    rec["drive_check"] = check_additive_schedule(
        np.asarray(model.params["emitter"].drive),
        probe_sched,
        np.array([f"{a}.{c}" for a, c in zip(area, cls)]),
    )

    n_chunks = int(round(TOTAL_MS / CHUNK_MS))
    chunk_steps = int(round(CHUNK_MS / DT))
    chunk_ms = int(round(CHUNK_MS))
    series = None
    y_trace = np.zeros((int(round(TOTAL_MS / DT)), y_n), dtype=np.float32)
    plastic = []
    nonfinite = {}

    for c in range(n_chunks):
        t0 = c * CHUNK_MS
        sched = jnp.asarray(schedule_chunk(t0, chunk_steps, n_neurons, lit_idx, np.float32))
        state, outputs = jtfne.run_continuation(step_fn, state, sched)
        # record_weight_trace=False drops the per-edge weight slot, so with the optional traces the
        # tuple is (v, spikes, sources, H, current, u). The hdp kernel ignores those two flags and
        # returns arity 4; that is an observability difference between arms, recorded not assumed.
        if len(outputs) == 6:
            v, spikes, H, current, u = (
                np.asarray(outputs[0]),
                np.asarray(outputs[1]),
                np.asarray(outputs[3]),
                np.asarray(outputs[4]),
                np.asarray(outputs[5]),
            )
            assert np.allclose(u[-1], np.asarray(state.dynamic.u), atol=1e-4), "u trace is not u"
        elif len(outputs) == 4:
            v, spikes, H = np.asarray(outputs[0]), np.asarray(outputs[1]), np.asarray(outputs[3])
            current = u = None
        else:
            raise ValueError(f"unexpected output arity {len(outputs)}")
        # A rule declaring h_shape=(k,) carries H as (T, N, k). Coordinate 0 is H itself; the rest
        # are its own state, recorded under their declared names rather than averaged into H.
        A = None
        if H.ndim == 3:
            if H.shape[2] != 2:
                raise ValueError(f"unexpected H coordinate count {H.shape[2]}; expected (H, A)")
            H, A = H[:, :, 0], H[:, :, 1]
        assert np.allclose(v[-1], np.asarray(state.dynamic.v), atol=1e-4), "v trace is not v"
        for label, arr in (("v", v), ("H", H), ("u", u), ("current", current)):
            if arr is None:
                continue
            bad = int((~np.isfinite(arr)).sum())
            if bad:
                nonfinite[label] = nonfinite.get(label, 0) + bad
        y_trace[c * chunk_steps : (c + 1) * chunk_steps] = v[:, y0:y1]
        chunk_summary = summarize(v, u, spikes, current, H, members, chunk_ms, A=A)
        if series is None:
            series = {g: {k: [] for k in row} for g, row in chunk_summary.items()}
        for g, row in chunk_summary.items():
            for k, arr in row.items():
                series[g][k].append(arr)
        w = np.asarray(state.dynamic.w)
        theta = np.asarray(state.dynamic.theta_S)
        Hs = np.asarray(state.dynamic.H)
        As = None
        if Hs.ndim == 2 and Hs.shape[1] == 2:
            Hs, As = Hs[:, 0], Hs[:, 1]
        us = np.asarray(state.dynamic.u)
        if not np.isfinite(us).all():
            nonfinite["u_state"] = nonfinite.get("u_state", 0) + int((~np.isfinite(us)).sum())
        plastic.append(
            {
                "t_ms": t0 + CHUNK_MS,
                "u_mean_state": float(us.mean()),
                "H_mean": float(Hs.mean()),
                "H_min": float(Hs.min()),
                "H_max": float(Hs.max()),
                "H_p05": float(np.percentile(Hs, 5)),
                "H_p95": float(np.percentile(Hs, 95)),
                "H_frac_above_5": float((Hs > 5.0).mean()),
                "w_mean": float(w.mean()),
                "w_std": float(w.std()),
                "w_min": float(w.min()),
                "w_max": float(w.max()),
                # theta_S is empty when the rule declares no such target; reporting a
                # mean over an empty slice would write NaN into the receipt.
                **(
                    {"theta_mean": float(theta.mean()), "theta_std": float(theta.std())}
                    if theta.size
                    else {"theta": "absent: rule declares no theta_S target"}
                ),
                **(
                    {}
                    if As is None
                    else {
                        "A_mean_state": float(As.mean()),
                        "A_min": float(As.min()),
                        "A_max": float(As.max()),
                        "A_frac_zero": float((As <= 0.0).mean()),
                    }
                ),
            }
        )

    series = {g: {k: np.concatenate(v) for k, v in s.items()} for g, s in series.items()}
    rep = {k: np.concatenate(v) for k, v in rep_acc.items()}
    y_npz_kwargs = dict(
        v=y_trace,
        dt_ms=DT,
        neuron_index=np.arange(y0, y1),
        settle_ms=SETTLE_MS,
        stimulus_ms=STIM_MS,
        post_stimulus_ms=POST_MS,
        rep_index=rep_idx,
        rep_identity=np.array([f"V1_1.L4.{c}" for c in ("E", "PV", "SST", "VIP")]),
        rep_v=rep["v"].astype(np.float32),
        rep_u=rep["u"].astype(np.float32),
        rep_current=rep["current"].astype(np.float32),
        rep_spikes=(rep["spikes"] > 0.5).astype(np.uint8),
        spikes_1ms=np.concatenate(spikes_1ms).astype(np.uint8),
    )

    # ---- gates, in the sealed order -------------------------------------------------
    gates = {}
    checked = (
        ["v", "H"]
        + ([] if u is None else ["u"])
        + ([] if current is None else ["synaptic current"])
    )
    gates["execution"] = {
        "nonfinite": nonfinite,
        "pass": not nonfinite,
        "checked": f"{', '.join(checked)} at every step of every chunk",
        "u_and_current_per_step": u is not None,
        "note": (
            None
            if u is not None
            else "the hdp kernel ignores record_current_trace and record_u_trace and "
            "returns arity 4, so u is checked per chunk from state and per-step "
            "synaptic current is unavailable"
        ),
    }

    e_groups = {
        g: s for g, s in series.items() if g.endswith(".E") and not g.startswith(retina.AREA)
    }
    areas = sorted({g.split(".")[0] for g in e_groups})

    def area_e_rate(a):
        """Area E population rate: spike-count weighted across that area's E layers."""
        parts = [(series[g]["rate_hz"], members[g].size) for g in e_groups if g.split(".")[0] == a]
        tot = sum(n for _, n in parts)
        return sum(r * n for r, n in parts) / tot

    e_rate = {a: area_e_rate(a) for a in areas}
    final = {a: window(r, TOTAL_MS - 1000.0, TOTAL_MS) for a, r in e_rate.items()}
    low = {a: float(r.mean()) for a, r in final.items() if float(r.mean()) < 0.1}
    gates["collapse"] = {
        "final_1000ms_E_rate_hz": {a: round(float(r.mean()), 4) for a, r in final.items()},
        "below_0.1_hz": low,
        "pass": not low,
    }

    hot = {}
    for g, s in series.items():
        run = np.convolve((s["rate_hz"] > 200.0).astype(int), np.ones(100, dtype=int), mode="valid")
        sat = np.convolve((s["f_active"] >= 1.0).astype(int), np.ones(100, dtype=int), mode="valid")
        if (run == 100).any() or (sat == 100).any():
            hot[g] = {
                "max_rate_hz": float(s["rate_hz"].max()),
                "max_f_active": float(s["f_active"].max()),
                "sustained_over_200hz": bool((run == 100).any()),
                "sustained_f_active_1": bool((sat == 100).any()),
            }
    gates["runaway"] = {
        "violations": hot,
        "pass": not hot,
        "max_rate_hz_overall": round(float(max(s["rate_hz"].max() for s in series.values())), 3),
    }

    cv = {}
    for a in areas:
        r = window(e_rate[a], TOTAL_MS - 1000.0, TOTAL_MS)
        mu = float(r.mean())
        cv[a] = float(r.std() / mu) if mu > 0 else float("inf")
    gates["synchrony"] = {
        "cv_final_1000ms": {a: round(v, 4) for a, v in cv.items()},
        "threshold": 3.0,
        "violations": {a: v for a, v in cv.items() if v > 3.0},
        "pass": all(v <= 3.0 for v in cv.values()),
        "estimator": "coefficient of variation of the 1 ms-binned E population rate",
    }

    drift = {}
    t_lo, t_hi = TOTAL_MS - 1500.0, TOTAL_MS
    for a in areas:
        r = window(e_rate[a], t_lo, t_hi)
        if r.size != int(round(t_hi - t_lo)):
            raise ValueError(
                f"drift window for {a} holds {r.size} ms, need {int(round(t_hi - t_lo))}"
            )
        binned = r.reshape(-1, 100).mean(axis=1)  # 100 ms bins
        t = np.arange(binned.size) * 0.1  # seconds
        slope = float(np.polyfit(t, binned, 1)[0])
        half = binned.size // 2
        s1 = float(np.polyfit(t[:half], binned[:half], 1)[0])
        s2 = float(np.polyfit(t[half:], binned[half:], 1)[0])
        sustained = abs(s1) > 0.5 and abs(s2) > 0.5 and np.sign(s1) == np.sign(s2)
        drift[a] = {
            "slope_hz_per_s": round(slope, 4),
            "first_half": round(s1, 4),
            "second_half": round(s2, 4),
            "sustained": bool(sustained),
            "counts_as_drift": bool(abs(slope) > 0.5 and sustained),
        }
    gates["drift"] = {
        "tolerance_hz_per_s": 0.5,
        "window_ms": [t_lo, t_hi],
        "by_area": drift,
        "pass": not any(d["counts_as_drift"] for d in drift.values()),
        "rule": "a slope counts only if both halves exceed tolerance with the same sign",
    }

    base = {a: float(window(e_rate[a], SETTLE_MS - 1000.0, SETTLE_MS).mean()) for a in areas}
    stim = {a: float(window(e_rate[a], SETTLE_MS, SETTLE_MS + STIM_MS).mean()) for a in areas}
    resp = {}
    for a in areas:
        band = max(1.0, 0.10 * abs(base[a]))
        resp[a] = {
            "baseline_hz": round(base[a], 4),
            "stimulus_hz": round(stim[a], 4),
            "delta_hz": round(stim[a] - base[a], 4),
            "band_hz": round(band, 4),
            "responded": bool(abs(stim[a] - base[a]) > band),
        }
    levels = {
        "V1_driven": resp["V1_1"]["responded"],
        "V4": resp["V4_1"]["responded"] or resp["V4_2"]["responded"],
        "frontal": resp["FEF"]["responded"] or resp["PFC"]["responded"],
    }
    gates["propagation"] = {
        "by_area": resp,
        "levels": levels,
        "pass": all(levels.values()),
        "lateral_V1_2": {
            **resp["V1_2"],
            "note": (
                "the spot lit no retinal unit projecting to V1.2, so any "
                "change here is transfer through X[lat] or the V4 level "
                "and is reported separately, not counted as propagation"
            ),
        },
        "band_authority": "sealed V2.1 convention max(1 Hz, 10%)",
        "qualification": (
            "NON_IDENTIFYING. This within-run window difference cannot identify "
            "transmission in this architecture: WS-AUTH-2 ran it at g = 0, where "
            "no current can reach FEF or PFC, and it still reported +0.7971 and "
            "+0.8406 Hz there. The numbers are retained unchanged so this arm "
            "stays comparable to earlier ones, and they are not evidence about "
            "the hierarchy. The identifying estimator is "
            "jomission.harness.propagation.causal_response, which needs the four "
            "runs of this design and cannot be computed inside a single run"
        ),
    }

    # Retinal receipt: did the interface actually drive anything?
    rg = series[f"{retina.AREA}.{retina.LAYER}.{retina.CLASS}"]
    lit_share = lit_idx.size / float(retina.N)
    rec["retinal_response"] = {
        "n_lit": int(lit_idx.size),
        "lit_fraction": round(lit_share, 5),
        "population_rate_baseline_hz": round(
            float(window(rg["rate_hz"], SETTLE_MS - 1000.0, SETTLE_MS).mean()), 4
        ),
        "population_rate_stimulus_hz": round(
            float(window(rg["rate_hz"], SETTLE_MS, SETTLE_MS + STIM_MS).mean()), 4
        ),
        "population_rate_post_hz": round(
            float(window(rg["rate_hz"], SETTLE_MS + STIM_MS, TOTAL_MS).mean()), 4
        ),
        # Undefined when nothing is lit: it divides the population rate by the lit share, and an OFF
        # run lights nothing. Reported as absent rather than as a number, and never as zero.
        "implied_lit_unit_rate_hz": (
            round(
                float(window(rg["rate_hz"], SETTLE_MS, SETTLE_MS + STIM_MS).mean()) / lit_share, 3
            )
            if lit_share > 0
            else None
        ),
        "implied_lit_unit_rate_note": (
            None
            if lit_share > 0
            else "undefined: no unit is lit in an OFF run, so there is no lit-unit rate to imply"
        ),
        "note": (
            "the retina is silent absent stimulus by construction, so the whole population rate during "
            "stimulation is carried by the lit units alone"
        ),
    }

    rec["plastic_state"] = {
        "by_chunk": plastic,
        "w_changed": bool(
            abs(plastic[-1]["w_mean"] - plastic[0]["w_mean"]) > 1e-12
            or abs(plastic[-1]["w_std"] - plastic[0]["w_std"]) > 1e-12
        ),
        "theta_changed": (
            bool(abs(plastic[-1]["theta_mean"] - plastic[0]["theta_mean"]) > 1e-12)
            if "theta_mean" in plastic[0]
            else "not applicable: no theta_S target"
        ),
        "A_changed": (
            bool(abs(plastic[-1]["A_mean_state"] - plastic[0]["A_mean_state"]) > 1e-12)
            if "A_mean_state" in plastic[0]
            else "not applicable: rule carries no A"
        ),
        "sampling": (
            "w, theta_S and H state are sampled at 500 ms chunk boundaries, not per ms: the per-edge "
            "weight trace would be 50000 steps x 246844 edges x 4 bytes = 4.9 GB. H is additionally "
            "recorded per 1 ms as a per-neuron trace"
        ),
        "interpretation": (
            "under kernel=baseline these are controls: static values confirm no hidden plastic "
            "stabilization occurred, and are not evidence that stabilization works. Under "
            "kernel=hdp they are the mechanism itself, and H reaching a bound is saturation, "
            "not regulation"
        ),
    }

    y = y_trace
    stim_slice = slice(int(SETTLE_MS / DT), int((SETTLE_MS + STIM_MS) / DT))
    base_slice = slice(int((SETTLE_MS - 1000.0) / DT), int(SETTLE_MS / DT))
    rec["y_FEF_L6_E"] = {
        "address": "FEF.L6.E.v(t)",
        "n_neurons": int(y_n),
        "dt_ms": DT,
        "artifact": str(out_y.relative_to(ROOT)).replace("\\", "/"),
        "retained": "population-resolved v_i(t); the mean is derived from it, never recorded in its place",
        "v_mean_baseline": round(float(y[base_slice].mean()), 4),
        "v_mean_stimulus": round(float(y[stim_slice].mean()), 4),
        "per_neuron_mean_baseline": [round(float(x), 3) for x in y[base_slice].mean(axis=0)],
        "per_neuron_mean_stimulus": [round(float(x), 3) for x in y[stim_slice].mean(axis=0)],
        "across_neuron_spread_baseline": round(float(y[base_slice].mean(axis=0).std()), 4),
        "across_neuron_spread_stimulus": round(float(y[stim_slice].mean(axis=0).std()), 4),
    }

    # The series is evidence but it is bulk, not a claim. It goes to the .npz beside this file
    # rather than into the record, whose size v1 showed is essentially all series.
    series_groups = sorted(series)
    series_keys = sorted({k for s in series.values() for k in s})
    ragged = [g for g in series_groups if sorted(series[g]) != series_keys]
    if ragged:
        raise ValueError(
            f"series groups do not share one key set; {len(ragged)} differ, "
            f"first {ragged[0]}: {sorted(series[ragged[0]])} vs {series_keys}"
        )
    series_arrays = {
        f"series_{k}": np.asarray([series[g][k] for g in series_groups], dtype=np.float32)
        for k in series_keys
    }
    rec["series_storage"] = {
        "where": str(out_y.relative_to(ROOT)).replace("\\", "/"),
        "layout": "one array per key, shape (n_groups, n_samples), rows ordered by series_groups",
        "groups": series_groups,
        "keys": series_keys,
        "decimation": 1,
        "sample_ms": 1.0,
        "note": (
            "1 ms summaries retained at full resolution: the E-I regime diagnosis "
            "re-measures per-class rates from exact 1 ms-binned spike counts, removing "
            "the subsampling caveat rather than working around it. Reviewer 2026-09-19"
        ),
    }

    # ---- the lineage's own question: per-class rates at 1 ms, then attribution ----
    CLASSES = ("E", "PV", "SST", "VIP")
    class_groups = {
        c: [g for g in series if g.split(".")[-1] == c and not g.startswith(retina.AREA)]
        for c in CLASSES
    }
    n_of = {c: int(sum(members[g].size for g in class_groups[c])) for c in CLASSES}
    cidx = {c: np.concatenate([members[g] for g in class_groups[c]]) for c in CLASSES}

    def class_rate_1ms(c):
        """Exact 1 ms class rate: spike-count weighted across the class's groups."""
        parts = [(series[g]["rate_hz"], members[g].size) for g in class_groups[c]]
        tot = sum(n for _, n in parts)
        return sum(r * n for r, n in parts) / tot

    rate_1ms = {c: class_rate_1ms(c) for c in CLASSES}
    phases = {
        "full": (0.0, TOTAL_MS),
        "settle": (0.0, SETTLE_MS),
        "stimulus": (SETTLE_MS, SETTLE_MS + STIM_MS),
        "post": (SETTLE_MS + STIM_MS, TOTAL_MS),
    }
    rates = {
        c: {p: round(float(window(r, *w).mean()), 4) for p, w in phases.items()}
        for c, r in rate_1ms.items()
    }
    counts = {
        c: int(round(float(window(r, 0.0, TOTAL_MS).sum()) * n_of[c] * 1e-3))
        for c, r in rate_1ms.items()
    }

    b1 = {}
    for c in CLASSES:
        lo, hi = B1_CLASS_TARGETS[c]["global_target_Hz"]
        b1[c] = {
            "band": [lo, hi],
            "measured_full": rates[c]["full"],
            "in_band": bool(lo <= rates[c]["full"] <= hi),
        }
    b1["ordering_E_lt_PV"] = bool(rates["E"]["full"] < rates["PV"]["full"])

    em = model.params["emitter"]
    drive = np.asarray(em.drive, dtype=float)
    tonic = {c: float(drive[cidx[c]].mean()) for c in CLASSES}
    cur = {}
    for c in CLASSES:
        parts = [(series[g]["current_mean"], members[g].size) for g in class_groups[c]]
        tot = sum(n for _, n in parts)
        cur[c] = float(sum(r.mean() * n for r, n in parts) / tot)
    realized = {c: tonic[c] + cur[c] for c in CLASSES}

    fbi = json.loads((ROOT / "results" / "battery_b1.json").read_text(encoding="utf-8"))["F"]

    def rheobase(c):
        I = np.asarray(fbi[c]["I"], dtype=float)
        r = np.asarray(fbi[c]["r"], dtype=float)
        hit = np.flatnonzero(r > 1.0)
        return float(I[hit[0]]) if hit.size else None

    rheo = {c: rheobase(c) for c in CLASSES}
    c1 = {
        "tonic_configured": {c: round(v, 4) for c, v in tonic.items()},
        "rheobase_fresh": rheo,
        "realized_effective": {c: round(v, 4) for c, v in realized.items()},
        "pv_sub_rheobase": rheo["PV"] is not None and realized["PV"] < rheo["PV"],
        "vip_sub_rheobase": rheo["VIP"] is not None and realized["VIP"] < rheo["VIP"],
    }
    c1["supported"] = bool(c1["pv_sub_rheobase"] and c1["vip_sub_rheobase"])

    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    tbl = model.neuron_table()
    cls = np.array([str(r.get("cell_type")) for r in tbl])

    def wmean(pc, qc):
        m = (cls[pre] == pc) & (cls[post] == qc)
        return float(w[m].mean()) if m.any() else 0.0

    def wsum(pc, qc):
        m = (cls[pre] == pc) & (cls[post] == qc)
        return float(w[m].sum())

    wee, wepv, wevip = wmean("E", "E"), wmean("E", "PV"), wmean("E", "VIP")
    c2 = {
        "mean_E_to_E": round(wee, 6),
        "mean_E_to_PV": round(wepv, 6),
        "mean_E_to_VIP": round(wevip, 6),
        "supported": bool(wepv < wee and wevip < wee),
    }

    def uniq_vals(k, c):
        arr = np.atleast_1d(np.asarray(getattr(em, k), dtype=float))
        if arr.size == 1:
            return [float(arr.flat[0])]
        return sorted(set(arr[cidx[c]].tolist()))

    pv_ab = (uniq_vals("a", "PV"), uniq_vals("b", "PV"))
    sst_ab = (uniq_vals("a", "SST"), uniq_vals("b", "SST"))
    vip_ab = (uniq_vals("a", "VIP"), uniq_vals("b", "VIP"))
    c3 = {
        "built_PV_ab": [pv_ab[0], pv_ab[1]],
        "built_SST_ab": [sst_ab[0], sst_ab[1]],
        "built_VIP_ab": [vip_ab[0], vip_ab[1]],
        "canonical_FS_ab": [[0.10], [0.20]],
        "canonical_LTS_ab": [[0.02], [0.25]],
        "pv_matches_FS": bool(pv_ab == ([0.10], [0.20])),
        "sst_matches_LTS": bool(sst_ab == ([0.02], [0.25])),
        "vip_note": "no canonical VIP type exists; built values reported without verdict contribution",
    }
    c3["supported"] = bool(not c3["pv_matches_FS"] or not c3["sst_matches_LTS"])

    early = (SETTLE_MS, SETTLE_MS + 2000.0)
    sst_e = float(window(rate_1ms["SST"], *early).mean())
    pv_e = float(window(rate_1ms["PV"], *early).mean())
    vip_e = float(window(rate_1ms["VIP"], *early).mean())
    sst_pv_sum, sst_vip_sum = wsum("SST", "PV"), wsum("SST", "VIP")
    c4 = {
        "sst_full": rates["SST"]["full"],
        "sst_above_B1_top": bool(rates["SST"]["full"] > 12.0),
        "sst_to_PV_weight_sum": round(sst_pv_sum, 4),
        "sst_to_VIP_weight_sum": round(sst_vip_sum, 4),
        "anatomical_path": bool(sst_pv_sum > 0.0 and sst_vip_sum > 0.0),
        "pv_net_current": round(cur["PV"], 4),
        "vip_net_current": round(cur["VIP"], 4),
        "inhibition_dominated": bool(
            cur["PV"] < 0.0 and cur["VIP"] < 0.0 and tonic["PV"] > 0.0 and tonic["VIP"] > 0.0
        ),
        "sst_early": round(sst_e, 4),
        "pv_early": round(pv_e, 4),
        "vip_early": round(vip_e, 4),
        "precedence": bool(sst_e > 12.0 and sst_e > 5.0 * max(pv_e, vip_e)),
    }
    c4["supported"] = bool(
        c4["sst_above_B1_top"]
        and c4["anatomical_path"]
        and c4["inhibition_dominated"]
        and c4["precedence"]
    )

    rec["ei_regime"] = {
        "rates_full_resolution": rates,
        "exact_spike_totals": counts,
        "n_cells": n_of,
        "b1_comparison": b1,
        "C1_tonic": c1,
        "C2_weights": c2,
        "C3_parameters": c3,
        "C4_suppression": c4,
    }
    order = [
        ("C1", c1["supported"]),
        ("C2", c2["supported"]),
        ("C3", c3["supported"]),
        ("C4", c4["supported"]),
    ]
    winners = [k for k, s in order if s]
    names = {"C1": "TONIC", "C2": "EI_WEIGHTS", "C3": "CELL_PARAMETERS", "C4": "SST_SUPPRESSION"}
    health_order = ["execution", "collapse", "runaway"]
    first = next((g for g in health_order if not gates[g]["pass"]), None)
    rec["gates"] = gates
    rec["gate_order"] = health_order
    if first is not None:
        rec["verdict"] = "EI_REGIME_EXECUTION_FAIL"
        rec["fail_boundary"] = f"run-health gate failed: {first}; rates below are degenerate"
    elif len(winners) == 1:
        rec["verdict"] = "EI_REGIME_ATTRIBUTED_TO_" + names[winners[0]]
        rec["fail_boundary"] = f"attributed to {winners[0]}; other candidates read negative"
    else:
        rec["verdict"] = "EI_REGIME_UNRESOLVED_" + ("NONE" if not winners else "MULTIPLE")
        rec["fail_boundary"] = (
            "no candidate supported" if not winners else f"several supported: {winners}"
        )
    np.savez_compressed(
        out_y, **y_npz_kwargs, **series_arrays, series_groups=np.asarray(series_groups)
    )
    rec["elapsed_s"] = round(time.time() - started, 1)
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")

    print(
        "cell %s (g=%g, stimulus %s, seed %d) | %s | %s | %.0fs"
        % (
            rec["cell"],
            g_auth,
            stimulus,
            seed,
            rec["verdict"],
            rec["fail_boundary"],
            rec["elapsed_s"],
        )
    )
    print("  enforcement:", enforcement)
    print(
        "  realization recheck: cortical",
        rec["realization_recheck"]["cortical_identities"],
        "missing",
        rec["realization_recheck"]["missing"],
        "extra",
        rec["realization_recheck"]["extra"],
        "| retinal edges",
        rec["realization_recheck"]["retinal_edges"],
    )
    print(
        "  retina:",
        rec["retinal_response"]["population_rate_baseline_hz"],
        "->",
        rec["retinal_response"]["population_rate_stimulus_hz"],
        "Hz (lit-unit",
        rec["retinal_response"]["implied_lit_unit_rate_hz"],
        "Hz)",
    )
    for g in health_order:
        print(f"  {g}: {'pass' if gates[g]['pass'] else 'FAIL'}")
    print("  per-class rates (full run, Hz):")
    for c in CLASSES:
        print(f"    {c}: {rates[c]['full']} (B1 band {b1[c]['band']}, in_band {b1[c]['in_band']})")
    print(
        "  attribution: C1 %s, C2 %s, C3 %s, C4 %s"
        % (c1["supported"], c2["supported"], c3["supported"], c4["supported"])
    )
    print(
        "  y FEF.L6.E v_mean:",
        rec["y_FEF_L6_E"]["v_mean_baseline"],
        "->",
        rec["y_FEF_L6_E"]["v_mean_stimulus"],
    )
    print(
        "  w changed:",
        rec["plastic_state"]["w_changed"],
        "| theta changed:",
        rec["plastic_state"]["theta_changed"],
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--g",
        type=float,
        required=True,
        help="long-range transfer authority; g_FF = g_FB = g_X = g",
    )
    ap.add_argument(
        "--stimulus",
        choices=("on", "off"),
        required=True,
        help="'off' lights no retinal unit; it is the paired control for 'on'",
    )
    ap.add_argument(
        "--ee-cut",
        action="store_true",
        help="zero every within-area E->E weight; the principal delta of SCI-WS-OSC-1",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="continuation PRNG key. ON and OFF of one pair must share it",
    )
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-y", default=None)
    ap.add_argument("--spec-commit", default=None)
    a = ap.parse_args()
    configure(a.spec)
    main(
        a.g,
        a.stimulus,
        a.seed,
        Path(a.out) if a.out else None,
        Path(a.out_y) if a.out_y else None,
        a.spec,
        a.spec_commit,
    )

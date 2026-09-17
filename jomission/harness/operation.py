"""Canonical V2 local-operation estimators and battery driver.

One implementation of the estimators named in manifests/gates/v2_local_operation.json
(`estimators`, `windows`), consumed by every prospective operation battery. Parameters
(dt, window length, ISI CV band, min spikes) are read from the gate file.

Shapes and units: a spike raster is (steps, n) bool or {0, 1}, one analysis window;
cls is (n,) class labels; rates in Hz; time in ms. Frozen executed batteries
(gate file `frozen_literal_implementations`) keep their own code;
tests/test_v2_operation_prospective.py checks this module against them.
"""

from __future__ import annotations

import subprocess

import numpy as np

from jomission.harness import gates
from jomission.harness.drive import check_additive_schedule

CLASSES = ("E", "PV", "SST", "VIP")


def _isi_params(gate: dict, profile: str) -> tuple[list[float], int]:
    for spec in gate["profiles"][profile]["checks"].values():
        if spec["kind"] in ("isi_fraction_pooled", "isi_fraction_by_class"):
            return spec["cv_band"], int(spec["min_spikes"])
    raise KeyError(f"profile {profile} has no ISI check")


def isi_cvs(raster, dt_ms: float, min_spikes: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron spike counts (n,) and ISI CV (n,), nan when fewer than min_spikes. CV = std/mean, ddof 0."""
    r = np.asarray(raster) > 0.5
    n = r.shape[1]
    nsp = r.sum(axis=0)
    cv = np.full(n, np.nan)
    for i in np.flatnonzero(nsp >= min_spikes):
        isi = np.diff(np.flatnonzero(r[:, i])).astype(np.float64) * dt_ms
        cv[i] = float(isi.std() / max(isi.mean(), 1e-9))
    return nsp, cv


def window_summary(raster, cls, profile: str = "prospective_v2", gate_id: str = "v2_local_operation") -> dict:
    """Window dict for gates.evaluate plus recorded (ungated) diagnostics."""
    gate = gates.load_gate(gate_id)
    dt_ms = float(gate["windows"]["dt_ms"])
    band, min_spikes = _isi_params(gate, profile)
    r = np.asarray(raster) > 0.5
    cls = np.asarray(cls)
    window_s = r.shape[0] * dt_ms / 1000.0
    nsp, cv = isi_cvs(r, dt_ms, min_spikes)
    rates_n = nsp / window_s
    ok = ~np.isnan(cv)
    inband = ok & (cv >= band[0]) & (cv <= band[1])
    out = {"rates": {}, "rate_cv": {}, "isi_fraction": {}, "cv_evaluable": {}, "active_fraction": {}, "silent_fraction": {}}
    for c in CLASSES:
        sel = cls == c
        rc = rates_n[sel]
        out["rates"][c] = float(rc.mean())
        out["rate_cv"][c] = float(rc.std() / max(rc.mean(), 1e-9))
        ne = int(ok[sel].sum())
        out["cv_evaluable"][c] = ne
        out["isi_fraction"][c] = float(inband[sel].sum() / ne) if ne else None
        out["active_fraction"][c] = float(ok[sel].mean())
        out["silent_fraction"][c] = float((~ok[sel]).mean())
    out["isi_fraction_pooled"] = float(inband.sum() / ok.sum()) if ok.any() else 0.0
    out["sync"] = float((r[:, cls == "E"].mean(axis=1) > 0.5).mean())
    return out


def drift(windows: list[dict]) -> dict:
    out = {}
    for c in CLASSES:
        rs = [w["rates"][c] for w in windows]
        out[c] = float((max(rs) - min(rs)) / max(sum(rs) / len(rs), 1e-9))
    return out


def gate_blob(gate_id: str = "v2_local_operation") -> str:
    path = gates.GATE_DIR / f"{gate_id}.json"
    return subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True).stdout.strip()


def evaluate(windows: list[dict], profile: str = "prospective_v2", gate_id: str = "v2_local_operation") -> dict:
    """Gate result record: checks, verdict label, profile, gate blob. A FAIL is data, not an exception."""
    d = drift(windows)
    checks = gates.evaluate(profile, windows, d, gate_id)
    label = gates.load_gate(gate_id)["gate_id"]
    return {"profile": profile, "gate_blob": gate_blob(gate_id), "windows": windows, "drift": d, "checks": checks,
            "verdict": f"{label}_{'PASS' if all(checks.values()) else 'FAIL'}"}


def run_battery(engine, step_fn, state, cls, emitter_drive, schedule_chunks, expected_schedule_mean=None,
                profile: str = "prospective_v2", gate_id: str = "v2_local_operation", spikes_index: int = 1) -> dict:
    """Run chunks through engine.run_continuation and evaluate the analysis windows.

    engine: object with run_continuation(step_fn, state, sched) -> (state, outputs), outputs[spikes_index]
    a (steps, n) raster. emitter_drive (n,) native units; schedule_chunks: iterable of (steps, n)
    additive schedules, one per chunk, each checked with check_additive_schedule. Chunk count and
    analysis windows come from the gate file. Keeps only analysis-window rasters, as bool.
    """
    gate = gates.load_gate(gate_id)
    windows_idx = set(gate["windows"]["analysis_chunks"])
    n_chunk = int(round(gate["windows"]["run_s"] / gate["windows"]["chunk_s"]))
    cls = np.asarray(cls)
    summaries, additivity = [], []
    chunks = iter(schedule_chunks)
    for k in range(n_chunk):
        sched = next(chunks)
        additivity.append(check_additive_schedule(emitter_drive, np.asarray(sched), cls, expected_schedule_mean))
        state, out = engine.run_continuation(step_fn, state, sched)
        if k in windows_idx:
            raster = np.asarray(out[spikes_index]) > 0.5
            if not np.isfinite(np.asarray(out[spikes_index], dtype=np.float64)).all():
                raise FloatingPointError(f"non-finite spike output in chunk {k}")
            summaries.append(window_summary(raster, cls, profile, gate_id))
    rec = evaluate(summaries, profile, gate_id)
    rec["drive_additivity"] = additivity
    return rec

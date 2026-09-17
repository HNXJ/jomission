"""SCI-V21-PV-MICROSTATE: read-only per-cell reanalysis of the SCI-V21-PV-PREDECESSOR traces.

Spec (committed and pushed before this runs): results/v21_pv_microstate_spec.json.

No simulation. The question is whether the fate-carrying information already exists in the
recorded microscopic state earlier than the macroscopic PV-rate divergence at 15 s in run A,
at epochs where the population mean of the same variable does not separate.

B and C share an initial state and a fate and differ only in the noise realization, so d(B, C)
is the fate-irrelevant scale against which d(A, B) and d(A, C) are judged.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_pv_microstate_spec.json"
OUT = ROOT / "results" / "v21_pv_microstate.json"
SEALED = ROOT / "results" / "v21_pv_predecessor.json"

EPOCH_MS = 1000
LEAD_EPOCHS = 15
PV_RATE_DIVERGENCE_MS = 15000.0
PRECEDE_MS = 1000
SEP_FACTOR = 5.0
SUSTAIN_EPOCHS = 3
MIN_CELLS_ISI = 20
MIN_SPIKES_ISI = 3
WARMUP_MS = 50
DT_MS = 0.1
RUNS = ("A_collapse", "B_active", "C_control")
CLASSES = ("E", "PV", "SST", "VIP")


def wasserstein1(x, y):
    """Equal-size 1-D samples: mean |sorted(x) - sorted(y)|."""
    return float(np.abs(np.sort(np.asarray(x, float)) - np.sort(np.asarray(y, float))).mean())


def energy_distance(x, y):
    """2 E|X-Y| - E|X-X'| - E|Y-Y'| on point clouds of shape (n, d)."""
    x, y = np.atleast_2d(np.asarray(x, float)), np.atleast_2d(np.asarray(y, float))

    def mean_norm(a, b):
        return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).mean())

    return 2.0 * mean_norm(x, y) - mean_norm(x, x) - mean_norm(y, y)


def load(scratch):
    data = {}
    for name in RUNS:
        f = Path(scratch) / f"{name}.npz"
        z = np.load(f)
        data[name] = {k: z[k] for k in z}
        data[name]["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    return data


def trace_identity(data):
    """The traces must reproduce the population-mean numbers sealed with the parent result."""
    a = data["A_collapse"]
    v, u, s = a["v"].mean(axis=1), a["u"].mean(axis=1), a["sources"].mean(axis=1)
    got = {"v_base_mu": v[:5000].mean(), "v_base_sd": v[:5000].std(), "v_late": v[50000:].mean(),
           "u_base_mu": u[:5000].mean(), "u_base_sd": u[:5000].std(), "u_late": u[50000:].mean(),
           "s_base_mu": s[:5000].mean(), "s_base_sd": s[:5000].std(), "s_late": s[50000:].mean()}
    want = {"v_base_mu": -63.7501, "v_base_sd": 5.6072, "v_late": -63.4949,
            "u_base_mu": -12.4109, "u_base_sd": 0.7976, "u_late": -12.6865,
            "s_base_mu": 3.3786, "s_base_sd": 0.4464, "s_late": 3.3866}
    dev = {k: float(got[k]) - want[k] for k in want}
    ok = all(abs(d) < 5e-4 for d in dev.values())
    spec_hashes = json.loads(SPEC.read_text(encoding="utf-8"))["inputs"]["traces"]
    hashes_ok = all(spec_hashes[f"{n}.npz"] == "sha256 " + data[n]["sha256"] for n in RUNS)
    return {"values": {k: float(v) for k, v in got.items()}, "deviation": dev,
            "within_print_precision": bool(ok), "sha256_match": bool(hashes_ok),
            "sealed_reference": str(SEALED.relative_to(ROOT))}


def epoch_slices(n_bins):
    return [slice(i * EPOCH_MS, (i + 1) * EPOCH_MS) for i in range(n_bins // EPOCH_MS)]


def per_cell_epoch_means(arr, warmup=0):
    """(bins, cells) -> (epochs, cells) means, optionally dropping the first warmup bins."""
    out = []
    for i, sl in enumerate(epoch_slices(arr.shape[0])):
        block = arr[sl]
        if i == 0 and warmup:
            block = block[warmup:]
        out.append(block.mean(axis=0))
    return np.array(out, dtype=np.float64)


def isi_features(pv_spikes, n_epochs, n_pv):
    """Per epoch: (cells kept, [mean ISI ms, ISI CV]) for cells with enough spikes."""
    t_ms = pv_spikes[:, 0] * DT_MS
    cell = pv_spikes[:, 1]
    feats = []
    for e in range(n_epochs):
        lo, hi = e * EPOCH_MS, (e + 1) * EPOCH_MS
        sel = (t_ms >= lo) & (t_ms < hi)
        te, ce = t_ms[sel], cell[sel]
        rows = []
        for c in range(n_pv):
            ts = np.sort(te[ce == c])
            if ts.size >= MIN_SPIKES_ISI:
                isi = np.diff(ts)
                m = isi.mean()
                rows.append([m, isi.std() / m if m > 0 else 0.0])
        feats.append(np.array(rows, dtype=np.float64) if rows else np.empty((0, 2)))
    return feats


def sustained_start(flags):
    """First index where flags is True for SUSTAIN_EPOCHS consecutive epochs."""
    f = np.asarray(flags, dtype=np.int32)
    if f.size < SUSTAIN_EPOCHS:
        return None
    run = np.convolve(f, np.ones(SUSTAIN_EPOCHS, dtype=np.int32), mode="valid")
    hit = np.flatnonzero(run == SUSTAIN_EPOCHS)
    return int(hit[0]) if hit.size else None


def evaluate_level(d_ab, d_ac, d_bc, n_lead):
    """Predeclared criterion: both A-distances exceed SEP_FACTOR x the largest same-fate distance."""
    scale = float(np.nanmax(d_bc[:n_lead])) if n_lead else float("nan")
    thr = SEP_FACTOR * scale
    flags = [bool(np.isfinite(a) and np.isfinite(b) and a > thr and b > thr) for a, b in zip(d_ab, d_ac)]
    start = sustained_start(flags[:n_lead])
    return {"scale_bc": scale, "threshold": thr, "flags": flags,
            "qualifying_epoch": start,
            "qualifying_time_ms": None if start is None else float(start * EPOCH_MS),
            "leads_macroscopic": bool(start is not None and start * EPOCH_MS <= PV_RATE_DIVERGENCE_MS - PRECEDE_MS)}


def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    scratch = json.loads(SEALED.read_text(encoding="utf-8"))["scratch_dir"]
    data = load(scratch)
    rec = {"spec": "results/v21_pv_microstate_spec.json", "traces": scratch,
           "sha256": {n: data[n]["sha256"] for n in RUNS}}
    rec["trace_identity"] = trace_identity(data)
    if not (rec["trace_identity"]["within_print_precision"] and rec["trace_identity"]["sha256_match"]):
        rec["verdict"], rec["reason"] = "TRACE_IDENTITY_FAILED", "the traces do not reproduce the sealed population-mean numbers"
        OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(rec["verdict"], rec["reason"])
        return

    n_bins, n_pv = data["A_collapse"]["v"].shape
    n_epochs = n_bins // EPOCH_MS
    n_lead = min(LEAD_EPOCHS, n_epochs)

    cells = {n: {"v": per_cell_epoch_means(data[n]["v"]), "u": per_cell_epoch_means(data[n]["u"])} for n in RUNS}
    for n in RUNS:
        tot = sum(data[n][f"syn_{c}"] for c in CLASSES)
        cells[n]["input"] = per_cell_epoch_means(tot, warmup=WARMUP_MS)
    isi = {n: isi_features(data[n]["pv_spikes"], n_epochs, n_pv) for n in RUNS}

    # level 3 standardization: the run-B epoch-0 spread across cells
    sv = float(cells["B_active"]["v"][0].std()) or 1.0
    su = float(cells["B_active"]["u"][0].std()) or 1.0

    def dist_1d(key):
        return [[wasserstein1(cells[x][key][e], cells[y][key][e]) for e in range(n_epochs)]
                for x, y in (("A_collapse", "B_active"), ("A_collapse", "C_control"), ("B_active", "C_control"))]

    def dist_joint():
        def pts(n, e):
            return np.stack([cells[n]["v"][e] / sv, cells[n]["u"][e] / su], axis=1)
        return [[energy_distance(pts(x, e), pts(y, e)) for e in range(n_epochs)]
                for x, y in (("A_collapse", "B_active"), ("A_collapse", "C_control"), ("B_active", "C_control"))]

    def dist_isi():
        out = []
        for x, y in (("A_collapse", "B_active"), ("A_collapse", "C_control"), ("B_active", "C_control")):
            row = []
            for e in range(n_epochs):
                a, b = isi[x][e], isi[y][e]
                eligible = all(isi[n][e].shape[0] >= MIN_CELLS_ISI for n in RUNS)
                row.append(energy_distance(a, b) if eligible else float("nan"))
            out.append(row)
        return out

    levels = {}
    order = [(1, "v", lambda: dist_1d("v")), (2, "u", lambda: dist_1d("u")),
             (3, "joint_v_u", dist_joint), (4, "isi", dist_isi), (5, "input", lambda: dist_1d("input"))]
    qualifying = None
    for idx, name, fn in order:
        if idx == 5 and qualifying is not None:
            levels[name] = {"level": 5, "evaluated": False,
                            "reason": "levels 1-4 produced a qualifying separation; level 5 is conditional"}
            continue
        d_ab, d_ac, d_bc = fn()
        res = evaluate_level(d_ab, d_ac, d_bc, n_lead)
        res.update({"level": idx, "evaluated": True,
                    "d_AB": d_ab[:n_lead], "d_AC": d_ac[:n_lead], "d_BC": d_bc[:n_lead]})
        # the parallel population-mean test on the same variable
        if name in ("v", "u", "input"):
            m = {n: cells[n][name].mean(axis=1) for n in RUNS}
        elif name == "joint_v_u":
            m = {n: np.hypot(cells[n]["v"].mean(axis=1) / sv, cells[n]["u"].mean(axis=1) / su) for n in RUNS}
        else:
            m = {n: np.array([f[e][:, 0].mean() if f[e].shape[0] else np.nan for e in range(n_epochs)])
                 for n, f in isi.items()}
        mab = np.abs(m["A_collapse"] - m["B_active"])
        mac = np.abs(m["A_collapse"] - m["C_control"])
        mbc = np.abs(m["B_active"] - m["C_control"])
        res["population_mean_test"] = evaluate_level(mab, mac, mbc, n_lead)
        levels[name] = res
        if qualifying is None and res["leads_macroscopic"]:
            qualifying = (idx, name, res)
    rec["levels"] = levels
    rec["hierarchy_order"] = [n for _, n, _ in order]

    if qualifying is None:
        label = "RECORDED_MICROSTATE_INSUFFICIENT"
        reason = ("no declared per-cell representation separates run A from runs B and C by the predeclared factor for "
                  f"{SUSTAIN_EPOCHS} consecutive epochs at least {PRECEDE_MS} ms before the macroscopic divergence")
    else:
        idx, name, res = qualifying
        pop = res["population_mean_test"]
        pop_also = pop["qualifying_epoch"] is not None and pop["qualifying_epoch"] <= res["qualifying_epoch"]
        if pop_also:
            label = "UNRESOLVED"
            reason = (f"level {idx} ({name}) separates from {res['qualifying_time_ms']} ms, but the population mean of the same "
                      f"variable qualifies at {pop['qualifying_time_ms']} ms, so the separation is not specific to the "
                      "microscopic representation")
        else:
            label = "PV_MICROSTATE_PREDECESSOR"
            reason = (f"level {idx} ({name}) separates run A from B and C from {res['qualifying_time_ms']} ms, leading the "
                      f"macroscopic PV-rate divergence at {PV_RATE_DIVERGENCE_MS} ms, while the population mean of the same "
                      "variable does not qualify")
    rec["verdict"], rec["reason"] = label, reason
    rec["interpretation_limit"] = spec["interpretation_limit"]
    rec["forbidden_respected"] = spec["forbidden"]
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(label, reason)


if __name__ == "__main__":
    main()

"""Visualization stages for lineage SCI-LANSA-GEOM-R2 (manifests/visualization_contract.json).

V0: rule-based construction schematic (copied from the rule table; it is not an HSPICE schematic, see the atlas).
V1: initial 1 s raster from the 1 s smoke run (sampled cells: the driver keeps a fixed sample of cells per block).
V2: final rasters from the scored 12 s runs, first 1 s (same window as V1) and last 1 s.
V3: provisional atlas (lineage not complete: K2 NOT_RUN). Built from the saved arrays only; no re-simulation.

Usage: python scripts/e2_r2_viz.py RES_DIR OUT_DIR SMOKE_NPZ SMOKE_JSON V0_DIR
  RES_DIR    results/laminar_e2 (npz, json, motif png, readout json)
  OUT_DIR    results/viz/SCI-LANSA-GEOM-R2
  V0_DIR     directory with V0_rule_based_schematic.png and V0_rule_table.json
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, OUT, SMOKE, SMOKE_JSON, V0DIR = (Path(p) for p in sys.argv[1:6])
DT_MS = 0.1  # driver dt (run record dt_ms)
AREAS = ["S", "M", "D"]
TYPES = ["E", "PV", "SOM"]
RASTER_N = {"E": 200, "PV": 50, "SOM": 50}  # run_e2_stack.py
COL = {"E": "#c44e52", "PV": "#2a9d8f", "SOM": "#8d6cc6"}
TAGS = [f"r2_p0_n4000_s{s}_T12" for s in (7, 8, 9)]
SMOKE_TAG = "r2_smoke_p0_n4000_s7_T1"
SIZE_SHARE_NOTE = ("Sampled raster: the driver keeps a fixed sample of cells per block "
                   "(E 200, PV 50, SOM 50; default_rng(0)); not the full population.")


def block_rows(rec: dict) -> list[tuple[str, str, int, int]]:
    """(area, type, row offset, n sampled rows) in V1 order: area, then class E, PV, SOM."""
    out, off = [], 0
    for a in AREAS:
        for t in TYPES:
            n = min(RASTER_N[t], int(rec["sizes"][a][t]))
            out.append((a, t, off, n))
            off += n
    return out


def spikes(z, a: str, t: str, t0_ms: float, t1_ms: float):
    b0, b1 = int(round(t0_ms / DT_MS)), int(round(t1_ms / DT_MS))
    ts = z[f"{a}_{t}_t"].astype(np.int64)
    cs = z[f"{a}_{t}_c"].astype(np.int64)
    sel = (ts >= b0) & (ts < b1)
    return ts[sel] * DT_MS, cs[sel]


def raster(z, rec: dict, t0_ms: float, t1_ms: float, title: str, path: Path) -> int:
    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=130)
    n_pts = 0
    rows = block_rows(rec)
    for a, t, off, n in rows:
        x, c = spikes(z, a, t, t0_ms, t1_ms)
        ax.scatter(x, off + c, s=0.4, c=COL[t], marker="|", linewidths=0.4)
        n_pts += x.size
    seen = None
    for a, t, off, n in rows:
        if a != seen:
            ax.axhline(off, color="#333", lw=0.9)
            ax.text(t0_ms + 0.006 * (t1_ms - t0_ms), off + n * 0.5, a, ha="left", va="center",
                    fontsize=10, fontweight="bold", bbox=dict(fc="white", ec="none", pad=1.5))
            seen = a
        else:
            ax.axhline(off, color="#bbb", lw=0.4)
    total = rows[-1][2] + rows[-1][3]
    ax.set_ylim(total, 0)
    ax.set_xlim(t0_ms, t1_ms)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("sampled units (area, then E, PV, SOM)")
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[plt.Line2D([], [], color=COL[k], marker="|", ls="", label=k) for k in TYPES],
              loc="upper right", fontsize=8, frameon=False)
    ax.text(0.0, -0.06, SIZE_SHARE_NOTE, transform=ax.transAxes, fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return n_pts


def same_first_second(z_smoke, z_scored, rec_smoke: dict, rec_scored: dict) -> dict:
    """Identity check: the scored run's first 1 s sample equals the smoke run's whole sample, block by block."""
    res = {}
    for a, t, _, _ in block_rows(rec_scored):
        ts_s = z_smoke[f"{a}_{t}_t"].astype(np.int64)
        cs_s = z_smoke[f"{a}_{t}_c"].astype(np.int64)
        ts_c = z_scored[f"{a}_{t}_t"].astype(np.int64)
        cs_c = z_scored[f"{a}_{t}_c"].astype(np.int64)
        keep = ts_c < int(round(1000.0 / DT_MS))
        res[f"{a}_{t}"] = bool(np.array_equal(ts_s, ts_c[keep]) and np.array_equal(cs_s, cs_c[keep]))
    counts_eq = bool(np.array_equal(z_smoke["counts"], z_scored["counts"][: z_smoke["counts"].shape[0]]))
    return {"blocks_equal": res, "all_blocks_equal": all(res.values()), "counts_first_1s_equal": counts_eq}


def rates_panel(zs: dict, recs: dict, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=120, sharex=True)
    for ax, a in zip(axes, AREAS):
        for t in TYPES:
            tot = None
            for tag in TAGS:
                z, rec = zs[tag], recs[tag]
                j = list(z["block_keys"]).index(f"{a}_{t}")
                cnt = z["counts"][:, j].reshape(-1, 100).sum(axis=1)  # 10 ms bins
                rate = cnt / (rec["sizes"][a][t] * 0.01)
                tot = rate if tot is None else np.vstack([tot, rate])
            m = tot.mean(axis=0)
            ax.plot(np.arange(m.size) * 0.01, m, color=COL[t], lw=0.8, label=t)
        ax.set_ylabel(f"area {a}\nrate (Hz)")
        ax.set_ylim(bottom=0)
        ax.axvline(2.0, color="#777", ls=":", lw=0.9)  # scorer discard_s = 2.0
    axes[0].legend(fontsize=8, frameon=False, ncol=3)
    axes[-1].set_xlabel("time (s), mean of seeds 7-9, 10 ms bins")
    axes[0].set_title("Population rate per block (mean over the three scored seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def lfp_panel(zs: dict, recs: dict, path: Path) -> None:
    # first 2 s discarded, as the scorer does (discard_s = 2.0): the onset transient (~1e8 pA) dominates otherwise
    fs = 1000.0 / DT_MS
    seg, step = int(fs), int(fs / 2)
    d0 = int(2.0 * fs)
    fig, (axt, axp) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=120)
    tag = TAGS[0]
    z = zs[tag]
    t = np.arange(z["lfp_pA"].shape[0]) * DT_MS / 1000.0
    for k, a in enumerate(AREAS):
        axt.plot(t[d0: d0 + int(2 * fs)], z["lfp_pA"][d0: d0 + int(2 * fs), k], lw=0.5, label=a)
    axt.set_xlabel("time (s), seed 7, from 2 s (first 2 s discarded)")
    axt.set_ylabel("current-based LFP proxy (pA)")
    axt.legend(fontsize=8, frameon=False)
    win = np.hanning(seg)
    for k, a in enumerate(AREAS):
        psds = []
        for tg in TAGS:
            x = zs[tg]["lfp_pA"][d0:, k].astype(np.float64)
            segs = [x[i:i + seg] for i in range(0, x.size - seg + 1, step)]
            P = np.mean([np.abs(np.fft.rfft((s - s.mean()) * win)) ** 2 for s in segs], axis=0)
            psds.append(P / (fs * (win ** 2).sum()))
        f = np.fft.rfftfreq(seg, 1.0 / fs)
        axp.loglog(f[1:], np.mean(psds, axis=0)[1:], lw=0.9, label=a)
    axp.axvspan(10, 30, color="#999", alpha=0.15, label="alpha-beta band 10-30 Hz")
    axp.axvspan(50, 150, color="#c44e52", alpha=0.10, label="gamma band 50-150 Hz")
    axp.set_xlim(1, 200)
    axp.set_xlabel("frequency (Hz)")
    axp.set_ylabel("PSD (pA^2/Hz), mean of seeds 7-9")
    axp.legend(fontsize=7, frameon=False)
    fig.suptitle("Current-based LFP proxy (not a calibrated potential; readout is a separate analysis)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def conductance_panel(zs: dict, recs: dict, path: Path) -> None:
    z, rec = zs[TAGS[0]], recs[TAGS[0]]
    g = z["g_nS"].astype(np.float64)
    tt = np.arange(g.shape[0]) * DT_MS / 1000.0
    k = 10  # 1 ms means over the first 2 s
    n = int(2000 / (DT_MS * k))
    tm = tt.reshape(-1, k).mean(axis=1)[:n]
    gm = g.reshape(-1, k, g.shape[1]).mean(axis=1)[:n]
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=120)
    for c, lab in enumerate(rec["g_columns"]):
        ax.plot(tm, gm[:, c], lw=0.7, label=f"{lab[0]} {lab[1]}")
    ax.set_xlabel("time (s), seed 7")
    ax.set_ylabel("conductance (nS), 1 ms mean")
    ax.legend(fontsize=8, ncol=3, frameon=False)
    ax.set_title("Synaptic conductance proxies recorded by the driver", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def readout_panel(path: Path) -> dict:
    conds = ["G0", "G1", "G2", "G3", "G1p", "K1"]
    seeds = [7, 8, 9]
    vals = {}
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    w = 0.26
    for i, s in enumerate(seeds):
        r = json.loads((RES / f"r2_readout_r2_p0_n4000_s{s}_T12.json").read_text(encoding="utf-8"))
        vals[s] = {c: r["conditions"][c]["median_r_gm"] for c in conds}
        ax.bar(np.arange(len(conds)) + (i - 1) * w, [vals[s][c] for c in conds], w, label=f"seed {s}")
    ax.axhline(0.6, color="k", ls="--", lw=1)
    ax.text(len(conds) - 0.5, 0.62, "r_gm threshold 0.6", ha="right", fontsize=8)
    ax.axhline(0, color="#777", lw=0.6)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(["G0", "G1", "G2", "G3", "G1'", "K1 (flat)"])
    ax.set_ylabel("median r_gm over 11 held-out areas")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Readout conditions (sealed R2 spectra). G1 is below 0.6 and below K1 on every seed. K2 NOT_RUN.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return vals


def write_atlas(atlas: Path, rules: dict, vals: dict) -> None:
    atlas.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td style='text-align:right'>{v:,}</td></tr>"
                   for k, v in rules["rules"].items())
    stages = [
        ("V0", "construction", "rule-based block schematic (not HSPICE; no edge list drawn)", "present"),
        ("V1", "initial 1 s raster", "sampled cells, seed 7, 0-1 s", "present"),
        ("V2", "final raster", "sampled cells, scored seeds 7-9: 0-1 s and 11-12 s", "present"),
        ("connectivity", "rule table", "33 construction rules, realised edge counts", "present"),
        ("raster", "raster", "as V2", "present"),
        ("rates", "population rates", "per block, mean over seeds 7-9", "present"),
        ("LFP proxy", "current-based proxy", "qualified: not a potential, not calibrated", "present, qualified"),
        ("conductances", "g_nS", "1 ms means, seed 7", "present"),
        ("membrane dynamics", "V(t)", "the driver does not record membrane potential", "NOT RECORDED"),
        ("I_net", "net synaptic current", "the driver does not record it", "NOT RECORDED"),
        ("H, A, m(H)", "H-state, adaptation", "the lineage has no H-state (AdEx populations)", "NOT APPLICABLE"),
        ("plastic weights", "effective weights", "the lineage has no plasticity", "NOT APPLICABLE"),
        ("diagnostics", "readout and motif", "sealed scorer; readout G0-G1' and K1; K2 NOT_RUN", "present"),
    ]
    srows = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{html.escape(c)}</td>"
                    f"<td><b>{html.escape(d)}</b></td></tr>" for a, b, c, d in stages)
    vrows = "".join(
        f"<tr><td>{s}</td>" + "".join(f"<td>{vals[s][c]:.3f}</td>" for c in
                                       ["G0", "G1", "G2", "G3", "G1p", "K1"]) + "</tr>" for s in (7, 8, 9))
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SCI-LANSA-GEOM-R2 atlas (provisional)</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; color: #111; background: #fff; max-width: 1200px; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 28px; }}
table {{ border-collapse: collapse; font-size: 13px; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
.warn {{ background: #fff4e5; border: 1px solid #e0a040; padding: 8px; }}
</style></head><body>
<h1>SCI-LANSA-GEOM-R2: visualization atlas (V3, provisional)</h1>
<p class="warn"><b>PROVISIONAL.</b> The lineage is not complete: K2 is NOT_RUN and its selection is UNDEFINED.
Contract V3 is due at lineage completion. This atlas is built from the saved arrays of the scored runs (seeds 7-9, 12 s,
engine JaxFNE 0.5.8, CPU, sealed spec at 46899e2). The G1 falsification (per seed, not pooled) is in the receipt
<code>docs/receipts/lansa_r2_scored_runs_2026-11-26.md</code>.</p>
<h2>Stage inventory (contract section V3)</h2>
<table><tr><th>contract item</th><th>what</th><th>status here</th><th>state</th></tr>{srows}</table>
<h2>V0: construction (rule-based)</h2>
<p>Labelled rule-based: built from the 33 construction rules and their realised counts. No edge list is drawn, and the
sealed construction was not modified (the HSPICE renderer needs an edge list, which the default route does not store).</p>
<img src="../V0_rule_based_schematic.png" alt="V0 rule-based schematic">
<h2>Connectivity: construction rules and realised edge counts (seed 7)</h2>
<table><tr><th>rule</th><th>realised edges</th></tr>{rows}</table>
<h2>V1: initial raster, 0-1 s (seed 7 smoke run)</h2>
<p>{html.escape(SIZE_SHARE_NOTE)}</p>
<img src="../V1_raster_initial_1000ms_s7.png" alt="V1">
<h2>V2: final raster, scored seeds (first and last second)</h2>
<img src="../V2_raster_first1s_s7.png" alt="V2 first 1 s seed 7">
<img src="../V2_raster_last1s_s7.png" alt="V2 last 1 s seed 7">
<p>Seeds 8 and 9 are in the parent directory as <code>V2_raster_*_s8.png</code> and <code>V2_raster_*_s9.png</code>.</p>
<h2>Rates</h2><img src="rates.png" alt="population rates">
<h2>LFP proxy (qualified)</h2><img src="lfp.png" alt="LFP proxy and PSD">
<h2>Conductances (seed 7)</h2><img src="conductance.png" alt="conductances">
<h2>Diagnostics: readout conditions</h2><img src="readout.png" alt="readout">
<table><tr><th>seed</th><th>G0</th><th>G1</th><th>G2</th><th>G3</th><th>G1'</th><th>K1</th></tr>{vrows}</table>
<h2>Diagnostics: motif maps (sealed scorer, per seed)</h2>
<img src="motif_s7.png" alt="motif seed 7"><img src="motif_s8.png" alt="motif seed 8"><img src="motif_s9.png" alt="motif seed 9">
</body></html>"""
    (atlas / "index.html").write_text(body, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = OUT / "atlas"
    atlas.mkdir(parents=True, exist_ok=True)
    zs = {tg: np.load(RES / f"{tg}.npz") for tg in TAGS}
    recs = {tg: json.loads((RES / f"{tg}.json").read_text(encoding="utf-8")) for tg in TAGS}
    zsm = np.load(SMOKE)
    recsm = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))

    # identity: the scored first second equals the smoke run, sample for sample
    ident = same_first_second(zsm, zs[TAGS[0]], recsm, recs[TAGS[0]])
    print("identity smoke vs scored s7 first 1 s:", json.dumps(ident))
    if not ident["all_blocks_equal"]:
        raise SystemExit("first second differs from the smoke run; V2 window is not the V1 window")

    n1 = raster(zsm, recsm, 0.0, 1000.0, "V1 initial 1 s raster, seed 7 (smoke run, sampled cells)",
                OUT / "V1_raster_initial_1000ms_s7.png")
    points = {"V1_s7_0_1s": n1}
    for tg, s in zip(TAGS, (7, 8, 9)):
        points[f"V2_first1s_s{s}"] = raster(zs[tg], recs[tg], 0.0, 1000.0,
                                            f"V2 final run, seed {s}: first 1 s (same window as V1)",
                                            OUT / f"V2_raster_first1s_s{s}.png")
        points[f"V2_last1s_s{s}"] = raster(zs[tg], recs[tg], 11000.0, 12000.0,
                                           f"V2 final run, seed {s}: last 1 s (11-12 s)",
                                           OUT / f"V2_raster_last1s_s{s}.png")
    shutil.copy(V0DIR / "V0_rule_based_schematic.png", OUT / "V0_rule_based_schematic.png")
    rules = json.loads((V0DIR / "V0_rule_table.json").read_text(encoding="utf-8"))
    rates_panel(zs, recs, atlas / "rates.png")
    lfp_panel(zs, recs, atlas / "lfp.png")
    conductance_panel(zs, recs, atlas / "conductance.png")
    vals = readout_panel(atlas / "readout.png")
    for s in (7, 8, 9):
        shutil.copy(RES / f"r2_p0_n4000_s{s}_T12_motif.png", atlas / f"motif_s{s}.png")
    write_atlas(atlas, rules, vals)
    # V1 and V2 rasters are referenced by the atlas at the lineage root
    print("points per figure:", json.dumps(points))
    print("written", OUT)


if __name__ == "__main__":
    main()

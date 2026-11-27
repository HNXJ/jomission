"""Visualization stages for lineage SCI-LANSA-GEOM-R2 (manifests/visualization_contract.json).

V0: rule-based construction schematic, from scripts/e2_r2_v0.py (not an HSPICE schematic; see the atlas).
V1: initial 1 s raster from the 1 s smoke run (sampled cells: the driver keeps a fixed sample of cells per block).
V2: final rasters from the scored 12 s runs, first 1 s (same window as V1) and last 1 s.
V3: provisional atlas (lineage not complete: K2 NOT_RUN). Built from the saved arrays only; no re-simulation.

The LFP panel uses the scored proxy and estimator: e2_motif_figure.rws (RWS on g_nS) and e2_scorer.unit_spectra
(1 kHz, DPSS 1 s windows, 2 Hz smoothing, first 2 s discarded), so the atlas shows what the sealed analysis scored.

Usage: python scripts/e2_r2_viz.py RES_DIR OUT_DIR SMOKE_NPZ SMOKE_JSON V0_DIR ARCHIVE_V1_PNG
  RES_DIR         results/laminar_e2 (npz, json, motif png, readout json)
  OUT_DIR         results/viz/SCI-LANSA-GEOM-R2
  V0_DIR          directory with V0_rule_based_schematic.png and V0_rule_table.json (may equal OUT_DIR)
  ARCHIVE_V1_PNG  the earlier full-population V1 image of the smoke run, copied to OUT_DIR/archive unchanged
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

RES, OUT, SMOKE, SMOKE_JSON, V0DIR, ARCHIVE_V1 = (Path(p) for p in sys.argv[1:7])
DT_MS = 0.1  # driver dt (run record dt_ms)
UNITS = ["S", "M", "D"]
TYPES = ["E", "PV", "SOM"]
RASTER_N = {"E": 200, "PV": 50, "SOM": 50}  # run_e2_stack.py
COL = {"E": "#c44e52", "PV": "#2a9d8f", "SOM": "#8d6cc6"}
TAGS = [f"r2_p0_n4000_s{s}_T12" for s in (7, 8, 9)]
CONDS = ["G0", "G1", "G2", "G3", "G1p", "K1"]
COND_LABEL = {"G0": "G0", "G1": "G1", "G2": "G2", "G3": "G3", "G1p": "G1'", "K1": "K1 (flat)"}
SIZE_SHARE_NOTE = ("Sampled raster: the driver keeps a fixed sample of cells per unit "
                   "(E 200, PV 50, SOM 50; default_rng(0)); not the full population.")


def block_rows(rec: dict) -> list[tuple[str, str, int, int]]:
    """(unit, type, row offset, n sampled rows) in V1 order: unit, then class E, PV, SOM."""
    out, off = [], 0
    for u in UNITS:
        for t in TYPES:
            n = min(RASTER_N[t], int(rec["sizes"][u][t]))
            out.append((u, t, off, n))
            off += n
    return out


def spikes(z, u: str, t: str, t0_ms: float, t1_ms: float):
    b0, b1 = int(round(t0_ms / DT_MS)), int(round(t1_ms / DT_MS))
    ts = z[f"{u}_{t}_t"].astype(np.int64)
    cs = z[f"{u}_{t}_c"].astype(np.int64)
    sel = (ts >= b0) & (ts < b1)
    return ts[sel] * DT_MS, cs[sel]


def raster(z, rec: dict, t0_ms: float, t1_ms: float, title: str, path: Path) -> int:
    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=130)
    n_pts = 0
    rows = block_rows(rec)
    for u, t, off, n in rows:
        x, c = spikes(z, u, t, t0_ms, t1_ms)
        ax.scatter(x, off + c, s=0.4, c=COL[t], marker="|", linewidths=0.4)
        n_pts += x.size
    seen = None
    for u, t, off, n in rows:
        if u != seen:
            ax.axhline(off, color="#333", lw=0.9)
            ax.text(t0_ms + 0.006 * (t1_ms - t0_ms), off + n * 0.5, f"unit {u}", ha="left", va="center",
                    fontsize=10, fontweight="bold", bbox=dict(fc="white", ec="none", pad=1.5))
            seen = u
        else:
            ax.axhline(off, color="#bbb", lw=0.4)
    total = rows[-1][2] + rows[-1][3]
    ax.set_ylim(total, 0)
    ax.set_xlim(t0_ms, t1_ms)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("sampled cells (unit, then E, PV, SOM)")
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[plt.Line2D([], [], color=COL[k], marker="|", ls="", label=k) for k in TYPES],
              loc="upper right", fontsize=8, frameon=False)
    ax.text(0.0, -0.06, SIZE_SHARE_NOTE, transform=ax.transAxes, fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return n_pts


def same_first_second(z_smoke, z_scored, rec_scored: dict) -> dict:
    """Identity check: the scored run's first 1 s sample equals the smoke run's whole sample, block by block."""
    res = {}
    for u, t, _, _ in block_rows(rec_scored):
        ts_s = z_smoke[f"{u}_{t}_t"].astype(np.int64)
        cs_s = z_smoke[f"{u}_{t}_c"].astype(np.int64)
        ts_c = z_scored[f"{u}_{t}_t"].astype(np.int64)
        cs_c = z_scored[f"{u}_{t}_c"].astype(np.int64)
        keep = ts_c < int(round(1000.0 / DT_MS))
        res[f"{u}_{t}"] = bool(np.array_equal(ts_s, ts_c[keep]) and np.array_equal(cs_s, cs_c[keep]))
    counts_eq = bool(np.array_equal(z_smoke["counts"], z_scored["counts"][: z_smoke["counts"].shape[0]]))
    return {"blocks_equal": res, "all_blocks_equal": all(res.values()), "counts_first_1s_equal": counts_eq}


def rates_panel(zs: dict, recs: dict, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=120, sharex=True)
    for ax, u in zip(axes, UNITS):
        for t in TYPES:
            tot = None
            for tag in TAGS:
                z, rec = zs[tag], recs[tag]
                j = list(z["block_keys"]).index(f"{u}_{t}")
                cnt = z["counts"][:, j].reshape(-1, 100).sum(axis=1)  # 10 ms bins
                rate = cnt / (rec["sizes"][u][t] * 0.01)
                tot = rate if tot is None else np.vstack([tot, rate])
            m = tot.mean(axis=0)
            ax.plot(np.arange(m.size) * 0.01, m, color=COL[t], lw=0.8, label=t)
        ax.set_ylabel(f"unit {u}\nrate (Hz)")
        ax.set_ylim(bottom=0)
        ax.axvline(2.0, color="#777", ls=":", lw=0.9)  # scorer discard_s = 2.0
    axes[0].legend(fontsize=8, frameon=False, ncol=3)
    axes[-1].set_xlabel("time (s), mean of seeds 7-9, 10 ms bins; dotted line = 2 s (scorer discard)")
    axes[0].set_title("Population rate per block (mean over the three scored seeds); onset transient at 0 s", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def lfp_panel(recs: dict, path: Path) -> None:
    """The scored proxy and estimator. Trace: RWS proxy, seed 7, 2-4 s. Spectrum: diagonal of the scored cross-spectrum
    (e2_scorer.unit_spectra: DPSS 1 s windows, 2 Hz smoothing, 2 s discarded), mean over windows and seeds 7-9."""
    fig, (axt, axp) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=120)
    rec7 = recs[TAGS[0]]
    z7 = np.load(RES / f"{TAGS[0]}.npz")
    x = mf.rws(z7["g_nS"].astype(np.float64), DT_MS, rec7, rec7["units"])  # [T - delay, U], pA
    t = np.arange(x.shape[0]) * DT_MS / 1000.0
    m = (t >= 2.0) & (t < 4.0)
    for k, u in enumerate(rec7["units"]):
        axt.plot(t[m], x[m, k], lw=0.5, label=f"unit {u}")
    axt.set_xlabel("time (s), seed 7")
    axt.set_ylabel("RWS proxy (pA), scored")
    axt.legend(fontsize=8, frameon=False)
    psd = []
    for tg in TAGS:
        f, S, _ = sc.unit_spectra(tg, 2.0, RES)
        diag = np.real(np.diagonal(S, axis1=2, axis2=3))  # [windows, f, U]
        psd.append(diag.mean(axis=0))
    P = np.mean(psd, axis=0)
    sel = (f >= 1) & (f <= 150)
    for k, u in enumerate(rec7["units"]):
        axp.loglog(f[sel], P[sel, k], lw=0.9, label=f"unit {u}")
    axp.axvspan(10, 30, color="#999", alpha=0.15, label="alpha-beta band 10-30 Hz")
    axp.axvspan(50, 150, color="#c44e52", alpha=0.10, label="gamma band 50-150 Hz")
    axp.set_xlabel("frequency (Hz)")
    axp.set_ylabel("PSD (pA^2/Hz), scored estimator, mean of seeds 7-9")
    axp.legend(fontsize=7, frameon=False)
    fig.suptitle("Scored RWS proxy (current-based model LFP; not a calibrated potential)", fontsize=10)
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
    ax.set_xlabel("time (s), seed 7 (includes the onset transient)")
    ax.set_ylabel("conductance (nS), 1 ms mean")
    ax.legend(fontsize=8, ncol=3, frameon=False)
    ax.set_title("Synaptic conductance proxies recorded by the driver", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def readout_panel(path: Path) -> dict:
    seeds = [7, 8, 9]
    data = {}
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    w = 0.26
    for i, s in enumerate(seeds):
        r = json.loads((RES / f"r2_readout_r2_p0_n4000_s{s}_T12.json").read_text(encoding="utf-8"))
        data[s] = r
        vals = [r["conditions"][c]["median_r_gm"] for c in CONDS]
        ax.bar(np.arange(len(CONDS)) + (i - 1) * w, vals, w, label=f"seed {s}")
    ax.axhline(0.6, color="k", ls="--", lw=1)
    ax.text(len(CONDS) - 0.5, 0.62, "r_gm threshold 0.6 (the claim)", ha="right", fontsize=8)
    ax.axhline(0, color="#777", lw=0.6)
    ax.set_xticks(np.arange(len(CONDS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
    ax.set_ylabel("median r_gm over 11 held-out areas")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Readout conditions (sealed R2 spectra). G1 is below 0.6 and below K1 on every seed. K2 NOT_RUN.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return data


def readout_table(data: dict) -> str:
    rows = []
    for s in (7, 8, 9):
        r = data[s]
        for c in CONDS:
            cc = r["conditions"][c]
            gm, ab = cc["median_r_gm_ge"], cc["median_r_ab_ge"]
            cross = cc["crossing_2se"]
            cross_txt = "not defined" if cross is None else str(cross).lower()
            rows.append(
                f"<tr><td>{s}</td><td>{COND_LABEL[c]}</td><td>{cc['median_r_gm']:.3f}</td>"
                f"<td>{'yes' if gm['0.5'] else 'no'}</td><td>{'yes' if gm['0.6'] else 'no'}</td>"
                f"<td>{'yes' if gm['0.7'] else 'no'}</td><td>{cc['median_r_ab']:.3f}</td>"
                f"<td>{'yes' if ab['0.6'] else 'no'}</td><td>{cross_txt}</td></tr>")
    head = ("<tr><th>seed</th><th>condition</th><th>median r_gm</th><th>r_gm &ge; 0.5</th><th>r_gm &ge; 0.6</th>"
            "<th>r_gm &ge; 0.7</th><th>median r_ab</th><th>r_ab &ge; 0.6</th><th>crossing_2se (model profile)</th></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def write_atlas(atlas: Path, rules: dict, data: dict) -> None:
    atlas.mkdir(parents=True, exist_ok=True)
    rule_rows = "".join(f"<tr><td>{html.escape(k)}</td><td style='text-align:right'>{v:,}</td></tr>"
                        for k, v in rules["rules"].items())
    stages = [
        ("V0", "construction", "rule-based block schematic; units S, M, D; depths are thickness fractions relative to L4; "
         "not HSPICE; contract elements not drawn are listed on the figure", "present, partial"),
        ("V1", "initial 1 s raster", "sampled cells, seed 7 smoke run, 0-1 s; the earlier full-population image is kept in "
         "archive/", "present, sampled"),
        ("V2", "final raster", "sampled cells, scored seeds 7-9: 0-1 s (same window as V1) and 11-12 s", "present, sampled"),
        ("connectivity", "rule table", "33 construction rules, realised edge counts (seed 7)", "present"),
        ("raster", "raster", "as V2", "present, sampled"),
        ("rates", "population rates", "per block, mean over seeds 7-9, with the scorer's 2 s discard marked", "present"),
        ("field / LFP proxy", "RWS proxy and scored spectra", "the scored proxy and estimator; qualified: not a potential",
         "present, qualified"),
        ("conductances", "g_nS", "1 ms means, seed 7, first 2 s", "present"),
        ("membrane dynamics", "V(t)", "the driver does not record membrane potential", "NOT RECORDED"),
        ("I_net", "net synaptic current", "the driver does not record it", "NOT RECORDED"),
        ("H, A, m(H)", "H-state, adaptation", "the lineage has no H-state (AdEx populations)", "NOT APPLICABLE"),
        ("plastic weights", "effective weights", "the lineage has no plasticity", "NOT APPLICABLE"),
        ("diagnostics", "readout, crossing, sensitivity", "sealed scorer; readout G0 to G1' and K1; crossing per condition "
         "(K1 not defined); K2 NOT_RUN", "partial (K2 NOT_RUN)"),
    ]
    srows = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{html.escape(c)}</td>"
                    f"<td><b>{html.escape(d)}</b></td></tr>" for a, b, c, d in stages)
    deviations = [
        "V0 is rule-based and is not an HSPICE schematic. The default route stores no edge list, so the renderer cannot run.",
        "V0 does not draw x/y placement, O and X connections, FF/FB direction, E/I projection sign, or connection authority g or w.",
        "V1 and V2 show the driver's fixed cell sample (E 200, PV 50, SOM 50 per unit), not all 14251 units.",
        "The V1 full-population image of the smoke run is kept in archive/ and is not the V1 shown above.",
        "V3 is provisional: the lineage is not complete (K2 NOT_RUN). Membrane dynamics, I_net and H/A/m(H) are not recorded or not applicable.",
        "The LFP panel is the scored RWS proxy (not lfp_pA, which the analysis does not use).",
    ]
    dev = "".join(f"<li>{html.escape(d)}</li>" for d in deviations)
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
<h2>Deviations from the contract</h2><ul>{dev}</ul>
<h2>V0: construction (rule-based)</h2>
<img src="../V0_rule_based_schematic.png" alt="V0 rule-based schematic">
<h2>Connectivity: construction rules and realised edge counts (seed 7)</h2>
<table><tr><th>rule</th><th>realised edges</th></tr>{rule_rows}</table>
<h2>V1: initial raster, 0-1 s (seed 7 smoke run, sampled cells)</h2>
<p>{html.escape(SIZE_SHARE_NOTE)}</p>
<img src="../V1_raster_initial_1000ms_s7.png" alt="V1">
<h2>V2: final raster, scored seeds (first and last second)</h2>
<img src="../V2_raster_first1s_s7.png" alt="V2 first 1 s seed 7">
<img src="../V2_raster_last1s_s7.png" alt="V2 last 1 s seed 7">
<p>Seeds 8 and 9 are in the parent directory as <code>V2_raster_*_s8.png</code> and <code>V2_raster_*_s9.png</code>.</p>
<h2>Rates</h2><img src="rates.png" alt="population rates">
<h2>Field: scored RWS proxy (qualified)</h2><img src="lfp.png" alt="scored RWS proxy and spectra">
<h2>Conductances (seed 7)</h2><img src="conductance.png" alt="conductances">
<h2>Diagnostics: readout conditions</h2><img src="readout.png" alt="readout">
{readout_table(data)}
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

    # identity gate: the scored first second equals the smoke run, sample for sample and count for count
    ident = same_first_second(zsm, zs[TAGS[0]], recs[TAGS[0]])
    ok = (ident["all_blocks_equal"] and ident["counts_first_1s_equal"]
          and recsm["sizes"] == recs[TAGS[0]]["sizes"] and zsm["counts"].shape[0] == 10000)
    print("identity smoke vs scored s7 first 1 s:", json.dumps(ident), "sizes equal:",
          recsm["sizes"] == recs[TAGS[0]]["sizes"])
    if not ok:
        raise SystemExit("identity gate failed: V2 first second is not the V1 window; nothing written")

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
    src, dst = V0DIR / "V0_rule_based_schematic.png", OUT / "V0_rule_based_schematic.png"
    if src.resolve() != dst.resolve():
        shutil.copy(src, dst)
    rules = json.loads((V0DIR / "V0_rule_table.json").read_text(encoding="utf-8"))
    (OUT / "archive").mkdir(exist_ok=True)
    shutil.copy(ARCHIVE_V1, OUT / "archive" / "V1_fullpopulation_smoke_s7_archive.png")
    rates_panel(zs, recs, atlas / "rates.png")
    lfp_panel(recs, atlas / "lfp.png")
    conductance_panel(zs, recs, atlas / "conductance.png")
    data = readout_panel(atlas / "readout.png")
    for s in (7, 8, 9):
        shutil.copy(RES / f"r2_p0_n4000_s{s}_T12_motif.png", atlas / f"motif_s{s}.png")
    write_atlas(atlas, rules, data)
    print("points per figure:", json.dumps(points))
    print("written", OUT)


if __name__ == "__main__":
    main()

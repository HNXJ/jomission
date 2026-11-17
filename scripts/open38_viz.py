"""OPEN-38 V0-V3 (manifests/visualization_contract.json), drawn retrospectively from recorded evidence.

Usage (JaxFNE 0.5.9 importable, as for the runs):
  python -B scripts/open38_viz.py

Nothing is simulated. Every figure is drawn from files the executed runs wrote, after checking them:
  V0  the reference model of seed 1 rebuilt with caminos_stack.construct (construction only), accepted only when
      its edge table equals the one the run recorded; drawn by jaxfne_vis.network_hspice.
  V1  the recorded raster sample of the reference run, seed 1, 0-1000 ms; jaxfne_vis.network_raster.
  V2  the same call and window for every arm, seed 1 (the state after the composition intervention).
  V3  atlas/index.html: the scorer's relative-power maps (seed means), per-run metrics from the verdict files,
      block rates from the recorded counts, LFP-proxy traces, the recorded edge table.
Each .npz must hash to the npz_sha256 its stage's integrity receipt recorded, and each run record must carry its
stage's execution SHA. The raster sample's cells are re-derived exactly as run_e2_stack.py chose them (rng 0).
Writes results/viz/SCI-OPEN38-O38S/ and its provenance.json.
"""
from __future__ import annotations

import datetime
import hashlib
import html
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402
import run_e2_stack as run  # noqa: E402
from jomission.laminar import caminos_e0 as c0  # noqa: E402
from jomission.laminar import caminos_stack as cs  # noqa: E402
from jomission.visualization import jaxfne_vis as vis  # noqa: E402

LINEAGE = "SCI-OPEN38-O38S"
D = ROOT / "results" / "laminar_e2"
OUT = ROOT / "results" / "viz" / LINEAGE
STAGES = {
    "stage1": {"exec": "53f8b89fdb1ec4d67e3c972dd24ee4c899a680cc", "integrity": "o38s_pc_integrity.json",
               "arms": ("ref", "flat", "rev"), "tag": "o38s_pc_{arm}_n4000_s{s}_T8"},
    "stage2": {"exec": "0d7162210553127482659d758b102f3b8f5ea71c", "integrity": "o38s_s2_integrity.json",
               "arms": ("v1", "mt", "mst", "teo", "chc", "shape"), "tag": "o38s_{arm}_n4000_s{s}_T8"},
}
SEEDS = (1, 2, 3, 4, 5)
UNITS = ("S", "M", "D")
DISCARD_S = 2.0  # the scorer's discard, used for the maps and the block rates
WINDOW_MS = (0.0, 1000.0)
NOTE = "drawn retrospectively {date} from recorded outputs; executed at {exe}"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def runs() -> list[dict]:
    """Every run of both stages with its verified sources."""
    out = []
    for stage, s in STAGES.items():
        integ = json.loads((D / s["integrity"]).read_text(encoding="utf-8"))
        recorded = {r["tag"]: r["npz_sha256"] for r in integ["runs"]}
        for arm in s["arms"]:
            for seed in SEEDS:
                tag = s["tag"].format(arm=arm, s=seed)
                rec = json.loads((D / f"{tag}.json").read_text(encoding="utf-8"))
                npz = D / f"{tag}.npz"
                h = sha256(npz)
                if h != recorded[tag]:
                    raise SystemExit(f"{tag}.npz sha256 {h} != integrity receipt {recorded[tag]}")
                if rec["git_sha"] != s["exec"] or rec["seed"] != seed:
                    raise SystemExit(f"{tag}: run record git_sha {rec['git_sha']} seed {rec['seed']}")
                out.append({"stage": stage, "arm": arm, "seed": seed, "tag": tag, "exec": s["exec"], "rec": rec,
                            "npz": npz, "npz_sha256": h, "json_sha256": sha256(D / f"{tag}.json")})
    return out


class View:
    """A jaxfne model seen with each unit (S, M, D) as an area, so the renderers draw inter-unit projections.

    The model has one area C whose layers are the units; the renderers treat same-area edges as local and would
    draw none of the S/M/D coupling. Only labels change; ids, edges and weights are the model's own."""

    def __init__(self, rows: list[dict], params=None):
        self.rows, self.params = rows, params

    def neuron_table(self):
        return self.rows


def unit_rows(model) -> list[dict]:
    t = model.neuron_table()
    rows = list(t.to_dict("records")) if hasattr(t, "to_dict") else list(t)
    return [{"neuron_id": int(r["neuron_id"]), "area": str(r["layer"]), "layer": str(r["layer"]),
             "cell_type": str(r["cell_type"])} for r in rows]


def construct(rec: dict):
    lay, seed = rec["layout"], rec["seed"]
    sizes, _, p_inter, _ = cs.layout(lay)
    mods = cs.variant(lay)[1]
    if sizes != rec["sizes"]:
        raise SystemExit(f"layout {lay} sizes differ from the run record")
    model, _ = cs.construct(seed, sizes, p_inter, cs.local_p(lay, sizes), tau_sst_e_ms=cs.sst_taus(lay, sizes),
                            k_scale=mods.get("k", 1.0), deep_size=mods.get("dsz", 1.0),
                            gaba_scale=mods.get("gi", 1.0), triple_drive=cs.drive_entries(lay, sizes),
                            w_scales=mods.get("w"))
    if json.loads(json.dumps(cs.edge_table(model))) != rec["edges"]:
        raise SystemExit(f"{lay} seed {seed}: rebuilt edge table differs from the run record")
    return model


def raster_sample(r: dict) -> tuple[View, np.ndarray]:
    """The recorded raster sample as (n_steps, n_cells), with the cells run_e2_stack.py picked."""
    rec, z = r["rec"], np.load(r["npz"])
    n_steps = int(z["counts"].shape[0])
    if n_steps != round(rec["T_s"] * 1000.0 / c0.DT_MS) or rec["dt_ms"] != c0.DT_MS:
        raise SystemExit(f"{r['tag']}: {n_steps} steps != T_s {rec['T_s']} / dt {rec['dt_ms']}")
    sl = cs.block_slices(rec["sizes"])
    rng = np.random.default_rng(0)  # run_e2_stack.py's raster pick, same order of blocks
    rows, cols = [], []
    for m in rec["units"]:
        for t in cs.POPS:
            idx = np.arange(sl[(m, t)].start, sl[(m, t)].stop)
            pick = np.sort(rng.choice(idx, size=min(run.RASTER_N[t], idx.size), replace=False))
            st, cell = z[f"{m}_{t}_t"], z[f"{m}_{t}_c"]
            if st.size and (st.min() < 0 or st.max() >= n_steps or cell.max() >= pick.size):
                raise SystemExit(f"{r['tag']} {m}_{t}: spike index outside the run or the picked cells")
            base = len(rows)
            rows += [{"neuron_id": base + i, "area": m, "layer": m, "cell_type": t, "model_id": int(p)}
                     for i, p in enumerate(pick)]
            cols.append((base, st, cell))
    spikes = np.zeros((n_steps, len(rows)), dtype=bool)
    for base, st, cell in cols:
        spikes[st, base + cell.astype(np.int64)] = True
    return View(rows), spikes


def draw_raster(r: dict, path: Path, title: str, date: str) -> dict:
    view, spikes = raster_sample(r)
    n = {t: run.RASTER_N[t] for t in cs.POPS}
    sub = (f"{r['tag']}: recorded raster sample, {n['E']} E, {n['PV']} PV, {n['SOM']} SOM cells per unit "
           f"(rng 0, as run_e2_stack.py); t 0 = simulation start, so this window is the start-up transient "
           f"(the scorer discards the first {DISCARD_S:.0f} s); " + NOTE.format(date=date, exe=r["exec"][:7]))
    info = vis.network_raster(view, spikes, window_ms=WINDOW_MS, dt_ms=c0.DT_MS, areas=list(UNITS), title=title,
                              subtitle=sub, path=path, class_order=cs.POPS)
    info["n_steps_recorded"] = int(spikes.shape[0])
    return info


def rule_table(model) -> list[tuple]:
    """Every construction rule: source -> target block, edge count, mean weight, receptor (kernel arrays)."""
    st = model.static["adex"]
    rid, pre, post = np.asarray(st["rule_id"]), np.asarray(st["pre"]), np.asarray(st["post"])
    w, rec = np.asarray(st["weight_nS"]), np.asarray(st["receptor"])
    rows_ = unit_rows(model)
    blk = np.array([f"{r['area']}.{r['cell_type']}" for r in sorted(rows_, key=lambda r: r["neuron_id"])])
    out = []
    for i, (name, _) in enumerate(st["rules"]):
        m = rid == i
        if not m.any():
            out.append((name, "-", 0, float("nan"), "-"))
            continue
        src, dst, rc = set(blk[pre[m]]), set(blk[post[m]]), set(rec[m].tolist())
        if len(src) != 1 or len(dst) != 1 or len(rc) != 1:
            raise SystemExit(f"rule {name} spans several blocks or receptors: {src} {dst} {rc}")
        out.append((name, f"{src.pop()} -> {dst.pop()}", int(m.sum()), round(float(w[m].mean()), 4),
                    st["receptor_names"][rc.pop()]))
    return out


def compose_v0(blocks_png: Path, rules: list[tuple], out: Path) -> None:
    im = plt.imread(blocks_png)
    n = len(rules)
    fig = plt.figure(figsize=(18, 9 + 0.22 * n))
    ax = fig.add_axes([0, 0.22 * n / (9 + 0.22 * n) + 0.02, 1, 1 - 0.22 * n / (9 + 0.22 * n) - 0.02])
    ax.imshow(im)
    ax.axis("off")
    tab = fig.add_axes([0.12, 0.01, 0.76, 0.22 * n / (9 + 0.22 * n)])
    tab.axis("off")
    t = tab.table(cellText=[[r[0], r[1], f"{r[2]:,}", f"{r[3]:.4f}", r[4]] for r in rules],
                  colLabels=["rule", "source -> target block", "edges", "mean w (nS)", "receptor"],
                  loc="upper center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.1)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def tile(paths: list[Path], out: Path, title: str) -> None:
    imgs = [plt.imread(p) for p in paths]
    fig, axes = plt.subplots(3, 3, figsize=(24, 15))
    for ax, im in zip(axes.flat, imgs):
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def block_rates(r: dict) -> dict:
    z = np.load(r["npz"])
    keys = [str(k) for k in z["block_keys"]]
    a = int(round(DISCARD_S * 1000.0 / c0.DT_MS))
    T = (z["counts"].shape[0] - a) * c0.DT_MS * 1e-3
    out = {}
    for j, k in enumerate(keys):
        m, t = k.split("_")
        out[k] = round(float(z["counts"][a:, j].sum()) / (r["rec"]["sizes"][m][t] * T), 3)
    return out


def main() -> None:
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "atlas").mkdir(exist_ok=True)
    allr = runs()
    by = {(r["arm"], r["seed"]): r for r in allr}
    arms = [a for s in STAGES.values() for a in s["arms"]]
    ref1 = by[("ref", 1)]

    # V0
    model = construct(ref1["rec"])
    st = model.static["adex"]  # the AdEx kernel's own edge arrays (params["edge_list"] is None for this model)
    edge_list = type("Edges", (), {"pre": st["pre"], "post": st["post"], "weight": st["weight_nS"],
                                   "receptor_index": st["receptor"]})
    inhib = tuple(i for i, n in enumerate(st["receptor_names"]) if n.startswith("gaba"))
    view = View(unit_rows(model), {"edge_list": edge_list})
    v0 = vis.network_hspice(
        view, inhibitory_receptors=inhib,
        x="Poisson drive to E and PV (c0.drive_spec)", y="LFP proxy per unit", stages=[list(UNITS)],
        title=(f"OPEN-38 reference (prescribed SOM/PV ramp), seed 1: construction of {ref1['tag']}\n"
               f"units S, M, D drawn as blocks (the model holds them as layers of one area C), so below 'within-area' "
               f"means within a unit and 'across areas' between units; w = mean weight in nS;\n"
               f"overlapping arrow labels hide 10 of the 12 inter-unit projections: the rule table is the full "
               f"topology; "
               + NOTE.format(date=date, exe=ref1["exec"][:7])),
        path=OUT / "_hspice_blocks.png", class_order=cs.POPS)
    rule_rows = rule_table(model)
    if sum(r[2] for r in rule_rows) != v0["n_edges_total"]:
        raise SystemExit("rule table does not account for every edge")
    compose_v0(OUT / "_hspice_blocks.png", rule_rows, OUT / "jaxfne_network_hspice.png")
    (OUT / "_hspice_blocks.png").unlink()
    v0["rules"] = [dict(zip(("rule", "blocks", "n_edges", "mean_w_nS", "receptor"), r)) for r in rule_rows]

    # V1, V2
    v1 = draw_raster(ref1, OUT / "raster_initial_1000ms.png",
                     "V1 initial state: reference (prescribed ramp), seed 1, 0-1000 ms", date)
    tiles, v2_parts = [], {}
    for a in arms:
        p = OUT / "atlas" / f"raster_{a}_s1.png"
        v2_parts[a] = draw_raster(by[(a, 1)], p, f"{a}, seed 1, 0-1000 ms", date)
        tiles.append(p)
    tile(tiles, OUT / "raster_final.png",
         "V2 after the composition intervention: every arm, seed 1, the V1 window and call (ref repeated top left)\n"
         "arm against arm at seed 1, not one model before and after; the window is the start-up transient; "
         + f"drawn retrospectively {date} from recorded raster samples")

    # V3: maps (scorer, seed means), metrics, rates, LFP proxy
    ext = [float(sc.F[0]), float(sc.F[-1]), float(mf.CHANNELS[-1]), float(mf.CHANNELS[0])]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True, constrained_layout=True)
    metrics = []
    for ax, a in zip(axes.flat, arms):
        maps = []
        for s in SEEDS:
            r = by[(a, s)]
            _, S, K = sc.unit_spectra(r["tag"], DISCARD_S, D)
            maps.append(sc.rp_map(S.mean(axis=0), K))
            v = json.loads((D / f"{r['tag']}_verdict.json").read_text(encoding="utf-8"))
            metrics.append({"tag": r["tag"], "arm": a, "stage": r["stage"], "exec": r["exec"][:7],
                            **{k: v[k] for k in ("r_ab", "r_gm", "r_proto", "crossing_2se")},
                            "rates_hz": block_rates(r)})
        mv = [m for m in metrics if m["arm"] == a]
        im = ax.imshow(np.mean(maps, axis=0), aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=ext)
        ax.set_title(f"{a}: mean r_ab {np.mean([m['r_ab'] for m in mv]):+.3f} (5 seeds)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("depth rel. mid-L4")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="relative power (P / max over depth)")
    fig.suptitle(f"Relative-power maps, scorer e2_scorer.rp_map, {DISCARD_S:.0f} s discard, seed means; "
                 f"drawn retrospectively {date}", fontsize=10)
    fig.savefig(OUT / "atlas" / "sl_maps.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, a in zip(axes.flat, arms):
        z = np.load(by[(a, 1)]["npz"])
        b = int(WINDOW_MS[1] / c0.DT_MS)
        t = np.arange(b) * c0.DT_MS
        for j, u in enumerate(UNITS):
            ax.plot(t, z["lfp_pA"][:b, j], lw=0.5, label=u)
        ax.set_title(f"{a}, seed 1", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.supxlabel("time from simulation start (ms)")
    fig.supylabel("LFP proxy (pA)")
    fig.suptitle(f"LFP proxy: {ref1['rec']['lfp_proxy']}; drawn retrospectively {date}", fontsize=9)
    fig.savefig(OUT / "atlas" / "lfp_proxy_s1.png", dpi=150)
    plt.close(fig)

    def table(head, rows_):
        return ("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"
                + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows_)
                + "</table>")

    rate_keys = list(metrics[0]["rates_hz"])
    edges = ref1["rec"]["edges"]
    edge_rows = [[json.dumps(e)] for e in edges] if isinstance(edges, list) else \
        [[k, json.dumps(v)] for k, v in edges.items()]
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{LINEAGE} atlas</title>
<style>body{{font-family:Arial,sans-serif;font-size:13px;max-width:1300px;margin:20px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:2px 6px;font-size:11px}}img{{max-width:100%}}</style></head><body>
<h1>{LINEAGE}: OPEN-38 spectrum sensitivity (V3 atlas)</h1>
<p><b>Retrospective.</b> Created {created} from recorded outputs; no simulation was run to make it. Executions:
stage 1 at {STAGES['stage1']['exec'][:7]}, stage 2 at {STAGES['stage2']['exec'][:7]}. Each .npz matched the
npz_sha256 in its stage's integrity receipt. Sources and hashes: <a href="../provenance.json">provenance.json</a>.</p>
<h2>Construction (V0)</h2><img src="../jaxfne_network_hspice.png">
<h3>Recorded edge table, {html.escape(ref1['tag'])}</h3>{table(['edges'] if edge_rows and len(edge_rows[0]) == 1 else ['block', 'record'], edge_rows)}
<h2>Rasters (V1, V2)</h2><img src="../raster_initial_1000ms.png"><img src="../raster_final.png">
<h2>Relative-power maps</h2><img src="sl_maps.png">
<h2>Field: LFP proxy (recurrent synaptic current onto E cells, not an LFP)</h2><img src="lfp_proxy_s1.png">
<h2>Per run: verdict metrics and block rates (Hz, after {DISCARD_S:.0f} s)</h2>
{table(['tag', 'stage', 'exec', 'r_ab', 'r_gm', 'r_proto', 'crossing_2se'] + rate_keys,
       [[m['tag'], m['stage'], m['exec'], f"{m['r_ab']:+.4f}", f"{m['r_gm']:+.4f}", f"{m['r_proto']:+.4f}",
         m['crossing_2se']] + [m['rates_hz'][k] for k in rate_keys] for m in metrics])}
<h2>Not in this atlas</h2><p>Membrane potential, I_net, H, A, m(H) and plastic weights: run_e2_stack.py did not record
them, and the model has no plasticity. Weights are the static constructed weights shown in V0 and the edge table.</p>
</body></html>"""
    (OUT / "atlas" / "index.html").write_text(page, encoding="utf-8")

    prov = {
        "lineage_id": LINEAGE, "created_utc": created, "generator": "scripts/open38_viz.py",
        "nature": "retrospective visualization of recorded evidence; no simulation was run to make it",
        "executions": {k: v["exec"] for k, v in STAGES.items()},
        "raster_window_note": "0-1000 ms from simulation start (contract V1 'first 1000 ms'): the start-up "
                              f"transient; the scored state begins after the {DISCARD_S:.0f} s discard. V2 compares "
                              "arms with the reference at seed 1, not one model before and after.",
        "v0_note": "the full topology is the rule table (all rules, edge counts, mean weights, receptors); the "
                   "drawn arrows label 2 of the 12 inter-unit projections legibly",
        "V0": {"source_run": ref1["tag"], "procedure": "caminos_stack.construct from the run record's layout and "
               "seed; edge table equal to the recorded one", **{k: v0[k] for k in ("n_neurons", "n_edges_total",
                                                                                   "n_projections", "rules")}},
        "V1": {"source_run": ref1["tag"], **{k: v1[k] for k in ("window_ms", "n_units", "n_spikes",
                                                                 "n_steps_recorded", "thinned")}},
        "V2": {a: {"source_run": by[(a, 1)]["tag"], **{k: v2_parts[a][k] for k in ("window_ms", "n_units",
                                                                                     "n_spikes", "thinned")}}
               for a in arms},
        "runs": [{k: r[k] for k in ("tag", "stage", "exec", "npz_sha256", "json_sha256")} for r in allr],
        "jaxfne_version": __import__("jaxfne").__version__,
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"V0": v0["n_projections"], "V1": v1["n_spikes"], "runs": len(allr)}))


if __name__ == "__main__":
    main()

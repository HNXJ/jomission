"""VISUALIZATION_CONTRACT renderers: the V0 schematic and the V1/V2 rasters.

The contract is manifests/visualization_contract.json; this module is what satisfies it.
Everything here reads a realized model and its recorded signals. No science is computed and
no model value is modified.

  V0  hspice_schematic   construct -> block schematic, before simulation
  V1  raster             the native regime, first 1000 ms, before interpretation
  V2  raster             the same figure after the authorized work, same ordering and window
  V3  the atlas, built by jomission.visualization.jaxfne_suite_atlas

V1 and V2 share one ordering function on purpose: a before/after comparison is only readable
when it is the same figure twice.
"""

from __future__ import annotations

import collections
import pathlib
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Drawing conventions, shared by every stage so figures from different lineages compare.
CLASS_COLOR = {"E": "#d1495b", "PV": "#2a9d8f", "SST": "#457b9d", "VIP": "#e9c46a"}
CHANNEL_COLOR = {"ff": "#2f6fb0", "fb": "#b5651d", "lat": "#6a5acd", "in": "#4a4a4a"}
CHANNEL_STYLE = {"ff": "-", "fb": "--", "lat": "-", "in": "-"}
LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6")
CLASSES = ("E", "PV", "SST", "VIP")
EXCITATORY_RECEPTOR = 0

# Which layer pair realizes which TFNE channel. Mirrors jomission.tfne.o_authority.RELATION.
RELATION = {("L2", "L4"): "ff", ("L3", "L4"): "ff", ("L6", "L1"): "fb", ("L3", "L3"): "lat"}


def stage_paths(lineage_id: str, contract: dict) -> dict[str, str]:
    """The four contract paths for one lineage, under the manifest's path convention."""
    root = contract["path_convention"].replace("<lineage_id>", lineage_id)
    return {s: root.replace("<artifact>", contract["stages"][s]["artifact"])
            for s in contract["order"]}


def _owner_map(index: dict[str, tuple], n_total: int) -> np.ndarray:
    owner = np.empty(n_total, dtype=object)
    for group, (a, b, _n) in index.items():
        owner[a:b] = group
    return owner


def describe(model: Any, index: dict[str, tuple]) -> dict:
    """Structure of the realized circuit: populations, and projections aggregated to blocks.

    Returns counts per (area, layer, cell class) and one entry per
    (source area, target area, source group, target group) with edge count, mean weight and
    the receptor that sets the projection's sign. No individual neuron appears.
    """
    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    weight = np.asarray(edges.weight)
    receptor = np.asarray(getattr(edges, "receptor_index", np.zeros_like(pre)))
    n_total = int(max(pre.max(), post.max())) + 1
    owner = _owner_map(index, n_total)

    pops: dict[str, dict[str, dict[str, int]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    for group, (_a, _b, n) in index.items():
        parts = group.split(".")
        if len(parts) == 3:
            pops[parts[0]][parts[1]][parts[2]] = int(n)
        else:
            pops[parts[0]].setdefault("-", {})[parts[-1]] = int(n)

    src = np.array([owner[i] if owner[i] is not None else "?" for i in pre], dtype=object)
    dst = np.array([owner[i] if owner[i] is not None else "?" for i in post], dtype=object)
    n_edge: collections.Counter = collections.Counter()
    w_sum: collections.Counter = collections.Counter()
    for s, d, w, r in zip(src, dst, weight, receptor):
        sa, da = str(s).split(".")[0], str(d).split(".")[0]
        if sa == da:
            continue
        n_edge[(sa, da, str(s), str(d), int(r))] += 1
        w_sum[(sa, da, str(s), str(d), int(r))] += float(w)

    projections = []
    for key, n in sorted(n_edge.items()):
        sa, da, s, d, r = key
        sl, dl = s.split(".")[1], d.split(".")[1]
        projections.append({
            "source_area": sa, "target_area": da, "source": s, "target": d,
            "channel": RELATION.get((sl, dl), "in" if sa != da else "local"),
            "n_edges": int(n), "mean_weight": round(w_sum[key] / n, 6), "receptor": r,
            "sign": "excitatory" if r == EXCITATORY_RECEPTOR else "inhibitory",
        })
    local = int(sum(1 for s, d in zip(src, dst)
                    if str(s).split(".")[0] == str(d).split(".")[0]))
    return {"populations": {a: dict(v) for a, v in pops.items()},
            "projections": projections,
            "n_edges_total": int(pre.size),
            "n_edges_local": local,
            "n_edges_long_range": int(pre.size) - local}


def _draw_area(ax, x, y, w, h, area, layers, *, fontsize=7):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="#f7f7f9", edgecolor="#333333", linewidth=1.4, zorder=2))
    ax.text(x + w / 2, y + h + 0.012, area, ha="center", va="bottom",
            fontsize=fontsize + 3, fontweight="bold", zorder=3)
    present = [L for L in LAYERS if L in layers]
    if not present:
        return {}
    row_h = h / len(present)
    rows = {}
    for i, L in enumerate(present):
        ry = y + h - (i + 1) * row_h
        rows[L] = ry + row_h / 2
        ax.add_patch(mpatches.Rectangle((x, ry), w, row_h, facecolor="none",
                                        edgecolor="#c9c9cf", linewidth=0.6, zorder=3))
        ax.text(x + 0.006, ry + row_h / 2, L, ha="left", va="center",
                fontsize=fontsize, color="#555555", zorder=4)
        chips = [(c, layers[L][c]) for c in CLASSES if c in layers[L]]
        cw = (w - 0.030) / max(len(chips), 1)
        for j, (cls, n) in enumerate(chips):
            cx = x + 0.026 + j * cw
            ax.add_patch(mpatches.Rectangle((cx, ry + row_h * 0.18), cw * 0.88, row_h * 0.64,
                                            facecolor=CLASS_COLOR[cls], alpha=0.85,
                                            edgecolor="none", zorder=4))
            ax.text(cx + cw * 0.44, ry + row_h / 2, f"{cls} {n}", ha="center", va="center",
                    fontsize=fontsize - 0.5, color="white", fontweight="bold", zorder=5)
    return rows


def _arrow(ax, p0, p1, channel, label, *, rad=0.0, lw=1.6, fontsize=6.5, label_pos=0.5):
    """An excitatory projection ends in an arrowhead; an inhibitory one ends in a bar."""
    style = "-|>" if channel != "inhibitory" else "-["
    ax.annotate("", xy=p1, xytext=p0, zorder=6, annotation_clip=False,
                arrowprops=dict(arrowstyle=style, color=CHANNEL_COLOR.get(channel, "#444"),
                                linestyle=CHANNEL_STYLE.get(channel, "-"), linewidth=lw,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    mx = p0[0] + (p1[0] - p0[0]) * label_pos
    my = p0[1] + (p1[1] - p0[1]) * label_pos + rad * 0.35
    ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
            color=CHANNEL_COLOR.get(channel, "#444"), zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.88,
                      edgecolor="none"))


def hspice_schematic(model: Any, index: dict[str, tuple], out_path: str | pathlib.Path, *,
                     title: str, subtitle: str = "", stages: list[list[str]] | None = None,
                     x_label: str = "x", y_label: str = "y",
                     authority: dict | None = None, dpi: int = 150) -> dict:
    """V0: the construction receipt. Blocks and projections only, never individual neurons.

    ``stages`` is the left-to-right column order, each column a list of area names.
    ``authority`` optionally maps channel -> g so the figure states the authority it was
    realized under alongside the realized w.
    """
    d = describe(model, index)
    pops = d["populations"]
    cortical = [a for a in pops if any(L in pops[a] for L in LAYERS)]
    if stages is None:
        order = ["V1_1", "V1_2", "V4_1", "V4_2", "FEF", "PFC"]
        known = [a for a in order if a in cortical]
        stages = [[a for a in known if a.startswith("V1")],
                  [a for a in known if a.startswith("V4")],
                  [a for a in known if a in ("FEF", "PFC")]]
        stages = [s for s in stages if s]
    placed = {a for col in stages for a in col}
    # External means "not placed in a column", not "has no layers": an input interface may well
    # carry a layer name. Anything unplaced is drawn as drive into the first column.
    external = [a for a in pops if a not in placed]

    fig, ax = plt.subplots(figsize=(19.5, 10.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_col = len(stages)
    left, right = 0.085, 0.915
    col_w = (right - left) / n_col
    box_w = col_w * 0.62
    rows: dict[str, dict[str, float]] = {}
    cx_of: dict[str, float] = {}
    span_of: dict[str, tuple[float, float]] = {}
    for ci, col in enumerate(stages):
        cx = left + ci * col_w + (col_w - box_w) / 2
        n = len(col)
        box_h = 0.70 / n - 0.055
        for ri, area in enumerate(col):
            by = 0.20 + (n - 1 - ri) * (box_h + 0.085)
            rows[area] = _draw_area(ax, cx, by, box_w, box_h, area, pops[area])
            cx_of[area] = cx + box_w / 2
            span_of[area] = (cx, cx + box_w)
        ax.text(cx + box_w / 2, 0.945, f"object {ci + 1}", ha="center", va="center",
                fontsize=9, color="#777777", style="italic")

    # x on the far left, y on the far right.
    ax.add_patch(mpatches.FancyBboxPatch((0.006, 0.47), 0.066, 0.10,
                                         boxstyle="round,pad=0.006,rounding_size=0.01",
                                         facecolor="#eef3fa", edgecolor="#2f6fb0",
                                         linewidth=1.6, zorder=2))
    ax.text(0.039, 0.545, "x", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.039, 0.505, x_label, ha="center", va="center", fontsize=7.5, color="#2f6fb0")
    ax.add_patch(mpatches.FancyBboxPatch((0.928, 0.47), 0.066, 0.10,
                                         boxstyle="round,pad=0.006,rounding_size=0.01",
                                         facecolor="#fdf1e7", edgecolor="#b5651d",
                                         linewidth=1.6, zorder=2))
    ax.text(0.961, 0.545, "y", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.961, 0.505, y_label, ha="center", va="center", fontsize=7.5, color="#b5651d")

    # External drive into the first column.
    for area in stages[0]:
        ext = [p for p in d["projections"] if p["source_area"] in external
               and p["target_area"] == area]
        if not ext:
            continue
        n_e = sum(p["n_edges"] for p in ext)
        w_e = sum(p["mean_weight"] * p["n_edges"] for p in ext) / n_e
        tgt = ext[0]["target"].split(".")
        yv = rows[area].get(tgt[1], 0.5)
        _arrow(ax, (0.074, 0.52), (span_of[area][0], yv), "ff",
               f"{ext[0]['source_area']}\n{tgt[1]}.{tgt[2]}  n={n_e}  w={w_e:.3f}",
               rad=0.10, lw=2.0, label_pos=0.52)

    def label_for(p):
        g = (authority or {}).get(p["channel"])
        gs = f"  g={g}" if g is not None else ""
        return (f"{p['channel'].upper()}  {p['source'].split('.', 1)[1]}"
                f"→{p['target'].split('.', 1)[1]}\n"
                f"n={p['n_edges']}  w={p['mean_weight']:.3f}{gs}")

    drawn: set = set()
    # Several projections share one pair of blocks (ff carries two layer identities, and fb
    # runs back along the same gap). Stagger their labels along the arrow so none is buried.
    slot: collections.Counter = collections.Counter()
    FF_POS = (0.17, 0.29, 0.41, 0.53)
    FB_POS = (0.64, 0.74, 0.84, 0.94)
    for p in d["projections"]:
        sa, da, ch = p["source_area"], p["target_area"], p["channel"]
        if sa in external or da in external:
            continue
        col_s = next(i for i, c in enumerate(stages) if sa in c)
        col_d = next(i for i, c in enumerate(stages) if da in c)
        sign = "inhibitory" if p["sign"] == "inhibitory" else ch
        if ch == "lat":
            # X: lateral/cross, inside one object. Drawn once per unordered pair.
            key = ("lat", tuple(sorted((sa, da))))
            if key in drawn:
                continue
            drawn.add(key)
            ys, yd = rows[sa].get("L3", 0.5), rows[da].get("L3", 0.5)
            if col_s == col_d:
                ax.annotate("", xy=(cx_of[da], yd), xytext=(cx_of[sa], ys), zorder=6,
                            arrowprops=dict(arrowstyle="<|-|>", color=CHANNEL_COLOR["lat"],
                                            linewidth=2.0, shrinkA=3, shrinkB=3))
                ax.text(cx_of[sa] + 0.004, (ys + yd) / 2, label_for(p), ha="left", va="center",
                        fontsize=6.5, color=CHANNEL_COLOR["lat"], zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.9,
                                  edgecolor="none"))
            continue
        # O: between objects. FF runs left to right above, FB right to left below.
        ys = rows[sa].get(p["source"].split(".")[1], 0.5)
        yd = rows[da].get(p["target"].split(".")[1], 0.5)
        gap = (min(col_s, col_d), max(col_s, col_d), sa[-1] == da[-1])
        k = slot[(gap, ch)]
        slot[(gap, ch)] += 1
        pool = FF_POS if col_d > col_s else FB_POS
        if col_d > col_s:
            _arrow(ax, (span_of[sa][1], ys), (span_of[da][0], yd), sign, label_for(p),
                   rad=0.16 if sa[-1] != da[-1] else 0.04, label_pos=pool[k % len(pool)])
        else:
            _arrow(ax, (span_of[sa][0], ys), (span_of[da][1], yd), sign, label_for(p),
                   rad=0.18 if sa[-1] != da[-1] else 0.06, label_pos=pool[k % len(pool)])

    # Readout.
    last = stages[-1][0]
    ax.annotate("", xy=(0.926, 0.52), xytext=(span_of[last][1], rows[last].get("L6", 0.4)),
                zorder=6, arrowprops=dict(arrowstyle="-|>", color="#b5651d", linewidth=2.0,
                                          connectionstyle="arc3,rad=-0.12"))

    handles = [mpatches.Patch(color=CLASS_COLOR[c], label=f"{c} population") for c in CLASSES]
    handles += [
        plt.Line2D([], [], color=CHANNEL_COLOR["ff"], lw=2, ls="-",
                   label="O feedforward (FF), excitatory"),
        plt.Line2D([], [], color=CHANNEL_COLOR["fb"], lw=2, ls="--",
                   label="O feedback (FB), excitatory"),
        plt.Line2D([], [], color=CHANNEL_COLOR["lat"], lw=2, ls="-",
                   label="X lateral / cross, excitatory"),
        plt.Line2D([], [], color="#4a4a4a", lw=2, ls="-",
                   label="inhibitory projection (bar end)"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
              bbox_to_anchor=(0.5, -0.015))
    ax.text(0.5, 0.995, title, ha="center", va="top", fontsize=15, fontweight="bold")
    foot = (f"{d['n_edges_total']} edges: {d['n_edges_local']} local, "
            f"{d['n_edges_long_range']} long-range across "
            f"{len(d['projections'])} block projections. Blocks and projections only; "
            f"no individual neuron is drawn.")
    ax.text(0.5, 0.968, subtitle or foot, ha="center", va="top", fontsize=8.5, color="#555555")
    if subtitle:
        ax.text(0.5, 0.020, foot, ha="center", va="bottom", fontsize=8, color="#777777")

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(out), **{k: d[k] for k in
                                 ("n_edges_total", "n_edges_local", "n_edges_long_range")},
            "n_projections": len(d["projections"])}


def unit_order(index: dict[str, tuple], *, areas: list[str] | None = None) -> tuple:
    """Rows ordered area -> layer -> cell type. V1 and V2 both call this, unchanged."""
    groups = []
    for group, (a, b, n) in index.items():
        parts = group.split(".")
        area = parts[0]
        layer = parts[1] if len(parts) == 3 else "-"
        cls = parts[-1]
        groups.append((area, layer, cls, int(a), int(b), int(n), group))
    if areas is not None:
        rank = {x: i for i, x in enumerate(areas)}
        groups = [g for g in groups if g[0] in rank]
        groups.sort(key=lambda g: (rank[g[0]], LAYERS.index(g[1]) if g[1] in LAYERS else 99,
                                   CLASSES.index(g[2]) if g[2] in CLASSES else 99))
    else:
        groups.sort(key=lambda g: (g[0], LAYERS.index(g[1]) if g[1] in LAYERS else 99,
                                   CLASSES.index(g[2]) if g[2] in CLASSES else 99))
    rows = np.concatenate([np.arange(g[3], g[4]) for g in groups]) if groups else np.array([])
    return rows, groups


def raster(spikes: np.ndarray, index: dict[str, tuple], out_path: str | pathlib.Path, *,
           title: str, dt_ms: float = 1.0, t0_ms: float = 0.0, window_ms: float = 1000.0,
           areas: list[str] | None = None, subtitle: str = "", dpi: int = 150,
           max_points: int = 400_000) -> dict:
    """V1 and V2: a spike raster ordered area -> layer -> cell type.

    ``spikes`` is (n_time, n_neurons), the same layout the drivers record.
    """
    rows, groups = unit_order(index, areas=areas)
    a = int(round(t0_ms / dt_ms))
    b = int(round((t0_ms + window_ms) / dt_ms))
    if b > spikes.shape[0] or a < 0 or b <= a:
        raise ValueError(f"window {t0_ms}-{t0_ms + window_ms} ms outside recorded "
                         f"{spikes.shape[0] * dt_ms} ms at dt {dt_ms} ms")
    block = np.asarray(spikes[a:b][:, rows])
    t_idx, u_idx = np.nonzero(block)
    n_spikes = int(t_idx.size)
    if n_spikes > max_points:  # thin for file size, and say so rather than silently dropping
        keep = np.linspace(0, n_spikes - 1, max_points).astype(int)
        t_idx, u_idx = t_idx[keep], u_idx[keep]
    t_ms = t0_ms + t_idx * dt_ms

    pos = np.empty(rows.size, dtype=object)
    at = 0
    boundaries = []
    for area, layer, cls, ga, gb, n, _g in groups:
        pos[at:at + n] = cls
        boundaries.append((at, at + n, area, layer, cls))
        at += n
    colors = np.array([CLASS_COLOR.get(c, "#999999") for c in pos], dtype=object)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(16, 10), height_ratios=[4.2, 1.0], sharex=True,
        gridspec_kw={"hspace": 0.06})
    ax.scatter(t_ms, u_idx, s=0.45, c=list(colors[u_idx]), marker="|", linewidths=0.45)
    seen_area = None
    for a0, b0, area, layer, cls in boundaries:
        if area != seen_area:
            ax.axhline(a0, color="#222222", linewidth=1.0, alpha=0.8)
            ax.text(t0_ms - window_ms * 0.012, (a0 + b0) / 2, area, ha="right", va="center",
                    fontsize=9, fontweight="bold")
            seen_area = area
        elif cls == "E":
            ax.axhline(a0, color="#bbbbbb", linewidth=0.5, alpha=0.7)
    ax.set_ylim(0, rows.size)
    ax.set_yticks([])  # rows are units; the area labels on the left are the ordinate
    ax.set_ylabel("unit  (area → layer → cell type)", labelpad=34)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=22)
    if subtitle:
        ax.text(0.0, 1.008, subtitle, transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.legend(handles=[mpatches.Patch(color=CLASS_COLOR[c], label=c) for c in CLASSES],
              loc="upper right", ncol=4, frameon=False, fontsize=8)

    rate = block.sum(axis=1) / (rows.size * dt_ms * 1e-3)
    axr.plot(t0_ms + np.arange(block.shape[0]) * dt_ms, rate, color="#333333", linewidth=0.8)
    axr.set_xlabel("time (ms)")
    axr.set_ylabel("population\nrate (Hz)")
    axr.set_xlim(t0_ms, t0_ms + window_ms)
    axr.margins(x=0)

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(out), "n_units": int(rows.size), "n_groups": len(groups),
            "window_ms": [t0_ms, t0_ms + window_ms], "dt_ms": dt_ms,
            "n_spikes": n_spikes, "thinned": n_spikes > max_points,
            "mean_rate_hz": round(float(rate.mean()), 4)}

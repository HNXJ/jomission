"""Deck figures drawn from the manuscript's records (no simulation, no new fit).

Usage: python paper/jomission_lansa/deck/make_figs.py   (cwd = repo root)

Every plotted value is read from a record of Table S6 (sections/supplement.tex); profiles of the readout optima are
rebuilt from the power shares and coherences stored in D3/D5 and checked against the stored r and mid (assert
within TOL; the check fails if a sign or share is wrong). Output: deck/figs/*.pdf at slide size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_readout_capacity as rc  # noqa: E402
import e2_scorer as sc  # noqa: E402

DATA = ROOT / "paper/jomission_lansa/data"
E2 = ROOT / "results/laminar_e2"
OUT = Path(__file__).resolve().parent / "figs"
TOL = 2e-3
COL = {"S": "#1565c0", "M": "#ff9800", "D": "#00acc1", "obs": "black", "gold": "#cfb87c", "red": "#e53935"}
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
                     "lines.linewidth": 1.5})
W, H = 15.0 / 2.54, 6.2 / 2.54  # slide figure box, inches
FTICKS, ZTICKS = [10, 50, 100, 150], [-0.4, 0.0, 0.4]  # one tick set for every map and depth axis of a map
Z = rc.mf.CHANNELS


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def depth_axis(ax) -> None:
    ax.set_ylim(0.42, -0.62)
    ax.set_ylabel("depth (fraction of thickness, 0 = mid-L4)")
    ax.axhspan(-0.2, 0.0, color=COL["gold"], alpha=0.25, lw=0)


# ---- D1: mid-depth relative gamma, 12 runs vs observed ----
def fig_mid() -> None:
    d = load(DATA / "mid_gamma.json")
    keys = ["S [-0.6,-0.3]", "mid [-0.2,0]", "D [0.15,0.4]"]
    labels = ["S window\n-0.6 to -0.3", "mid window\n-0.2 to 0", "D window\n0.15 to 0.4"]
    runs = d["runs"]
    assert len(runs) == 12, len(runs)
    fig, ax = plt.subplots(figsize=(W * 0.6, H), layout="constrained")
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        v = np.array([runs[t][k] for t in runs])
        ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=14, color=COL["S"], zorder=3,
                   label="model, 12 runs" if i == 0 else None)
        ax.plot([i - 0.3, i + 0.3], [d["observed"][k]] * 2, color=COL["obs"], lw=2,
                label="observed, 11-area mean" if i == 0 else None)
    ax.set_xticks(range(3), labels)
    ax.set_xlabel("depth window (fraction of thickness, 0 = mid-L4)")
    ax.set_ylabel("relative 50-150 Hz power")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    save(fig, "mid_gamma")


# ---- R5-R8: every scored circuit run against the threshold and readout ceilings ----
def fig_value() -> None:
    groups = {"size ladder": ("lfs_n*_verdict.json", "sw_n4000_dens_k1_kd1_sst15_s1_T8_verdict.json"),
              "search": ("t1_[ks]*_n4000_s*_T8_verdict.json",),
              "confirmation": ("t1_confirm_*_verdict.json",),
              "weight runs, exploratory": ("e2x_*_verdict.json",)}
    expect = {"size ladder": 9, "search": 63, "confirmation": 3, "weight runs, exploratory": 12}
    style = {"size ladder": ("o", COL["S"]), "search": (".", "#9e9e9e"),
             "confirmation": ("D", COL["red"]), "weight runs, exploratory": ("s", COL["M"])}
    ceil = load(E2 / "readout_ceiling.json")["bands"]["image_gm"]
    fig, ax = plt.subplots(figsize=(W * 0.62, H), layout="constrained")
    n_all, n_pass = 0, 0
    for g, pats in groups.items():
        files = sorted({p for pat in pats for p in E2.glob(pat)})
        v = [load(p) for p in files]
        assert len(v) == expect[g], (g, len(v))
        n_all += len(v)
        n_pass += sum(bool(x["pass"]) for x in v)
        mk, c = style[g]
        ax.scatter([x["r_ab"] for x in v], [x["r_gm"] for x in v], marker=mk, color=c, s=14 if mk != "." else 30,
                   label=f"{g} ({len(v)})", zorder=4 if g == "size ladder" else 3)
    ax.axhspan(0.6, 1.0, xmin=(0.6 + 1) / 2, xmax=1, color=COL["gold"], alpha=0.35, lw=0)
    ax.text(0.8, 0.97, "both \u2265 0.6", ha="center", va="top", fontsize=6.5)
    ax.axhline(ceil["coherent"], color="black", lw=0.8, ls="--")
    ax.axhline(ceil["incoherent"], color="black", lw=0.8, ls=":")
    ax.text(-0.98, ceil["coherent"] + 0.015, f"readout ceiling, coherent {ceil['coherent']:.3f}", fontsize=7)
    ax.text(-0.98, ceil["incoherent"] + 0.015, f"incoherent {ceil['incoherent']:.3f}", fontsize=7)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 1)
    ax.set_xlabel("$r_{ab}$ (alpha-beta depth correlation)")
    ax.set_ylabel("$r_{gm}$ (gamma depth correlation)")
    ax.legend(loc="lower left", fontsize=6.5)
    save(fig, "value_scatter")
    print(f"value_scatter: {n_all} verdicts, {n_pass} pass")


# ---- readout capacity, step by step (D3, D5, D7) ----
def target():
    y = sc.PROTO[:, rc.GM].mean(axis=1)
    return y / y.max()


def profile(share, coh_signs, s=0.15):
    """Rank-one (|coh| = 1) or diagonal C from stored shares; coh_signs None = diagonal."""
    K = rc.kern(rc.Z3, s)
    a = np.asarray(share, float)
    if coh_signs is None:
        P = (K ** 2) @ a
    else:
        v = np.sqrt(a) * np.asarray(coh_signs, float)
        P = (K @ v) ** 2
    return P / P.max()


def fig_kernel(y) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.7, H), layout="constrained", sharey=True)
    axs[0].plot(y, Z, color=COL["obs"], marker="o", ms=3)
    axs[0].set_xlabel("observed relative 50-150 Hz power")
    axs[0].set_title("target y: 11-area band-mean profile", fontsize=8)
    depth_axis(axs[0])
    K = rc.kern(rc.Z3, 0.15)
    zz = np.linspace(-0.6, 0.4, 201)
    for i, u in enumerate("SMD"):
        axs[1].plot(np.exp(-((zz - rc.Z3[i]) ** 2) / 0.15 ** 2), zz, color=COL[u], label=u)
    assert K.shape == (21, 3)
    axs[1].set_xlabel("kernel weight $K^2$ ($\\sigma$ = 0.15)")
    axs[1].set_title("readout: three Gaussian sources", fontsize=8)
    for i, u in enumerate("SMD"):
        axs[1].text(1.03, rc.Z3[i], u, color=COL[u], va="center", fontsize=8, fontweight="bold")
    axs[1].set_xlim(0, 1.12)
    axs[1].axhspan(-0.2, 0.0, color=COL["gold"], alpha=0.25, lw=0)
    save(fig, "step1_target_kernel")


def fig_steps(y, d3) -> None:
    s15 = d3["sigma"]["0.15"]
    single = profile([0, 1, 0], None)
    assert abs(rc.mid(single) - s15["single_M"]) < 1e-9
    auto = profile(s15["maxr_auto"]["share"], None)
    full_signs = [1, np.sign(s15["maxr_full"]["coh_SM"]), np.sign(s15["maxr_full"]["coh_SD"])]
    full = profile(s15["maxr_full"]["share"], full_signs)
    for P, rec in ((auto, s15["maxr_auto"]), (full, s15["maxr_full"])):
        assert abs(rc.corr(P, y) - rec["r"]) < TOL and abs(rc.mid(P) - rec["mid"]) < TOL, (rc.corr(P, y), rec)
    curves = [("single source at M", single, COL["M"], f"mid {s15['single_M']:.3f}"),
              ("independent S, M, D (best r)", auto, COL["S"],
               f"r {s15['maxr_auto']['r']:.3f}, mid {s15['maxr_auto']['mid']:.2f}"),
              ("coherent S, M, D (best r)", full, COL["red"],
               f"r {s15['maxr_full']['r']:.3f}, mid {s15['maxr_full']['mid']:.2f}")]
    for k in (1, 2, 3):
        fig, axs = plt.subplots(1, 2, figsize=(W * 0.85, H), layout="constrained",
                                gridspec_kw={"width_ratios": [1.1, 1]})
        ax = axs[0]
        ax.plot(y, Z, color=COL["obs"], marker="o", ms=3, label="observed y")
        for i, (lab, P, c, _) in enumerate(curves[:k]):
            ax.plot(P, Z, color=c, lw=1.8 if i == k - 1 else 1.2, label=lab)
        depth_axis(ax)
        ax.set_xlabel("relative 50-150 Hz power")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.5)
        b = axs[1]
        names = ["observed\n(map)", "single\nsource M", "independent\nS, M, D", "coherent\nS, M, D"][:k + 1]
        vals = [d3["map_mid"]] + [rc.mid(c[1]) for c in curves[:k]]
        cols = [COL["obs"]] + [c[2] for c in curves[:k]]
        b.barh(range(len(vals)), vals, color=cols, height=0.6)
        for i, v in enumerate(vals):
            b.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
        b.set_yticks(range(len(vals)), names, fontsize=6.5)
        b.invert_yaxis()
        b.set_xlim(0, 1.15)
        b.set_xticks([0, 0.5, 1])
        b.set_xlabel("mid-depth relative gamma")
        save(fig, f"step{k + 1}_profiles")


def fig_joint(y, d3, d5, d7) -> None:
    s15, j = d3["sigma"]["0.15"], d5["joint"]["0.15"]
    jf = j["map_mid_full"]
    P = profile(jf["share"], [1, np.sign(jf["coh_SM"]), np.sign(jf["coh_SD"])])
    assert abs(rc.corr(P, y) - jf["r"]) < TOL and abs(rc.mid(P) - jf["mid"]) < TOL, (rc.corr(P, y), jf)
    assert jf["feasible"] and not j["map_mid_auto"]["feasible"]
    fig, axs = plt.subplots(1, 2, figsize=(W, H), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    t, eps = d5["targets"]["map_mid"], d5["eps"]
    ax.axvspan(t - eps, t + eps, color=COL["gold"], alpha=0.45, lw=0)
    ax.text(t, 0.42, f"target mid\n{t:.3f} \u00b1 {eps}", ha="center", fontsize=6.5)
    amax = d7["sigma"]["0.15"]["auto_mid_max"]
    ax.axvspan(amax, 1.0, color="#e0e0e0", alpha=0.6, lw=0, zorder=0)
    ax.axvline(amax, color=COL["S"], lw=0.8, ls=":")
    ax.text(amax + 0.005, 0.95, f"grey: mid > {amax:.3f},\nbeyond independent sources", fontsize=6.5, color=COL["S"],
            va="top")
    pts = [("independent, best r", s15["maxr_auto"], COL["S"], "o"),
           ("coherent, best r", s15["maxr_full"], COL["red"], "o"),
           ("coherent, mid on target", jf, COL["red"], "*")]
    for lab, rec, c, mk in pts:
        ax.scatter(rec["mid"], rec["r"], color=c, marker=mk, s=60 if mk == "*" else 25, zorder=4, label=lab)
        ax.annotate(f"r {rec['r']:.3f}", (rec["mid"], rec["r"]), (4, -9), textcoords="offset points", fontsize=6.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("mid-depth relative gamma of the profile")
    ax.set_ylabel("r with observed profile y")
    ax.legend(loc="lower left", fontsize=6.5)
    b = axs[1]
    b.plot(y, Z, color=COL["obs"], marker="o", ms=3, label="observed y")
    b.plot(P, Z, color=COL["red"], label="joint optimum")
    depth_axis(b)
    b.set_ylabel("depth (fraction of thickness, 0 = mid-L4)")
    b.set_xlabel("relative 50-150 Hz power")
    sh = jf["share"]
    b.set_title(f"shares S {sh[0]:.2f}, M {sh[1]:.2f}, D {sh[2]:.2f}; S\u2013M +1, D \u22121", fontsize=7)
    b.legend(loc="lower right", fontsize=6.5)
    save(fig, "step5_joint")


def fig_sigma(d3, d4) -> None:
    sig = ["0.10", "0.15", "0.20", "0.25"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(W * 0.75, H), layout="constrained")
    for off, kind, c in ((-0.12, "full", COL["red"]), (0.12, "auto", COL["S"])):
        ins = [d3["sigma"][s][f"maxr_{kind}"]["r"] for s in sig]
        ho = [d4["sigma"][s][f"{kind}_test_r"]["mean"] for s in sig]
        sd = [d4["sigma"][s][f"{kind}_test_r"]["sd"] for s in sig]
        lab = "coherent" if kind == "full" else "independent"
        ax.scatter(x + off, ins, marker="_", s=120, color=c, lw=2, label=f"{lab}, in-sample max r (11 areas)")
        ax.errorbar(x + off, ho, yerr=sd, fmt="o", ms=4, color=c, mfc="white", capsize=2, lw=1,
                    label=f"{lab}, held-out area (mean \u00b1 SD, n 11 folds)")
    tm = [d4["sigma"][s]["template_r"]["mean"] for s in sig]
    ax.plot(x, tm, color="#9e9e9e", lw=1, ls="--", label="10-area mean profile vs held-out")
    ax.axhline(0.6, color="black", lw=0.6)
    ax.text(3.45, 0.61, "0.6", fontsize=7)
    ax.set_xticks(x, [f"{s}{' (declared)' if s == '0.15' else ''}" for s in sig])
    ax.set_xlabel("readout kernel width $\\sigma$ (fraction of thickness)")
    ax.set_ylabel("Pearson r with gamma profile")
    ax.set_ylim(0.2, 1.05)
    ax.set_xlim(-0.5, 3.8)
    ax.legend(loc="lower right", fontsize=6)
    save(fig, "step6_sigma_loao")


def fig_circuit(d5) -> None:
    c = d5["circuit"]
    rows = [(f"ref s{i + 1}", v) for i, v in enumerate(c["reference"].values())]
    rows += [(f"selected s{i + 4}", v) for i, v in enumerate(c["t1_winner_fresh"].values())]
    opt = d5["joint"]["0.15"]["map_mid_full"]
    rows.append(("optimum", {"share": opt["share"], "recoh_SM": opt["coh_SM"], "recoh_SD": opt["coh_SD"],
                             "recoh_MD": opt["coh_MD"]}))
    fig, axs = plt.subplots(1, 2, figsize=(W, H), layout="constrained")
    x = np.arange(len(rows))
    bot = np.zeros(len(rows))
    for i, u in enumerate("SMD"):
        v = np.array([r[1]["share"][i] for r in rows])
        axs[0].bar(x, v, bottom=bot, color=COL[u], label=u, width=0.7)
        bot += v
    axs[0].set_xticks(x, [r[0] for r in rows], rotation=45, ha="right", fontsize=6.5)
    axs[0].set_ylabel("share of 50-150 Hz source power")
    axs[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for k, (pair, mk) in enumerate((("SM", "o"), ("SD", "s"), ("MD", "^"))):
        v = [r[1][f"recoh_{pair}"] for r in rows]
        axs[1].scatter(x + (k - 1) * 0.24, v, marker=mk, s=18, color=["#424242", COL["S"], COL["D"]][k],
                       label=f"{pair[0]}-{pair[1]}")
    axs[1].axhline(0, color="black", lw=0.5)
    axs[1].set_ylim(-1.1, 1.1)
    axs[1].set_xticks(x, [r[0] for r in rows], rotation=45, ha="right", fontsize=6.5)
    axs[1].set_ylabel("Re coherence, band mean")
    axs[1].legend(loc="lower left", fontsize=6.5)
    save(fig, "step7_circuit_vs_opt")


CROPS = {  # name: (source figure, (left, top, right, bottom) px, white-out boxes in crop px)
    # column: title row (run stem) dropped; ref_dynamics: panel letters and the "RWS, z" label whited out
    "ref_dynamics": ("results/viz/paper_w3/sw_n4000_dens_k1_kd1_sst15_s1_T8_block.png", (144, 36, 1460, 885),
                     [(70, 10, 120, 60), (700, 10, 750, 60), (668, 300, 702, 480), (240, 812, 570, 849)]),
}
LEFT_LABEL = {"ref_dynamics": "sampled cells by unit (depth from mid-L4)"}
MID_LABEL = {"ref_dynamics": (704, "LFP proxy (z)", 0.40)}  # x in crop px, y in figure fraction (between M and D)
BOTTOM_LABEL = {"ref_dynamics": (403, 832, "Time (s)")}  # replaces "Time (s); sampled cells"; crop px


def crops() -> None:
    from PIL import Image, ImageDraw
    for name, (src, box, blank) in CROPS.items():
        im = Image.open(ROOT / src).convert("RGB")
        assert box[2] <= im.width and box[3] <= im.height, (name, im.size)
        im = im.crop(box)
        d = ImageDraw.Draw(im)
        for b in blank:
            d.rectangle(b, fill="white")
        w, h = im.size
        pad = 90 if name in LEFT_LABEL else 0
        fig = plt.figure(figsize=((w + pad) / 300, h / 300), dpi=300)
        ax = fig.add_axes([pad / (w + pad), 0, w / (w + pad), 1])
        ax.imshow(im)
        ax.axis("off")
        if name in LEFT_LABEL:
            fig.text(35 / (w + pad), 0.55, LEFT_LABEL[name], rotation=90, ha="center", va="center", fontsize=7)
        if name in MID_LABEL:
            x, s, yf = MID_LABEL[name]
            fig.text((x + pad) / (w + pad), yf, s, rotation=90, ha="center", va="center", fontsize=6)
        if name in BOTTOM_LABEL:
            x, y, s = BOTTOM_LABEL[name]
            fig.text((x + pad) / (w + pad), 1 - y / h, s, ha="center", va="center", fontsize=7)
        fig.savefig(OUT / f"{name}.png", dpi=300)
        plt.close(fig)


def fig_density() -> None:
    """Page 2: LANSA densities; page 3: built cell counts per depth bin against LANSA, one panel per type."""
    import e2_circuit_schema as cs_fig
    z, types, Xm, _, _ = cs_fig.lansa()
    fig, ax = plt.subplots(figsize=(W * 0.4, H), layout="constrained")
    for i, t in enumerate(types):
        lab, col, ls = cs_fig.LANSA_STYLE[t]
        ax.plot(Xm[:, i] / Xm[:, i].max(), z, color=col, ls=ls, lw=1.5, label=lab)
    ax.set_ylim(0.43, -0.63)
    ax.set_yticks(ZTICKS)
    ax.set_xlabel("density / max (median of 11 areas)")
    ax.set_ylabel("depth (fraction of thickness, 0 = mid-L4)")
    ax.set_title("cell densities (LANSA)", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6.5, handlelength=1.6,
              columnspacing=0.8)
    save(fig, "densities")
    rec = load(E2 / f"{REF_TAGS[0]}.json")
    bp = cs_fig.built_profile(rec, z)
    assert sum(sum(u.values()) for u in rec["sizes"].values()) == 14251
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(1, 3, figsize=(W * 0.46, H), layout="constrained", sharey=True)
    for ax, (lt, t) in zip(axs, cs_fig.LANSA_TO_MODEL.items()):
        col = cs_fig.TYPE_COLOR[t]
        ax.barh(z, bp[t] / bp[t].max(), height=0.04, color=col, alpha=0.35, lw=0)
        ax.plot(Xm[:, types.index(lt)] / Xm[:, types.index(lt)].max(), z, color=col, lw=1.4)
        ax.set_title(t if lt == t else f"{t} / {lt}", fontsize=7.5)
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 1])
    axs[0].set_ylim(0.43, -0.63)
    axs[0].set_yticks(ZTICKS)
    axs[0].set_ylabel("depth (fraction of thickness, 0 = mid-L4)")
    axs[1].set_xlabel("cells / max per type")
    axs[1].legend([Patch(color="#9e9e9e", alpha=0.5, lw=0), Line2D([], [], color="#616161", lw=1.4)],
                  ["built", "LANSA"], loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6.5)
    save(fig, "built")


def atlas(name, panels, rows, cols, figsize) -> None:
    """Slide-size redraw of a manuscript atlas from the same maps and verdict means (larger type)."""
    ext = [float(sc.F[0]), float(sc.F[-1]), float(Z[-1]), float(Z[0])]
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True, layout="constrained")
    for ax, (title, R) in zip(axes.flat, panels):
        im = ax.imshow(R, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=ext)
        ax.set_title(title, fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("depth (fraction of\nthickness, 0 = mid-L4)")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
        ax.set_xticks(FTICKS)
    axes[0, 0].set_yticks(ZTICKS)
    fig.colorbar(im, ax=axes, shrink=0.7, label="relative power (P / max over depth)")
    save(fig, name)


def fig_atlases() -> None:
    import e2_select as sel
    import e2_t1_atlas as t1a
    import e2_t8_atlas as t8a
    rt = lambda m: f"$r_{{ab}}$ {m['r_ab']:.2f}, $r_{{gm}}$ {m['r_gm']:.2f}".replace("-", "−")  # noqa: E731
    panels = [("observed, 11-area mean", sc.PROTO)]
    for lv in sel.LEVELS["kd"]:
        seeds, maps, vs = t1a.level_maps("kd", 4000, lv)
        assert seeds == [1, 2, 3], (lv, seeds)
        m = {k: float(np.mean([v[k] for v in vs])) for k in ("r_ab", "r_gm")}
        panels.append((f"B = {lv.replace('p', '.')}" + (" (ref)" if lv == sel.REF["kd"] else "") + (" (kept)" if lv == "1p41" else "") + f"\n{rt(m)}",
                       np.mean(maps, axis=0)))
    atlas("t1_kd_atlas", panels, 2, 4, (W, 6.6 / 2.54))
    panels = [("observed, 11-area mean", sc.PROTO)]
    for a, nm in t8a.ARMS.items():
        R, m = t8a.arm_map(a)
        panels.append((f"{nm.replace('S<->D', 'S' + chr(8596) + 'D')}\n{rt(m)}", R))
    atlas("t8_atlas", panels, 2, 3, (W, 6.6 / 2.54))


MAP_LABEL = "relative power (P / max over depth)"
REF_TAGS = ["sw_n4000_dens_k1_kd1_sst15_s1_T8", "lfs_n4000_s2_T8", "lfs_n4000_s3_T8"]
T1_TAG = "t1_kd1p41_n4000_s1_T8"


def draw_map(ax, R, title) -> object:
    """One rendering for every observed and model map: linear 1-150 Hz, viridis, 0-1."""
    im = ax.imshow(R, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[float(sc.F[0]), float(sc.F[-1]), float(Z[-1]), float(Z[0])])
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_xticks(FTICKS)
    ax.set_yticks(ZTICKS)
    return im


def model_map(tags):
    maps, vs = [], []
    for t in tags:
        _, S, K = sc.unit_spectra(t, 2.0, rc.mf.D)
        maps.append(sc.rp_map(S.mean(axis=0), K))
        vs.append(load(E2 / f"{t}_verdict.json"))
    return np.mean(maps, axis=0), {k: float(np.mean([v[k] for v in vs])) for k in ("r_ab", "r_gm", "r_proto")}


def band_profiles(ax, R, solid_model=True) -> None:
    for (a, b), c, nm in (((10, 30), COL["S"], "10-30 Hz"), ((50, 150), COL["red"], "50-150 Hz")):
        m = (sc.F >= a) & (sc.F <= b)
        ax.plot(R[:, m].mean(axis=1), Z, color=c, label=nm)
        ax.plot(sc.PROTO[:, m].mean(axis=1), Z, color=c, ls="--")
    ax.set_ylim(Z[-1], Z[0])
    ax.set_xlabel("band-mean rel. power")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=6.5, ncol=2, handlelength=1.2,
              columnspacing=0.8, title="solid model, dashed observed", title_fontsize=6.5)


def fig_obs_maps() -> None:
    Yd = load(ROOT / "docs/laminar_motif/data/lansa_lfp_Y.json")
    Y = np.asarray(Yd["Y"])
    areas = Yd["axes"]["areas"]
    assert len(areas) == 11 and np.allclose(Yd["axes"]["freq_hz"][:150], sc.F), Yd["axes"]["freq_hz"][:3]
    assert np.allclose(Y.mean(axis=2), sc.PROTO)
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.62, H), layout="constrained", sharey=True)
    draw_map(axs[0], sc.PROTO, f"mean of {len(areas)} areas")
    im = draw_map(axs[1], Y[:, :, areas.index("V1")], "V1")
    axs[0].set_ylabel("depth (fraction of thickness, 0 = mid-L4)")
    fig.colorbar(im, ax=axs, shrink=0.85, label=MAP_LABEL)
    save(fig, "obs_maps")


def fig_map_triplet(name, tags, title) -> dict:
    R, m = model_map(tags)
    fig, axs = plt.subplots(1, 3, figsize=(W, H * 1.05), layout="constrained",
                            gridspec_kw={"width_ratios": [1, 1, 0.75]})
    draw_map(axs[0], sc.PROTO, "observed, 11-area mean")
    im = draw_map(axs[1], R, title)
    axs[0].set_ylabel("depth (fraction of thickness, 0 = mid-L4)")
    axs[1].set_yticklabels([])
    fig.colorbar(im, ax=axs[:2], shrink=0.85, label=MAP_LABEL)
    band_profiles(axs[2], R)
    axs[2].set_yticks(ZTICKS)
    axs[2].set_title("band profiles", fontsize=8)
    save(fig, name)
    return m


def fig_mechanism() -> None:
    e1 = load(ROOT / "results/laminar_e1/e1_result.json")
    rho = {"S": "0.25", "M": "1.0", "D": "2.0"}  # the stack's imposed SOM/PV ramp
    pred = {u: [e1["per_seed"][s][rho[u]]["peak_freq_E_Hz"] for s in ("1", "2", "3")] for u in "SMD"}
    lr = load(E2 / "lfs_rule.json")["stats"]["4000"]
    obs = {u: [p[u] for p in lr["peaks"]] for u in "SMD"}
    assert [round(np.mean(obs[u])) for u in "SMD"] == [43, 11, 12], lr["peak_mean"]
    assert [round(np.mean(pred[u])) for u in "SMD"] == [46, 34, 27], pred
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.98, H), layout="constrained", gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    rhos = [0.25, 0.5, 1.0, 2.0]
    for s, mk in zip(("1", "2", "3"), ("o", "s", "^")):
        ax.plot(rhos, [e1["per_seed"][s][f"{r}"]["peak_freq_E_Hz"] for r in rhos], marker=mk, color="#616161",
                lw=1.0, ms=4, label=f"seed {s}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("SOM/PV cell ratio $\\rho$")
    ax.set_ylabel("E peak frequency (Hz)")
    ax.set_ylim(10, 50)
    ax.axhspan(10, 30, color=COL["S"], alpha=0.08, lw=0)
    ax.set_title("one isolated unit", fontsize=8)
    ax.legend(loc="upper right")
    b = axs[1]
    for i, u in enumerate("SMD"):
        b.scatter([i - 0.12] * 3, pred[u], marker="o", facecolor="white", edgecolor=COL[u], s=28, zorder=3,
                  label="isolated unit at its $\\rho$" if i == 0 else None)
        b.scatter([i + 0.12] * 3, obs[u], marker="o", color=COL[u], s=28, zorder=3,
                  label="unit inside the column" if i == 0 else None)
        b.annotate("", (i + 0.12, np.mean(obs[u])), (i - 0.12, np.mean(pred[u])),
                   arrowprops={"arrowstyle": "->", "color": "#757575", "lw": 0.8})
    b.axhspan(10, 30, color=COL["S"], alpha=0.08, lw=0)
    b.axhspan(50, 52, color=COL["red"], alpha=0.0, lw=0)
    b.axhline(50, color=COL["red"], lw=0.6, ls=":")
    b.text(2.45, 51, "scored gamma band starts at 50 Hz", fontsize=6.5, color=COL["red"], ha="right")
    b.text(2.45, 28.5, "alpha-beta 10-30 Hz", fontsize=6.5, color=COL["S"], ha="right")
    b.set_xticks(range(3), [f"S ($\\rho$ 0.25)", "M ($\\rho$ 1)", "D ($\\rho$ 2)"])
    b.set_ylabel("E peak frequency (Hz)")
    b.set_ylim(5, 55)
    b.set_xlim(-0.5, 2.5)
    b.set_title("3 seeds each, N = 4000", fontsize=8)
    b.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93))
    save(fig, "mechanism")


def fig_t8_summary() -> None:
    d = load(DATA / "t8_summary.json")
    arms = [("ref", "ref."), ("dflat", "dens.\nflat"), ("drev", "dens.\nS↔D"),
            ("rflat", "ratio\nflat"), ("rrev", "ratio\nrev.")]
    dom = {"ref": "", "dflat": "D", "drev": "D", "rflat": "S", "rrev": "S"}
    keys = [("r_ab", "$r_{ab}$, alpha-beta depth profile"), ("r_gm", "$r_{gm}$, gamma depth profile"),
            ("mid_gamma", "mid-depth relative gamma")]
    fig, axs = plt.subplots(1, 3, figsize=(W, H * 1.05), layout="constrained", sharex=True)
    for ax, (k, lab) in zip(axs, keys):
        for i, (a, _) in enumerate(arms):
            v = [r[k] for r in d["runs"][a]]
            assert len(v) == 3 and not any(r["missing"] for r in d["runs"][a])
            c = "black" if a == "ref" else (COL["D"] if dom[a] == "D" else COL["S"])
            ax.scatter([i - 0.12, i, i + 0.12], v, color=c, s=14, zorder=3)  # offset so equal values stay visible
            ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=c, lw=1.6)
        if k == "mid_gamma":
            ax.axhline(0.848, color=COL["gold"], lw=2)
            ax.text(4.4, 0.80, "observed 0.848", ha="right", fontsize=7)
            ax.set_ylim(0, 1)
        else:
            ax.axhline(0.6, color="#9e9e9e", lw=0.8, ls="--")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylim(-1, 1)
            ax.text(-0.4, 0.63, "pass 0.6", ha="left", fontsize=6.5, color="#616161")
        ax.set_title(lab, fontsize=8)
        ax.set_xticks(range(5), [n for _, n in arms], fontsize=7)
    axs[0].set_ylabel("value per seed (dots), mean (bar)")
    save(fig, "t8_summary")


WEIGHT_SHOWN = {"e2x_weeM1p68_wesS0p5_wesM0p5_kd1p41_n4000_s1_T8": "no crossing",  # settings in the slide caption
                "e2x_weeM1p68_wesS0p5_wpeM1p25_kd1p41_n4000_s1_T8": "crossing"}


def fig_weights() -> None:
    """Exploratory middle-unit weight runs (R8): r_ab vs r_gm with the crossing, and two of their maps."""
    tags = [T1_TAG] + sorted(p.name[:-len("_verdict.json")] for p in E2.glob("e2x_*_verdict.json"))
    v = {t: load(E2 / f"{t}_verdict.json") for t in tags}
    assert len(tags) == 13 and sum(bool(x["pass"]) for x in v.values()) == 0
    assert sum(bool(v[t]["crossing_2se"]) for t in tags) == 7
    fig, axs = plt.subplots(1, 3, figsize=(W, H * 1.05), layout="constrained",
                            gridspec_kw={"width_ratios": [1.05, 1, 1]})
    ax = axs[0]
    ax.axhspan(0.6, 1.0, xmin=(0.6 + 0.6) / 1.6, xmax=1, color=COL["gold"], alpha=0.35, lw=0)
    for t in tags:
        c = v[t]["crossing_2se"]
        ax.scatter(v[t]["r_ab"], v[t]["r_gm"], s=30 if t in WEIGHT_SHOWN else 18, zorder=3,
                   facecolor=COL["M"] if c else "white", edgecolor="black" if t == T1_TAG else COL["M"],
                   lw=1.2 if t in WEIGHT_SHOWN or t == T1_TAG else 0.8)
    for k, t in enumerate(WEIGHT_SHOWN):
        ax.annotate("AB"[k], (v[t]["r_ab"], v[t]["r_gm"]), (6, 4), textcoords="offset points", fontsize=8,
                    fontweight="bold")
    ax.annotate("start", (v[T1_TAG]["r_ab"], v[T1_TAG]["r_gm"]), (4, -9), textcoords="offset points", fontsize=7)
    ax.axhline(0.6, color="#9e9e9e", lw=0.8, ls="--")
    ax.axvline(0.6, color="#9e9e9e", lw=0.8, ls="--")
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("$r_{ab}$ (alpha-beta depth correlation)")
    ax.set_ylabel("$r_{gm}$ (gamma depth correlation)")
    ax.scatter([], [], facecolor=COL["M"], edgecolor=COL["M"], s=18, label="crossing > 2 SE")
    ax.scatter([], [], facecolor="white", edgecolor=COL["M"], s=18, label="no crossing")
    ax.legend(loc="lower left", fontsize=6.5)
    ax.set_title("13 settings, seed 1", fontsize=8)
    for k, (t, nm) in enumerate(WEIGHT_SHOWN.items()):
        R, m = model_map([t])
        assert abs(m["r_gm"] - v[t]["r_gm"]) < 1e-12
        im = draw_map(axs[k + 1], R, f"{'AB'[k]}: {nm}\n" + f"$r_{{ab}}$ {m['r_ab']:.2f}, $r_{{gm}}$ {m['r_gm']:.2f}")
        axs[k + 1].title.set_fontsize(7)
    axs[1].set_ylabel("depth (fraction of thickness, 0 = mid-L4)")
    axs[2].set_yticklabels([])
    fig.colorbar(im, ax=axs[1:], shrink=0.85, label=MAP_LABEL)
    save(fig, "weights")


def fig_column() -> None:
    """Page 3 schema, redrawn from the reference record: one dot per 20 cells, E/PV/SOM loops, E links."""
    import e2_circuit_schema as cs_fig
    from matplotlib.patches import FancyArrowPatch
    from jomission.laminar import caminos_stack as cst
    z, _, _, _, _ = cs_fig.lansa()
    rec = load(E2 / f"{REF_TAGS[0]}.json")
    tc = cs_fig.TYPE_COLOR
    fig, ax = plt.subplots(figsize=(W * 0.5, H), layout="constrained")
    rng = np.random.default_rng(0)
    arrow = lambda a, b, c, rad=0.3, lw=1.0: ax.add_patch(FancyArrowPatch(  # noqa: E731
        a, b, arrowstyle="-|>", mutation_scale=7, color=c, lw=lw, connectionstyle=f"arc3,rad={rad}",
        shrinkA=5, shrinkB=5))
    for m, sel in cst.MODULE_BINS.items():
        zb = z[sel(z)]
        lo, hi = zb.min() - 0.025, zb.max() + 0.025
        ax.axhspan(lo, hi, xmin=0, xmax=0.55, color="#eeeeee" if m == "M" else "#f6f6f6", zorder=0)
        for t in tc:
            n = int(round(rec["sizes"][m][t] / 20))
            ax.scatter(rng.uniform(0.02, 0.53, n), rng.uniform(lo, hi, n), s=2, color=tc[t], lw=0, zorder=2)
        zc = rec["depth_rel_L4"][m]
        pe, pp, ps = (0.68, zc), (0.84, zc - 0.07), (0.84, zc + 0.07)
        for q, t in ((pe, "E"), (pp, "PV"), (ps, "SOM")):
            ax.scatter(*q, s=70, color=tc[t], zorder=3)
        for a, b in ((pe, pp), (pp, pe), (pe, ps), (ps, pe)):
            arrow(a, b, "#424242")
        ax.text(0.92, zc, f"{m}\nN {sum(rec['sizes'][m].values())}", va="center", fontsize=7.5)
    units = list(rec["sizes"])
    for a, b in zip(units[:-1], units[1:]):
        za, zb_ = rec["depth_rel_L4"][a], rec["depth_rel_L4"][b]
        arrow((0.63, za + 0.03), (0.63, zb_ - 0.03), tc["E"], rad=-0.3, lw=0.9)
        arrow((0.61, zb_ - 0.03), (0.61, za + 0.03), tc["E"], rad=-0.3, lw=0.9)
    for t in tc:
        ax.scatter([], [], s=20, color=tc[t], label=t)
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.02), ncol=3, fontsize=7, handletextpad=0.2)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(0.43, -0.63)
    ax.set_yticks(ZTICKS)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("depth (fraction of thickness, 0 = mid-L4)")
    ax.set_title("model column (one dot per 20 cells)", fontsize=8)
    save(fig, "column")


def fig_geometry() -> None:
    """Readout-depth confound (depth_placement.json): spectra fixed, readout depths moved; frequency-shift null."""
    d = load(E2 / "depth_placement.json")
    sw = d["sweep"][d["select_on"]]
    own, sel = sw["own"]["r_proto"], sw["selected"][0]
    assert round(own, 2) == 0.38 and round(sel, 3) == 0.720 and sw["n_placements"] == 1330, (own, sel)
    ho = d["held_out"]
    assert len(ho) == 8
    real, n_real = d["shift_null"]["diag_real"][0][0], d["shift_null"]["diag_real"][1]
    draws = d["shift_null"]["draws"]
    found = [r for r, n in draws if r is not None]
    assert len(draws) == 20 and len(found) == 8 and sum(r > real for r in found) == 3, (len(found), real)
    fig, axs = plt.subplots(1, 3, figsize=(W, H), layout="constrained", gridspec_kw={"width_ratios": [0.8, 1.2, 1]})
    a = axs[0]
    a.bar([0, 1], [own, sel], color=["#9e9e9e", COL["S"]], width=0.6)
    for i, v in enumerate((own, sel)):
        a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    a.set_xticks([0, 1], ["declared\ndepths", "selected\n(1330 tried)"])
    a.set_ylim(0, 1)
    a.set_ylabel("map similarity $r_{proto}$")
    a.set_title("selection run, spectra fixed", fontsize=8)
    b = axs[1]
    rng = np.random.default_rng(0)
    for i, k in enumerate(("r_proto", "r_ab", "r_gm")):
        v = [x[k] for x in ho.values()]
        b.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=[COL["S"], COL["D"], COL["red"]][i], zorder=3)
    b.axhline(0.6, color="#9e9e9e", lw=0.8, ls="--")
    b.set_xticks(range(3), ["$r_{proto}$", "$r_{ab}$", "$r_{gm}$"])
    b.set_ylim(0, 1)
    b.set_ylabel("score at the selected depths")
    b.set_title("8 other ladder runs, same depths", fontsize=8)
    c = axs[2]
    c.scatter(rng.uniform(-0.1, 0.1, len(found)), found, s=16, color="#757575", zorder=3,
              label=f"shifted ({len(found)}/20 placed)")
    c.axhline(real, color=COL["S"], lw=1.6, label=f"unshifted {real:.3f}")
    c.set_xlim(-0.5, 0.5)
    c.set_xticks([])
    c.set_ylim(0.45, 0.85)
    c.set_ylabel("best $r_{proto}$")
    c.set_title("shift null: 8 of 20 draws met both\nband thresholds; best of 1330 each", fontsize=7)
    c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    save(fig, "geometry")


def main() -> None:
    crops()
    fig_density()
    fig_column()
    fig_geometry()
    fig_atlases()
    fig_obs_maps()
    m = fig_map_triplet("ref_triplet", REF_TAGS, "reference model, mean of 3 seeds")
    assert abs(m["r_ab"] - np.mean([0.839, 0.857, 0.827])) < 2e-3, m
    m = fig_map_triplet("scorer_triplet", [T1_TAG], "model, seed 1")
    assert abs(m["r_gm"] - 0.423) < 1e-3, m
    fig_mechanism()
    fig_t8_summary()
    fig_weights()
    y = target()
    d3, d4 = load(DATA / "readout_capacity.json"), load(DATA / "readout_capacity_loao.json")
    d5, d7 = load(DATA / "readout_joint.json"), load(DATA / "readout_joint_grid.json")
    assert abs(rc.mid(y) - d3["y_mid"]) < 1e-9, "target profile differs from D3"
    fig_mid()
    fig_value()
    fig_kernel(y)
    fig_steps(y, d3)
    fig_joint(y, d3, d5, d7)
    fig_sigma(d3, d4)
    fig_circuit(d5)
    print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()

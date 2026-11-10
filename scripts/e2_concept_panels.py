"""Concept-form panel per scored E2 run, for the Pages dashboard (observed vs model relative-power maps).

Usage:
  python scripts/e2_concept_panels.py [TAG ...]     (default: every results/laminar_e2/*_verdict.json)

Per run, one 7.2 x 2.6 in figure: A observed map (PROTO, mean of 11 LANSA areas); B model map, the scorer's rp_map of
the run (same 21 x 150 grid, same viridis scale); C depth profiles of the alpha-beta (10-30 Hz) and gamma (50-150 Hz)
band means, model solid and observed dashed; D the run's configuration and verdict. Writes
results/viz/spectrolaminar/runs/{TAG}.png (150 dpi) and site-src/data/spectrolaminar_runs.json, one row per run with
its configuration and verdict (the dashboard's table). Reads results/laminar_e2/{TAG}.json, .npz, _verdict.json.
No number is computed here that the scorer does not compute: B is e2_scorer.rp_map of e2_scorer.unit_spectra.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

OUT = mf.ROOT / "results" / "viz" / "spectrolaminar"
# a site table, not a result: kept out of results/ (every tracked results/*.json must be a registered result)
TABLE = mf.ROOT / "site-src" / "data" / "spectrolaminar_runs.json"
PT = 7
C_AB, C_GM = "#1565c0", "#e53935"


def band_profile(R: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    sel = (sc.F >= lo) & (sc.F <= hi)
    return R[:, sel].mean(axis=1)


def _per_unit(x) -> dict:
    """Older records store one scalar for all units; newer ones a dict per unit."""
    return x if isinstance(x, dict) else ({"all": x} if x is not None else {})


def config_row(tag: str, rec: dict, v: dict) -> dict:
    mods = rec.get("modifiers") or {}
    w = mods.get("w") or {}
    return {"tag": tag, "layout": rec.get("layout", ""), "seed": rec.get("seed"), "T_s": rec.get("T_s"),
            "N": int(sum(sum(s.values()) for s in rec["sizes"].values())),
            "sizes": " ".join(f"{u}:{s['E']}/{s['PV']}/{s['SOM']}" for u, s in rec["sizes"].items()),
            "p_local": " ".join(f"{u}:{p:.3f}" for u, p in _per_unit(rec.get("p_local")).items()),
            "tau_sst_ms": " ".join(f"{u}:{t:g}" for u, t in _per_unit(rec.get("tau_sst_e_ms")).items()),
            "w_mods": " ".join(f"{k}x{x:g}" for k, x in w.items()) or "-",
            "edge_delivery": rec.get("edge_delivery") or "dense",
            "r_proto": round(v["r_proto"], 3), "r_ab": round(v["r_ab"], 3), "r_gm": round(v["r_gm"], 3),
            "crossing_2se": v["crossing_2se"], "perm_rank": v["perm_rank"],
            "smooth_p95": round(v["smooth_p95"], 3), "pass": v["pass"]}


def _map(ax, R, title):
    im = ax.pcolormesh(sc.F, mf.CHANNELS, R, cmap="viridis", vmin=0, vmax=1, shading="nearest")
    ax.set_xscale("log")
    ax.set_xticks([2, 10, 50, 150], ["2", "10", "50", "150"])
    ax.set_ylim(mf.CHANNELS[-1] + 0.025, mf.CHANNELS[0] - 0.025)
    ax.set_xlabel("frequency (Hz)")
    ax.set_title(title, fontsize=PT + 0.5)
    return im


def panel(tag: str, out: Path = OUT) -> dict:
    rd = mf.D
    rec = json.loads((rd / f"{tag}.json").read_text(encoding="utf-8"))
    v = json.loads((rd / f"{tag}_verdict.json").read_text(encoding="utf-8"))
    _, S, K = sc.unit_spectra(tag, float(v["discard_s"]), rd)
    R = sc.rp_map(S.mean(axis=0), K)
    r_now = sc._r_over_bins(R, sc.PROTO)[0]
    if not np.isclose(r_now, v["r_proto"], rtol=0, atol=1e-9):
        raise ValueError(f"{tag}: map r_proto {r_now} != verdict {v['r_proto']} (stale verdict?)")
    row = config_row(tag, rec, v)
    plt.rcParams.update({"font.size": PT, "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(6.8, 2.7), constrained_layout=True)  # print size: full text width
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.75, 1.3])
    a = fig.add_subplot(gs[0])
    _map(a, sc.PROTO, "observed, 11-area mean")
    a.set_ylabel("depth from mid-L4")
    b = fig.add_subplot(gs[1], sharey=a)
    im = _map(b, R, "model, scorer map")
    b.tick_params(labelleft=False)
    fig.colorbar(im, ax=b, shrink=0.85, label="relative power")
    c = fig.add_subplot(gs[2], sharey=a)
    for band, col, name in zip(mf.BANDS[mf.FIG_BANDS], (C_AB, C_GM), ("10-30 Hz", "50-150 Hz")):
        c.plot(band_profile(sc.PROTO, band), mf.CHANNELS, color=col, ls="--", lw=1.2)
        c.plot(band_profile(R, band), mf.CHANNELS, color=col, lw=1.5, label=name)
    c.set_xlabel("relative power")
    c.set_title("band profiles", fontsize=PT + 0.5)
    c.tick_params(labelleft=False)
    c.legend(frameon=False, fontsize=PT - 0.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
             handlelength=1.2, columnspacing=0.6, title="solid model, dashed observed", title_fontsize=PT - 0.5)
    d = fig.add_subplot(gs[3])
    d.axis("off")
    k = tag.find("_kd") if len(tag) > 34 else -1  # wrap long tags before the base-layout tokens
    lines = [tag[:k], "  " + tag[k:]] if k > 0 else [tag]
    lines += [f"seed {row['seed']}, {row['T_s']:g} s, N {row['N']:,}, {row['edge_delivery']}",
             "E/PV/SOM per unit", *("  " + s for s in row["sizes"].split()), "p " + row["p_local"], "SOM->E tau " + row["tau_sst_ms"] + " ms",
             "w " + row["w_mods"], "",
             f"r_proto {row['r_proto']:.3f} (null p95 {row['smooth_p95']:.3f})",
             f"r_ab {row['r_ab']:.3f}   r_gm {row['r_gm']:.3f}  (>= 0.6)",
             f"crossing {row['crossing_2se']}   perm rank {row['perm_rank']}",
             f"verdict: {'PASS' if row['pass'] else 'FAIL'}"]
    d.text(0, 1, "\n".join(lines), va="top", fontsize=PT - 1, family="DejaVu Sans Mono", transform=d.transAxes)
    for ax_, letter in zip((a, b, c, d), "ABCD"):
        t = ax_.get_title()
        ax_.set_title("")
        ax_.set_title(rf"$\bf{{{letter}}}$  {t}", loc="left", fontsize=PT + 0.5)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "runs" / f"{tag}.png", dpi=300)
    plt.close(fig)
    return row


def main(tags: list[str]) -> None:
    full = not tags  # the table is written only from the full run set, never from a named subset
    if full:
        # symlinked verdicts are axis aliases of a carried winner (e.g. t1_sst15 -> t1_kd1p41): one panel per run
        tags = sorted(p.name[: -len("_verdict.json")] for p in mf.D.glob("*_verdict.json") if not p.is_symlink())
    rows = []
    for t in tags:
        rows.append(panel(t))
        print(t, rows[-1]["r_proto"], rows[-1]["pass"], flush=True)
    if not full:
        return
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps({"runs": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

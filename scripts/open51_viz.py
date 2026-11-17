"""OPEN-51: retrospective, partial V0 and V3 for E0 (SCI-LAMINAR-MOTIF-1) and E1 (SCI-LAMINAR-E1).

  python -B scripts/open51_viz.py COUNTS_DIR

COUNTS_DIR holds laminar_e0/ and laminar_e1/ with the original *_counts.npy (read only; the salvage copy on F:).
Nothing is constructed or simulated. Drawn from preserved evidence only:
- V0: the declared construction (jomission/laminar/caminos_e0.py constants and RULES, unchanged since 3257ffc).
  Realized edges and weights were never recorded and are not drawn.
- V3 (partial): population rates, E-rate spectra and the per-run estimands, from the original counts. Each counts
  file must match its sha256 below (docs/laminar_motif/OPEN51_E0_E1_VIZ_FEASIBILITY_2026-11-17.md) and reproduce its
  committed run JSON before anything is drawn.
V1 and V2 are not drawn: the runs saved no per-cell spikes.
"""
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from jomission.laminar import caminos_analysis as ca  # noqa: E402
from jomission.laminar import caminos_e0 as ce  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = ce.DT_MS / 1000.0
STAMP = "RETROSPECTIVE, PARTIAL: drawn 2026-11-17 from preserved evidence after interpretation (OPEN-51)"
SHA = {
    "laminar_e0/e0_seed1_counts.npy": "ac5a4415af817472262c57bc272657770927ed4ff1382b93c7ff94d2de3966a7",
    "laminar_e0/e0_seed2_counts.npy": "f7d2a3b1a68c87ccefa19b374f79de6988053b7c981710d09119661392ac14ae",
    "laminar_e0/e0_seed3_counts.npy": "e2b70bdc99ed611e2d732d8fb94a0c40978649f6550a57cfc7821af2bac95e9b",
    "laminar_e0/e0_seed4_counts.npy": "9158654942bd56eec654072a26859d24e0409c56783872c0f2a517afdb249be3",
    "laminar_e0/e0_seed5_counts.npy": "5549416ecd38032215409bc359573558c0bad98537b269dcf12a3d4814e97ff2",
    "laminar_e1/e1_rho0.25_seed1_counts.npy": "eed03c5eda28078b6ffbf0a640c769deec0fa1a2bce87bff41bbd0dccf0cd9f9",
    "laminar_e1/e1_rho0.25_seed2_counts.npy": "fd0e731abc0e3301cc0e02628bcf42d0f9578281089e85825a15d89735cb8fa2",
    "laminar_e1/e1_rho0.25_seed3_counts.npy": "5a50fbe94e76cd1f9e6085579c702f40f1f23d41eab34f4009e5f4a88b06b3eb",
    "laminar_e1/e1_rho0.5_seed1_counts.npy": "10b95ab35ad605856ecaa0f0f653e63bff261df3a54dcaf092824ca9a53a28d9",
    "laminar_e1/e1_rho0.5_seed2_counts.npy": "8a8eeaf858da40d482631f59f9a6c0bbcb73a824c0f0b50bb0bbee79bcbd36e2",
    "laminar_e1/e1_rho0.5_seed3_counts.npy": "6f49c57caa1313a8a5cb451a17e53a80368623199bf26f0332446837ef3ab9bf",
    "laminar_e1/e1_rho1_seed1_counts.npy": "ac5a4415af817472262c57bc272657770927ed4ff1382b93c7ff94d2de3966a7",
    "laminar_e1/e1_rho1_seed2_counts.npy": "f7d2a3b1a68c87ccefa19b374f79de6988053b7c981710d09119661392ac14ae",
    "laminar_e1/e1_rho1_seed3_counts.npy": "e2b70bdc99ed611e2d732d8fb94a0c40978649f6550a57cfc7821af2bac95e9b",
    "laminar_e1/e1_rho2_seed1_counts.npy": "9ff48508f2ef1ded10218cd9a1d9536b0d47a6d2ee0b3ef281807233857b3417",
    "laminar_e1/e1_rho2_seed2_counts.npy": "a3a3e819a0ed30e9817029f2db7046698640fd2243c5dd81dc8f5ab185353738",
    "laminar_e1/e1_rho2_seed3_counts.npy": "5dde113bb719900e5fa86441c394925185bb32c2edbc76009c70e53b558389e2",
    "laminar_e1/e1_rho4_seed1_counts.npy": "024a24a0beece805be80a179217d0185941fe142ec7e4ef711220ad418163ecd",
    "laminar_e1/e1_rho4_seed2_counts.npy": "868d748ccde9110213223789521233ad73b2fe841729f66a18d4ee09185c8b1b",
    "laminar_e1/e1_rho4_seed3_counts.npy": "0648b217445440ae259ad3afce544a7d23e11cba69f015e5ee8bbd0321303ce7",
}
LINEAGES = {
    "SCI-LAMINAR-MOTIF-1": {"title": "E0: CAMINOS E-PV-SST reference module, p 0.07, seeds 1-5",
                            "runs": [("laminar_e0", f"e0_seed{s}", s, 1.0) for s in range(1, 6)],
                            "commit": "ea249f1"},
    "SCI-LAMINAR-E1": {"title": "E1: SST/PV ratio rho sweep at fixed N_inh 1500, seeds 1-3",
                       "runs": [("laminar_e1", f"e1_rho{r}_seed{s}", s, float(r))
                                for r in ("0.25", "0.5", "1", "2", "4") for s in (1, 2, 3)],
                       "commit": "8200fac"},
}
CHECK_KEYS = ("peak_freq_E_Hz", "peak_excess_E", "oscillates", "lag_PV_SOM_ms")


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_run(counts_dir, folder, stem):
    rel = f"{folder}/{stem}_counts.npy"
    raw = (counts_dir / rel).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA[rel]:
        raise SystemExit(f"{rel}: sha256 {digest} differs from the recorded {SHA[rel]}")
    counts = np.load(counts_dir / rel)
    rec = json.loads((ROOT / "results" / folder / f"{stem}.json").read_text(encoding="utf-8"))
    n_sst = int(rec.get("n_sst", ce.NSST))
    sizes = ce.sizes(n_sst)
    got = ca.analyze_run(counts, DT, sizes)
    want = rec["analysis"]
    if got["spikes_kept"] != want["spikes_kept"]:
        raise SystemExit(f"{rel}: spikes_kept {got['spikes_kept']} != JSON {want['spikes_kept']}")
    for k in CHECK_KEYS:
        a, b = got[k], want[k]
        same = a == b if isinstance(b, bool) or b is None else abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(b))
        if not same:
            raise SystemExit(f"{rel}: recomputed {k} {a} != JSON {b}")
    return {"rel": rel, "sha256": digest, "json": f"results/{folder}/{stem}.json", "counts": counts, "sizes": sizes,
            "rec": rec, "analysis": want}


def draw_v0(path, lineage, size_rows):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    pos = {"E": (7.0, 7.0), "PV": (4.6, 3.9), "SOM": (9.4, 3.9)}
    names = {"E": "E", "PV": "PV", "SOM": "SST (JaxFNE: SOM)"}
    bw, bh = 2.6, 1.15
    for k, (x, y) in pos.items():
        col = "#f4cccc" if k == "E" else "#cfe2f3"
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.05", fc=col, ec="k"))
        ax.text(x, y + 0.25, names[k], ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x, y - 0.2, size_rows[k], ha="center", va="center", fontsize=7.5)
    ax.add_patch(FancyBboxPatch((2.9, 2.75), 8.2, 5.95, boxstyle="round,pad=0.1", fc="none", ec="0.5", ls="--"))
    ax.text(3.0, 8.45, "module C: one CAMINOS module, no layers", fontsize=9, color="0.3")
    # x: drive on the far left; y: recorded output on the far right
    ax.add_patch(FancyBboxPatch((0.1, 4.6), 2.2, 1.7, boxstyle="round,pad=0.05", fc="#eeeeee", ec="k"))
    ax.text(1.2, 5.45, f"x: Poisson drive\n{ce.ext_source_count(0.07)} private sources\nper E and PV cell\n"
            f"{ce.EXT_RATE_HZ:g} Hz, g {ce.Q_EXT} nS, ampa\nSST undriven", ha="center", va="center", fontsize=8)
    for tgt in ("E", "PV"):
        tx, ty = pos[tgt]
        ax.add_patch(FancyArrowPatch((2.35, 5.45), (tx - bw / 2 - 0.05, ty), arrowstyle="-|>", mutation_scale=12,
                                     color="#c00000", lw=1.2))
    ax.add_patch(FancyBboxPatch((11.7, 4.6), 2.2, 1.7, boxstyle="round,pad=0.05", fc="#eeeeee", ec="k"))
    ax.text(12.8, 5.45, "y: per-step\npopulation counts\nE, PV, SOM, dt 0.1 ms\n(the only saved output)",
            ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((11.2, 5.45), (11.65, 5.45), arrowstyle="-|>", mutation_scale=12, color="k"))
    pairs = {(r[1], r[2]) for r in ce.RULES}
    for i, (name, pre, post, g, mech) in enumerate(ce.RULES, 1):
        col = "#c00000" if mech == "ampa" else "#1f4e9c"
        style = "-|>" if mech == "ampa" else "-["
        (x0, y0), (x1, y1) = pos[pre], pos[post]
        if pre == post:
            dx = 0.0 if pre == "E" else -1.0
            sx, sy = (x0 - 0.5, y0 + bh / 2 + 0.05) if pre == "E" else (x0 - bw / 2 - 0.05, y0 + 0.3)
            ex, ey = (x0 + 0.5, y0 + bh / 2 + 0.05) if pre == "E" else (x0 - bw / 2 - 0.05, y0 - 0.3)
            ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), connectionstyle="arc3,rad=-1.6" if pre == "E"
                                         else "arc3,rad=1.6", arrowstyle=style, mutation_scale=12, color=col, lw=1.4))
            lx, ly = (x0, y0 + bh / 2 + 0.75) if pre == "E" else (x0 - bw / 2 - 0.75 + dx * 0, y0)
            ax.text(lx, ly, f"{i}", ha="center", va="center", fontsize=9, color=col, weight="bold",
                    bbox={"fc": "white", "ec": col, "boxstyle": "circle,pad=0.2"})
            continue
        v = np.array([x1 - x0, y1 - y0])
        nrm = np.array([-v[1], v[0]]) / np.hypot(*v)
        off = 0.22 if (post, pre) in pairs else 0.0
        a = np.array([x0, y0]) + off * nrm
        b = np.array([x1, y1]) + off * nrm
        u = (b - a) / np.hypot(*(b - a))
        # leave each box at its edge: smallest t with |u_x t| = bw/2 or |u_y t| = bh/2, plus a margin
        t = min(bw / 2 / abs(u[0]) if u[0] else np.inf, bh / 2 / abs(u[1]) if u[1] else np.inf) + 0.1
        ax.add_patch(FancyArrowPatch(tuple(a + t * u), tuple(b - t * u), arrowstyle=style, mutation_scale=12,
                                     color=col, lw=1.4, shrinkA=0, shrinkB=0))
        m = (a + b) / 2 + (0.32 if off else 0.3) * nrm
        ax.text(*m, f"{i}", ha="center", va="center", fontsize=9, color=col, weight="bold",
                bbox={"fc": "white", "ec": col, "boxstyle": "circle,pad=0.2"})
    ax.text(7.0, 3.0, "absent by declaration: PV->SST, SST->SST", ha="center", fontsize=8.5, style="italic")
    lines = ["#  rule    pre -> post   g (nS)  receptor  sign"]
    for i, (name, pre, post, g, mech) in enumerate(ce.RULES, 1):
        sign = "excitatory" if mech == "ampa" else "inhibitory"
        lines.append(f"{i}  {name:7s} {pre:>3s} -> {post:<4s}   {g:5.1f}   {mech:8s}  {sign}")
    ax.text(0.2, 2.45, "\n".join(lines), family="monospace", fontsize=8, va="top")
    ax.text(6.3, 2.45, f"All blocks: Bernoulli p 0.07, delay {ce.DELAY_MS} ms, edge_seed = run seed.\n"
            f"ampa tau {ce.TAU_E_MS} ms, E_rev {ce.REV_E_MV} mV; gaba tau {ce.TAU_I_MS} ms, E_rev {ce.REV_I_MV} mV.\n"
            "Red arrow excitatory, blue bar inhibitory; g is the declared connection authority.\n"
            "Projection direction: arrows run pre -> post (one module, so no FF/FB axis).", fontsize=8, va="top")
    ax.text(6.3, 1.1, "Realized edge counts and weights were never recorded. This schematic is the declaration\n"
            "(caminos_e0.py, constants unchanged since 3257ffc), not a construction receipt; no construction ran.",
            fontsize=8, va="top", color="#a00000")
    fig.suptitle(f"{lineage}  V0 construction (declared)\n{STAMP}", fontsize=10, color="#a00000")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def rate_trace(counts, n, t0_s, t1_s, bin_steps=10):
    a, b = int(round(t0_s / DT)), int(round(t1_s / DT))
    c = counts[a:b, :].astype(float).reshape(-1, bin_steps, 3).sum(axis=1)
    t = t0_s + (np.arange(c.shape[0]) + 0.5) * bin_steps * DT
    return t, c / (np.array([n["E"], n["PV"], n["SOM"]], float) * bin_steps * DT)


def draw_rates(path, lineage, runs):
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 1.5 * len(runs) + 0.8), sharex=True, squeeze=False)
    for ax, r in zip(axes[:, 0], runs):
        t, rates = rate_trace(r["counts"], r["sizes"], 2.0, 2.5)
        for i, (pop, col) in enumerate(zip(("E", "PV", "SST"), ("#c00000", "#1f4e9c", "#2e7d32"))):
            ax.plot(t * 1000, rates[:, i], color=col, lw=0.8, label=pop)
        ax.set_ylabel(f"{r['label']}\nHz", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1, 0].set_xlabel("time from simulation start (ms); 2000-2500 ms, first window after the 2 s discard; "
                           "1 ms bins, population rate per cell", fontsize=8)
    fig.suptitle(f"{lineage}  population rates from the saved counts\n{STAMP}", fontsize=9, color="#a00000")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_spectra(path, lineage, runs):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis", len(runs))
    for i, r in enumerate(runs):
        keep = r["counts"][ca.kept_slice(r["counts"].shape[0], DT)]
        f, p = ca.e_welch(keep[:, 0] / DT / r["sizes"]["E"], DT)
        sel = (f >= 5) & (f <= 120)
        ax.semilogy(f[sel], p[sel], color=cmap(i), lw=0.9, label=r["label"])
        fpk = r["analysis"]["peak_freq_E_Hz"]
        if fpk is not None:
            ax.axvline(fpk, color=cmap(i), lw=0.5, ls=":")
    ax.axvspan(ca.BAND_LO_HZ, ca.BAND_HI_HZ, color="0.92", zorder=0)
    ax.set_xlabel("frequency (Hz); grey: peak search band 20-80 Hz; dotted: recorded f_pk", fontsize=8)
    ax.set_ylabel("E-rate Welch PSD (1 s Hann, 50% overlap, after 2 s)", fontsize=8)
    ax.legend(fontsize=6.5, ncol=3)
    fig.suptitle(f"{lineage}  E-rate spectra from the saved counts\n{STAMP}", fontsize=9, color="#a00000")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_fpk(path, runs):
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, mk in zip((1, 2, 3), ("o", "s", "^")):
        pts = [(r["rho"], r["analysis"]) for r in runs if r["seed"] == s]
        x = [np.log2(rho) for rho, _ in pts]
        y = [a["peak_freq_E_Hz"] for _, a in pts]
        osc = [a["oscillates"] for _, a in pts]
        ax.plot(x, y, color="0.6", lw=0.7)
        ax.scatter([xi for xi, o in zip(x, osc) if o], [yi for yi, o in zip(y, osc) if o], marker=mk, color="k",
                   label=f"seed {s}, oscillates")
        ax.scatter([xi for xi, o in zip(x, osc) if not o], [yi for yi, o in zip(y, osc) if not o], marker=mk,
                   facecolor="none", edgecolor="r", label=f"seed {s}, below threshold")
    ax.set_xlabel("log2 rho (SST/PV)")
    ax.set_ylabel("recorded f_pk (Hz)")
    ax.legend(fontsize=6.5)
    fig.suptitle(f"SCI-LAMINAR-E1  recorded f_pk against rho\n{STAMP}", fontsize=8, color="#a00000")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def atlas(out, lineage, meta, runs, figs):
    rows = "".join(
        f"<tr><td>{html.escape(r['label'])}</td><td>{r['sizes']['PV']}/{r['sizes']['SOM']}</td>"
        + "".join(f"<td>{r['analysis']['rates_Hz'][k]:.3f}</td>" for k in ("E", "PV", "SOM"))
        + f"<td>{r['analysis']['peak_freq_E_Hz']}</td><td>{r['analysis']['peak_excess_E']:.3f}</td>"
        f"<td>{r['analysis']['oscillates']}</td><td>{r['analysis']['lag_PV_SOM_ms']}</td>"
        f"<td>{r['rec']['wall_s']:.0f}</td><td><code>{r['sha256'][:16]}</code></td></tr>" for r in runs)
    imgs = "".join(f"<h2>{html.escape(t)}</h2><img src='{f}' style='max-width:100%'>" for t, f in figs)
    page = f"""<!doctype html><meta charset="utf-8"><title>{lineage} V3 (partial, retrospective)</title>
<body style="font-family:sans-serif;max-width:1100px;margin:auto">
<h1>{lineage}: {html.escape(meta['title'])} (V3 atlas, partial)</h1>
<p style="color:#a00000"><b>{STAMP}.</b> Result commit {meta['commit']}. Every counts file matched its recorded sha256
and reproduced its committed run JSON (spikes kept, f_pk, peak excess, oscillation call, PV-SST lag) before drawing.
Sources and hashes: <a href="provenance.json">provenance.json</a>.</p>
<h2>Construction (V0, declared)</h2><img src="../jaxfne_network_hspice.png" style="max-width:100%">
{imgs}
<h2>Per run: recorded estimands (results JSON)</h2>
<table border=1 cellpadding=3 style="border-collapse:collapse;font-size:12px">
<tr><th>run</th><th>N_PV/N_SST</th><th>E Hz</th><th>PV Hz</th><th>SST Hz</th><th>f_pk Hz</th><th>peak excess</th>
<th>oscillates</th><th>lag PV-SST ms</th><th>wall s</th><th>counts sha256</th></tr>{rows}</table>
<h2>Not in this atlas</h2>
<p>Raster, per-cell spike times, membrane potential, I_net, H, A, m(H), and realized edges and weights: the runs
saved only per-step population counts (caminos_e0.py run() sums the raster in memory), so none of these exists.
The model has no plasticity. No field or LFP proxy was recorded. V1 and V2 are absent for the same reason; this
lineage's visualization set stays incomplete under manifests/visualization_contract.json.</p></body>"""
    (out / "atlas" / "index.html").write_text(page, encoding="utf-8", newline="\n")


def main(counts_dir: Path) -> None:
    head = git("rev-parse", "HEAD")
    for lineage, meta in LINEAGES.items():
        # outside results/: unregistered candidates (no lineage-record change is authorized; results_coverage)
        out = ROOT / "docs" / "laminar_motif" / "open51_viz" / lineage
        (out / "atlas").mkdir(parents=True, exist_ok=True)
        runs = []
        for folder, stem, seed, rho in meta["runs"]:
            r = load_run(counts_dir, folder, stem)
            r.update(seed=seed, rho=rho, label=stem)
            runs.append(r)
        if lineage == "SCI-LAMINAR-E1":
            by_rho = sorted({(r["rho"], r["sizes"]["PV"], r["sizes"]["SOM"]) for r in runs})
            head = "rho " + ", ".join(f"{rho:g}" for rho, _, _ in by_rho)
            size_rows = {"E": f"N = {ce.NE}", "PV": f"N at {head}:\n" + ", ".join(str(p) for _, p, _ in by_rho),
                         "SOM": f"N at {head}:\n" + ", ".join(str(s) for _, _, s in by_rho)}
        else:
            size_rows = {"E": f"N = {ce.NE}", "PV": f"N = {ce.NPV}", "SOM": f"N = {ce.NSST}"}
        draw_v0(out / "jaxfne_network_hspice.png", lineage, size_rows)
        figs = [("Population rates", "rates.png"), ("E-rate spectra", "spectra.png")]
        sample = runs if lineage == "SCI-LAMINAR-MOTIF-1" else [r for r in runs if r["seed"] == 1]
        draw_rates(out / "atlas" / "rates.png", lineage, sample)
        draw_spectra(out / "atlas" / "spectra.png", lineage, runs)
        if lineage == "SCI-LAMINAR-E1":
            draw_fpk(out / "atlas" / "fpk_rho.png", runs)
            figs.append(("Lineage diagnostic: f_pk against rho", "fpk_rho.png"))
        atlas(out, lineage, meta, runs, figs)
        prov = {"schema": "jomission.open51_viz.v1", "lineage": lineage, "label": STAMP, "drawn_at_head": head,
                "script": "scripts/open51_viz.py", "declared_construction": "jomission/laminar/caminos_e0.py "
                "(constants and RULES unchanged since 3257ffc)", "result_commit": meta["commit"],
                "jaxfne_build": "e0160dfc (declared, E0_SEAL.md Amendment 1; observed for E1 by compiled-code "
                                "equality, see the OPEN-51 note)",
                "checks": "sha256 per counts file; analyze_run(counts) equals the run JSON on spikes_kept and "
                          + ", ".join(CHECK_KEYS),
                "inputs": [{"counts": r["rel"], "sha256": r["sha256"], "run_json": r["json"]} for r in runs],
                "absent": ["V1", "V2", "raster", "membrane", "I_net", "H", "A", "m(H)", "realized edges/weights"]}
        (out / "atlas" / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8",
                                                       newline="\n")
        print(f"{lineage}: {len(runs)} runs verified; V0 and partial V3 written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

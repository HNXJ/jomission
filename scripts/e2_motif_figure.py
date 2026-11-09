"""Spectrolaminar motif figure for a model run, in the layout of the published panels d-i.

Usage:
  python scripts/e2_motif_figure.py TAG [DISCARD_S]

Row 1 (d-f): model LFP proxy per module as recorded. Row 2 (g-i): the same proxy with each module's
amplitude weighted by the LANSA NG (pyramidal proxy) mean density of its bins, median over the 11 areas
(module bins as scripts/lansa_module_density.py; weights normalized to D = 1; power scales by weight^2).
Relative power: P(d, f) / max_d P(d, f). Bands as in the published figure: alpha-beta 10-30 Hz,
gamma 50-150 Hz. Writes results/laminar_e2/<TAG>_motif.png and <TAG>_motif.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.signal.windows import dpss  # noqa: E402

SMOOTH_HZ = 2.0  # multitaper half-bandwidth, as the published analysis (1 s windows, 2 Hz smoothing)


def multitaper(sig: np.ndarray, fs: float, nper: int) -> tuple[np.ndarray, np.ndarray]:
    """[time, channel] -> (f, P[channel, f]): non-overlapping windows of nper samples, DPSS NW = W*T,
    K = 2NW - 1 tapers, power averaged over tapers and windows; each window demeaned."""
    nw = SMOOTH_HZ * nper / fs
    tapers = dpss(nper, nw, Kmax=max(1, int(2 * nw) - 1))  # [K, nper]
    nwin = sig.shape[0] // nper
    acc = 0.0
    for i in range(nwin):
        x = sig[i * nper:(i + 1) * nper]
        x = x - x.mean(axis=0)
        acc = acc + (np.abs(np.fft.rfft(tapers[:, :, None] * x[None], axis=1)) ** 2).mean(axis=0)
    return np.fft.rfftfreq(nper, 1.0 / fs), (acc / max(nwin, 1)).T

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "laminar_e2"
MODS = ("S", "M", "D")
AB, GM = (10.0, 30.0), (50.0, 150.0)
CHANNELS = np.round(np.linspace(-0.6, 0.4, 21), 3)  # virtual probe at the LANSA bin centers
SIGMA = 0.15  # volume-conduction kernel width (assumption, relative depth units)
C_SUP, C_DEEP ="#D9822B", "#7030A0"  # panel d/g superficial orange, deep purple
C_AB, C_GM = "#0000FF", "#FF0000"  # band bars and panel f/i profiles
# MATLAB parula, 9 anchors (approximation of the published colormap)
PARULA = LinearSegmentedColormap.from_list("parula", [
    "#352A87", "#0F5CDD", "#1481D6", "#06A4CA", "#2EB7A4", "#87BF77", "#D1BB59", "#FEC832", "#F9FB0E"])


def lansa_weights() -> tuple[dict[str, float], dict[str, float]]:
    d = json.loads((ROOT / "docs/laminar_motif/data/lansa_density_X.json").read_text(encoding="utf-8"))
    X = np.asarray(d["X"], dtype=float)  # [bin, type, area]
    z = np.round(np.asarray(d["axes"]["bin_center_rel_L4"]), 3)
    ng = d["axes"]["types"].index("NG")
    groups = {"S": z < -0.175, "M": (z >= -0.175) & (z <= 0.175), "D": z > 0.175}
    dens = {m: float(np.median(X[g, ng, :].mean(axis=0))) for m, g in groups.items()}
    depth = {m: float(z[g].mean()) for m, g in groups.items()}
    return {m: dens[m] / dens["D"] for m in MODS}, depth


def band_mean(f, P, lo, hi):
    s = (f >= lo) & (f <= hi)
    return P[:, s].mean(axis=1)


def row(axes, f, P, depth, letters, title):
    ax_d, ax_e, ax_f = axes
    R = P / P.max(axis=0, keepdims=True)  # relative power per frequency across depth
    s = (f >= 0) & (f <= 150)
    ax_d.plot(f[s], R[0, s], c=C_SUP, lw=1.6)
    ax_d.plot(f[s], R[-1, s], c=C_DEEP, lw=1.6)
    ax_d.text(0.98, 0.93, "Superficial", color="k", ha="right", transform=ax_d.transAxes, fontsize=10)
    ax_d.text(0.98, 0.25, "Deep", color="k", ha="right", transform=ax_d.transAxes, fontsize=10)
    ax_d.set_xlim(0, 150)
    ax_d.set_ylim(0, 1.05)
    ax_d.set_xticks([0, 50, 100, 150])
    ax_d.set_xlabel("Frequency (Hz)")
    ax_d.set_ylabel("Relative power")
    ys = np.asarray(depth, dtype=float)
    edges =np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2, [ys[-1] + (ys[-1] - ys[-2]) / 2]])
    im = ax_e.imshow(R[:, s], aspect="auto", cmap=PARULA, vmin=0.3, vmax=1.0, interpolation="nearest",
                     extent=(0, 150, edges[-1], edges[0]))
    ax_e.set_xticks([0, 50, 100, 150])
    ax_e.set_xlabel("Frequency (Hz)")
    ax_e.set_ylabel("Laminar depth (rel. mid-L4)")
    cb = plt.colorbar(im, ax=ax_e, fraction=0.05, pad=0.03, ticks=[0.3, 1])
    cb.set_label("Relative power", rotation=270, labelpad=10)
    ab, gm = band_mean(f, R, *AB), band_mean(f, R, *GM)  # band mean of relative power, not renormalized
    ax_f.plot(ab, ys, c=C_AB, lw=1.8)
    ax_f.plot(gm, ys, c=C_GM, lw=1.8)
    ax_f.axhline(0, c="k", lw=1.6)
    ax_f.set_ylim(edges[-1], edges[0])
    ax_f.set_xlim(0.0, 1.02)
    ax_f.set_xticks([0, 0.3, 0.7, 1])
    ax_f.set_xticklabels(["0", "0.3", "0.7", "1"])
    ax_f.set_xlabel("Relative power")
    ax_f.set_ylabel("Laminar depth (rel. mid-L4)")
    ax_f.text(0.2, 1.01, "Alpha-\nbeta", color="k", ha="center", va="bottom", fontsize=10, transform=ax_f.transAxes)
    ax_f.text(0.62, 1.01, "Gamma", color="k", ha="center", va="bottom", fontsize=10, transform=ax_f.transAxes)
    for ax in (ax_d, ax_e):
        tr = ax.get_xaxis_transform()
        ax.plot(AB, [1.03, 1.03], c=C_AB, lw=6, transform=tr, clip_on=False, solid_capstyle="butt")
        ax.plot(GM, [1.03, 1.03], c=C_GM, lw=6, transform=tr, clip_on=False, solid_capstyle="butt")
        ax.text(np.mean(AB), 1.07, "Alpha-\nbeta", transform=tr, ha="center", va="bottom", fontsize=10)
        ax.text(np.mean(GM), 1.07, "Gamma", transform=tr, ha="center", va="bottom", fontsize=10)
    for ax, L in zip(axes, letters):
        ax.text(-0.22, 1.18, L, transform=ax.transAxes, fontsize=16, fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    ax_d.set_title(title, loc="left", fontsize=9, pad=38)
    d = ab - gm
    sign_change = np.nonzero(np.diff(np.sign(d)))[0]
    cross = None
    if sign_change.size:
        i = int(sign_change[0])
        cross = float(ys[i] + (ys[i + 1] - ys[i]) * d[i] / (d[i] - d[i + 1]))
    return {"depth": ys.tolist(), "alpha_beta_profile": ab.tolist(), "gamma_profile": gm.tolist(),
            "alpha_beta_peak_depth": float(ys[np.argmax(ab)]), "gamma_peak_depth": float(ys[np.argmax(gm)]),
            "motif": bool(ys[np.argmax(gm)] < ys[np.argmax(ab)]), "crossing_depth": cross}


def main(tag: str, discard_s: float) -> None:
    z = np.load(D / f"{tag}.npz")
    rec = json.loads((D / f"{tag}.json").read_text(encoding="utf-8"))
    dt = rec["dt_ms"] / 1000.0
    lfp = z["lfp_pA"].astype(np.float64)[int(round(discard_s / dt)):]
    k10 = int(round(1e-3 / dt))  # average to 1 kHz before the PSD
    n1 = lfp.shape[0] // k10
    lfp = lfp[: n1 * k10].reshape(n1, k10, -1).mean(axis=1)
    fs = 1000.0
    nper = min(1000, n1)
    w, depth = lansa_weights()
    zm = np.array([depth[m] for m in MODS])
    chans = CHANNELS
    K = np.exp(-((chans[:, None] - zm[None, :]) ** 2) / (2 * SIGMA ** 2))  # [channel, module] volume conduction

    def probe_psd(amp):
        sig = (lfp * amp[None, :]) @ K.T  # [time, channel]
        return multitaper(sig, fs, nper)

    f, P = probe_psd(np.ones(len(MODS)))
    _, Pw = probe_psd(np.array([w[m] for m in MODS]))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
    fig, axs = plt.subplots(2, 3, figsize=(14, 10.5), gridspec_kw={"width_ratios": [1, 1.35, 0.9], "hspace": 0.6, "wspace": 0.42})
    r1 = row(axs[0], f, P, chans, "def", f"Model LFP proxy (E2 stack, seed {rec['seed']}, {rec['T_s']:g} s)")
    r2 = row(axs[1], f, Pw, chans, "ghi", "Same run, module amplitude x LANSA NG density "
             + ", ".join(f"{m} {w[m]:.2f}" for m in MODS))
    fig.savefig(D / f"{tag}_motif.png", dpi=150, bbox_inches="tight")
    out = {"tag": tag, "discard_s": discard_s, "bands": {"alpha_beta": AB, "gamma": GM},
           "probe": {"channels_rel_L4": chans.tolist(), "gaussian_sigma": SIGMA, "module_depth_rel_L4": depth},
           "lansa_ng_weight": w, "raw": r1, "lansa_weighted": r2}
    (D / f"{tag}_motif.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (1, 2):
        raise SystemExit(__doc__)
    main(a[0], float(a[1]) if len(a) == 2 else 2.0)

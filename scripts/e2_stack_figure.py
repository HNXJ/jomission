"""E2 stack analysis and figure: per-module rasters, PSDs, band power by depth.

Usage:
  python scripts/e2_stack_figure.py TAG [DISCARD_S]

Reads results/laminar_e2/<TAG>.npz and .json; writes results/laminar_e2/<TAG>_summary.json and
results/laminar_e2/<TAG>_fig.png. Bands: alpha-beta 8-30 Hz, gamma 31-150 Hz (PLAN_LAMINAR_MOTIF.md goal).
Peak frequency and oscillation call per module use the E0 estimator (caminos_analysis.analyze_run).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("JAXFNE_SITE"):
    sys.path.insert(0, os.environ["JAXFNE_SITE"])
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

from jomission.laminar import caminos_analysis as an  # noqa: E402

D = ROOT / "results" / "laminar_e2"
MODS = ("S", "M", "D")
POPS = ("E", "PV", "SOM")
BANDS = {"alpha_beta": (8.0, 30.0), "gamma": (31.0, 150.0)}
COL = {"E": "#3a3a3a", "PV": "#d55e00", "SOM": "#0072b2"}


def main(tag: str, discard_s: float) -> None:
    z = np.load(D / f"{tag}.npz")
    rec = json.loads((D / f"{tag}.json").read_text(encoding="utf-8"))
    dt = rec["dt_ms"] / 1000.0
    fs = 1.0 / dt
    counts = z["counts"]
    lfp = z["lfp_pA"].astype(np.float64)
    cut = int(round(discard_s / dt))
    nper = min(int(round(1.0 / dt)), lfp.shape[0] - cut)

    out = {"tag": tag, "discard_s": discard_s, "modules": {}}
    psd = {}
    for k, m in enumerate(MODS):
        c = counts[:, 3 * k:3 * k + 3]
        sz = rec["sizes"][m]
        if discard_s == an.DISCARD_S and c.shape[0] * dt > discard_s + 1.0:
            est = an.analyze_run(c, dt, sz)
        else:
            est = {"note": "run too short for the E0 estimator"}
        x = lfp[cut:, k] - lfp[cut:, k].mean()
        f, p = welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2)
        r = c[cut:, 0].astype(np.float64) / dt / sz["E"]
        fr, pr = welch(r - r.mean(), fs=fs, window="hann", nperseg=nper, noverlap=nper // 2)
        psd[m] = (f, p, fr, pr)
        bp = {b: float(np.trapezoid(p[(f >= lo) & (f <= hi)], f[(f >= lo) & (f <= hi)])) for b, (lo, hi) in BANDS.items()}
        sel = (f >= 4) & (f <= 150)
        out["modules"][m] = {
            "rho": rec["rho"][m],
            "estimator_E_rate": est,
            "lfp_peak_Hz_4_150": float(f[sel][np.argmax(p[sel])]),
            "lfp_band_power": bp,
            "lfp_gamma_over_alpha_beta": bp["gamma"] / bp["alpha_beta"] if bp["alpha_beta"] > 0 else None,
        }
    for b in BANDS:
        vals = np.array([out["modules"][m]["lfp_band_power"][b] for m in MODS])
        rp = vals / vals.max() if vals.max() > 0 else vals
        for m, v in zip(MODS, rp):
            out["modules"][m].setdefault("lfp_relative_power", {})[b] = float(v)
        out[f"{b}_peak_module"] = MODS[int(np.argmax(vals))]
    (D / f"{tag}_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.6, 1.2, 0.9])
    t_lo = discard_s
    t_hi = min(t_lo + 1.0, counts.shape[0] * dt)
    for k, m in enumerate(MODS):
        ax = fig.add_subplot(gs[k, 0])
        off = 0
        for t in POPS:
            ts = z[f"{m}_{t}_t"] * dt
            cs_ = z[f"{m}_{t}_c"]
            sel = (ts >= t_lo) & (ts < t_hi)
            ax.scatter(ts[sel], cs_[sel] + off, s=0.6, c=COL[t], marker="|", linewidths=0.5)
            off += int(cs_.max()) + 1 if cs_.size else 0
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylabel(f"{m}  rho={rec['rho'][m]:g}\ncell (E, PV, SST)")
        ax.set_yticks([])
        if k == 2:
            ax.set_xlabel("time (s)")
        if k == 0:
            ax.set_title("raster, 1 s after discard (sample: 200 E, 50 PV, 50 SST)")
        ax2 = fig.add_subplot(gs[k, 1])
        f, p, fr, pr = psd[m]
        s = (f >= 2) & (f <= 150)
        ax2.semilogy(f[s], p[s], c="k", lw=1, label="LFP proxy")
        ax2b = ax2.twinx()
        ax2b.semilogy(fr[s], pr[s], c="#999999", lw=0.8, label="E rate")
        ax2b.set_yticks([])
        for lo, hi, cc in ((8, 30, "#56b4e9"), (31, 150, "#e69f00")):
            ax2.axvspan(lo, hi, color=cc, alpha=0.12, lw=0)
        mm = out["modules"][m]
        fpk = mm["estimator_E_rate"].get("peak_freq_E_Hz")
        ax2.set_title(f"{m}: LFP peak {mm['lfp_peak_Hz_4_150']:.0f} Hz; E-rate f_pk {fpk if fpk is not None else 'n/a'}", fontsize=9)
        ax2.set_ylabel("PSD (pA^2/Hz)")
        if k == 2:
            ax2.set_xlabel("frequency (Hz)")
    ax3 = fig.add_subplot(gs[:, 2])
    y = np.arange(len(MODS))[::-1]
    for b, cc in (("alpha_beta", "#56b4e9"), ("gamma", "#e69f00")):
        ax3.plot([out["modules"][m]["lfp_relative_power"][b] for m in MODS], y, "o-", c=cc, label=b.replace("_", "-"))
    ax3.set_yticks(y, [f"{m} (rho {rec['rho'][m]:g})" for m in MODS])
    ax3.set_xlabel("relative LFP-proxy power (max over modules = 1)")
    ax3.set_xlim(0, 1.05)
    ax3.legend(loc="lower left", fontsize=9)
    ax3.set_title("band power by depth")
    fig.suptitle(f"E2 stack {tag}: seed {rec['seed']}, {rec['T_s']:g} s, p_inter {rec['p_inter']:g}, JaxFNE {rec['jaxfne_version']}")
    fig.savefig(D / f"{tag}_fig.png", dpi=130)
    print(json.dumps({m: {kk: out["modules"][m][kk] for kk in ("lfp_peak_Hz_4_150", "lfp_gamma_over_alpha_beta", "lfp_relative_power")} for m in MODS}))
    print("peaks:", out["alpha_beta_peak_module"], out["gamma_peak_module"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (1, 2):
        raise SystemExit(__doc__)
    main(a[0], float(a[1]) if len(a) == 2 else an.DISCARD_S)

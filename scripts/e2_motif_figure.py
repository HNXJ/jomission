"""Spectrolaminar motif figure for a model stack run, in the layout of the published panels d-i, with a
null-tested score (revision after the 2026-11-09 review).

Usage:
  python scripts/e2_motif_figure.py TAG [DISCARD_S]

LFP proxy per unit (Mazzoni et al. 2015 RWS form): |I_ampa(t - 6 ms)| - 1.65 |I_gaba(t)| onto the unit's E
cells, I = g (V - E_rev) at V fixed -65 mV, from the recorded recurrent conductances g_nS. Row 1 (d-f):
mean per E cell. Row 2 (g-i): sum over E cells (unit size enters through the model's own cell counts; no
post-hoc density weight). A virtual 21-contact probe at the LANSA bin centers sees each unit through a
Gaussian volume-conduction kernel (sigma 0.15). Spectra: multitaper, 1 s windows, 2 Hz smoothing, from the
unit cross-spectra, so P(c, f) = sum_uv K_cu K_cv S_uv(f). Relative power R(c, f) = P / max_c P.

Score per band set, on R averaged in each band:
  contrast M = slope over depth of (alpha-beta - gamma); M > 0 means alpha-beta deeper than gamma (motif).
  SE: jackknife over 1 s windows; z = M / SE (flat null M = 0: identical units give R independent of f).
  Depth-shuffle null: M over unit-depth permutations (all when U <= 6, else 2000 draws, seed 0); p = share >= M.
  Crossing: (ab - gm) below -2 SE at the top contact and above +2 SE at the bottom contact; depth = first
  zero of the full-data difference.
Peak frequency per contact and per unit: argmax of absolute power in 5-150 Hz.
Writes results/laminar_e2/<TAG>_motif.png and <TAG>_motif.json.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.signal.windows import dpss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jomission.laminar import caminos_e0 as c0  # noqa: E402
from jomission.laminar import caminos_stack as cs  # noqa: E402

D = ROOT / "results" / "laminar_e2"
SMOOTH_HZ = 2.0  # multitaper half-bandwidth, as the published analysis (1 s windows, 2 Hz smoothing)
BANDS = {"image": ((10.0, 30.0), (50.0, 150.0)), "manuscript": ((8.0, 30.0), (31.0, 150.0))}
FIG_BANDS = "image"  # verdict bands (owner ruling 2026-11-09); "manuscript" is scored as secondary only
PEAK_RANGE = (5.0, 150.0)
V_REF_MV = -65.0
RWS_DELAY_MS, RWS_ALPHA = 6.0, 1.65  # Mazzoni et al. 2015
CHANNELS = np.round(np.linspace(-0.6, 0.4, 21), 3)  # virtual probe at the LANSA bin centers
SIGMA = 0.15  # volume-conduction kernel width (assumption, relative depth units)
N_SHUFFLE = 2000
C_SUP, C_DEEP = "#D9822B", "#7030A0"  # panel d/g superficial orange, deep purple
C_AB, C_GM = "#0000FF", "#FF0000"  # band bars and panel f/i profiles
# MATLAB parula, 9 anchors (approximation of the published colormap)
PARULA = LinearSegmentedColormap.from_list("parula", [
    "#352A87", "#0F5CDD", "#1481D6", "#06A4CA", "#2EB7A4", "#87BF77", "#D1BB59", "#FEC832", "#F9FB0E"])


def rws(g: np.ndarray, dt_ms: float, rec: dict, units: list[str]) -> np.ndarray:
    """g [T, columns] (nS summed onto E cells) -> [T - d, U] pA, RWS proxy. Columns from rec['g_columns']
    (unit, receptor) with reversals; runs without it hold (ampa, gaba) per unit. Every receptor with
    reversal below -40 mV counts as GABA."""
    d = int(round(RWS_DELAY_MS / dt_ms))
    if "g_columns" in rec:
        cols, rev = rec["g_columns"], np.asarray(rec["g_reversal_mV"])
    else:
        cols = [[u, r] for u in units for r in ("ampa", "gaba")]
        rev = np.tile([c0.REV_E_MV, c0.REV_I_MV], len(units))
    I = np.abs(g * (V_REF_MV - rev)[None])
    unit = np.array([c[0] for c in cols])
    ia = np.stack([I[:, (unit == u) & (rev > -40)].sum(axis=1) for u in units], axis=1)
    ig = np.stack([I[:, (unit == u) & (rev <= -40)].sum(axis=1) for u in units], axis=1)
    return ia[:-d] - RWS_ALPHA * ig[d:]


def cross_spectra(x: np.ndarray, fs: float, nper: int) -> tuple[np.ndarray, np.ndarray]:
    """[T, U] -> (f, S[window, f, U, U]); DPSS NW = W*T, K = 2NW - 1 tapers, each window demeaned."""
    nw = SMOOTH_HZ * nper / fs
    tapers = dpss(nper, nw, Kmax=max(1, int(2 * nw) - 1))  # [K, nper]
    S = []
    for i in range(x.shape[0] // nper):
        seg = x[i * nper:(i + 1) * nper]
        F = np.fft.rfft(tapers[:, :, None] * (seg - seg.mean(axis=0))[None], axis=1)  # [K, f, U]
        S.append(np.einsum("kfu,kfv->fuv", F, F.conj()) / F.shape[0])
    return np.fft.rfftfreq(nper, 1.0 / fs), np.asarray(S)


def probe_power(S: np.ndarray, K: np.ndarray) -> np.ndarray:
    """S [..., f, U, U], K [C, U] -> P [..., C, f]."""
    return np.einsum("cu,...fuv,cv->...cf", K, S, K).real


def band_profiles(f, P, bands):
    R = P / P.max(axis=-2, keepdims=True)
    (a0, a1), (g0, g1) = bands
    return R[..., (f >= a0) & (f <= a1)].mean(axis=-1), R[..., (f >= g0) & (f <= g1)].mean(axis=-1)


def slope(ys, v):
    """Least-squares slope of v[..., C] over depth ys[C]."""
    yc = ys - ys.mean()
    return (v * yc).sum(axis=-1) / (yc ** 2).sum()


def score(f, S, K, ys, bands) -> dict:
    """Contrast, jackknife SE, depth-shuffle null and crossing for one band set."""
    Sm = S.mean(axis=0)
    ab, gm = band_profiles(f, probe_power(Sm, K), bands)
    dlt = ab - gm
    M = float(slope(ys, dlt))
    n = S.shape[0]
    se_M, se_d = None, None
    if n >= 2:
        loo = (S.sum(axis=0)[None] - S) / (n - 1)
        a_l, g_l = band_profiles(f, probe_power(loo, K), bands)
        d_l = a_l - g_l
        se_d = np.sqrt((n - 1) / n * ((d_l - d_l.mean(axis=0)) ** 2).sum(axis=0))
        m_l = slope(ys, d_l)
        se_M = float(np.sqrt((n - 1) / n * ((m_l - m_l.mean()) ** 2).sum()))
    U = Sm.shape[-1]
    if U <= 6:
        perms = [np.array(p) for p in itertools.permutations(range(U))]
    else:
        rng = np.random.default_rng(0)
        perms = [rng.permutation(U) for _ in range(N_SHUFFLE)]
    null = []
    for p in perms:
        a_p, g_p = band_profiles(f, probe_power(Sm[:, p][:, :, p], K), bands)
        null.append(float(slope(ys, a_p - g_p)))
    null = np.asarray(null)
    cross, cross_depth = None, None
    if se_d is not None:
        cross = bool(dlt[0] < -2 * se_d[0] and dlt[-1] > 2 * se_d[-1])
    sc = np.nonzero(np.diff(np.sign(dlt)))[0]
    if sc.size:
        i = int(sc[0])
        cross_depth = float(ys[i] + (ys[i + 1] - ys[i]) * dlt[i] / (dlt[i] - dlt[i + 1]))
    return {"bands": bands, "alpha_beta_profile": ab.tolist(), "gamma_profile": gm.tolist(),
            "diff_se": None if se_d is None else se_d.tolist(),
            "contrast_M": M, "jackknife_se": se_M, "z": None if not se_M else M / se_M,
            "shuffle": {"n": len(perms), "exhaustive": U <= 6, "p_ge": float((null >= M).mean()),
                        "null_mean": float(null.mean()), "null_sd": float(null.std())},
            "crossing_2se": cross, "zero_depth_first": cross_depth, "n_windows": n}


def peak_freq(f, P):
    s = (f >= PEAK_RANGE[0]) & (f <= PEAK_RANGE[1])
    return f[s][np.argmax(P[..., s], axis=-1)]


def row(axes, f, P, ys, sc, letters, title):
    ax_d, ax_e, ax_f = axes
    AB, GM = BANDS[FIG_BANDS]
    R = P / P.max(axis=0, keepdims=True)
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
    edges = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2, [ys[-1] + (ys[-1] - ys[-2]) / 2]])
    im = ax_e.imshow(R[:, s], aspect="auto", cmap=PARULA, vmin=0.3, vmax=1.0, interpolation="nearest",
                     extent=(0, 150, edges[-1], edges[0]))
    ax_e.plot(peak_freq(f, P), ys, "o", ms=3, mfc="w", mec="k", mew=0.5)  # absolute peak per contact
    ax_e.set_xticks([0, 50, 100, 150])
    ax_e.set_xlabel("Frequency (Hz)")
    ax_e.set_ylabel("Laminar depth (rel. mid-L4)")
    cb = plt.colorbar(im, ax=ax_e, fraction=0.05, pad=0.03, ticks=[0.3, 1])
    cb.set_label("Relative power", rotation=270, labelpad=10)
    ab, gm = np.asarray(sc["alpha_beta_profile"]), np.asarray(sc["gamma_profile"])
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
        ax.spines[["top", "right"]].set_visible(False)
    se = sc["jackknife_se"]
    stat = f"M = {sc['contrast_M']:+.3f}" + (f" ± {se:.3f} (z {sc['z']:+.1f})" if se else "") \
        + f", shuffle p {sc['shuffle']['p_ge']:.3f} (n {sc['shuffle']['n']}), crossing {sc['crossing_2se']}"
    ax_d.figure.text(0.1, ax_d.get_position().y1 + 0.085, f"{title}\n{stat}", fontsize=9, va="bottom")


def main(tag: str, discard_s: float) -> None:
    z = np.load(D / f"{tag}.npz")
    rec = json.loads((D / f"{tag}.json").read_text(encoding="utf-8"))
    units = list(rec.get("units", rec["sizes"]))  # runs before 128d6c2 record no units: 3-unit stack
    depth = rec.get("depth_rel_L4") or cs.layout("e2_3unit")[3]
    if len(units) < 2:
        raise SystemExit("motif needs at least 2 units")
    dt_ms = rec["dt_ms"]
    x = rws(z["g_nS"].astype(np.float64), dt_ms, rec, units)[int(round(discard_s * 1000.0 / dt_ms)):]
    k = int(round(1.0 / dt_ms))  # average to 1 kHz before the spectra
    n1 = x.shape[0] // k
    x = x[: n1 * k].reshape(n1, k, -1).mean(axis=1)
    fs, nper = 1000.0, min(1000, n1)
    n_e = np.array([rec["sizes"][u]["E"] for u in units], dtype=float)
    zu = np.array([depth[u] for u in units])
    K = np.exp(-((CHANNELS[:, None] - zu[None, :]) ** 2) / (2 * SIGMA ** 2))  # [contact, unit]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
    fig, axs = plt.subplots(2, 3, figsize=(14, 11), gridspec_kw={"width_ratios": [1, 1.35, 0.9], "hspace": 0.7, "wspace": 0.42})
    out = {"tag": tag, "discard_s": discard_s, "units": units, "unit_depth_rel_L4": zu.tolist(),
           "probe": {"channels_rel_L4": CHANNELS.tolist(), "gaussian_sigma": SIGMA},
           "proxy": f"RWS |I_ampa(t-{RWS_DELAY_MS:g} ms)| - {RWS_ALPHA} |I_gaba(t)| onto E, V {V_REF_MV} mV",
           "fig_bands": FIG_BANDS}
    for r, (key, scale, letters, label) in enumerate((("per_cell_mean", 1.0 / n_e, "def", "mean per E cell"),
                                                      ("sum_over_cells", np.ones_like(n_e), "ghi", "sum over E cells"))):
        f, S = cross_spectra(x * scale[None], fs, nper)
        Sm = S.mean(axis=0)
        P = probe_power(Sm, K)
        scores = {b: score(f, S, K, CHANNELS, BANDS[b]) for b in BANDS}
        unit_pow = np.einsum("fuu->uf", Sm).real
        out[key] = {"scores": scores, "peak_hz_contact": peak_freq(f, P).tolist(),
                    "peak_hz_unit": dict(zip(units, peak_freq(f, unit_pow).tolist()))}
        row(axs[r], f, P, CHANNELS, scores[FIG_BANDS], letters,
            f"Model LFP proxy (RWS, {label}); {rec.get('layout', 'e2_3unit')}, seed {rec['seed']}, {rec['T_s']:g} s")
    fig.savefig(D / f"{tag}_motif.png", dpi=150, bbox_inches="tight")
    (D / f"{tag}_motif.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key in ("per_cell_mean", "sum_over_cells"):
        for b, sc in out[key]["scores"].items():
            print(key, b, {k: sc[k] for k in ("contrast_M", "jackknife_se", "z", "crossing_2se", "zero_depth_first")},
                  "shuffle_p", sc["shuffle"]["p_ge"])
        print(key, "peak_hz_unit", out[key]["peak_hz_unit"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (1, 2):
        raise SystemExit(__doc__)
    main(a[0], float(a[1]) if len(a) == 2 else 2.0)

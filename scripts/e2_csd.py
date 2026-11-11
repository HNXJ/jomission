"""CSD from the model's synaptic currents, forward-modelled to the E2 virtual probe.

Usage:
  python scripts/e2_csd.py TAG [--results-dir DIR] [--out-dir DIR] [--discard S]

Column current I_k(t) = g_k(t) (V - E_rev,k) at V = -65 mV, block-averaged to 1 ms (nS x mV = pA). Signed as the
membrane current: positive = outward (source), negative = inward (sink). No readout delay and no abs(): this is the
current itself, not the RWS LFP proxy of e2_motif_figure.py.
Each column is split over PLACEMENT (offsets from the unit depth z_u). Each placement gets an equal and opposite
return at z_u, or at z_u + RETURN_AT_SOMA when the placement sits at the soma (offset 0).
phi(c, t) = sum_s J_s(t) / (4 pi SIGMA_E sqrt((z_c - d_s)^2 + H^2)), J_s the signed source or return current.
CSD estimate = -(second difference of phi over contacts, end contacts duplicated) / dz^2.
True CSD = net source current J_s binned to the nearest contact.
Trigger average: 10-30 Hz band-pass of phi on the contact with the largest band power; troughs of that trace, at
least one 30 Hz period apart; phi and CSD averaged over +-100 ms around each trough.
Refuses a run without a gaba_sst column (gaba would mix PV and SST) and a run whose g_nS shape differs from g_columns.
Depth units are those of depth_rel_L4; phi and CSD are left unconverted (arbitrary units).
The first --discard seconds (default 2.0, start-up transient) are dropped from the 1 ms currents before any
computation; every output (npz, correlation, troughs, rank, PNG) is post-discard. Time stamps are absolute run time.
Writes <TAG>_csd.npz and <TAG>_csd.png (2 x 2: 1 s window; trigger averages).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, find_peaks, sosfiltfilt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_motif_figure import CHANNELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V_MV = -65.0
SIGMA_E = 0.3  # S/m, ASSUMED
H = 0.05  # depth units, ASSUMED
BAND = (10.0, 30.0)
TRIG_MS = 100.0
DISCARD_S = 2.0  # start-up transient; dropped before any computation (CLI --discard)
WIN_MS = 1000.0  # figure: window in the top row, from the first retained sample
RETURN_AT_SOMA = -0.05  # depth offset of the return of a source at the soma

# ASSUMED placements: parent design, not measured. Per receptor: (offset from unit depth z_u, fraction of I_k).
PLACEMENT = {
    "ampa": ((0.0, 0.5), (-0.10, 0.5)),  # half basal at z_u, half apical at z_u - 0.10
    "gaba": ((0.0, 1.0),),  # PV, perisomatic at z_u
    "gaba_sst": ((-0.15, 1.0),),  # SST, distal apical at z_u - 0.15
}


def column_currents(g: np.ndarray, dt_ms: float, rev_mV: np.ndarray) -> np.ndarray:
    """g [T, K] nS -> I [T // n, K] pA at 1 ms: block mean of g, then g (V - E_rev)."""
    k = int(round(1.0 / dt_ms))
    if abs(k * dt_ms - 1.0) > 1e-9:
        raise ValueError(f"dt_ms {dt_ms} does not divide 1 ms")
    n = g.shape[0] // k
    gb = g[: n * k].reshape(n, k, -1).mean(axis=1)
    return gb * (V_MV - np.asarray(rev_mV, dtype=float))[None, :]


def source_table(cols: list, depth: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cols [(unit, receptor)], depth {unit: z_u} -> (column index, signed weight, depth) per source and return."""
    col, w, d = [], [], []
    for k, (unit, rec) in enumerate(cols):
        if rec not in PLACEMENT:
            raise ValueError(f"no placement for receptor {rec!r}")
        zu = float(depth[unit])
        for off, frac in PLACEMENT[rec]:
            ret = zu if off != 0.0 else zu + RETURN_AT_SOMA
            col += [k, k]
            w += [frac, -frac]
            d += [zu + off, ret]
    return np.array(col), np.array(w), np.array(d)


def csd_estimate(phi: np.ndarray, dz: float) -> np.ndarray:
    """phi [T, C] -> -(second difference over contacts) / dz^2, end contacts duplicated (Vaknin)."""
    p = np.concatenate([phi[:, :1], phi, phi[:, -1:]], axis=1)
    return -(p[:, :-2] - 2.0 * p[:, 1:-1] + p[:, 2:]) / dz ** 2


def forward(I: np.ndarray, cols: list, depth: dict) -> dict:
    """I [T, K] pA at 1 ms -> phi, CSD estimate and true CSD on the CHANNELS contacts, all [T, C]."""
    col, w, de = source_table(cols, depth)
    zc = np.asarray(CHANNELS, dtype=float)
    dz = float(np.diff(zc).mean())
    if not np.allclose(np.diff(zc), dz):
        raise ValueError("contacts must be uniformly spaced")
    J = I[:, col] * w[None, :]
    G = 1.0 / (4.0 * np.pi * SIGMA_E * np.sqrt((zc[:, None] - de[None, :]) ** 2 + H ** 2))  # [C, E]
    phi = J @ G.T
    B = np.zeros((de.size, zc.size))
    B[np.arange(de.size), np.abs(de[:, None] - zc[None, :]).argmin(axis=1)] = 1.0
    return {"phi": phi, "csd_est": csd_estimate(phi, dz), "csd_true": J @ B, "dz": dz}


def trigger_average(phi: np.ndarray, csd: np.ndarray, fs_hz: float = 1000.0) -> dict:
    """Trough-triggered averages of phi and CSD on the contact with the largest 10-30 Hz power."""
    sos = butter(4, BAND, btype="bandpass", fs=fs_hz, output="sos")
    bp = sosfiltfilt(sos, phi, axis=0)
    power = (bp ** 2).mean(axis=0)
    c = int(np.argmax(power))
    troughs_all, _ = find_peaks(-bp[:, c], distance=int(round(fs_hz / BAND[1])))
    w = int(round(TRIG_MS * fs_hz / 1000.0))
    troughs = troughs_all[(troughs_all >= w) & (troughs_all < phi.shape[0] - w)]
    lags = np.arange(-w, w + 1)
    if troughs.size:
        idx = troughs[:, None] + lags[None, :]
        trig_phi, trig_csd = phi[idx].mean(axis=0), csd[idx].mean(axis=0)
    else:
        trig_phi = trig_csd = np.full((lags.size, phi.shape[1]), np.nan)
    return {"contact": c, "power": power, "n_troughs_all": int(troughs_all.size), "troughs": troughs,
            "lag_ms": lags, "trig_phi": trig_phi, "trig_csd": trig_csd}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over all entries of csd_est and csd_true: a forward/inverse consistency check, not validation.
    It depends on H against dz and on off-grid placement (true CSD is binned to the nearest contact)."""
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def numerical_rank(M: np.ndarray, tol: float = 1e-6) -> int:
    """rank_mixing: singular values > tol x largest. For csd_est = I M^T with T >= K, this is rank(M), the
    source-to-CSD mixing matrix."""
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0


def energy_rank(M: np.ndarray, frac: float = 1e-2) -> int:
    """rank_energy: singular values >= frac x largest."""
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s >= frac * s[0])) if s.size and s[0] > 0 else 0


def load_run(results_dir: Path, tag: str) -> tuple[dict, np.ndarray, list, np.ndarray]:
    rec = json.loads((results_dir / f"{tag}.json").read_text(encoding="utf-8"))
    g = np.load(results_dir / f"{tag}.npz")["g_nS"].astype(np.float64)
    cols = [(u, r) for u, r in rec["g_columns"]]
    if g.ndim != 2 or g.shape[1] != len(cols):
        raise SystemExit(f"STOP: g_nS shape {g.shape} does not match {len(cols)} g_columns")
    if not any(r == "gaba_sst" for _, r in cols):
        raise SystemExit("refused: no gaba_sst column; gaba would mix PV and SST")
    rev = np.asarray(rec["g_reversal_mV"], dtype=float)
    if rev.size != len(cols):
        raise SystemExit(f"STOP: {rev.size} g_reversal_mV for {len(cols)} g_columns")
    return rec, g, cols, rev


# The caption (assumed source depths, circular trigger contact, per-panel colour limits, demeaned trigger panels)
# is in the manuscript's figure legend; the figure is drawn at print size (full text width).
FIGSIZE_IN = (6.8, 4.6)
FONT_PT = 7


def _demean(field: np.ndarray) -> np.ndarray:
    """Subtract each contact's mean over the lag window (axis 0); all-NaN input (no troughs) passes through."""
    if not np.isfinite(field).any():
        return field
    return field - np.nanmean(field, axis=0, keepdims=True)


def _heat(ax, fig, field: np.ndarray, t0: float, t1: float, zc: np.ndarray, dz: float, title: str) -> None:
    """field [time, contact]; sinks (negative) red, sources (positive) blue; limits at the 99th percentile of |field|."""
    v = float(np.nanpercentile(np.abs(field), 99)) if np.isfinite(field).any() else 1.0
    im = ax.imshow(field.T, origin="lower", aspect="auto", cmap="RdBu", vmin=-v, vmax=v, interpolation="nearest",
                   extent=(t0, t1, zc[0] - dz / 2, zc[-1] + dz / 2))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_ylabel("depth rel. mid-L4")
    cb = fig.colorbar(im, ax=ax)
    cb.set_ticks([-v, 0.0, v])
    cb.set_ticklabels(["sink", "0", "source"])
    cb.set_label("a.u.")


def plot(path: Path, tag: str, phi: np.ndarray, csd: np.ndarray, dz: float, trig: dict, t_offset_ms: float,
         win_ms: float = WIN_MS) -> None:
    """2 x 2 figure from post-discard arrays. Top: phi and CSD estimate, 1 s window from the first retained sample.
    Bottom: the trigger averages from trigger_average (+-100 ms). Time stamps are absolute run time (ms)."""
    zc = np.asarray(CHANNELS, dtype=float)
    t0, t1 = float(t_offset_ms), float(t_offset_ms + win_ms)
    win = slice(0, int(win_ms))
    n = int(trig["troughs"].size)
    lags = trig["lag_ms"]
    plt.rcParams.update({"font.family": "Arial", "font.size": FONT_PT, "axes.titlesize": FONT_PT,
                         "xtick.labelsize": FONT_PT - 1, "ytick.labelsize": FONT_PT - 1,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE_IN, layout="constrained")
    _heat(axs[0, 0], fig, phi[win], t0, t1, zc, dz, f"phi (forward model), {t0 / 1000:g}-{t1 / 1000:g} s")
    _heat(axs[0, 1], fig, csd[win], t0, t1, zc, dz, f"CSD estimate, {t0 / 1000:g}-{t1 / 1000:g} s")
    tl = (float(lags[0]), float(lags[-1]))  # trigger axis is lag, not run time
    _heat(axs[1, 0], fig, _demean(trig["trig_phi"]), *tl, zc, dz, f"phi, trough-triggered (n = {n} troughs)")
    _heat(axs[1, 1], fig, _demean(trig["trig_csd"]), *tl, zc, dz, f"CSD, trough-triggered (n = {n} troughs)")
    for ax in axs[0]:
        ax.set_xlabel("time (ms)")
    for ax in axs[1]:
        ax.set_xlabel("lag from trough (ms)")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "laminar_e2")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--discard", type=float, default=DISCARD_S, help="seconds dropped from the start of the run")
    a = ap.parse_args(argv)
    if a.discard < 0:
        raise SystemExit("--discard must be >= 0")
    out_dir = a.out_dir or a.results_dir
    rec, g, cols, rev = load_run(a.results_dir, a.tag)
    I = column_currents(g, float(rec["dt_ms"]), rev)
    n_drop = int(round(a.discard * 1000.0))  # 1 ms bins
    if n_drop >= I.shape[0]:
        raise SystemExit("--discard removes the whole run")
    I = I[n_drop:]
    depth = {u: float(z) for u, z in rec["depth_rel_L4"].items()}
    F = forward(I, cols, depth)
    trig = trigger_average(F["phi"], F["csd_est"])  # trough indices are post-discard (absolute ms = n_drop + index)
    corr_consistency = corr(F["csd_est"], F["csd_true"])
    rank_mixing, rank_energy = numerical_rank(F["csd_est"]), energy_rank(F["csd_est"])
    rank_true = numerical_rank(F["csd_true"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cont = np.asarray(CHANNELS, dtype=float)
    np.savez(out_dir / f"{a.tag}_csd.npz", phi=F["phi"], csd_est=F["csd_est"], csd_true=F["csd_true"],
             contacts=cont, t_ms=n_drop + np.arange(F["phi"].shape[0]), discard_s=a.discard,
             trig_lag_ms=trig["lag_ms"], trig_phi=trig["trig_phi"], trig_csd=trig["trig_csd"],
             trig_contact=cont[trig["contact"]], trig_band_power=trig["power"],
             trough_ms=n_drop + trig["troughs"], n_troughs_all=trig["n_troughs_all"],
             corr_consistency=corr_consistency, rank_mixing=rank_mixing, rank_energy=rank_energy,
             rank_true=rank_true)
    plot(out_dir / f"{a.tag}_csd.png", a.tag, F["phi"], F["csd_est"], F["dz"], trig, float(n_drop))
    print(f"{a.tag} discard={a.discard:g}s corr_consistency={corr_consistency:.4f} "
          f"troughs_all={trig['n_troughs_all']} troughs_used={trig['troughs'].size} "
          f"trig_contact={float(cont[trig['contact']]):+.2f} "
          f"rank_mixing={rank_mixing} rank_energy={rank_energy} rank_true={rank_true}")


if __name__ == "__main__":
    main()

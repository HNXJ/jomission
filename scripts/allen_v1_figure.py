"""Spectrolaminar motif figure for an Allen V1 replica run (ALLEN_V1_REPLICA_SPEC.md 7), panels d-i layout.

Usage:
  python scripts/allen_v1_figure.py TAG [DISCARD_S]

Reads results/allen_v1/<TAG>.npz (g_nS [T, 8]: per E layer L2/3, L4, L5, L6 the summed AMPA then GABA conductance
onto that layer's E cells) and <TAG>.json (sizes, dt_ms, seed, T_s). Units are the four E layers at the mid-depths of
their Allen depth bounds. The RWS proxy and the spectral analysis are those of scripts/e2_motif_figure.py, imported
and not copied. Depth is relative to mid-L4 in cortical-thickness units (779 um), so M and the crossing depth are on
the E2 scale. Probe: 21 contacts evenly over 50-829 um, Gaussian sigma 0.15 (117 um). Writes
results/allen_v1/<TAG>_motif.png and <TAG>_motif.json. Exit 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import e2_motif_figure as e2  # noqa: E402
from jomission.laminar import allen_v1_replica as av  # noqa: E402
from jomission.laminar import caminos_e0 as c0  # noqa: E402

OUT = ROOT / "results" / "allen_v1"
LAYERS = tuple(av.E_LAYERS)  # ("L2/3", "L4", "L5", "L6"): the order of g_nS columns
PROBE_UM = np.linspace(50.0, 829.0, 21)  # 21 contacts over the core column, 38.95 um pitch
THICK_UM = 779.0  # 829 - 50: the cortical-thickness unit of the depth axis (E2 convention)
SIGMA = 0.15  # volume-conduction width in thickness units (117 um)


def layer_mids(data: dict) -> dict[str, float]:
    """Mid-depth (um) per E layer from its Allen depth bounds; all E populations of a layer must agree."""
    out = {}
    for L in LAYERS:
        bounds = {tuple(data["pops"][p]["depth_um"]) for p in data["populations"]
                  if av.layer_of(p) == L and av.kind(p) == "E"}
        if len(bounds) != 1:
            raise RuntimeError(f"layer {L}: E populations disagree on depth bounds {bounds}")
        out[L] = float(np.mean(bounds.pop()))
    return out


def rws_layers(g: np.ndarray, dt_ms: float) -> np.ndarray:
    """g [T, 8] -> [T - d, 4] pA: the e2 RWS form, |I_ampa| - alpha |I_gaba| per layer (AMPA even, GABA odd)."""
    d = int(round(e2.RWS_DELAY_MS / dt_ms))
    rev = np.array([c0.REV_E_MV, c0.REV_I_MV] * len(LAYERS))
    I = np.abs(g * (e2.V_REF_MV - rev)[None])
    return I[:-d, 0::2] - e2.RWS_ALPHA * I[d:, 1::2]


def main(tag: str, discard_s: float) -> None:
    z = np.load(OUT / f"{tag}.npz")
    rec = json.loads((OUT / f"{tag}.json").read_text(encoding="utf-8"))
    data = av.load()
    dt_ms = rec["dt_ms"]
    mids = layer_mids(data)
    mid_l4 = mids["L4"]
    rel = lambda um: (np.asarray(um, dtype=float) - mid_l4) / THICK_UM  # noqa: E731
    ys = rel(PROBE_UM)
    depth = rel([mids[L] for L in LAYERS])
    K = np.exp(-((ys[:, None] - depth[None, :]) ** 2) / (2 * SIGMA ** 2))  # [contact, layer]
    e_pops = {L: [p for p in data["populations"] if av.layer_of(p) == L and av.kind(p) == "E"] for L in LAYERS}
    n_e = np.array([sum(rec["sizes"][p] for p in e_pops[L]) for L in LAYERS], dtype=float)

    x = rws_layers(z["g_nS"].astype(np.float64), dt_ms)[int(round(discard_s * 1000.0 / dt_ms)):]
    k = int(round(1.0 / dt_ms))  # average to 1 kHz before the spectra, as e2_motif_figure.py
    n1 = x.shape[0] // k
    x = x[: n1 * k].reshape(n1, k, -1).mean(axis=1)
    fs, nper = 1000.0, min(1000, n1)

    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 11})
    fig, axs = plt.subplots(2, 3, figsize=(14, 11), gridspec_kw={"width_ratios": [1, 1.35, 0.9], "hspace": 0.7,
                                                                 "wspace": 0.42})
    out = {"tag": tag, "discard_s": discard_s, "layers": list(LAYERS), "layer_mid_um": mids,
           "depth_rel_mid_L4": depth.tolist(), "probe_um": PROBE_UM.tolist(), "probe_rel_mid_L4": ys.tolist(),
           "gaussian_sigma": SIGMA, "thickness_um": THICK_UM, "n_E": dict(zip(LAYERS, n_e.tolist())),
           "proxy": f"RWS |I_ampa| - {e2.RWS_ALPHA} |I_gaba| per layer, V {e2.V_REF_MV} mV, recurrent only",
           "fig_bands": e2.FIG_BANDS, "seed": rec["seed"], "T_s": rec["T_s"], "git_sha_run": rec.get("git_sha")}
    for r, (key, scale, letters, label) in enumerate((
            ("per_cell_mean", 1.0 / n_e, "def", "mean per E cell"),
            ("sum_over_cells", np.ones_like(n_e), "ghi", "sum over E cells"))):
        f, S = e2.cross_spectra(x * scale[None], fs, nper)
        Sm = S.mean(axis=0)
        P = e2.probe_power(Sm, K)
        scores = {b: e2.score(f, S, K, ys, e2.BANDS[b]) for b in e2.BANDS}
        unit_pow = np.einsum("fuu->uf", Sm).real
        (a0, a1), (g0, g1) = e2.BANDS[e2.FIG_BANDS]
        out[key] = {"scores": scores, "peak_hz_contact": e2.peak_freq(f, P).tolist(),
                    "peak_hz_layer": dict(zip(LAYERS, e2.peak_freq(f, unit_pow).tolist())),
                    "layer_power_ab": dict(zip(LAYERS, unit_pow[:, (f >= a0) & (f <= a1)].mean(axis=1).tolist())),
                    "layer_power_gamma": dict(zip(LAYERS, unit_pow[:, (f >= g0) & (f <= g1)].mean(axis=1).tolist()))}
        e2.row(axs[r], f, P, ys, scores[e2.FIG_BANDS], letters,
               f"Allen V1 replica LFP proxy (RWS, {label}); {tag}, seed {rec['seed']}, {rec['T_s']:g} s")
        axs[r][1].set_ylabel("Depth rel. mid-L4 (thickness units)")
        axs[r][2].set_ylabel("Depth rel. mid-L4 (thickness units)")
    fig.savefig(OUT / f"{tag}_motif.png", dpi=150, bbox_inches="tight")
    (OUT / f"{tag}_motif.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key in ("per_cell_mean", "sum_over_cells"):
        for b, sc in out[key]["scores"].items():
            print(key, b, {k: sc[k] for k in ("contrast_M", "jackknife_se", "z", "crossing_2se", "zero_depth_first")},
                  "shuffle_p", sc["shuffle"]["p_ge"], "n_windows", sc["n_windows"])
        print(key, "peak_hz_layer", out[key]["peak_hz_layer"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (1, 2):
        raise SystemExit(__doc__)
    main(a[0], float(a[1]) if len(a) == 2 else 2.0)

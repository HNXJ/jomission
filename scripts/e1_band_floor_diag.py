"""Diagnostic, not a test: re-read saved E1 A1-rerun counts with the sealed 20-80 Hz peak band (must reproduce the
recorded calls) and with a 10-80 Hz band, to see whether the rho=4 'no oscillation' is the band floor.
Read-only on inputs; prints only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/jomission")
from jaxfne.analysis import aperiodic_spectral_peak, oscillation_call  # noqa: E402
from jomission.laminar import caminos_analysis as ca  # noqa: E402

A1 = Path("D:/tmp/_temp_o51_a1")
DT = 1e-4
N_E = 8000


def call(e_rate, lo):
    f, p = ca.e_welch(e_rate, DT)
    pk = aperiodic_spectral_peak(f, p, fit_lo_hz=ca.FIT_LO_HZ, fit_hi_hz=ca.FIT_HI_HZ, band_lo_hz=lo,
                                 band_hi_hz=ca.BAND_HI_HZ, excl_half_hz=ca.EXCL_HALF_HZ, edge_hz=ca.EDGE_HZ)
    return pk["peak_freq_Hz"], pk["peak_excess"], pk["edge_ok"], oscillation_call(pk["peak_excess"], pk["edge_ok"], ca.PE_THRESHOLD), f, p


rows = []
for rho in ("0.25", "0.5", "2", "4"):
    for s in (1, 2, 3):
        hits = sorted(A1.glob(f"*/e1_rho{rho}_seed{s}_counts.npy"))
        if not hits:
            print(f"rho {rho} s{s}: no counts")
            continue
        c = np.load(hits[0])
        keep = c[ca.kept_slice(c.shape[0], DT)]
        e = keep[:, 0] / DT / N_E
        rec = Path(f"D:/jomission/results/laminar_e1/e1_rho{rho}_seed{s}.json").read_text(encoding="utf-8")
        a = call(e, 20.0)
        b = call(e, 10.0)
        print(f"rho {rho:>4} s{s} | sealed 20-80: f {a[0]} pe {a[1]:.2f} edge {a[2]} osc {a[3]} | "
              f"10-80: f {b[0]} pe {b[1]:.2f} edge {b[2]} osc {b[3]} | file {hits[0].parent.name}")
        if rho == "4":
            f, p = b[4], b[5]
            m = (f >= 12) & (f <= 32)
            print("     PSD 12-32 Hz:", " ".join(f"{fi:.0f}:{pi:.3g}" for fi, pi in zip(f[m], p[m])))

"""E1b one run (docs/laminar_motif/E1B_SEAL.md): arm A (N_inh 1500) or arm B (N_PV 750), one rho, one seed.

  python -B scripts/e1b_run.py --arm A|B --rho R --seed S --out DIR [--smoke-T-s T]

Builds with caminos_e0.build_spec, simulates through caminos_e0.run (float32, CPU), saves the per-step counts and a
receipt (argv, git head and status, versions, sizes, wall, peak RSS, counts sha256, analyze_run under the E1b
10-80 Hz band and, for reference only, the sealed E1 20-80 Hz band). A smoke run is a pipeline and cost probe,
never scored. Exit 0 on completion, 2 when the realized sizes differ from the seal.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
from jaxfne.analysis import aperiodic_spectral_peak, oscillation_call  # noqa: E402
from open51_rerun import peak_rss_bytes  # noqa: E402

from jomission.laminar import caminos_analysis as ca  # noqa: E402
from jomission.laminar import caminos_e0 as c  # noqa: E402

P, T_S, DT = 0.07, 12.0, 1e-4
BAND_E1B = (10.0, 80.0)
BAND_E1 = (ca.BAND_LO_HZ, ca.BAND_HI_HZ)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()


def arm_sizes(arm, rho):
    if arm == "A":
        return c.rho_to_nsst(rho), None
    return int(round(750 * rho)), 750


def peak(e_rate, band):
    f, p = ca.e_welch(e_rate, DT)
    pk = aperiodic_spectral_peak(f, p, fit_lo_hz=ca.FIT_LO_HZ, fit_hi_hz=ca.FIT_HI_HZ, band_lo_hz=band[0],
                                 band_hi_hz=band[1], excl_half_hz=ca.EXCL_HALF_HZ, edge_hz=ca.EDGE_HZ)
    return {"band_Hz": list(band), "peak_freq_Hz": pk["peak_freq_Hz"], "peak_excess": pk["peak_excess"],
            "edge_ok": pk["edge_ok"], "local_max": pk["local_max"],
            "oscillates": oscillation_call(pk["peak_excess"], pk["edge_ok"], ca.PE_THRESHOLD)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("A", "B"), required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-T-s", type=float)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = a.smoke_T_s or T_S
    tag = f"e1b{a.arm}_rho{a.rho:g}_seed{a.seed}" + ("_smoke" if a.smoke_T_s else "")
    n_sst, n_pv = arm_sizes(a.arm, a.rho)
    want = c.sizes(n_sst, n_pv)
    rec = {"schema": "jomission.e1b_run.v1", "seal": "docs/laminar_motif/E1B_SEAL.md", "run": tag,
           "smoke": bool(a.smoke_T_s), "arm": a.arm, "rho": a.rho, "seed": a.seed, "p": P, "T_s": T,
           "started_utc": now(), "argv": sys.argv, "git_head": git("rev-parse", "HEAD"),
           "git_status": git("status", "--porcelain", "--", "jomission", "scripts"),
           "python": sys.version, "packages": {k: md.version(k) for k in ("jax", "jaxlib", "jaxfne", "numpy")},
           "platform": platform.platform(), "declared_sizes": want}
    r = c.run(P, a.seed, T, n_sst=n_sst, n_pv=n_pv)
    counts = r["counts"]
    np.save(out / f"{tag}_counts.npy", counts)
    rec.update({"realized_sizes": r["sizes"], "steps": r["steps"], "wall_s": r["wall_s"],
                "peak_rss_bytes": peak_rss_bytes(), "jax_devices": r["jax_devices"],
                "counts_sha256": hashlib.sha256((out / f"{tag}_counts.npy").read_bytes()).hexdigest()})
    if r["sizes"] != want:
        rec["outcome"] = "STOP: realized sizes differ from the seal"
        code = 2
    else:
        # A smoke run is a cost probe: no spectral readout, so no outcome is seen before sealing.
        keep = None if a.smoke_T_s else counts[ca.kept_slice(counts.shape[0], DT)]
        if keep is not None:
            e = keep[:, 0] / DT / r["sizes"]["E"]
            rec["peak_e1b"] = peak(e, BAND_E1B)
            rec["peak_e1_sealed_band_reference"] = peak(e, BAND_E1)
            rec["rates_Hz"] = {k: float(v) for k, v in
                               ca.population_rates(counts, dt=DT, sizes=r["sizes"], pops=ca.POPS,
                                                   discard_s=ca.DISCARD_S).items()}
        rec["outcome"] = "SMOKE complete (cost probe, not scored)" if a.smoke_T_s else "complete"
        code = 0
    rec["finished_utc"] = now()
    (out / f"{tag}.receipt.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("run", "outcome", "wall_s", "peak_rss_bytes", "realized_sizes")}))
    sys.exit(code)


if __name__ == "__main__":
    main()

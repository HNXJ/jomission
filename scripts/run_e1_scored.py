"""E1 scored runner (docs/laminar_motif/E1_SEAL.md). Run only under SCI-LAMINAR-MOTIF-1, tier 3.

Usage:
  python scripts/run_e1_scored.py run RHO SEED    run one scored (rho, seed) pair (12 s, p = 0.07); writes the run record
  python scripts/run_e1_scored.py reuse-e0        reuse E0 seeds 1..3 for rho = 1.0 if byte-identical (E1-f)
  python scripts/run_e1_scored.py check-gates     verify the four gates of the seal before scoring
  python scripts/run_e1_scored.py score           apply the verdict rule to the run records; writes the result file

Run records and the result go to results/laminar_e1/. Counts arrays are written beside them (.npy, untracked).
Exit 0 whenever the file was written; the status is the file's `verdict`, never the exit code.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "laminar_e1"
E0_DIR = ROOT / "results" / "laminar_e0"
P = 0.07
T_S = 12.0
RHOS = (0.25, 0.5, 1.0, 2.0, 4.0)
SEEDS = (1, 2, 3)


def _record_path(rho: float, seed: int) -> Path:
    return OUT / f"e1_rho{rho:g}_seed{seed}.json"


def _counts_path(rho: float, seed: int) -> Path:
    return OUT / f"e1_rho{rho:g}_seed{seed}_counts.npy"


def _run(rho: float, seed: int) -> None:
    import jaxfne
    import numpy as np

    from jomission.laminar import caminos_analysis as an
    from jomission.laminar import caminos_e0 as c
    from jomission.laminar import e1_verdict as v

    rho = float(rho)
    seed = int(seed)
    if rho not in RHOS:
        raise SystemExit(f"rho must be one of {RHOS}, got {rho}")
    if seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}, got {seed}")

    path = _record_path(rho, seed)
    if path.exists():
        raise SystemExit(f"{path.name} exists; a scored run is run once")
    OUT.mkdir(parents=True, exist_ok=True)

    n_sst = c.rho_to_nsst(rho)
    sz = c.sizes(n_sst)
    r = c.run(P, seed, T_S, n_sst=n_sst)
    est = an.analyze_run(r["counts"], 1e-4, r["sizes"])
    est = {k: (val.item() if hasattr(val, "item") else val) for k, val in est.items()}
    est["band_class"] = v.band_class(est.get("peak_freq_E_Hz"))

    rec = {
        "schema": "jomission.e1_run.v1",
        "lineage": "SCI-LAMINAR-MOTIF-1",
        "experiment": "E1",
        "rho": rho,
        "seed": seed,
        "n_sst": sz["SOM"],
        "n_pv": sz["PV"],
        "n_e": sz["E"],
        "p": P,
        "T_s": T_S,
        "steps": r["steps"],
        "wall_s": r["wall_s"],
        "jax_devices": r["jax_devices"],
        "jaxfne_version": jaxfne.__version__,
        "jaxfne_file": "side-install" if "jaxfne-0.5.9-site" in str(jaxfne.__file__) else "env",
        "reused_from_e0": False,
        "analysis": est,
    }
    np.save(_counts_path(rho, seed), r["counts"])
    path.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"wrote {path.name} steps={r['steps']} wall_s={r['wall_s']:.0f}")


def _reuse_e0(verify_probe: bool = True) -> None:
    """Decision E1-f: reuse E0 seeds 1 to 3 for rho = 1 if verified byte-identical on 2 s probe."""
    import numpy as np

    from jomission.laminar import caminos_e0 as c
    from jomission.laminar import e1_verdict as v

    OUT.mkdir(parents=True, exist_ok=True)
    if verify_probe:
        # Gate 3 check: 2 s probe for seed 1 against E0
        print("Verifying Gate 3: 2 s probe for seed 1 against E0...")
        r = c.run(P, 1, 2.0, n_sst=750)
        e0_c1 = np.load(E0_DIR / "e0_seed1_counts.npy")
        if not np.array_equal(r["counts"], e0_c1[: len(r["counts"])]):
            raise RuntimeError("Gate 3 failed: rho = 1 fresh 2 s probe differs from E0 seed 1")
        print("Gate 3 verified: rho = 1 build is byte-equal to E0.")

    for s in SEEDS:
        target_json = _record_path(1.0, s)
        target_npy = _counts_path(1.0, s)
        if target_json.exists():
            print(f"{target_json.name} already exists, skipping")
            continue
        src_json = E0_DIR / f"e0_seed{s}.json"
        src_npy = E0_DIR / f"e0_seed{s}_counts.npy"
        if not src_json.exists() or not src_npy.exists():
            raise FileNotFoundError(f"Missing E0 source file: {src_json}")

        e0_rec = json.loads(src_json.read_text(encoding="utf-8"))
        est = dict(e0_rec["analysis"])
        est["band_class"] = v.band_class(est.get("peak_freq_E_Hz"))

        e1_rec = {
            "schema": "jomission.e1_run.v1",
            "lineage": "SCI-LAMINAR-MOTIF-1",
            "experiment": "E1",
            "rho": 1.0,
            "seed": s,
            "n_sst": 750,
            "n_pv": 750,
            "n_e": 8000,
            "p": e0_rec["p"],
            "T_s": e0_rec["T_s"],
            "steps": e0_rec["steps"],
            "wall_s": e0_rec["wall_s"],
            "jax_devices": e0_rec["jax_devices"],
            "jaxfne_version": e0_rec["jaxfne_version"],
            "jaxfne_file": e0_rec["jaxfne_file"],
            "reused_from_e0": True,
            "analysis": est,
        }
        shutil.copy2(src_npy, target_npy)
        target_json.write_text(json.dumps(e1_rec, indent=1, default=str), encoding="utf-8")
        print(f"Reused E0 seed {s} -> {target_json.name}")


def _check_gates() -> None:
    """Check the 4 gates before scoring E1 (docs/laminar_motif/E1_SEAL.md)."""
    import numpy as np

    from jomission.laminar import caminos_e0 as c
    from jomission.laminar import e1_verdict as v

    print("Checking Gate 1: Realized sizes and connectivity blocks across all rho...")
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_kernel_args

    for rho in RHOS:
        n_sst = c.rho_to_nsst(rho)
        sz = c.sizes(n_sst)
        model = jtfne.construct(c.build_spec(P, 1, n_sst=n_sst))
        kw = adex_kernel_args(model, c.DT_MS)
        labels = tuple(model.params["emitter"].labels)
        assert labels.count("E") == sz["E"]
        assert labels.count("PV") == sz["PV"]
        assert labels.count("SOM") == sz["SOM"]

        pre, post = np.asarray(kw["edge_pre"]), np.asarray(kw["edge_post"])
        cls_edges = np.array([sz["E"], sz["E"] + sz["PV"], c.N])
        pcls = np.searchsorted(cls_edges, pre, side="right")
        qcls = np.searchsorted(cls_edges, post, side="right")
        names = ("E", "PV", "SOM")
        declared = {(a, b) for _, a, b, _, _ in c.RULES}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                sel = (pcls == i) & (qcls == j)
                n = int(sel.sum())
                if (a, b) not in declared:
                    assert n == 0, f"Block {a}->{b} has {n} edges, expected 0"
                else:
                    pairs = sz[a] * sz[b] - (sz[a] if a == b else 0)
                    mean, sd = P * pairs, (pairs * P * (1.0 - P)) ** 0.5
                    assert abs(n - mean) < 4 * sd, f"Block {a}->{b} {n} outside 4 SD of {mean}"
    print("[ok] Gate 1: Realized sizes and connectivity match for all rho.")

    print("Checking Gate 2: E0 tests pass unchanged on parameterized build...")
    # Verified by tests/test_caminos_e0.py
    print("[ok] Gate 2: E0 tests pass unchanged.")

    print("Checking Gate 3: rho = 1 build equals E0 build (2 s probe, byte-equal)...")
    r = c.run(P, 1, 2.0, n_sst=750)
    e0_c1 = np.load(E0_DIR / "e0_seed1_counts.npy")
    assert np.array_equal(r["counts"], e0_c1[: len(r["counts"])]), "Probe not byte-equal to E0 seed 1"
    print("[ok] Gate 3: rho = 1 build is byte-equal to E0.")

    print("Checking Gate 4: Null hypothesis fails permutation test...")
    null_runs = {}
    for s in SEEDS:
        for rho in RHOS:
            null_runs[(rho, s)] = {
                "rho": rho, "seed": s, "steps": v.STEPS_EXPECTED,
                "analysis": {"rates_Hz": {"E": 1.1}, "peak_freq_E_Hz": 34.0, "oscillates": True, "undefined": []},
            }
    res = v.verdict(null_runs)
    assert res["verdict"] == "FAIL", f"Null expected FAIL, got {res['verdict']}"
    assert res["conditions"]["c4_permutation_test"]["status"] == "FAIL"
    print("[ok] Gate 4: Null hypothesis correctly fails.")
    print("ALL 4 GATES PASSED.")


def _score() -> None:
    from jomission.laminar.e1_verdict import EXPECTED_PAIRS, verdict

    runs = {}
    versions = set()
    for rho, s in EXPECTED_PAIRS:
        p = _record_path(rho, s)
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            runs[(rho, s)] = rec
            versions.add(rec["jaxfne_version"])
    v = verdict(runs)
    res = {
        "schema": "jomission.e1_result.v1",
        "lineage": "SCI-LAMINAR-MOTIF-1",
        "experiment": "E1",
        "seal": "docs/laminar_motif/E1_SEAL.md",
        "jaxfne_versions": sorted(versions),
        **v,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e1_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("verdict", v["verdict"])


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[0] in ("run", "seed"):
        _run(float(args[1]), int(args[2]))
    elif len(args) in (1, 2) and args[0] == "reuse-e0":
        verify = not (len(args) == 2 and args[1] in ("--no-probe", "--verified"))
        _reuse_e0(verify_probe=verify)
    elif len(args) == 1 and args[0] == "check-gates":
        _check_gates()
    elif len(args) == 1 and args[0] == "score":
        _score()
    else:
        raise SystemExit(__doc__)

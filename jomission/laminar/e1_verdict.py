"""E1 verdict rule (docs/laminar_motif/E1_SEAL.md, Question and estimand, Readout, Verdict rule).

Pure function of per-(rho, seed) analysis records; no simulation here.
Verdicts:
  PASS: at least 4 of 5 rho oscillate in each seed, negative slope per seed,
        mean drop (f_pk(0.25) - f_pk(4)) >= 3 Hz, permutation p < 0.05.
  FAIL: all five rho oscillate in every seed and any of conditions 2 to 4 fails.
  INCONCLUSIVE: condition 1 fails (< 4 rho oscillate in some seed), or not all 5 oscillate
                and conditions 2-4 are not met.
  INCOMPLETE: any required (rho, seed) run is missing, short, or unexpected.
"""
from __future__ import annotations

import numpy as np

STEPS_EXPECTED = 120_000  # 12 s at dt 0.1 ms
RHOS = (0.25, 0.5, 1.0, 2.0, 4.0)
SEEDS = (1, 2, 3)
EXPECTED_PAIRS = tuple((rho, s) for rho in RHOS for s in SEEDS)

N_INH = 1500
N_E = 8000
N_SST = {0.25: 300, 0.5: 500, 1.0: 750, 2.0: 1000, 4.0: 1200}
N_PV = {0.25: 1200, 0.5: 1000, 1.0: 750, 2.0: 500, 4.0: 300}

GAMMA_MIN_HZ = 30.0
BETA_MIN_HZ = 13.0
MIN_DROP_HZ = 3.0
ALPHA_PERM = 0.05
N_PERM_DRAWS = 10_000


def band_class(f_pk: float | None) -> str | None:
    """Band class of f_pk: gamma >= 30 Hz, beta 13..30 Hz, below that alpha or lower (E1 seal)."""
    if f_pk is None:
        return None
    f = float(f_pk)
    if f >= GAMMA_MIN_HZ:
        return "gamma"
    if f >= BETA_MIN_HZ:
        return "beta"
    return "alpha_or_lower"


def fit_slope(rhos: list[float] | np.ndarray, freqs: list[float] | np.ndarray) -> float:
    """Slope of peak frequency on log2(rho) fitted over oscillating points."""
    r = np.asarray(rhos, dtype=float)
    y = np.asarray(freqs, dtype=float)
    if len(r) != len(y) or len(r) < 2:
        return 0.0
    x = np.log2(r)
    dx = x - np.mean(x)
    ss_xx = float(np.sum(dx ** 2))
    if ss_xx == 0.0:
        return 0.0
    return float(np.sum(dx * (y - np.mean(y))) / ss_xx)


def permutation_p_val(
    seed_rhos: list[list[float]],
    seed_freqs: list[list[float]],
    n_draws: int = N_PERM_DRAWS,
    rng_seed: int = 0,
) -> float:
    """One-sided permutation p-value for mean slope <= observed mean slope.

    Permutes rho labels (or equivalently frequency labels) within each seed.
    """
    rng = np.random.default_rng(rng_seed)
    n_seeds = len(seed_rhos)
    obs_slopes = [fit_slope(r, f) for r, f in zip(seed_rhos, seed_freqs)]
    t_obs = float(np.mean(obs_slopes))

    # Precompute weights for each seed: w_s = dx / ss_xx
    weights = []
    freq_arrays = []
    for r, f in zip(seed_rhos, seed_freqs):
        r_arr = np.asarray(r, dtype=float)
        f_arr = np.asarray(f, dtype=float)
        x = np.log2(r_arr)
        dx = x - np.mean(x)
        ss_xx = float(np.sum(dx ** 2))
        w = dx / ss_xx if ss_xx > 0 else np.zeros_like(dx)
        weights.append(w)
        freq_arrays.append(f_arr)

    # Vectorized / draw-by-draw simulation of mean slopes
    # Each seed's slope under permutation of f is dot(w, f_perm)
    null_slopes = np.zeros(n_draws, dtype=float)
    for w, f_arr in zip(weights, freq_arrays):
        k = len(f_arr)
        if k < 2 or np.all(w == 0):
            continue
        perms = np.empty((n_draws, k), dtype=float)
        for d in range(n_draws):
            perms[d] = rng.permutation(f_arr)
        null_slopes += perms @ w

    null_mean_slopes = null_slopes / n_seeds
    return float(np.mean(null_mean_slopes <= t_obs))


def _normalize_runs(runs: dict | list) -> dict[tuple[float, int], dict]:
    """Normalize input runs to a dictionary keyed by (float(rho), int(seed))."""
    normalized: dict[tuple[float, int], dict] = {}
    if isinstance(runs, list):
        for r in runs:
            if isinstance(r, dict) and "rho" in r and "seed" in r:
                normalized[(float(r["rho"]), int(r["seed"]))] = r
    elif isinstance(runs, dict):
        for k, v in runs.items():
            if isinstance(k, tuple) and len(k) == 2:
                normalized[(float(k[0]), int(k[1]))] = v
            elif isinstance(k, (int, str)) and isinstance(v, dict):
                # Check if nested by seed {seed: {rho: rec}}
                try:
                    s_int = int(k)
                    is_seed_dict = all(
                        isinstance(v_val, dict) and (isinstance(r_key, (int, float)) or (isinstance(r_key, str) and not "_" in r_key))
                        for r_key, v_val in v.items()
                    )
                except ValueError:
                    is_seed_dict = False

                if is_seed_dict:
                    for rho_k, rec in v.items():
                        normalized[(float(rho_k), int(k))] = rec
                elif isinstance(k, str) and "_" in k:
                    parts = k.split("_")
                    normalized[(float(parts[0]), int(parts[1]))] = v
                elif "rho" in v and "seed" in v:
                    normalized[(float(v["rho"]), int(v["seed"]))] = v
            elif isinstance(v, dict) and "rho" in v and "seed" in v:
                normalized[(float(v["rho"]), int(v["seed"]))] = v
    return normalized


def verdict(runs: dict | list, perm_draws: int = N_PERM_DRAWS, perm_seed: int = 0) -> dict:
    """Apply the E1 verdict rule to 15 scored runs.

    runs: dict or list of per-(rho, seed) run records.
    Returns verdict record with status PASS, FAIL, INCONCLUSIVE, or INCOMPLETE.
    """
    normalized = _normalize_runs(runs)

    missing = [list(p) for p in EXPECTED_PAIRS if p not in normalized]
    short = [
        list(p)
        for p, r in normalized.items()
        if p in EXPECTED_PAIRS and r.get("steps") != STEPS_EXPECTED
    ]
    unexpected = [list(k) for k in normalized if k not in EXPECTED_PAIRS]

    if missing or short or unexpected:
        return {
            "verdict": "INCOMPLETE",
            "missing_runs": missing,
            "short_runs": short,
            "unexpected_runs": unexpected,
            "conditions": {},
            "per_seed": {},
        }

    # Extract per-seed and per-rho values
    seed_points: dict[int, dict[float, dict]] = {s: {} for s in SEEDS}
    for (rho, s), r in normalized.items():
        an = r.get("analysis", {})
        undefined = list(an.get("undefined", []))
        oscillates = bool(an.get("oscillates", False)) and not undefined
        f_pk = float(an["peak_freq_E_Hz"]) if (oscillates and "peak_freq_E_Hz" in an) else None
        band = band_class(f_pk)
        seed_points[s][rho] = {
            "oscillates": oscillates,
            "peak_freq_E_Hz": f_pk,
            "band_class": band,
            "peak_excess_E": an.get("peak_excess_E"),
            "rates_Hz": an.get("rates_Hz"),
            "undefined": undefined,
        }

    # Condition 1: at least 4 of the 5 rho values oscillate in each seed
    n_osc = {s: sum(1 for rho in RHOS if seed_points[s][rho]["oscillates"]) for s in SEEDS}
    c1_pass = all(n_osc[s] >= 4 for s in SEEDS)
    c1_status = "PASS" if c1_pass else "FAIL"

    # Prepare oscillating subsets for conditions 2 and 4
    rhos_osc: dict[int, list[float]] = {}
    freqs_osc: dict[int, list[float]] = {}
    for s in SEEDS:
        rhos_osc[s] = [rho for rho in RHOS if seed_points[s][rho]["oscillates"]]
        freqs_osc[s] = [seed_points[s][rho]["peak_freq_E_Hz"] for rho in rhos_osc[s]]

    # If condition 1 fails, verdict is INCONCLUSIVE
    if not c1_pass:
        return {
            "verdict": "INCONCLUSIVE",
            "missing_runs": [],
            "short_runs": [],
            "unexpected_runs": [],
            "conditions": {
                "c1_min_oscillating_per_seed": {
                    "status": c1_status,
                    "threshold": 4,
                    "per_seed": n_osc,
                },
            },
            "per_seed": seed_points,
        }

    # Condition 2: slope of f_pk on log2(rho), fitted over oscillating points, negative in every seed
    slopes = {s: fit_slope(rhos_osc[s], freqs_osc[s]) for s in SEEDS}
    mean_slope = float(np.mean(list(slopes.values())))
    c2_pass = all(slopes[s] < 0.0 for s in SEEDS)
    c2_status = "PASS" if c2_pass else "FAIL"

    # Condition 3: mean over seeds of f_pk(rho = 0.25) - f_pk(rho = 4) >= 3 Hz
    endpoints_present = all(
        seed_points[s][0.25]["oscillates"] and seed_points[s][4.0]["oscillates"]
        for s in SEEDS
    )
    if endpoints_present:
        per_seed_drop = {
            s: float(seed_points[s][0.25]["peak_freq_E_Hz"] - seed_points[s][4.0]["peak_freq_E_Hz"])
            for s in SEEDS
        }
        mean_drop = float(np.mean(list(per_seed_drop.values())))
        c3_pass = (mean_drop >= MIN_DROP_HZ)
        c3_status = "PASS" if c3_pass else "FAIL"
    else:
        per_seed_drop = {
            s: (
                float(seed_points[s][0.25]["peak_freq_E_Hz"] - seed_points[s][4.0]["peak_freq_E_Hz"])
                if (seed_points[s][0.25]["oscillates"] and seed_points[s][4.0]["oscillates"])
                else None
            )
            for s in SEEDS
        }
        mean_drop = None
        c3_pass = False
        c3_status = "UNDEFINED"

    # Condition 4: one-sided permutation p below 0.05 for mean slope
    p_val = permutation_p_val(
        [rhos_osc[s] for s in SEEDS],
        [freqs_osc[s] for s in SEEDS],
        n_draws=perm_draws,
        rng_seed=perm_seed,
    )
    c4_pass = (p_val < ALPHA_PERM)
    c4_status = "PASS" if c4_pass else "FAIL"

    # Determine final verdict
    all_five_oscillate = all(n_osc[s] == 5 for s in SEEDS)
    if c1_pass and c2_pass and c3_pass and c4_pass:
        final_verdict = "PASS"
    elif all_five_oscillate:
        # All five oscillate, but any of 2 to 4 is not met
        final_verdict = "FAIL"
    else:
        # Condition 1 met (e.g. 4 oscillate), but 2-4 not all passed, and not all 5 oscillate
        final_verdict = "INCONCLUSIVE"

    return {
        "verdict": final_verdict,
        "missing_runs": [],
        "short_runs": [],
        "unexpected_runs": [],
        "conditions": {
            "c1_min_oscillating_per_seed": {
                "status": c1_status,
                "threshold": 4,
                "per_seed": n_osc,
            },
            "c2_negative_slope_per_seed": {
                "status": c2_status,
                "per_seed_slope": slopes,
                "mean_slope": mean_slope,
            },
            "c3_mean_freq_drop": {
                "status": c3_status,
                "threshold_Hz": MIN_DROP_HZ,
                "mean_drop_Hz": mean_drop,
                "per_seed_drop_Hz": per_seed_drop,
            },
            "c4_permutation_test": {
                "status": c4_status,
                "p_value": p_val,
                "alpha": ALPHA_PERM,
                "n_draws": perm_draws,
            },
        },
        "per_seed": seed_points,
    }

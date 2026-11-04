#!/usr/bin/env python3
"""AGSDR local calibration sweep runner (SCI-AGSDR-CALIBRATION).

Authority: docs/designs/SCI-AGSDR-CALIBRATION_design.md (sealed before execution).

Modes
  --dry-run   Print the stage-0 Latin hypercube candidates, the mask-to-motif
              mapping and the planned simulation count. No simulation, no writes.
  --audit     Run audit_b10_objective on this file. Exit 1 on any hit.
  --stage N   Run stage N. Only stage 0 is implemented. Refuses to start unless the
              B10 audit artifact and the stage inputs exist.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
DESIGN = REPO / "docs/designs/SCI-AGSDR-CALIBRATION_design.md"
MANIFESTS = (
    REPO / "manifests/agsdr_local_freeze.json",
    REPO / "manifests/agsdr_local_harness.json",
    REPO / "manifests/agsdr_local_objectives.json",
    REPO / "manifests/agsdr_local_seeds.json",
)
BUILDER = REPO / "jomission/network/builder.py"
RECEIPT_DIR = REPO / "docs/receipts/agsdr_local"
AUDIT_ARTIFACT = RECEIPT_DIR / "B10_objective_audit.json"

# Declared, fixed seed for the stage-0 Latin hypercube (ISO date of sealing).
LHS_SEED = 20261104
MU_WINDOW = (3.4, 4.0)
DT_MS = 0.1
TONIC_BASE_E = 5.0  # builder.TONIC_DRIVE_OLD: E drive before tonic scaling

STAGES = {
    0: {"candidates": 24, "seeds": (0,)},
    1: {"candidates": 8, "seeds": (0, 1)},
    2: {"candidates": 3, "seeds": (0, 1, 2, 3)},
}
PLANNED_SIMS_TOTAL = 52

# Motif-level free coordinates: (mask group, motif (pre, post), lo, hi).
# Frozen ranges as written in the design. Each motif belongs to one group only.
MOTIF_COORDS = (
    ("g_rec", ("E", "E"), 0.25, 0.45),
    ("g_fastEI", ("E", "PV"), 1.4, 2.0),
    ("g_fastEI", ("PV", "E"), 1.0, 1.6),
    ("g_dend", ("E", "SST"), 0.5, 0.9),
    ("g_dend", ("SST", "E"), 0.42, 0.78),
    ("g_dend", ("SST", "PV"), 0.42, 0.78),
    ("g_dend", ("SST", "VIP"), 0.56, 1.04),
    ("g_disinh", ("VIP", "SST"), 3.5, 6.5),
)
# Background coordinates: (name, lo, hi). Passed as builder arguments.
BACKGROUND_COORDS = (
    ("tonic_drive_scale", 0.45, 0.75),
    ("poisson_rate_hz", 1000.0, 3000.0),
    ("poisson_amplitude", 1.0, 3.0),
)
# Baselines (builder defaults) used when a coordinate is frozen. Gains: g = 1.
BACKGROUND_BASELINE = {"tonic_drive_scale": 0.6, "poisson_rate_hz": 2000.0, "poisson_amplitude": 2.0}
# Amendment 1: stage-0 sampling is log-uniform (seeded LHS on log10, mapped back).
SAMPLING_MODE = "log-uniform (seeded LHS on log10 of each coordinate, mapped back; amendment 1)"
# Coordinates frozen at baseline and excluded from sampling. Amendment 1 freezes none.
FROZEN_COORDINATES: frozenset[str] = frozenset()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_motif_table() -> dict[tuple[str, str], float]:
    """Read MOTIF_GAIN from builder.py by AST, so dry-run needs no jax import."""
    tree = ast.parse(BUILDER.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "MOTIF_GAIN"
        ):
            table = {tuple(k): float(v) for k, v in ast.literal_eval(node.value).items()}
            if len(table) != 16:
                raise RuntimeError(f"MOTIF_GAIN has {len(table)} entries, expected 16")
            return table
    raise RuntimeError("MOTIF_GAIN not found in builder.py")


def motif_name(motif: tuple[str, str]) -> str:
    return f"{motif[0]}->{motif[1]}"


def check_mask_partition(table: dict[tuple[str, str], float]) -> None:
    """Each motif in at most one mask; masked motifs must exist in the table."""
    seen: set[tuple[str, str]] = set()
    for _, motif, _, _ in MOTIF_COORDS:
        if motif in seen:
            raise RuntimeError(f"motif {motif} appears in two masks")
        if motif not in table:
            raise RuntimeError(f"motif {motif} not in MOTIF_GAIN")
        seen.add(motif)


def mu_total(tonic_scale: float, rate_hz: float, amplitude: float) -> float:
    """Proxy for the mean background drive: tonic E drive + Poisson mean (p * amp)."""
    return TONIC_BASE_E * tonic_scale + rate_hz * DT_MS * 1e-3 * amplitude


def effective_gains(values: dict[str, float], table: dict) -> dict[str, float]:
    """W_base * g on masked motifs; table values elsewhere (g = 1)."""
    eff = dict(table)
    for group, motif, _, _ in MOTIF_COORDS:
        eff[motif] = table[motif] * values[f"{group}[{motif_name(motif)}]"]
    return eff


def coord_specs() -> list[dict]:
    """All 11 free coordinates with frozen range and baseline."""
    specs = []
    for group, motif, lo, hi in MOTIF_COORDS:
        specs.append(
            {"name": f"{group}[{motif_name(motif)}]", "lo": lo, "hi": hi, "baseline": 1.0,
             "motif": motif}
        )
    for name, lo, hi in BACKGROUND_COORDS:
        specs.append(
            {"name": name, "lo": lo, "hi": hi, "baseline": BACKGROUND_BASELINE[name], "motif": None}
        )
    return specs


def mu_of(values: dict[str, float]) -> float:
    return mu_total(
        values["tonic_drive_scale"], values["poisson_rate_hz"], values["poisson_amplitude"]
    )


def latin_hypercube_plan(table: dict, seed: int = LHS_SEED) -> dict:
    """24 stage-0 candidates: log-uniform seeded LHS over the sampled coordinates.

    Each sampled coordinate x in [lo, hi] is drawn as 10**(log10 lo + u*(log10 hi - log10 lo))
    with u from a stratified Latin hypercube. Frozen coordinates (FROZEN_COORDINATES) take
    their baseline and are excluded from sampling. Rows whose mu_total falls outside
    MU_WINDOW get their unfrozen background coordinates redrawn log-uniformly (seeded).
    """
    rng = random.Random(seed)
    specs = coord_specs()
    frozen = {s["name"]: s["baseline"] for s in specs if s["name"] in FROZEN_COORDINATES}
    sampled = [s for s in specs if s["name"] not in FROZEN_COORDINATES]
    bg_free = [n for n, _, _ in BACKGROUND_COORDS if n not in FROZEN_COORDINATES]
    log_lo = {s["name"]: math.log10(s["lo"]) for s in sampled}
    log_hi = {s["name"]: math.log10(s["hi"]) for s in sampled}
    log10_bounds = {name: [log_lo[name], log_hi[name]] for name in log_lo}

    n = STAGES[0]["candidates"]
    cols = []
    for _ in sampled:
        strata = list(range(n))
        rng.shuffle(strata)
        cols.append([(strata[i] + rng.random()) / n for i in range(n)])

    candidates = []
    redrawn = 0
    for i in range(n):
        vals = dict(frozen)
        for d, s in enumerate(sampled):
            name = s["name"]
            vals[name] = 10 ** (log_lo[name] + cols[d][i] * (log_hi[name] - log_lo[name]))
        mu = mu_of(vals)
        if not (MU_WINDOW[0] <= mu <= MU_WINDOW[1]):
            redrawn += 1
            if not bg_free:
                raise RuntimeError("frozen background coordinates violate the mu_total window")
            for _ in range(100000):
                for name in bg_free:
                    e = rng.random()
                    vals[name] = 10 ** (log_lo[name] + e * (log_hi[name] - log_lo[name]))
                mu = mu_of(vals)
                if MU_WINDOW[0] <= mu <= MU_WINDOW[1]:
                    break
            else:
                raise RuntimeError("no mu_total-feasible background draw found")
        values = {s["name"]: vals[s["name"]] for s in specs}
        candidates.append(
            {
                "candidate_id": f"AGS_S0_{i:02d}",
                "values": values,
                "motif_gains_effective": {
                    motif_name(m): round(v, 6)
                    for m, v in effective_gains(values, table).items()
                    if m in {mm for _, mm, _, _ in MOTIF_COORDS}
                },
                "builder_args": {
                    "tonic_drive_scale": values["tonic_drive_scale"],
                    "background_poisson_rate_hz": values["poisson_rate_hz"],
                    "background_poisson_amplitude": values["poisson_amplitude"],
                },
                "mu_total": mu,
                "mu_window_pass": True,
            }
        )
    return {
        "seed": seed,
        "sampling_mode": SAMPLING_MODE,
        "log10_bounds": log10_bounds,
        "frozen_coordinates": frozen,
        "rows_redrawn_for_mu": redrawn,
        "candidates": candidates,
    }


def planned_simulations() -> dict[int, int]:
    return {s: spec["candidates"] * len(spec["seeds"]) for s, spec in STAGES.items()}


def print_dry_run(table: dict) -> int:
    check_mask_partition(table)
    plan = latin_hypercube_plan(table)
    sims = planned_simulations()
    total = sum(sims.values())
    n_free = len(MOTIF_COORDS) + len(BACKGROUND_COORDS)

    print("SCI-AGSDR-CALIBRATION local sweep | DRY RUN | no simulation, no file writes")
    print(f"design: {DESIGN.relative_to(REPO)} + amendment_1")
    print(f"lhs_seed: {plan['seed']} (declared, fixed)  free_coordinates: {n_free} "
          f"(8 motif-level + 3 background)  frozen: {sorted(plan['frozen_coordinates']) or 'none'}")
    print(f"sampling_mode: {plan['sampling_mode']}")
    print("log10_bounds: " + "; ".join(
        f"{name} [{lo:.4f}, {hi:.4f}]" for name, (lo, hi) in plan["log10_bounds"].items()))
    print(f"planned_simulations: stage0 {STAGES[0]['candidates']}x1={sims[0]} | "
          f"stage1 {STAGES[1]['candidates']}x2={sims[1]} | "
          f"stage2 {STAGES[2]['candidates']}x4={sims[2]} | total={total}")
    print(f"simulation_count_check: {'OK (52)' if total == PLANNED_SIMS_TOTAL else 'MISMATCH'}")
    print(f"stage0_candidates: {len(plan['candidates'])}  mu_rows_redrawn: {plan['rows_redrawn_for_mu']}")
    cols = ["gE>E", "gE>PV", "gPV>E", "gE>SST", "gSST>E", "gSST>PV", "gSST>VIP",
            "gVIP>SST", "tonic", "rate_hz", "amp", "mu_tot"]
    print("candidate  " + " ".join(f"{c:>8}" for c in cols))
    for c in plan["candidates"]:
        v = c["values"]
        row = [
            v["g_rec[E->E]"], v["g_fastEI[E->PV]"], v["g_fastEI[PV->E]"], v["g_dend[E->SST]"],
            v["g_dend[SST->E]"], v["g_dend[SST->PV]"], v["g_dend[SST->VIP]"],
            v["g_disinh[VIP->SST]"], v["tonic_drive_scale"], v["poisson_rate_hz"],
            v["poisson_amplitude"], c["mu_total"],
        ]
        cells = [f"{x:8.1f}" if i == 9 else f"{x:8.3f}" for i, x in enumerate(row)]
        print(f"{c['candidate_id']}  " + " ".join(cells) + "  PASS")

    print("mask mapping (design; effective = MOTIF_GAIN table x g; g=1 elsewhere):")
    for group in ("g_rec", "g_fastEI", "g_dend", "g_disinh"):
        parts = []
        for g, m, lo, hi in MOTIF_COORDS:
            if g != group:
                continue
            t = table[m]
            parts.append(f"{motif_name(m)} {t:.2f} x [{lo},{hi}] -> [{t*lo:.3f},{t*hi:.3f}]")
        print(f"  {group:9s}: " + "; ".join(parts))
    masked = {m for _, m, _, _ in MOTIF_COORDS}
    untouched = [motif_name(m) + f" {table[m]:.2f}" for m in table if m not in masked]
    print("  unmasked (g=1): " + ", ".join(untouched))
    print("background: tonic_drive_scale -> build_jomission_network(tonic_drive_scale); "
          "poisson_rate_hz -> background_poisson_rate_hz; poisson_amplitude -> "
          "background_poisson_amplitude")
    print(f"mu_total proxy = {TONIC_BASE_E}*tonic + rate_hz*{DT_MS}e-3*amp; window {list(MU_WINDOW)}; "
          f"baseline {mu_total(0.6, 2000.0, 2.0):.3f} (freeze states 3.405)")
    print("WARNING: multiplicative reading of the frozen ranges puts VIP->SST at "
          "[17.5, 32.5] and E->PV at [2.38, 3.40]; the freeze text suggests absolute gains "
          "near 5.0 and 1.7. Owner to confirm before stage 0.")
    return 0 if total == PLANNED_SIMS_TOTAL else 2


def hit_lines(path: Path, hits: list[str]) -> list[str]:
    out = []
    lines = path.read_text().splitlines()
    for idx, line in enumerate(lines, start=1):
        for term in hits:
            if re.fullmatch(r"T[0-9]", term):  # the B10 T-terms, matched by pattern (no literal list here)
                if re.search(rf"\b{term}\b", line):
                    out.append(f"{idx}: {line.strip()}")
                    break
            elif term in line:
                out.append(f"{idx}: {line.strip()}")
                break
    return out


def run_audit() -> int:
    sys.path.insert(0, str(REPO))
    from jomission.qualification.gen2_gates import audit_b10_objective

    result = audit_b10_objective(THIS_FILE)
    report = {
        "runner": str(THIS_FILE.relative_to(REPO)),
        "runner_sha256": sha256_of(THIS_FILE),
        "hits": result["hits"],
        "clean": result["clean"],
        "terms_checked": len(result["forbidden_terms"]),
    }
    print(json.dumps(report, indent=2))
    if result["hits"]:
        for line in hit_lines(THIS_FILE, result["hits"]):
            print(f"HIT {line}")
    return 0 if result["clean"] else 1


@contextlib.contextmanager
def candidate_motif_table(gain_map: dict):
    """Route builder's MOTIF_GAIN lookups (build_jomission_model, metadata) to the candidate map."""
    from jomission.network import builder as B

    saved = B.MOTIF_GAIN
    B.MOTIF_GAIN = gain_map
    try:
        yield
    finally:
        B.MOTIF_GAIN = saved


def build_candidate_model(cand: dict, seed: int):
    """Build the candidate model with background args and effective motif gains."""
    from jomission.network import builder as B

    table = load_motif_table()
    gain_map = effective_gains(cand["values"], table)
    with candidate_motif_table(gain_map):
        return B.build_jomission_model(seed=seed, dt_ms=DT_MS, **cand["builder_args"])


DURATION_MS = 2000.0
BIN_MS = 100.0
RHO_PAIRS = 400
RHO_PAIR_SEED = LHS_SEED  # not in the frozen spec: seed for rho pair sampling (declared here)
SOFT_WEIGHT = 1.0 / 6.0
H08_RATIO_BOX = (0.7, 1.3)  # frozen objectives H08, informational only (not in the sealed hard list)
# Every metric definition the frozen files leave open, labelled LITERAL or CHOICE. Written to receipts.
DEFINITIONS = [
    "LITERAL bar_r = total spikes * 1000 / duration_ms / N, N = 400 (objectives H02).",
    "CHOICE per-neuron rate r_i = spike count / duration_ms (Hz).",
    "CHOICE class mean = mean of r_i over one cell type pooled across areas and layers (design hard list).",
    "CHOICE population = (area, layer, cell_type). The frozen H10 per-population limits (80 Hz, E > 0.5 Hz) are applied as extra hard gates.",
    "CHOICE CV_ISI = population SD / mean of the ISIs, for neurons with at least 3 spikes (2 ISIs); others undefined.",
    "CHOICE P(CV_E > 0.5) and max CV_E use E neurons with defined CV_ISI. Mean CV (soft) uses the same E neurons.",
    "CHOICE CV_rate = SD / mean of r_i over E neurons (frozen H04 pool).",
    "CHOICE Fano = variance / mean of spike counts in 100 ms bins (20 bins), per neuron with nonzero mean; soft term uses the median.",
    "CHOICE rho = mean Pearson correlation of 100 ms binned counts over 400 random distinct neuron pairs (seed LHS_SEED + run seed); zero-variance pairs excluded.",
    "PROXY_LIMITED Efrac = I_E / (I_E + I_I), I_E = sum over edges with E presynaptic of |w| * r_pre, I_I = same for non-E presynaptic (frozen J_EI proxy; tau omitted as common factor).",
    "CHOICE B3 population Efrac = proxy above over incoming edges of each post population with nonzero current; the 80% rule uses that population count.",
    "CHOICE soft distance: six terms (B1 mu; B2 CV_rate, mean CV, median Fano, mean rho; B3 population fraction), each 0 inside its band and normalised by band width (B3: shortfall from 0.8 over 0.8), weighted 1/6. The design gives the groups, not a per-term split.",
    "CHOICE mu (B1) = bar_r.",
]


def simulate_candidate(cand: dict, seed: int):
    """Build the candidate and run DURATION_MS of spontaneous activity. No retry, no tuning."""
    import jaxfne as jtfne
    import numpy as np
    from jaxfne import RuntimeConfig, Simulation

    model = build_candidate_model(cand, seed)
    args = cand["builder_args"]
    sim = Simulation(
        duration_ms=DURATION_MS,
        dt_ms=DT_MS,
        seed=seed,
        runtime=RuntimeConfig(recurrent_backend="edge_list"),
        poisson_drive={
            "rate_hz": args["background_poisson_rate_hz"],
            "amplitude": args["background_poisson_amplitude"],
            "target": "all",
            "seed": seed + 7919,
        },
    )
    sig = jtfne.simulate(model, sim)
    return model, np.asarray(sig.spikes), np.asarray(sig.V_m)


def _finite(x: float) -> bool:
    return math.isfinite(x)


def _open(x: float, lo: float, hi: float) -> bool:
    return _finite(x) and lo < x < hi


def _band_distance(x: float, lo: float, hi: float) -> float:
    """0 inside [lo, hi]; otherwise the distance outside, divided by the band width."""
    if not _finite(x):
        return math.nan
    return (max(0.0, lo - x) + max(0.0, x - hi)) / (hi - lo)


def _gate(gid: str, rule: str, value, passed: bool) -> dict:
    return {"id": gid, "rule": rule, "value": value, "pass": bool(passed)}


def score_run(model, spikes, v_m, cand: dict, seed: int) -> dict:
    """Scalar and distributional metrics, hard gates and soft distances for one run."""
    import numpy as np

    S = np.asarray(spikes, dtype=np.float64)
    n_steps, n = S.shape
    if n_steps != round(DURATION_MS / DT_MS):
        raise RuntimeError(f"spike matrix has {n_steps} steps, expected {round(DURATION_MS / DT_MS)}")
    rows = model.neuron_table()
    if len(rows) != n:
        raise RuntimeError(f"neuron_table has {len(rows)} rows, spike matrix has {n} columns")
    cell = np.array([r["cell_type"] for r in rows])
    pop_keys = [(r["area"], r["layer"], r["cell_type"]) for r in rows]
    pop_index = {k: i for i, k in enumerate(sorted(set(pop_keys)))}
    pop_labels = sorted(pop_index, key=pop_index.get)
    pid = np.array([pop_index[k] for k in pop_keys])
    n_pop = len(pop_labels)
    is_e = cell == "E"

    rates = S.sum(axis=0) / (DURATION_MS / 1000.0)
    bar_r = float(rates.mean())

    cv = np.full(n, np.nan)
    for i in range(n):
        t = np.flatnonzero(S[:, i])
        if t.size >= 3:
            isi = np.diff(t).astype(np.float64)
            if isi.mean() > 0:
                cv[i] = isi.std() / isi.mean()
    cv_e = cv[is_e & np.isfinite(cv)]
    p_cv_e = float(np.mean(cv_e > 0.5)) if cv_e.size else math.nan
    max_cv_e = float(cv_e.max()) if cv_e.size else math.nan
    mean_cv_e = float(cv_e.mean()) if cv_e.size else math.nan
    r_e = rates[is_e]
    cv_rate = float(r_e.std() / r_e.mean()) if r_e.size and r_e.mean() > 0 else math.nan

    n_bins = round(DURATION_MS / BIN_MS)
    B = S.reshape(n_bins, n_steps // n_bins, n).sum(axis=1)
    b_mean = B.mean(axis=0)
    b_var = B.var(axis=0)
    has = b_mean > 0
    fano = np.full(n, np.nan)
    fano[has] = b_var[has] / b_mean[has]
    median_fano = float(np.nanmedian(fano)) if has.any() else math.nan

    rng = np.random.default_rng(RHO_PAIR_SEED + seed)
    pi = rng.integers(0, n, RHO_PAIRS)
    pj = rng.integers(0, n - 1, RHO_PAIRS)
    pj = pj + (pj >= pi)
    Bc = B - b_mean
    sd = np.sqrt((Bc ** 2).sum(axis=0))
    den = sd[pi] * sd[pj]
    num = (Bc[:, pi] * Bc[:, pj]).sum(axis=0)
    ok = den > 0
    mean_rho = float((num[ok] / den[ok]).mean()) if ok.any() else math.nan

    el = model.params["edge_list"]
    pre = np.asarray(el.pre).astype(np.int64)
    post = np.asarray(el.post).astype(np.int64)
    w = np.abs(np.asarray(el.weight, dtype=np.float64))
    pre_e = is_e[pre]
    contrib = w * rates[pre]
    cur_e = np.where(pre_e, contrib, 0.0)
    cur_i = np.where(pre_e, 0.0, contrib)
    I_e, I_i = float(cur_e.sum()), float(cur_i.sum())
    efrac = I_e / (I_e + I_i) if I_e + I_i > 0 else math.nan
    e_post = np.bincount(pid[post], weights=cur_e, minlength=n_pop)
    i_post = np.bincount(pid[post], weights=cur_i, minlength=n_pop)
    has_cur = (e_post + i_post) > 0
    pop_efrac = e_post[has_cur] / (e_post[has_cur] + i_post[has_cur])

    pop_n = np.bincount(pid, minlength=n_pop)
    pop_rate = np.bincount(pid, weights=rates, minlength=n_pop) / pop_n
    pop_e = np.array([lbl[2] == "E" for lbl in pop_labels])
    class_mean = {c: float(rates[cell == c].mean()) for c in sorted(set(cell.tolist()))}

    finite = bool(np.isfinite(S).all() and np.isfinite(np.asarray(v_m)).all() and np.isfinite(rates).all())
    gates = [
        _gate("H01_finite", "all spikes, V_m and rates finite", finite, finite),
        _gate("H02_bar_r_open_1_30", "1 < bar_r < 30 Hz", bar_r, _open(bar_r, 1.0, 30.0)),
        _gate("H03_mean_rho_below_0p60", "mean rho < 0.60", mean_rho,
              _finite(mean_rho) and mean_rho < 0.60),
        _gate("H04_CV_rate_above_0p15", "CV_rate (E) > 0.15", cv_rate,
              _finite(cv_rate) and cv_rate > 0.15),
        _gate("H05_P_CV_E_gt_0p5_at_least_0p15", "P(CV_E > 0.5) >= 0.15", p_cv_e,
              _finite(p_cv_e) and p_cv_e >= 0.15),
        _gate("H06_max_CV_E_at_least_1p5", "max CV_E >= 1.5", max_cv_e,
              _finite(max_cv_e) and max_cv_e >= 1.5),
        _gate("H09_Efrac_open_0p05_0p95", "0.05 < Efrac < 0.95 (proxy)", efrac,
              _open(efrac, 0.05, 0.95)),
        _gate("H10a_class_mean_not_above_80", "no cell-type mean > 80 Hz", max(class_mean.values()),
              max(class_mean.values()) <= 80.0),
        _gate("H10b_E_class_mean_above_0p5", "E cell-type mean > 0.5 Hz", class_mean.get("E", math.nan),
              class_mean.get("E", math.nan) > 0.5),
        _gate("H10c_population_mean_not_above_80", "no (area,layer,type) mean > 80 Hz",
              float(pop_rate.max()), float(pop_rate.max()) <= 80.0),
        _gate("H10d_E_population_mean_above_0p5", "every E (area,layer) mean > 0.5 Hz",
              float(pop_rate[pop_e].min()), float(pop_rate[pop_e].min()) > 0.5),
    ]
    failed = [g["id"] for g in gates if not g["pass"]]
    hard_pass = not failed

    frac_in_band = float(np.mean((pop_efrac >= 0.15) & (pop_efrac <= 0.60))) if pop_efrac.size else math.nan
    soft_raw = {
        "B1_mu_band_3_15": _band_distance(bar_r, 3.0, 15.0),
        "B2_CV_rate_above_0p3": max(0.0, 0.3 - cv_rate) / 0.3 if _finite(cv_rate) else math.nan,
        "B2_mean_CV_E_band_0p5_1p5": _band_distance(mean_cv_e, 0.5, 1.5),
        "B2_median_Fano_band_0p7_2": _band_distance(median_fano, 0.7, 2.0),
        "B2_mean_rho_band_m0p05_0p2": _band_distance(mean_rho, -0.05, 0.2),
        "B3_population_Efrac_frac_at_least_0p8": (
            max(0.0, 0.8 - frac_in_band) / 0.8 if _finite(frac_in_band) else math.nan),
    }
    soft_ok = all(_finite(v) for v in soft_raw.values())
    soft_vector = [v for v in soft_raw.values()] if soft_ok else None
    soft_distance = SOFT_WEIGHT * sum(soft_vector) if soft_ok else None

    h08 = {}
    for group, motif, _, _ in MOTIF_COORDS:
        g = cand["values"][f"{group}[{motif_name(motif)}]"]
        h08[motif_name(motif)] = {"g_multiplier": g, "within_0p7_1p3": H08_RATIO_BOX[0] <= g <= H08_RATIO_BOX[1]}

    metrics = {
        "n_neurons": n,
        "total_spikes": int(S.sum()),
        "bar_r_Hz": bar_r,
        "class_mean_Hz": class_mean,
        "E_CV_rate": cv_rate,
        "E_P_CV_gt_0p5": p_cv_e,
        "E_max_CV": max_cv_e,
        "E_mean_CV": mean_cv_e,
        "E_defined_CV_count": int(cv_e.size),
        "all_median_Fano": median_fano,
        "mean_rho": mean_rho,
        "rho_pairs_used": int(ok.sum()),
        "Efrac_proxy_global": efrac,
        "I_E_proxy": I_e,
        "I_I_proxy": I_i,
        "population_count": n_pop,
        "population_Efrac_defined_count": int(pop_efrac.size),
        "population_Efrac_frac_in_0p15_0p60": frac_in_band,
        "population_mean_Hz_min": float(pop_rate.min()),
        "population_mean_Hz_max": float(pop_rate.max()),
        "E_population_mean_Hz_min": float(pop_rate[pop_e].min()),
        "mu_total_proxy": cand["mu_total"],
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "failed": failed,
        "hard_pass": hard_pass,
        "soft_terms": soft_raw,
        "soft_vector": soft_vector,
        "soft_distance": soft_distance,
        "informational": {"H08_motif_ratio_box_0p7_1p3": h08,
                          "H08_all_within_box": all(v["within_0p7_1p3"] for v in h08.values())},
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (bool, str)) or type(obj).__name__ == "bool_":
        return bool(obj) if type(obj).__name__ == "bool_" else obj
    if isinstance(obj, int):
        return int(obj)
    x = float(obj)
    return x if math.isfinite(x) else None


def execute_run(cand: dict, seed: int) -> dict:
    """Simulate one candidate, score it, write its receipt. Errors propagate (no silent retry)."""
    model, spikes, v_m = simulate_candidate(cand, seed)
    result = score_run(model, spikes, v_m, cand, seed)
    receipt = {
        "candidate_id": cand["candidate_id"],
        "seed": seed,
        "stage": 0,
        "status": "FEASIBLE" if result["hard_pass"] else "REJECTED_HARD",
        "design": "docs/designs/SCI-AGSDR-CALIBRATION_design.md + amendment_1 (sealed)",
        "sampling_mode": SAMPLING_MODE,
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "values": cand["values"],
        "builder_args": cand["builder_args"],
        "motif_gains_effective": cand["motif_gains_effective"],
        "metrics": result["metrics"],
        "hard_gates": result["gates"],
        "failed_hard_gates": result["failed"],
        "soft_terms": result["soft_terms"],
        "soft_distance": result["soft_distance"],
        "informational": result["informational"],
        "definitions": DEFINITIONS,
    }
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path(cand["candidate_id"], seed).write_text(
        json.dumps(_jsonable(receipt), indent=2, allow_nan=False))
    return {
        "candidate_id": cand["candidate_id"],
        "seed": seed,
        "hard_pass": result["hard_pass"],
        "failed": result["failed"],
        "soft_distance": result["soft_distance"],
        "soft_vector": result["soft_vector"],
        "soft_terms": result["soft_terms"],
    }


def write_summary(rows: list[dict], plan0: dict) -> dict:
    feasible = [r for r in rows if r["hard_pass"]]

    def dominates(a: list[float], b: list[float]) -> bool:
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    nondominated = {
        r["candidate_id"]: not any(
            dominates(o["soft_vector"], r["soft_vector"]) for o in feasible if o is not r)
        for r in feasible
    }
    ranked = sorted(feasible, key=lambda r: (r["soft_distance"], r["candidate_id"]))
    summary = {
        "stage": 0,
        "seeds": [0],
        "simulations_run": len(rows),
        "lhs_seed": plan0["seed"],
        "sampling_mode": plan0["sampling_mode"],
        "log10_bounds": plan0["log10_bounds"],
        "frozen_coordinates": plan0["frozen_coordinates"],
        "rows_redrawn_for_mu": plan0["rows_redrawn_for_mu"],
        "runner_sha256": sha256_of(THIS_FILE),
        "hard_pass_count": len(feasible),
        "rejected": [{"candidate_id": r["candidate_id"], "failed_hard_gates": r["failed"]}
                     for r in rows if not r["hard_pass"]],
        "feasible_ranked": [{"candidate_id": r["candidate_id"], "soft_distance": r["soft_distance"],
                             "pareto_nondominated": nondominated[r["candidate_id"]],
                             "soft_terms": r["soft_terms"]} for r in ranked],
        "best_feasible": (
            {"candidate_id": ranked[0]["candidate_id"], "soft_distance": ranked[0]["soft_distance"]}
            if ranked else None),
        "null_result": not feasible,
        "not_gated_at_stage0": [
            "H07 locality and H11 CL1-A hash audit (not in the sealed hard list; not computed)",
            "H08 motif-ratio box 0.7x-1.3x (informational flag per candidate; not in the sealed hard list)",
        ],
        "definitions": DEFINITIONS,
        "stage1_selection": "not run",
    }
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    (RECEIPT_DIR / "stage0_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=False))
    return summary


def receipt_path(candidate_id: str, seed: int) -> Path:
    return RECEIPT_DIR / f"candidate_{candidate_id}_seed{seed}.json"


def run_stage(stage: int) -> int:
    if stage != 0:
        raise NotImplementedError("only stage 0 is implemented; stages 1 and 2 are not run by this change")
    spec = STAGES[stage]
    missing: list[str] = [str(p) for p in (DESIGN, *MANIFESTS) if not p.exists()]
    if not AUDIT_ARTIFACT.exists():
        missing.append(f"{AUDIT_ARTIFACT} (B10 audit artifact, hits=[])")
    else:
        art = json.loads(AUDIT_ARTIFACT.read_text())
        if art.get("hits") != []:
            missing.append(f"{AUDIT_ARTIFACT} reports hits {art.get('hits')}")
        if art.get("runner_sha256") != sha256_of(THIS_FILE):
            missing.append(f"{AUDIT_ARTIFACT} is stale for the current runner source")
    survivors_file = RECEIPT_DIR / f"stage{stage - 1}_survivors.json"
    plan0 = None
    if stage == 0:
        plan0 = latin_hypercube_plan(load_motif_table())
        candidates = plan0["candidates"]
    elif survivors_file.exists():
        candidates = json.loads(survivors_file.read_text())["candidates"]
    else:
        candidates = []
        missing.append(str(survivors_file))
    if missing:
        print(f"stage {stage} blocked; missing or invalid inputs:")
        for m in missing:
            print(f"  - {m}")
        return 2
    if len(candidates) != spec["candidates"]:
        print(f"stage {stage} expects {spec['candidates']} candidates, found {len(candidates)}")
        return 2
    # Stage-0 plan receipt: sampling mode, log10 bounds and frozen coordinates (amendment 1).
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    record = {k: plan0[k] for k in (
        "seed", "sampling_mode", "log10_bounds", "frozen_coordinates", "rows_redrawn_for_mu")}
    record["candidate_ids"] = [c["candidate_id"] for c in candidates]
    (RECEIPT_DIR / "stage0_plan.json").write_text(json.dumps(record, indent=2))
    rows: list[dict] = []
    with (RECEIPT_DIR / "trace.jsonl").open("w") as trace:
        trace.write(json.dumps({
            "event": "stage0_start", "sampling_mode": plan0["sampling_mode"],
            "lhs_seed": plan0["seed"], "log10_bounds": plan0["log10_bounds"],
            "runner_sha256": sha256_of(THIS_FILE)}) + "\n")
        for cand in candidates:
            for seed in spec["seeds"]:
                row = execute_run(cand, seed)
                rows.append(row)
                trace.write(json.dumps(_jsonable({
                    "event": "candidate_scored", "candidate_id": row["candidate_id"],
                    "seed": seed, "hard_pass": row["hard_pass"],
                    "failed_hard_gates": row["failed"], "soft_distance": row["soft_distance"]})) + "\n")
    summary = write_summary(rows, plan0)
    print(f"stage 0 done: simulated={summary['simulations_run']} "
          f"hard_pass={summary['hard_pass_count']} best={summary['best_feasible']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--audit", action="store_true")
    group.add_argument("--stage", type=int, choices=sorted(STAGES))
    args = ap.parse_args(argv)
    if args.dry_run:
        return print_dry_run(load_motif_table())
    if args.audit:
        return run_audit()
    return run_stage(args.stage)


if __name__ == "__main__":
    sys.exit(main())

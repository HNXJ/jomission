#!/usr/bin/env python3
"""AGSDR local calibration sweep runner (SCI-AGSDR-CALIBRATION).

Authority: docs/designs/SCI-AGSDR-CALIBRATION_design.md (sealed before execution).

Modes
  --dry-run   Print the stage-0 Latin hypercube candidates, the mask-to-motif
              mapping and the planned simulation count. No simulation, no writes.
  --audit     Run audit_b10_objective on this file. Exit 1 on any hit.
  --stage N   Run stage N (0, 1 or 2). Refuses to start unless the audit
              artifact and the stage inputs exist. Not executed in this task.
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


def execute_run(cand: dict, seed: int) -> None:
    build_candidate_model(cand, seed)
    raise NotImplementedError(
        "simulate-and-score is not implemented in this runner; stage execution is blocked. "
        "No receipt was written."
    )


def receipt_path(candidate_id: str, seed: int) -> Path:
    return RECEIPT_DIR / f"candidate_{candidate_id}_seed{seed}.json"


def run_stage(stage: int) -> int:
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
    for cand in candidates:
        for seed in spec["seeds"]:
            execute_run(cand, seed)
    if plan0 is not None:
        # Stage-0 receipt: sampling mode, log10 bounds and frozen coordinates (amendment 1).
        RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        record = {k: plan0[k] for k in (
            "seed", "sampling_mode", "log10_bounds", "frozen_coordinates", "rows_redrawn_for_mu")}
        record["candidate_ids"] = [c["candidate_id"] for c in candidates]
        (RECEIPT_DIR / "stage0_plan.json").write_text(json.dumps(record, indent=2))
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

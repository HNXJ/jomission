"""Exhaustive read-only audit of the allocation policy R for CTX[jomission_v0].

    python scripts/ctx_allocation_audit.py

TFNE/2 requires a declared deterministic R that preserves exact cardinality:

    for every layer l:  sum_c N[l.c] = N[l],   and   sum_l N[l] = N.

Adopting the engine's policy as CTX[jomission_v0].R is admissible only if that holds for every
population this definition can realize, so the audit is exhaustive over layer totals rather than
checked at the one size the architecture happens to use. Writes results/ctx_allocation_audit.json.
No construction and no simulation.

Two candidate readings of the engine's behaviour are audited separately, because they are not the
same function: the independent per-class rounding int(round(N_l * P[l][c])), and the sequential
policy the engine actually implements.
"""

from __future__ import annotations

import json
from pathlib import Path

from jaxfne._config import _counts_from_fractions as engine_counts

from jomission.tfne import ctx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ctx_allocation_audit.json"
MAX_LAYER_TOTAL = 2000


def independent_round(total, proportions):
    """int(round(N_l * P[l][c])) per class, with no coupling between classes."""
    return {c: int(round(total * p)) for c, p in proportions.items()}


def audit(policy, proportions, limit):
    """Every layer total from 0 to limit. Returns violations and the first one, if any."""
    bad = []
    for total in range(limit + 1):
        counts = policy(total, proportions)
        got = sum(counts.values())
        if got != total:
            bad.append({"layer_total": total, "sum_c": got, "counts": counts})
    return bad


def main():
    P = ctx.layer_cell_type_fractions()
    rec = {
     "spec": "results/ctx_allocation_audit_spec.json",
     "question": ("is the JaxFNE 0.4.24 allocation policy admissible as CTX[jomission_v0].R, i.e. does "
                  "it preserve sum_c N[l.c] = N[l] for every population this definition can realize"),
     "read_only": "no model is constructed and no simulation is run",
     "layers": list(ctx.LAYERS), "classes": list(ctx.CLASSES), "P": P,
     "max_layer_total": MAX_LAYER_TOTAL,
     "cases_per_policy": len(ctx.LAYERS) * (MAX_LAYER_TOTAL + 1),
     "policies": {},
    }

    for name, policy, note in (
        ("engine_sequential", engine_counts,
         "jaxfne/_config.py:130 _counts_from_fractions: int(round(total*frac)) in declaration order, "
         "clamped to the remaining budget, with the last key assigned the remainder"),
        ("independent_round", independent_round,
         "int(round(N_l * P[l][c])) per class with no coupling; the literal per-class formula"),
    ):
        per_layer = {}
        for lay in ctx.LAYERS:
            bad = audit(policy, P[lay], MAX_LAYER_TOTAL)
            per_layer[lay] = {"violations": len(bad), "first_violation": bad[0] if bad else None}
        total_bad = sum(v["violations"] for v in per_layer.values())
        rec["policies"][name] = {"note": note, "by_layer": per_layer, "violations": total_bad,
                                 "preserves_exact_cardinality": total_bad == 0}

    # The architecture's own size, reported explicitly rather than left to the exhaustive sweep.
    n = ctx.N_PER_INSTANCE
    layer_fracs = ctx.layer_population_fractions()
    layer_counts = engine_counts(n, layer_fracs)
    at_size = {lay: {"layer_total": layer_counts[lay], "counts": engine_counts(layer_counts[lay], P[lay])}
               for lay in ctx.LAYERS}
    rec["at_architecture_size"] = {
     "N": n, "layer_fractions": layer_fracs, "layer_counts": layer_counts,
     "sum_layers": sum(layer_counts.values()),
     "by_layer": at_size,
     "sum_all_classes": sum(sum(v["counts"].values()) for v in at_size.values())}

    ok = (rec["policies"]["engine_sequential"]["preserves_exact_cardinality"]
          and rec["at_architecture_size"]["sum_layers"] == n
          and rec["at_architecture_size"]["sum_all_classes"] == n)
    rec["verdict"] = "ALLOCATION_POLICY_" + ("PASS" if ok else "FAIL")
    rec["adoptable_as_R"] = ok
    rec["correction"] = (
     "the two policies are different functions. The independent per-class formula does not preserve "
     "exact cardinality and is therefore not admissible as R; the sequential policy the engine "
     "actually implements does. CTX[jomission_v0].R adopts the sequential policy")
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(rec["verdict"])
    for name, p in rec["policies"].items():
        print(f"  {name}: {p['violations']} violations of sum_c N = N_l in "
              f"{rec['cases_per_policy']} cases -> admissible as R: {p['preserves_exact_cardinality']}")
    print("  at N =", n, "layer counts", layer_counts, "sum", rec["at_architecture_size"]["sum_layers"])
    print("  sum over all layers and classes:", rec["at_architecture_size"]["sum_all_classes"])


if __name__ == "__main__":
    main()

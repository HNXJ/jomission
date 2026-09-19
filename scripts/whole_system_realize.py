"""R3/R4: emit the architecture receipt, realize it in JaxFNE, and reconcile the two.

    python scripts/whole_system_realize.py [--out results/<file>.json]

No simulation. The model is constructed and inspected; no step function is compiled and no
trajectory is run. Writes results/whole_system_realization.json.

Order matters: the expected graph is emitted from NF(A) *before* construction, and projection
identities are compared before edge counts, because a matching edge count over a mismatched
identity set is not a realization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jomission.tfne import architecture, ctx, nf, realize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "whole_system_realization.json"
SPEC = "results/whole_system_realization_spec.json"


def population_reconciliation(index, objects):
    """What TFNE owns: per-instance totals, absent populations empty, present ones non-empty,
    and the realized class counts inside each layer equal to allocate(layer_total, P[layer])."""
    decl = ctx.declaration()
    problems, per_area = [], {}
    for obj in objects:
        area = realize.area_name(tuple(obj["path"]))
        total = sum(v[2] for k, v in index.items() if k.split(".")[0] == area)
        per_layer = {}
        for lay in ctx.LAYERS:
            got = {c: index.get(f"{area}.{lay}.{c}", (0, 0, 0))[2] for c in ctx.CLASSES}
            layer_total = sum(got.values())
            want = ctx.expected_class_counts(layer_total, lay) if layer_total else dict.fromkeys(ctx.CLASSES, 0)
            per_layer[lay] = {"realized": got, "expected_from_P": want, "layer_total": layer_total}
            for c in ctx.CLASSES:
                if want.get(c, 0) != got[c]:
                    problems.append(f"{area}.{lay}.{c}: realized {got[c]}, P requires {want.get(c, 0)}")
                if f"{lay}.{c}" in decl["absent_populations"] and got[c]:
                    problems.append(f"{area}.{lay}.{c}: declared absent but realized {got[c]}")
                if f"{lay}.{c}" not in decl["absent_populations"] and got[c] == 0:
                    problems.append(f"{area}.{lay}.{c}: declared present but realized empty")
        per_area[area] = {"total": total, "expected_total": decl["N"], "by_layer": per_layer}
        if total != decl["N"]:
            problems.append(f"{area}: realized {total} neurons, N declares {decl['N']}")
    return per_area, problems


def main(out=OUT):
    chain, rules, external, decl = architecture.build()
    normal = nf.normalize(chain, rules, external)

    rec = {
     "spec": SPEC,
     "lineage": "whole-system realization (R1-R4). No simulation.",
     "tfne": {"algebra": "tfne/2", "authority": "docs/tfne/TFNE_V2_SPEC.md (SEALED)",
              "expression": "x : {V1^2 X[lat]} O[fffb] {V4^2 X[lat]} O[fffb] {FEF X[lat] PFC} : y",
              "x": architecture.RETINA, "y": architecture.OUTPUT_DECLARATION,
              "rules": {k: {kk: vv for kk, vv in v.items() if kk != "clauses"} | {"clauses": v["clauses"]}
                        for k, v in rules.items()},
              "mapping_authority": architecture.MAPPING_AUTHORITY},
     "definition": decl,
     "normalized": {k: v for k, v in normal.items() if not k.startswith("_")},
    }

    # R3 invariant: an ordered chain relates only adjacent levels.
    labels = {c.label: [i.name() for i in c.instances()] for c in chain.items}
    member_of = {name: label for label, names in labels.items() for name in names}
    skips = []
    for p in normal["_projection_objects"]:
        if p.source.path == ("Retina",):
            continue
        a, b = member_of[".".join(p.source.path)], member_of[".".join(p.target.path)]
        if a == b:
            continue
        order = [c.label for c in chain.items]
        if abs(order.index(a) - order.index(b)) > 1:
            skips.append(p.as_dict())
    rec["adjacency_invariant"] = {
     "non_adjacent_pairs": [list(t) for t in nf.non_adjacent_pairs(chain)],
     "skip_projections": skips,
     "holds": not skips}

    # R4: realize, then reconcile identities before counts.
    cfg, areas = realize.scaffold(normal)
    cfg = realize.declare_mechanisms(cfg, normal)
    cfg = realize.declare_projections(cfg, normal)
    model = realize.construct(cfg)
    index, (area, layer, cls) = realize.index_map(model)
    realized, n_edges = realize.realized_projections(model, area, layer, cls)
    expected = realize.expected_projections(normal)

    missing = sorted(k for k in expected if k not in realized)
    extra = sorted(k for k in realized if k not in expected)
    rec["projection_reconciliation"] = {
     "expected": len(expected), "realized_identities": len(realized),
     "missing": [list(k) for k in missing], "extra": [list(k) for k in extra],
     "edge_counts": {f"{s} -> {t}": int(n) for (s, t), n in
                     sorted(realized.items(), key=lambda kv: (kv[0][0], kv[0][1]))},
     "total_edges": n_edges,
     "cross_area_edges": int(sum(realized.values()))}

    # Classify every extra identity rather than narrating it: is it the superficial-layer
    # selector expansion, or something else?
    superficial = {"L2", "L3", "L2/3", "L23"}
    def superficial_variant(k):
        want = [(s, t) for (s, t) in expected
                if s.split(".")[0] == k[0].split(".")[0] and t.split(".")[0] == k[1].split(".")[0]
                and s.split(".")[2] == k[0].split(".")[2] and t.split(".")[2] == k[1].split(".")[2]]
        return any(s.split(".")[1] in superficial and k[0].split(".")[1] in superficial
                   and t.split(".")[1] in superficial and k[1].split(".")[1] in superficial
                   for s, t in want)
    unexplained = [k for k in extra if not superficial_variant(k)]
    rec["projection_reconciliation"]["extra_classification"] = {
     "superficial_selector_expansion": [list(k) for k in extra if superficial_variant(k)],
     "unexplained": [list(k) for k in unexplained],
     "engine_receipt": ("jaxfne/_construct_connectivity.py:28 _interarea_layer_set maps any of "
                        "{L2, L3, L2/3, L23} to the whole superficial set, and "
                        "_connection_selector_mask:128 applies it to every connections() selector, "
                        "so a declared L3.E source or target also selects L2.E")}

    mech, mech_meta = realize.mechanism_receipt(model, area, layer, cls, normal)
    declared_mech = sorted({p.mechanism for p in normal["_projection_objects"]})
    mech_problems = [k for k, v in mech.items() if v["receptor_index"] != [0]]
    rec["mechanism_reconciliation"] = {
     "declared": declared_mech, "per_projection": mech, "edge_list": mech_meta,
     "receptor_index_0": "the sole declared mechanism, AMPA, occupies index 0 of the mechanism table",
     "problems": mech_problems,
     "observability_limit": ("tau_ms is empty on the concatenated edge list, so the declared 2.0 ms "
                             "is configured and the receptor identity is realized, but the per-edge "
                             "time constant is not readable from the constructed model")}

    per_area, pop_problems = population_reconciliation(index, normal["objects"])
    rec["population_reconciliation"] = {"by_area": per_area, "problems": pop_problems}

    # The declared output must resolve, and to exactly the intended neurons.
    y_key = ".".join((realize.area_name(("FEF",)), "L6", "E"))
    y = index.get(y_key)
    if y:
        sel = np.flatnonzero((area == realize.area_name(("FEF",))) & (layer == "L6") & (cls == "E"))
        exact = bool(sel.size == y[2] and int(sel.min()) == y[0] and int(sel.max()) + 1 == y[1])
    else:
        exact = False
    rec["output_interface"] = {"address": "FEF.L6.E", "jaxfne_key": y_key, "index_range": y,
                               "resolves_nonempty": bool(y and y[2] > 0),
                               "metadata_exact": exact,
                               "declaration": architecture.OUTPUT_DECLARATION}

    # The retinal input interface: where 1024 inputs will enter, established structurally.
    targets = [str(a) for a in chain.items[0].frontier("in")]
    entry = {t: index.get(realize.area_name(tuple(t.split(".")[:2])) + "." + ".".join(t.split(".")[2:]))
             for t in targets}
    rec["input_interface"] = {
     "x": architecture.RETINA,
     "targets": targets, "target_index_ranges": entry,
     "n_target_neurons": int(sum(v[2] for v in entry.values() if v)),
     "realized_here": False,
     "note": ("x is declared and its entry points resolved; no retinal population is constructed in "
              "this lineage, which realizes A only. The simulation lineage drives these indices")}

    ok = (rec["adjacency_invariant"]["holds"] and not missing and not extra and not pop_problems
          and not mech_problems
          and rec["output_interface"]["resolves_nonempty"] and rec["output_interface"]["metadata_exact"]
          and all(v for v in entry.values()))
    rec["deltas"] = {"objects": 0 if len(normal["objects"]) == len(areas) else len(normal["objects"]) - len(areas),
                     "populations": len(pop_problems), "projections": len(missing) + len(extra),
                     "index": 0 if rec["output_interface"]["metadata_exact"] else 1}
    rec["verdict"] = "WHOLE_SYSTEM_REALIZATION_" + ("PASS" if ok else "FAIL")
    rec["reason"] = ("every declared object, population, projection and interface reconciles with the "
                     "realized model" if ok else
                     f"missing {len(missing)}, extra {len(extra)}, population problems {len(pop_problems)}, "
                     f"skips {len(skips)}, mechanism problems {len(mech_problems)}, "
                     f"output resolves {rec['output_interface']['resolves_nonempty']}")
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(rec["verdict"], rec["reason"])
    print("  objects:", [o["name"] for o in normal["objects"]])
    print("  projections expected/realized:", len(expected), len(realized),
          "| cross-area edges:", rec["projection_reconciliation"]["cross_area_edges"],
          "| total edges:", n_edges)
    print("  skips:", len(skips), "| population problems:", len(pop_problems))
    print("  y = FEF.L6.E ->", y)
    if missing[:5]:
        print("  missing:", missing[:5])
    if extra[:5]:
        print("  extra:", extra[:5])
    if pop_problems[:5]:
        print("  population:", pop_problems[:5])


if __name__ == "__main__":
    # A correction lineage writes a new file: the earlier result is immutable evidence.
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    main(Path(ap.parse_args().out))

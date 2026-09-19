"""TFNE v2 conformance arithmetic: allocation, canonical ordering, index map, chain expansion.

    python scripts/tfne/v2_conformance.py        # derived JSON on stdout

Not a parser and not a compiler. It takes already-normalized corpus declarations (the
appendix cases of docs/tfne/TFNE_V2_SPEC.md) and derives the numbers the spec states:
integer allocation under R, canonical natural-key traversal, and the index map I.
tests/test_tfne_v2_spec.py checks the spec against this derivation.
"""
import json
import re

# Corpus declarations. members: declared proportion map; order: explicit biological order
# overriding natural ordering (enters NF); instances: replicated family size.
CASES = {
    "C2_LGN": {"object": "LGN", "N": 1000, "members": {"relay": 0.8, "inter": 0.2}},
    "C3_V1": {"object": "V1", "N": 10000,
              "members": {"L1": 0.02, "L2": 0.15, "L3": 0.20, "L4": 0.22, "L5": 0.23, "L6": 0.18},
              "children": {"L1": {"PV": 0.10, "SST": 0.35, "VIP": 0.55},
                           "L2": {"E": 0.70, "PV": 0.15, "SST": 0.10, "VIP": 0.05},
                           "L3": {"E": 0.70, "PV": 0.15, "SST": 0.10, "VIP": 0.05},
                           "L4": {"E": 0.70, "PV": 0.15, "SST": 0.10, "VIP": 0.05},
                           "L5": {"E": 0.70, "PV": 0.15, "SST": 0.10, "VIP": 0.05},
                           "L6": {"E": 0.70, "PV": 0.15, "SST": 0.10, "VIP": 0.05}}},
    "C4_CORD": {"object": "CORD", "instances": ("SEG", 8), "N_instance": 500,
                "members": {"inter": 0.6, "motor": 0.2, "sensory": 0.2},
                "chain": {"rule": "spinal", "src": "out", "dst": "in", "out": "inter", "in": "inter"}},
    "N4_valid": {"object": "NUC", "N": 101, "members": {"E": 0.8, "PV": 0.2}},
}


def natural_key(name):
    """Typed path token: (stem, numeric suffix or index, remainder). L2 < L10; SEG.2 < SEG.10."""
    m = re.fullmatch(r"([^0-9]*)([0-9]*)(.*)", str(name))
    stem, num, rest = m.group(1), m.group(2), m.group(3)
    return (stem, int(num) if num else -1, rest)


def allocate(total, proportions):
    """Largest remainder; ties to the natural-key-first member. Integer counts summing to total."""
    keys = sorted(proportions, key=natural_key)
    exact = {k: total * proportions[k] for k in keys}
    counts = {k: int(exact[k] // 1) for k in keys}
    short = total - sum(counts.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - counts[k]), natural_key(k)))[:short]:
        counts[k] += 1
    return counts


def layout(counts_tree, start=0):
    """Index ranges in canonical traversal order. counts_tree: name -> int or nested dict."""
    out, cursor = {}, start
    for name in sorted(counts_tree, key=natural_key):
        node = counts_tree[name]
        if isinstance(node, dict):
            sub, end = layout(node, cursor)
            out[name] = [cursor, end]
            out.update({f"{name}.{k}": v for k, v in sub.items()})
            cursor = end
        else:
            out[name] = [cursor, cursor + node]
            cursor += node
    return out, cursor


def chain(prefix, n, rule, src_pop, dst_pop):
    """O[rule](A^n): projections between adjacent replicated instances. n - 1 identities."""
    return [{"src": f"{prefix}.{i}.{src_pop}", "dst": f"{prefix}.{i + 1}.{dst_pop}", "rule": rule}
            for i in range(1, n)]


def derive():
    out = {}
    lgn = CASES["C2_LGN"]
    counts = allocate(lgn["N"], lgn["members"])
    out["C2_LGN"] = {"counts": counts, "index_map": layout(counts)[0]}

    v1 = CASES["C3_V1"]
    layer_counts = allocate(v1["N"], v1["members"])
    tree = {ln: allocate(layer_counts[ln], v1["children"][ln]) for ln in layer_counts}
    ranges, total = layout(tree)
    out["C3_V1"] = {"layer_counts": layer_counts, "L4_counts": tree["L4"], "L1_counts": tree["L1"],
                    "total": total, "index_map_selected": {k: ranges[k] for k in
                                                           ("L1", "L4", "L4.E", "L4.PV", "L4.SST", "L4.VIP")}}

    cord = CASES["C4_CORD"]
    prefix, n = cord["instances"]
    per = allocate(cord["N_instance"], cord["members"])
    tree = {f"{prefix}.{i}": dict(per) for i in range(1, n + 1)}
    ranges, total = layout(tree)
    ch = cord["chain"]
    out["C4_CORD"] = {"per_instance_counts": per, "total": total,
                      "index_map_selected": {k: ranges[k] for k in
                                             (f"{prefix}.3", f"{prefix}.3.inter", f"{prefix}.3.motor", f"{prefix}.3.sensory")},
                      "projections": chain(prefix, n, ch["rule"], ch["out"], ch["in"])}

    nuc = CASES["N4_valid"]
    out["N4_valid"] = {"exact": {k: round(nuc["N"] * v, 4) for k, v in nuc["members"].items()},
                       "counts": allocate(nuc["N"], nuc["members"])}
    out["natural_order_probe"] = sorted(["L10", "L2", "L1", "SEG.10", "SEG.2"], key=natural_key)
    return out


if __name__ == "__main__":
    print(json.dumps(derive(), indent=1, sort_keys=True))

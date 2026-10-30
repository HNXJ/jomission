"""Compare the SCI-EI-REPLAY-058 receipts with their sealed counterparts.

Usage (from the repo root): python scripts/ei_replay_058_compare.py

Pairs each replay receipt in results/ei_replay_058/ with its sealed receipt, compares every
leaf (numbers within 1e-9 absolute and relative, strings and nulls exactly), recomputes the
interaction verdict from the replay per-cell rates by the sealed viability rule, and writes
results/ei_replay_058/comparison.json. A text leaf whose sealed and replay values are both .npz
artifact paths is a location difference: listed apart and not counted (spec amendment 3).
Classification: REPRODUCED (no leaf differs), ENGINE_SHIFT (a leaf differs, every verdict label
unchanged), VERDICT_CHANGED (a label differs). classification_as_run keeps the label before amendment 3.

Exit 0 when the comparison is written; exit 2 when a receipt is missing or unreadable.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPLAY = "results/ei_replay_058"
OUT = ROOT / REPLAY / "comparison.json"
CELLS = ("baseline", "w_only", "a_only", "both")
EXCLUDE = {"elapsed_s"}
TOL_ABS = 1e-9
TOL_REL = 1e-9
BAND = {"PV": (12.0, 25.0), "VIP": (8.0, 15.0), "SST": (6.0, 12.0)}
SEALED_INTERACTION = "results/whole_system_ei_interaction.json"
SEALED_REGIME = "results/whole_system_ei_regime_diagnosis.json"
REPLAY_REGIME = f"{REPLAY}/whole_system_ei_regime_diagnosis.json"
PAIRS = [(SEALED_REGIME, REPLAY_REGIME)] + [
    (
        f"results/whole_system_ei_interaction_intact_g000_on_s0_{c}.json",
        f"{REPLAY}/whole_system_ei_interaction_{c}.json",
    )
    for c in CELLS
]


def load(rel):
    path = ROOT / rel
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"unreadable receipt {rel}: {exc}")


def leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            if k not in EXCLUDE:
                yield from leaves(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + (str(i),))
    else:
        yield path, node


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_location(a, b):
    return isinstance(a, str) and isinstance(b, str) and a.endswith(".npz") and b.endswith(".npz")


def compare(sealed, replay):
    s = dict(leaves(sealed))
    r = dict(leaves(replay))
    numeric, text, location, missing = [], [], [], []
    max_abs = 0.0
    for p, a in s.items():
        key = "/".join(p)
        if p not in r:
            missing.append(key)
            continue
        b = r[p]
        if is_num(a) and is_num(b):
            d = abs(a - b)
            max_abs = max(max_abs, d)
            if d > TOL_ABS + TOL_REL * abs(a):
                numeric.append({"path": key, "sealed": a, "replay": b})
        elif a != b:
            entry = {"path": key, "sealed": a, "replay": b}
            (location if is_location(a, b) else text).append(entry)
    extra = ["/".join(p) for p in r if p not in s]
    return {
        "leaves": len(s),
        "max_abs_diff": max_abs,
        "numeric_diffs": numeric,
        "text_diffs": text,
        "location_diffs": location,
        "missing_in_replay": missing,
        "extra_in_replay": extra,
    }


def viability(doc):
    rates = doc["ei_regime"]["rates_full_resolution"]
    r = {c: float(rates[c]["full"]) for c in ("E", "PV", "SST", "VIP")}
    viable = all(lo <= r[c] <= hi for c, (lo, hi) in BAND.items()) and r["E"] < r["PV"]
    return {**r, "viable": viable}


def interaction_label(v):
    w, a, both = v["w_only"]["viable"], v["a_only"]["viable"], v["both"]["viable"]
    if w and a:
        return "MULTIPLE_SUFFICIENT"
    if w:
        return "WEIGHT_SUFFICIENT"
    if a:
        return "ADAPTATION_SUFFICIENT"
    if both:
        return "JOINT_REQUIRED"
    return "EI_REGIME_STILL_UNRESOLVED"


def classify(clean, labels_hold):
    if clean:
        return "REPRODUCED"
    return "ENGINE_SHIFT" if labels_hold else "VERDICT_CHANGED"


def main():
    results = {}
    for sealed_rel, replay_rel in PAIRS:
        results[replay_rel] = compare(load(sealed_rel), load(replay_rel))
    sealed_cells = {c: viability(load(f"results/whole_system_ei_interaction_intact_g000_on_s0_{c}.json")) for c in CELLS}
    replay_cells = {c: viability(load(f"{REPLAY}/whole_system_ei_interaction_{c}.json")) for c in CELLS}
    labels = {
        "regime_sealed": load(SEALED_REGIME)["verdict"],
        "regime_replay": load(REPLAY_REGIME)["verdict"],
        "interaction_sealed": load(SEALED_INTERACTION)["verdict"],
        "interaction_sealed_recomputed": interaction_label(sealed_cells),
        "interaction_replay_recomputed": interaction_label(replay_cells),
    }
    labels_hold = (
        labels["regime_sealed"] == labels["regime_replay"]
        and labels["interaction_sealed"] == labels["interaction_replay_recomputed"]
        and labels["interaction_sealed_recomputed"] == labels["interaction_replay_recomputed"]
    )
    clean = all(
        not v["numeric_diffs"] and not v["text_diffs"] and not v["missing_in_replay"]
        for v in results.values()
    )
    clean_as_run = all(
        not v["numeric_diffs"] and not v["text_diffs"] and not v["location_diffs"] and not v["missing_in_replay"]
        for v in results.values()
    )
    classification = classify(clean, labels_hold)
    out = {
        "lineage_id": "SCI-EI-REPLAY-058",
        "classification": classification,
        "classification_as_run": classify(clean_as_run, labels_hold),
        "labels": labels,
        "interaction_viability": {"sealed": sealed_cells, "replay": replay_cells},
        "pairs": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"classification {classification}")
    for k, v in labels.items():
        print(f"  {k}: {v}")
    for rel, v in results.items():
        print(
            f"  {rel}: leaves {v['leaves']}, numeric diffs {len(v['numeric_diffs'])}, "
            f"text diffs {len(v['text_diffs'])}, location diffs {len(v['location_diffs'])}, "
            f"max abs diff {v['max_abs_diff']:.3g}"
        )


if __name__ == "__main__":
    main()

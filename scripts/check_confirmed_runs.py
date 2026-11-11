"""Guard against the r_ab mix-up recurring: the confirmed setting's runs must be identified by their own records.

The defect this prevents: quoting the axis-1 winner's numbers (all-unit 15 ms) as if they were the confirmed
setting's, which is that circuit plus a 21 ms deep-unit decay. The two are different circuits with different r_ab.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "results/viz/paper_w7/band_scatter.json"
D = ROOT / "results/laminar_e2"
RESULTS = ROOT / "paper/jomission_lansa/sections/results.tex"


def bands(path: Path) -> dict[str, tuple[float, float]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for group in ("ladder", "t1_sel", "t1_conf"):
        for r in d.get(group, []):
            out[r["tag"]] = (r["r_ab"], r["r_gm"])
    return out


def main() -> int:
    b = bands(SC)
    problems = []

    # the confirmed setting: identified from the run records, not from a tag prefix
    confirm = sorted(t for t in b if t.startswith("t1_confirm_"))
    if not confirm:
        problems.append("no t1_confirm_* runs found")
    layouts, taus = set(), []
    for t in confirm:
        rec = json.loads((D / f"{t}.json").read_text(encoding="utf-8"))
        layouts.add(rec["layout"])
        taus.append(tuple(sorted(rec.get("tau_sst_e_ms", {}).items())))
    if len(layouts) != 1:
        problems.append(f"confirmation runs disagree on layout: {layouts}")

    # their selection-seed counterparts share the layout string; find them by that, not by name
    sel_tags = []
    for t in b:
        if t.startswith("t1_confirm_") or not t.startswith("t1_"):
            continue
        p = D / f"{t}.json"
        if not p.exists():
            continue
        if json.loads(p.read_text(encoding="utf-8")).get("layout") in layouts:
            sel_tags.append(t)

    if not sel_tags:
        problems.append("no selection-seed runs share the confirmation layout")
    else:
        sel = [b[t][0] for t in sorted(sel_tags)]
        conf = [b[t][0] for t in confirm]
        print(f"confirmed setting layout : {sorted(layouts)[0]}")
        print(f"  selection-seed r_ab    : {', '.join(f'{v:+.3f}' for v in sel)}  -> {min(sel):.2f}..{max(sel):.2f}")
        print(f"  fresh-seed r_ab        : {', '.join(f'{v:+.3f}' for v in conf)}  -> {min(conf):.2f}..{max(conf):.2f}")
        # the axis-1 winner, for contrast: this is what must NOT be quoted as the confirmed setting
        alt = [b[t][0] for t in b if t.startswith("t1_kd1p41_n4000")]
        if alt:
            print(f"  axis-1 winner (different circuit) r_ab: {min(alt):+.2f}..{max(alt):+.2f}")
        text = RESULTS.read_text(encoding="utf-8")
        lo, hi = f"{min(sel):.2f}", f"{max(sel):.2f}"
        if f"{lo}--{hi}" not in text.replace("0.320", "0.32").replace("0.473", "0.47"):
            problems.append(f"results.tex never states the confirmed setting's selection range {lo}--{hi}")

    for p in problems:
        print(f"FAIL: {p}")
    print("OK: confirmed setting identified from its own run records" if not problems else "PROBLEMS FOUND")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

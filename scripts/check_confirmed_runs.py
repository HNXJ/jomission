"""Guard against the r_ab mix-up recurring: the confirmed setting is identified by its run records.

The defect this prevents: quoting the axis-1 winner's numbers (all-unit 15 ms) as if they were the confirmed
setting's, which is that circuit plus a 21 ms deep-unit decay. The two are different circuits with different r_ab.

Checks, in order:
 1. Every confirmation run records the SAME layout and the SAME tau set; otherwise the guard refuses to proceed.
 2. The selection-seed runs are those whose layout AND tau set equal the confirmation's.
 3. Each of abstract.tex, results.tex and discussion.tex states the confirmed setting's selection range, and no
    sentence there pairs the setting with the axis-1 winner's range unless it also carries the confirmed range.

Check 3 exists because a bare "does the file contain 0.32--0.47" test passes the pre-fix text, which assigned
0.58--0.79 to the setting and mentioned 0.32--0.47 only in a different sentence.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "results/viz/paper_w7/band_scatter.json"
D = ROOT / "results/laminar_e2"
SEC = ROOT / "paper/jomission_lansa/sections"
SETTING_WORDS = ("the setting", "selected setting", "confirmed setting", "frozen")


def records(tags):
    out = {}
    for t in tags:
        p = D / f"{t}.json"
        if not p.exists():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[t] = (rec["layout"], tuple(sorted(rec.get("tau_sst_e_ms", {}).items())))
    return out


def rng(vals):
    return f"{min(vals):.2f}", f"{max(vals):.2f}"


def main() -> int:
    sc = json.loads(SC.read_text(encoding="utf-8"))
    b = {r["tag"]: r for g in ("ladder", "t1_sel", "t1_conf") for r in sc.get(g, [])}
    problems = []

    confirm = sorted(t for t in b if t.startswith("t1_confirm_"))
    crec = records(confirm)
    if not crec:
        print("FAIL: confirmation run record missing")
        return 1

    # 1. the confirmation runs must agree, or "the confirmed setting" has no single meaning
    layouts = {v[0] for v in crec.values()}
    tausets = {v[1] for v in crec.values()}
    if len(layouts) != 1:
        problems.append(f"confirmation runs disagree on layout: {sorted(layouts)}")
    if len(tausets) != 1:
        problems.append(f"confirmation runs disagree on tau set: {sorted(tausets)}")
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    layout, taus = next(iter(layouts)), next(iter(tausets))
    print(f"confirmed setting: layout {layout}")
    print("                   tau    " + ", ".join(f"{k} {v:g}" for k, v in taus))

    # 2. selection runs = same layout AND same tau set
    cand = [t for t in b if t.startswith("t1_") and not t.startswith("t1_confirm_")]
    srec = records(cand) or {}
    sel_tags = sorted(t for t, v in srec.items() if v == (layout, taus))
    if not sel_tags:
        print("FAIL: no selection-seed run matches the confirmed setting's layout and tau set")
        return 1

    sel_lo, sel_hi = rng([b[t]["r_ab"] for t in sel_tags])
    conf_lo, conf_hi = rng([b[t]["r_ab"] for t in confirm])
    alt_lo, alt_hi = rng([b[t]["r_ab"] for t in b if t.startswith("t1_kd1p41_n4000")])
    print(f"  selection runs  : {', '.join(sel_tags)}")
    print(f"  confirmation    : {', '.join(confirm)}")
    print(f"  selection r_ab  : {sel_lo}--{sel_hi}")
    print(f"  fresh r_ab      : {conf_lo}--{conf_hi}")
    print(f"  axis-1 winner, different circuit: {alt_lo}--{alt_hi}")
    if (sel_lo, sel_hi) == (alt_lo, alt_hi):
        print("FAIL: the two circuits score the same; this guard cannot tell them apart")
        return 1

    # 3. scoped text check
    for name in ("abstract", "results", "discussion"):
        p = SEC / f"{name}.tex"
        if not p.exists():
            continue
        flat = " ".join(re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8")).split())
        if f"{sel_lo}--{sel_hi}" not in flat:
            problems.append(f"{name}.tex never states the confirmed range {sel_lo}--{sel_hi}")
        for m in re.finditer(r"[^.]*\.", flat):
            s = m.group(0)
            if not any(w in s.lower() for w in SETTING_WORDS):
                continue
            if f"{alt_lo}--{alt_hi}" in s and f"{sel_lo}--{sel_hi}" not in s:
                problems.append(f"{name}.tex: a sentence about the setting quotes {alt_lo}--{alt_hi} "
                                f"(the axis-1 winner) without {sel_lo}--{sel_hi}: {s.strip()[:120]!r}")

    for pr in problems:
        print(f"FAIL: {pr}")
    print("\nOK: confirmed setting identified from its run records and the text is scoped to it"
          if not problems else "\nPROBLEMS FOUND")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

"""Positive-control thresholds from frozen T8 evidence (t8_summary.json). Usage: T8_SUMMARY_JSON"""
import json
import statistics as st
import sys

s = json.load(open(sys.argv[1], encoding="utf-8"))
ref = [r["r_ab"] for r in s["runs"]["ref"]]
print("T8 ref r_ab per seed", [round(x, 4) for x in ref])
m, sd = st.mean(ref), st.stdev(ref)
print(f"T8 ref mean {m:.4f} seed SD {sd:.4f}; band mean +- 3 SD = [{m - 3 * sd:.4f}, {m + 3 * sd:.4f}]")
for arm in ("rflat", "rrev"):
    d = [a["r_ab"] - b["r_ab"] for a, b in zip(s["runs"]["ref"], s["runs"][arm])]
    c = s["comparisons"][arm]["r_ab"]
    print(f"{arm}: d_s {[round(x, 4) for x in d]} mean {st.mean(d):.4f} SD {st.stdev(d):.4f};"
          f" summary mean {c['mean']:.4f} sd {c['sd']:.4f} decision {c['decision']}")

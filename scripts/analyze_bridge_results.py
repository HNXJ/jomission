import json
import numpy as np

with open('results/plasticity_30s_bridge_results.json') as f:
    res = json.load(f)

probes = res['probe_results']
print("=== PROBE METRICS ACROSS AGES (0, 10, 30 s) ===")
for t in ['0', '10', '30']:
    p = probes[t]
    print(f"Age {t:>2}s:")
    print(f"  Weak Probe (drive=1.0): Mean Rate = {p['weak_mean_rate']:.2f} Hz")
    print(f"    By Class: E = {p['weak_by_class']['E']:.2f} Hz, PV = {p['weak_by_class']['PV']:.2f} Hz, SST = {p['weak_by_class']['SST']:.2f} Hz, VIP = {p['weak_by_class']['VIP']:.2f} Hz")
    print(f"    By Area:  V1 = {p['weak_by_area']['V1']:.2f} Hz, V4 = {p['weak_by_area']['V4']:.2f} Hz, FEF = {p['weak_by_area']['FEF']:.2f} Hz, PFC = {p['weak_by_area']['PFC']:.2f} Hz")
    print(f"  Ordinary Probe (drive=3.0): Mean Rate = {p['ord_mean_rate']:.2f} Hz")
    print(f"    By Class: E = {p['ord_by_class']['E']:.2f} Hz, PV = {p['ord_by_class']['PV']:.2f} Hz, SST = {p['ord_by_class']['SST']:.2f} Hz, VIP = {p['ord_by_class']['VIP']:.2f} Hz")
    print(f"    By Area:  V1 = {p['ord_by_area']['V1']:.2f} Hz, V4 = {p['ord_by_area']['V4']:.2f} Hz, FEF = {p['ord_by_area']['FEF']:.2f} Hz, PFC = {p['ord_by_area']['PFC']:.2f} Hz")
    if t != '0':
        pw = p['weak_vs_0']
        po = p['ord_vs_0']
        print(f"  --> Dynamic Shifts vs t=0s:")
        print(f"      Weak Probe:     Relative Shift = {pw['relative_mean_shift']:+.2%}, Correlation = {pw['correlation']:.4f}")
        print(f"                      Max Abs Shift = {pw['max_abs_shift']:.2f} Hz, Mean Abs = {pw['mean_abs_shift']:.2f} Hz")
        print(f"                      Shift Quantiles: p10 = {pw['shift_quantiles']['p10']:+.2f} Hz, p50 = {pw['shift_quantiles']['p50']:+.2f} Hz, p90 = {pw['shift_quantiles']['p90']:+.2f} Hz")
        print(f"      Ordinary Probe: Relative Shift = {po['relative_mean_shift']:+.2%}, Correlation = {po['correlation']:.4f}")
        print(f"                      Max Abs Shift = {po['max_abs_shift']:.2f} Hz, Mean Abs = {po['mean_abs_shift']:.2f} Hz")
        print(f"                      Shift Quantiles: p10 = {po['shift_quantiles']['p10']:+.2f} Hz, p50 = {po['shift_quantiles']['p50']:+.2f} Hz, p90 = {po['shift_quantiles']['p90']:+.2f} Hz")

print("\n=== CLASS PAIRS (4x4) AT t=30s (BREAKDOWN ACROSS ALL 16 COMBINATIONS) ===")
sm30 = res['summaries']['30']
print(f"{'Class Pair':12s} | {'Edges':5s} | {'Gain':8s} | {'D2':7s} | {'Corr':6s} | {'Delta w':10s} | {'Rec Gain':8s} | {'FF Gain':8s} | {'FB Gain':8s}")
print("-" * 85)
for cp, m in sorted(sm30['by_class_pair'].items()):
    rec_g = sm30['by_class_pair_and_proj'].get(cp, {}).get('recurrent', {}).get('gain', float('nan'))
    ff_g = sm30['by_class_pair_and_proj'].get(cp, {}).get('FF', {}).get('gain', float('nan'))
    fb_g = sm30['by_class_pair_and_proj'].get(cp, {}).get('FB', {}).get('gain', float('nan'))
    rec_s = f"{rec_g:8.4f}" if not np.isnan(rec_g) else "     ---"
    ff_s = f"{ff_g:8.4f}" if not np.isnan(ff_g) else "     ---"
    fb_s = f"{fb_g:8.4f}" if not np.isnan(fb_g) else "     ---"
    print(f"{cp:12s} | {m['n_edges']:5d} | {m['gain']:8.4f} | {m['d2_displacement']:7.4f} | {m['correlation']:6.4f} | {m['delta_w']:+10.6f} | {rec_s} | {ff_s} | {fb_s}")

print("\n=== BOUND / SATURATION FRACTION & SIGN CHANGES ACROSS AGES ===")
print(f"{'Age':5s} | {'Sign Changes':12s} | {'At Floor':10s} | {'At Ceiling':10s} | {'p10':8s} | {'p50':8s} | {'p90':8s}")
print("-" * 75)
for t in res['checkpoints']:
    g = res['summaries'][str(t)]['global']
    sc = g.get('sign_changes', 0)
    af = g.get('frac_at_floor', 0.0)
    ac = g.get('frac_at_ceiling', 0.0)
    q = g.get('quantiles', {})
    print(f"{t:5d} | {sc:12d} | {af:10.4%} | {ac:10.4%} | {q.get('p10', 0):8.4f} | {q.get('p50', 0):8.4f} | {q.get('p90', 0):8.4f}")

import json

with open('results/plasticity_100s_extension_results.json') as f:
    res = json.load(f)

print("=== WEIGHT CHANGE CONCENTRATION AT t=100s ===")
conc = res['summaries']['100']['concentration']
for pct in ['1', '5', '10']:
    c = conc[f'top_{pct}pct']
    print(f"  Top {pct:>2}% ({c['k_edges']} edges): {c['fraction_l1']*100:.2f}% of ||Delta w||_1 | {c['fraction_l2_sq']*100:.2f}% of ||Delta w||_2^2")

print("\n=== PLASTICITY DIRECTION AT t=100s ===")
dir_m = res['summaries']['100']['direction']
e = dir_m['excitatory']
i = dir_m['inhibitory']
print(f"  Excitatory ({e['n_edges']} edges):")
print(f"    Mean Delta |w|:      {e['mean_delta_mag']:+.6f}")
print(f"    Mean Signed Delta w: {e['mean_signed_delta']:+.6f}")
print(f"    Strengthened:        {e['frac_strengthened']*100:.2f}%")
print(f"    Weakened:            {e['frac_weakened']*100:.2f}%")
print(f"    Unchanged:           {e['frac_unchanged']*100:.2f}%")
print(f"  Inhibitory ({i['n_edges']} edges):")
print(f"    Mean Delta |w|:      {i['mean_delta_mag']:+.6f} (magnitude strengthening)")
print(f"    Mean Signed Delta w: {i['mean_signed_delta']:+.6f} (hyperpolarizing shift)")
print(f"    Strengthened Inh:    {i['frac_strengthened_inhibition']*100:.2f}%")
print(f"    Weakened Inh:        {i['frac_weakened_inhibition']*100:.2f}%")
print(f"    Unchanged:           {i['frac_unchanged']*100:.2f}%")

print("\n=== SPARSE FUNCTIONAL RESPONDERS AT t=100s ===")
for p_name in ['weak', 'ord']:
    p = res['probe_results']['100'][f'{p_name}_vs_0']
    print(f"--- {p_name.upper()} PROBE ---")
    print(f"  Global Mean Shift: {p['relative_mean_shift']:+.2%}, Correlation: {p['correlation']:.4f}, Max Abs Shift: {p['max_abs_shift']:.2f} Hz")
    sr = p['sparse_responders']
    for thresh in ['1', '2', '5']:
        s = sr[f'ge_{thresh}Hz']
        print(f"  |Delta r| >= {thresh} Hz: {s['count']}/400 neurons ({s['fraction']*100:.2f}%)")
        print(f"    Neuron IDs: {s['neuron_ids']}")
        print(f"    By Class: {s['by_class']}")
        print(f"    By Area:  {s['by_area']}")
        print(f"    By Layer: {s['by_layer']}")

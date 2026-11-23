"""Generate example HTML for VIS_FOUNDATION_v0 — frozen C019."""
import pathlib

from jomission.network.builder import build_jomission_model
from jomission.visualization.network_viz import hierarchy_fig, motif_matrix_fig, spatial_fig, rf_fig

model = build_jomission_model(n_per_area=100, seed=0)
# also show seed 1 variant for vertical bottleneck exposure? use seed0 canonical

fig_h = hierarchy_fig(model)
fig_m_local = motif_matrix_fig(model, area="V1", conn_type="local", metric="weight")
fig_m_ff = motif_matrix_fig(model, area="V4", conn_type="FF", metric="edge_count", with_controls=False)
fig_m_fbs = motif_matrix_fig(model, area="V1", conn_type="FB", metric="probability")
fig_s_v1 = spatial_fig(model, area="V1", max_edges_plot=1500)
fig_s_all = spatial_fig(model, area=None, edge_distance_threshold=0.12, max_edges_plot=1200)
fig_rf = rf_fig(model)

# Combine into single HTML with sections
out = pathlib.Path("results/visualization/example_network_C019.html")
out.parent.mkdir(parents=True, exist_ok=True)

html_parts = [
    "<html><head><meta charset='utf-8'><title>C019 Network Visualization</title>",
    "<style>body{font-family:system-ui;margin:20px} h2{margin-top:40px} .note{color:#555;font-size:13px}</style>",
    "</head><body>",
    "<h1>VIS_FOUNDATION_v0 — V2 Network Visualization — C019 65b302e8c7cdceb5</h1>",
    "<p class='note'>Generated from builder.py:62,90,395 and jaxfne/emitters.py EdgeList; RF from network/rf.py:47; populations.py:31. Frozen C019 65b302e8c7cdceb5.</p>",
    "<p class='note'>Figure A hierarchy: width→|W| opacity→p dash→FF/FB/local/vertical. Motif matrix answers: What connects to SST in V1 L2/3? Spatial kernel σ=0.08 vs realized. RF 32×32 W_RF 400×1024 E_A/E_B/E_R/E_omission=0.</p>",
]

def add(fig, title):
    html_parts.append(f"<h2>{title}</h2>")
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

add(fig_h, "Figure A — Cortical hierarchy (V1 V4 FEF PFC × L1/L2/3/L4/L5/L6) — width→W opacity→p dash→type — would have made W4 vertical bottleneck obvious")
add(fig_m_local, "Motif matrix — V1 local — weight (source layer,class × target layer,class) — try SST in V1 L2/3 query")
add(fig_m_ff, "Motif matrix — V4 FF edges (V1→V4) — edge count")
add(fig_m_fbs, "Motif matrix — V1 FB probability")
add(fig_s_v1, "Spatial — V1 positions & edges (σ=0.08) — kernel vs realized p(edge|distance)")
add(fig_s_all, "Spatial — ALL areas distance threshold 0.12 — protects against all-to-all")
add(fig_rf, "RF — 32×32 W_RF 400×1024, centers/coverage/overlap, active maps A/B/R, energies E_A/E_B/E_R/E_omission=0 — would have exposed 185× problem")

html_parts.append("</body></html>")
out.write_text("\n".join(html_parts))
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB)")
# quick validation: print stats that would have flagged bottleneck
from jomission.visualization.network_viz import get_motif_stats
stats = get_motif_stats(model)
# vertical edges summary
vert = [(k,v) for k,v in stats.items() if v['conn_type']=='vertical']
ff = [(k,v) for k,v in stats.items() if v['conn_type']=='FF']
print(f"vertical motifs {len(vert)} total edges {sum(v['edges'] for _,v in vert)}")
print(f"FF motifs {len(ff)} total edges {sum(v['edges'] for _,v in ff)}")
# show weakest vertical
weak = sorted(vert, key=lambda kv: kv[1]['edges'])[:3]
for k,v in weak:
    print(f" weak vertical {k} edges {v['edges']} p {v['p']:.3f}")
print("RF energies:", end=" ")
from jomission.network.rf import RFConfig, RFOperator
rf = RFConfig(); op = RFOperator(rf, model)
for sid in ["stimulus_A","stimulus_B","random_stimulus","stimulus_omitted"]:
    d = op.drive_for_stimulus(sid)
    print(sid, float((d**2).sum()), end=" | ")
print()

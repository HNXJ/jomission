"""Master Build Script for the Jomission Interactive Reference Gallery."""

import os
import sys
import time

from jomission.visualization.theme import wrap_figure_with_provenance_html
from jomission.visualization.network_3d import build_3d_network_figure
from jomission.visualization.visual_field import build_visual_field_figure
from jomission.visualization.activity import build_activity_figure
from jomission.visualization.spectral import build_spectral_figure
from jomission.visualization.plasticity import build_plasticity_figure
from jomission.visualization.qualification import build_qualification_figure


def build_all_gallery_figures(out_dir: str = "docs/_static/plotly", docs_index_path: str = "docs/gallery/index.md"):
    print("=== BUILDING JOMISSION INTERACTIVE REFERENCE GALLERY ===")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(docs_index_path), exist_ok=True)

    tasks = [
        ("1. Flagship 3D Network Explorer", "network_3d.html", build_3d_network_figure, "OBSERVED"),
        ("2. Visual Field & RF Architecture", "visual_field_mapping.html", build_visual_field_figure, "DERIVED"),
        ("3. Interactive Raster & Population Rates", "raster_population.html", build_activity_figure, "OBSERVED"),
        ("4. Spectral Response & Time-Frequency", "spectral_response.html", build_spectral_figure, "DERIVED"),
        ("5. Plasticity Trajectory & Circuit Matrix", "plasticity_trajectory.html", build_plasticity_figure, "OBSERVED"),
        ("6. B1-B3 Qualification & Root Causes", "b1_b2_b3_dashboard.html", build_qualification_figure, "OBSERVED"),
    ]

    manifest = []

    for label, filename, builder_fn, evidence in tasks:
        t0 = time.time()
        print(f"Building {label}...")
        try:
            fig, caption, provenance = builder_fn()
            html_content = wrap_figure_with_provenance_html(fig, caption, provenance, evidence)
            out_file = os.path.join(out_dir, filename)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            size_kb = os.path.getsize(out_file) / 1024.0
            dt = time.time() - t0
            print(f"  -> Generated {filename} ({size_kb:.1f} KB in {dt:.2f}s)")
            manifest.append({
                "label": label,
                "filename": filename,
                "caption": caption,
                "evidence": evidence,
                "size_kb": size_kb,
            })
        except Exception as e:
            print(f"  FAILED {label}: {e}")
            raise e

    # Generate docs/gallery/index.md
    print(f"\nGenerating Gallery Index at {docs_index_path}...")
    index_content = """# Jomission Interactive Reference Gallery

Welcome to the interactive reference gallery for **Jomission** and **JaxFNE**. These standalone, publication-grade Plotly visualizations provide full interactive exploration of network architecture, receptive field mapping, population dynamics, spectrolaminar profiles, longitudinal plasticity, and qualification diagnostics.

---

## Flagship Interactive Figures

"""
    for item in manifest:
        title = item["label"].split(". ", 1)[1]
        badge_color = "success" if item["evidence"] == "OBSERVED" else "info"
        index_content += f"""### {item['label']}
<span class="badge badge-{badge_color}">{item['evidence']}</span> • [Open Full Screen Explorer (HTML)](../_static/plotly/{item['filename']})

{item['caption']}

<iframe src="../_static/plotly/{item['filename']}" width="100%" height="740" style="border: 1px solid #30363d; border-radius: 6px; background-color: #0d1117;" loading="lazy"></iframe>

---

"""

    with open(docs_index_path, "w", encoding="utf-8") as f:
        f.write(index_content)
    print(f"Successfully wrote {docs_index_path}")
    print("\n=== GALLERY BUILD COMPLETE: 6 / 6 INTERACTIVE FIGURES GENERATED ===")


if __name__ == "__main__":
    build_all_gallery_figures()

"""Spectrolaminar motif gallery: one page with every E2 motif figure and its null-tested score.

Usage:
  python scripts/e2_gallery.py

Reads results/laminar_e2/*_motif.json (+ the run's .json) and writes results/laminar_e2/gallery.html next to
the figures (untracked, like the figures). Rerun after new runs finish. Exit 0 when written.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "laminar_e2"


def fmt(x, p=3):
    return "–" if x is None else (f"{x:+.{p}f}" if isinstance(x, float) else html.escape(str(x)))


def main() -> None:
    rows, figs = [], []
    for mj in sorted(D.glob("*_motif.json"), key=lambda p: (p.name.startswith("smoke"), p.name.lower())):
        tag = mj.name[: -len("_motif.json")]
        if tag.startswith("_temp"):
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        rec_p = D / f"{tag}.json"
        rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.exists() else {}
        r = m.get("per_cell_mean")
        if r is None:  # pre-revision schema (argmax readout)
            continue
        sc = r["scores"][m.get("fig_bands", "image")]
        pk = r["peak_hz_unit"]
        pg, pa = r.get("unit_power_gamma"), r.get("unit_power_ab")
        units = m["units"]
        ratio_g = f"{pg[units[-1]] / pg[units[0]]:.3g}" if pg else "–"
        ratio_a = f"{pa[units[-1]] / pa[units[0]]:.3g}" if pa else "–"
        sizes = rec.get("sizes", {})
        rows.append(
            f"<tr><td><a href='#{tag}'>{html.escape(tag)}</a></td><td>{rec.get('T_s', '–')}</td>"
            f"<td>{html.escape(json.dumps(rec.get('modifiers', {})))}</td>"
            f"<td>{' / '.join(str(sizes[u]['E']) for u in units) if sizes else '–'}</td>"
            f"<td>{' / '.join(f'{pk[u]:.0f}' for u in units)}</td><td>{ratio_g}</td><td>{ratio_a}</td>"
            f"<td>{fmt(sc['contrast_M'])}</td><td>{fmt(sc['jackknife_se'])}</td><td>{fmt(sc['z'], 1)}</td>"
            f"<td>{sc['shuffle']['p_ge']:.3f}</td><td>{fmt(sc['crossing_2se'])}</td></tr>")
        png = base64.b64encode((D / f"{tag}_motif.png").read_bytes()).decode("ascii")  # self-contained page
        figs.append(f"<h3 id='{tag}'>{html.escape(tag)}</h3><img src='data:image/png;base64,{png}' alt='{tag} motif'>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>E2 motif gallery</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#111;background:#fff}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #ddd}}
.note{{font-size:13px;color:#444;max-width:1000px}}</style></head><body>
<h1>Spectrolaminar motif, model runs (E2, unsealed)</h1>
<p class="note">Readout: scripts/e2_motif_figure.py. RWS LFP proxy per E cell, 21-contact probe, sigma 0.15,
multitaper 2 Hz, bands 10-30 / 50-150 Hz. M = depth slope of (alpha-beta - gamma) relative power; M &gt; 0 is
the motif. z = M / jackknife SE (3-unit null SD of z is 1.6, so |z| &gt; 2 is not significant at 3 units);
shuffle p over unit-depth orders (floor 1/6 at 3 units); crossing needs +-2 SE at both ends.
Unit order superficial / middle / deep. Power ratios are deep / superficial.</p>
<table><tr><th>run</th><th>T (s)</th><th>modifiers</th><th>N_E</th><th>peak Hz</th><th>gamma D/S</th>
<th>ab D/S</th><th>M</th><th>SE</th><th>z</th><th>shuffle p</th><th>crossing</th></tr>
{''.join(rows)}</table>{''.join(figs)}</body></html>"""
    (D / "gallery.html").write_text(page, encoding="utf-8")
    print(f"wrote {D / 'gallery.html'}: {len(rows)} runs")


if __name__ == "__main__":
    main()

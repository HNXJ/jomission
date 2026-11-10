"""Spectrolaminar motif gallery: every E2 motif figure with its null-tested score and its similarity to the
observed LANSA (Mendoza-Halliday et al. 2024) map, sorted by similarity (highest first).

Usage:
  python scripts/e2_gallery.py

Reads results/laminar_e2/*_motif.json, the run's .json/.npz and <tag>_similarity.json (computed with
scripts/e2_motif_similarity.py when missing); writes results/laminar_e2/gallery.html next to the figures
(untracked, like the figures). Sort key: r_proto of the sum-over-cells row. Exit 0 when written.
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
D = Path(__file__).resolve().parents[1] / "results" / "laminar_e2"
SILENT_HZ = 0.1  # an E population below this rate makes the run's map unrepresentative (flagged)
DISCARD_S = 2.0


def fmt(x, p=3):
    return "–" if x is None else (f"{x:+.{p}f}" if isinstance(x, float) else html.escape(str(x)))


def e_rates(tag: str, rec: dict) -> dict[str, float] | None:
    npz = D / f"{tag}.npz"
    if not npz.exists() or "sizes" not in rec:
        return None
    z = np.load(npz)
    c = z["counts"][int(DISCARD_S * 1000 / rec["dt_ms"]):]
    if c.shape[0] == 0:  # run shorter than the discard
        return None
    T =c.shape[0] * rec["dt_ms"] / 1000
    out = {}
    for k, key in enumerate(z["block_keys"]):
        u, t = str(key).split("_")
        if t == "E":
            out[u] = float(c[:, k].sum() / T / rec["sizes"][u]["E"])
    return out


def similarity(tag: str) -> dict | None:
    p = D / f"{tag}_similarity.json"
    if not p.exists():
        import e2_motif_similarity as ms

        try:
            ms.main([tag], DISCARD_S)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"no similarity for {tag}: {e}")
            return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    items = []
    for mj in D.glob("*_motif.json"):
        tag = mj.name[: -len("_motif.json")]
        if tag.startswith("_temp"):
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        r = m.get("per_cell_mean")
        if r is None:  # pre-revision schema (argmax readout)
            continue
        rec_p = D / f"{tag}.json"
        rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.exists() else {}
        sim = similarity(tag)
        rs = sim["sum_over_cells"] if sim else None
        rc = sim["per_cell_mean"] if sim else None
        rates = e_rates(tag, rec)
        units = m["units"]
        silent = [u for u in units if rates and rates[u] < SILENT_HZ]
        sc = r["scores"][m.get("fig_bands", "image")]
        pk = r["peak_hz_unit"]
        pg, pa = r.get("unit_power_gamma"), r.get("unit_power_ab")
        ratio_g = f"{pg[units[-1]] / pg[units[0]]:.3g}" if pg else "–"
        ratio_a = f"{pa[units[-1]] / pa[units[0]]:.3g}" if pa else "–"
        sizes = rec.get("sizes", {})
        flag = f"silent E: {', '.join(silent)}" if silent else ""
        row = (
            f"<tr{' class=bad' if silent else ''}><td><a href='#{tag}'>{html.escape(tag)}</a></td>"
            f"<td><b>{fmt(rs['r_proto']) if rs else '–'}</b></td><td>{fmt(rs['null95']) if rs else '–'}</td>"
            f"<td>{fmt(rs['r_ab']) if rs else '–'}</td><td>{fmt(rs['r_gm']) if rs else '–'}</td>"
            f"<td>{fmt(rs['r_area']) if rs else '–'}</td><td>{fmt(rc['r_proto']) if rc else '–'}</td>"
            f"<td>{html.escape(flag)}</td><td>{rec.get('T_s', '–')}</td>"
            f"<td>{html.escape(json.dumps(rec.get('modifiers', {})))}</td>"
            f"<td>{' / '.join(str(sizes[u]['E']) for u in units) if sizes else '–'}</td>"
            f"<td>{' / '.join(f'{rates[u]:.2f}' for u in units) if rates else '–'}</td>"
            f"<td>{' / '.join(f'{pk[u]:.0f}' for u in units)}</td><td>{ratio_g}</td><td>{ratio_a}</td>"
            f"<td>{fmt(sc['contrast_M'])}</td><td>{fmt(sc['z'], 1)}</td>"
            f"<td>{sc['shuffle']['p_ge']:.3f}</td><td>{fmt(sc['crossing_2se'])}</td></tr>")
        png = base64.b64encode((D / f"{tag}_motif.png").read_bytes()).decode("ascii")  # self-contained page
        fig = (f"<h3 id='{tag}'>{html.escape(tag)} — r_proto {fmt(rs['r_proto']) if rs else '–'}"
               f"{' (' + html.escape(flag) + ')' if flag else ''}</h3>"
               f"<img src='data:image/png;base64,{png}' alt='{tag} motif'>")
        items.append((rs["r_proto"] if rs else -np.inf, tag, row, fig))
    items.sort(key=lambda t: (-t[0], t[1].lower()))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>E2 motif gallery</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#111;background:#fff}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #ddd}}
tr.bad td{{color:#999}}.note{{font-size:13px;color:#444;max-width:1100px}}</style></head><body>
<h1>Spectrolaminar motif, model runs (E2, unsealed), sorted by similarity to the observed map</h1>
<p class="note">Similarity: scripts/e2_motif_similarity.py against the LANSA observed relative-power maps
(Mendoza-Halliday et al. 2024; 21 bins x 1-150 Hz x 11 areas). r_proto = mean over frequency of Pearson r over
the 21 bins against the 11-area mean map, sum-over-cells row (sort key); null95 = 95th percentile over 1000 bin
permutations; r_ab, r_gm = r of the 8-30 and 31-150 Hz depth profiles; r_area = mean over areas; "cell r" =
per-cell-mean row. Even- vs odd-area observed means agree at 0.978. Readout sigma 0.15 (wider sigma raises
an all-units-equal control too, so it is not fitted here). Grey rows: an E population below {SILENT_HZ} Hz,
so the map does not represent that unit's dynamics. Motif columns (scripts/e2_motif_figure.py, per-cell row):
M = depth slope of (alpha-beta - gamma) relative power, z = M / jackknife SE (3-unit null SD of z 1.6), shuffle
p floor 1/6, crossing needs +-2 SE at both ends. Unit order superficial / middle / deep; ratios deep / superficial.</p>
<table><tr><th>run</th><th>r_proto</th><th>null95</th><th>r_ab</th><th>r_gm</th><th>r_area</th><th>cell r</th>
<th>flag</th><th>T (s)</th><th>modifiers</th><th>N_E</th><th>E rate Hz</th><th>peak Hz</th><th>gamma D/S</th>
<th>ab D/S</th><th>M</th><th>z</th><th>shuffle p</th><th>crossing</th></tr>
{''.join(t[2] for t in items)}</table>{''.join(t[3] for t in items)}</body></html>"""
    (D / "gallery.html").write_text(page, encoding="utf-8")
    print(f"wrote {D / 'gallery.html'}: {len(items)} runs; top {items[0][1]} {items[0][0]:+.3f}" if items else "no runs")


if __name__ == "__main__":
    main()

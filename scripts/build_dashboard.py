"""Build docs/DASHBOARD.html from docs/receipts/receipts.json and checklist-score.md.

Self-contained: inline SVG, no external scripts or fonts. Every number comes from the
receipts file or the checklist table; nothing is typed into the HTML by hand.
Usage: python scripts/build_dashboard.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Owner weights (2026-11-04). The build fails if a category's item count x 2 differs.
OWNER_WEIGHTS = {
    "Science program": 40.0,
    "Documentation and reproducibility": 30.0,
    "Dashboard and figures": 20.0,
    "Rest": 10.0,
}


def checklist_rows() -> list[tuple[int, str, str, float]]:
    """Rows of the v2 table: | # | Category | Objective | Score | Evidence |."""
    text = (ROOT / "checklist-score.md").read_text(encoding="utf-8")
    rows = []
    for m in re.finditer(r"^\| (\d+) \| (.*?) \| (.*?) \| ([\d.]+) \| (.*?) \|$", text, flags=re.M):
        rows.append((int(m.group(1)), m.group(2), m.group(3), float(m.group(4))))
    return rows


def check_checklist_summary(rows) -> None:
    """Fail if the 'By category' table or the header total disagree with the item rows."""
    text = (ROOT / "checklist-score.md").read_text(encoding="utf-8")
    cats: dict[str, float] = {}
    for _, c, _, p in rows:
        cats[c] = cats.get(c, 0.0) + p
    for m in re.finditer(r"^\| (.+?) \| ([\d.]+) / ([\d.]+) \| (\d+) \|$", text, flags=re.M):
        name, got, n_items = m.group(1), float(m.group(2)), int(m.group(4))
        if name in cats and abs(got - cats[name]) > 1e-9:
            raise ValueError(f"'By category' table: {name!r} says {got:g}, items sum {cats[name]:g}")
        if name in cats and n_items != sum(1 for r in rows if r[1] == name):
            raise ValueError(f"'By category' table: {name!r} item count {n_items} mismatches rows")
    total = sum(p for _, _, _, p in rows)
    m = re.search(r"Current score: ([\d.]+)/100", text)
    if m and abs(float(m.group(1)) - total) > 1e-9:
        raise ValueError(f"header score {m.group(1)} != item total {total:g}")


def open_stack() -> list[dict]:
    """Rows of docs/OPEN.md: id, status, what is open (first 240 chars), who decides."""
    out = []
    for line in (ROOT / "docs/OPEN.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| OPEN-"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split(" | ")]
        text = c[2] if len(c) >= 3 else ""
        out.append({"id": c[0], "status": c[1], "open": text[:240] + ("..." if len(text) > 240 else ""),
                    "decides": c[-2] if len(c) >= 5 else "-"})
    return out


def results_inventory() -> list[dict]:
    """Every top-level results/*.json: parse state, verdict field, sealed status. Sorted by path."""
    reg = json.loads((ROOT / "manifests/sealed_artifacts.json").read_text(encoding="utf-8"))
    sealed = {a["path"] for a in reg["artifacts"]}
    rows = []
    # Case-insensitive key: Windows path order, so the page is byte-identical on every OS.
    for f in sorted((ROOT / "results").glob("*.json"), key=lambda p: p.name.lower()):
        rel = f.relative_to(ROOT).as_posix()
        raw = f.read_bytes()
        enc, parsed, verdict = "utf-8", None, "-"
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            enc = "cp1252"
            try:
                parsed = json.loads(raw.decode("cp1252"))
            except Exception:
                enc = "unparsed"
        except Exception:
            enc = "unparsed"
        if isinstance(parsed, dict):
            for key in ("verdict", "status"):
                if isinstance(parsed.get(key), str):
                    verdict = parsed[key]
                    break
        rows.append({"path": rel, "bytes": len(raw), "encoding": enc, "verdict": verdict, "sealed": rel in sealed})
    return rows


def figures() -> list[tuple[str, int]]:
    out = []
    for f in sorted((ROOT / "docs/_static").rglob("*"), key=lambda p: p.as_posix().lower()):
        if f.is_file() and f.suffix.lower() in (".html", ".png", ".svg", ".pdf"):
            out.append((f.relative_to(ROOT).as_posix(), f.stat().st_size))
    return out


def bar(x: float, y: float, w: float, h: float, pct: float, label: str, color: str) -> str:
    filled = max(0.0, min(1.0, pct)) * w
    return (f'<text x="{x-8}" y="{y+h*0.7}" text-anchor="end" font-size="12">{html.escape(label)}</text>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#eee"/>'
            f'<rect x="{x}" y="{y}" width="{filled:.1f}" height="{h}" fill="{color}"/>')


def score_svg(rows) -> str:
    cats: dict[str, list[float]] = {}
    for _, c, _, p in rows:
        cats.setdefault(c, []).append(p)
    out, y = [], 30
    for name, pts in cats.items():
        got, mx = sum(pts), 2.0 * len(pts)
        if name in OWNER_WEIGHTS and mx != OWNER_WEIGHTS[name]:
            raise ValueError(f"category {name!r}: max {mx:g} != owner weight {OWNER_WEIGHTS[name]:g}")
        out.append(bar(250, y, 360, 22, got / mx if mx else 0, f"{name} {got:.0f}/{mx:.0f}", "#2b6cb0"))
        y += 40
    return f'<svg width="660" height="{y}" font-family="sans-serif">{"".join(out)}</svg>'


def tier_svg(tiers: list[dict]) -> str:
    out, y = [], 30
    for t in tiers:
        if t["passed"] is None:
            out.append(f'<text x="8" y="{y+16}" font-size="12">tier {t["tier"]}: {html.escape(t["reason"])}</text>')
        else:
            run = t["passed"] + t["failed"]  # skipped tests are not run, so they are not in the bar
            label = f'tier {t["tier"]} {t["passed"]}/{run}' + (f' (+{t["skipped"]} skipped)' if t["skipped"] else "")
            out.append(bar(190, y, 420, 22, t["passed"] / run if run else 0,
                           label, "#38a169" if t["failed"] == 0 else "#c53030"))
        y += 40
    return f'<svg width="660" height="{y}" font-family="sans-serif">{"".join(out)}</svg>'


def item_svg(rows) -> str:
    out, y = [], 16
    for n, _, _, p in rows:
        out.append(bar(180, y, 300, 10, p / 2.0, f"{n}", "#2b6cb0" if p >= 2.0 else "#d69e2e"))
        y += 14
    return f'<svg width="520" height="{y}" font-family="sans-serif" font-size="9">{"".join(out)}</svg>'


def main() -> None:
    rec = json.loads((ROOT / "docs/receipts/receipts.json").read_text(encoding="utf-8"))
    rows = checklist_rows()
    check_checklist_summary(rows)
    inv = results_inventory()
    inv_rows = "".join(f"<tr><td>{html.escape(r['path'])}</td><td>{r['bytes']}</td><td>{r['encoding']}</td><td>{html.escape(r['verdict'])}</td><td>{'yes' if r['sealed'] else 'no'}</td></tr>" for r in inv)
    fig_rows = "".join(f"<tr><td>{html.escape(p)}</td><td>{n}</td></tr>" for p, n in figures())
    total = sum(p for _, _, _, p in rows)
    ei = rec["edge_identity"]
    open_rows = "".join(f"<li>item {n}: {html.escape(obj)} ({pts:g}/2)</li>" for n, _, obj, pts in rows if pts < 2.0)
    stack = open_stack()
    stack_rows = "".join(
        f"<tr><td>{html.escape(r['id'])}</td><td>{html.escape(r['status'])}</td><td>{html.escape(r['open'])}</td><td>{html.escape(r['decides'])}</td></tr>"
        for r in stack)
    dec_rows = "".join(
        f"<tr><td>{html.escape(d['id'])}</td><td>{html.escape(d['text'])}</td><td>{html.escape(d['status'])}</td></tr>"
        for d in rec["owner_decisions"])
    tier_rows = "".join(
        f"<tr><td>{t['tier']}</td><td>{html.escape(t['name'])}</td><td>{t['passed'] if t['passed'] is not None else '-'}</td>"
        f"<td>{t['failed'] if t['failed'] is not None else '-'}</td><td>{t['skipped'] if t['skipped'] is not None else '-'}</td>"
        f"<td>{html.escape(t['status'])}</td><td>{html.escape(t.get('commit') or '-')}</td></tr>"
        for t in rec["tiers"])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>jomission dashboard</title>
<style>
:root{{--fg:#1a202c;--mut:#4a5568;--bg:#fff;--card:#f7fafc}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--fg:#e2e8f0;--mut:#a0aec0;--bg:#1a202c;--card:#2d3748}}}}
:root[data-theme="dark"]{{--fg:#e2e8f0;--mut:#a0aec0;--bg:#1a202c;--card:#2d3748}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:24px}}
.card{{background:var(--card);border-radius:8px;padding:12px;margin:10px 0;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #cbd5e0;padding:4px;text-align:left}}
.big{{font-size:34px;font-weight:700}} .mut{{color:var(--mut);font-size:13px}}
</style></head><body>
<h1>jomission dashboard: jaxfne 0.5.8 migration and checklist</h1>
<p class="mut">Generated by scripts/build_dashboard.py from docs/receipts/receipts.json and checklist-score.md. Engine pin {html.escape(rec['engine']['pin'])}. Each tier row carries the commit it was run on.</p>
<div class="card"><div class="big">{total:.1f} / 100</div><div class="mut">checklist score (v4, 50 items x 2, re-weighted 2026-11-04)</div>{score_svg(rows)}</div>
<h2>Test tiers</h2><div class="card">{tier_svg(rec['tiers'])}
<table><tr><th>tier</th><th>name</th><th>passed</th><th>failed</th><th>skipped</th><th>status</th><th>commit</th></tr>{tier_rows}</table></div>
<h2>Edge identity: jaxfne 0.4.25 vs 0.5.8</h2>
<div class="card"><p>Canonical network edge list: {ei['fields_compared']} fields compared, mismatched: {('none' if not ei['mismatched'] else ', '.join(ei['mismatched']))}; edges {ei['n_edges_0425']} (0.4.25) vs {ei['n_edges_058']} (0.5.8); exit code {ei['exit_code']}.</p>
<p class="mut">Method: {html.escape(ei['method'])}.</p>
<p class="mut">Frozen config_hash: 0.4.25 {rec['config_hash']['0.4.25']}, 0.5.8 {rec['config_hash']['0.5.8']}. {html.escape(rec['config_hash']['note'])}.</p></div>
<h2>CI and project check</h2><div class="card"><p>CI {html.escape(rec['ci']['job'])}: {html.escape(rec['ci']['result'])} (run {rec['ci']['check_run']}, head {html.escape(rec['ci']['head_observed'])}). Project check: {html.escape(rec['project_check']['result'])} at {html.escape(rec['project_check']['commit'])}.</p></div>
<h2>Lite model</h2><div class="card"><p>Areas {', '.join(rec['lite_model']['areas'])}, n_per_area {rec['lite_model']['n_per_area']}, {rec['lite_model']['neurons']} neurons: {html.escape(rec['lite_model']['status'])}. Test: {html.escape(rec['lite_model']['test'])}.</p></div>
<h2>Checklist items</h2><div class="card">{item_svg(rows)}<p class="mut">Blue: full credit. Amber: partial or open. Numbers are checklist item numbers.</p></div>
<h2>Owner decisions</h2><div class="card"><table><tr><th>id</th><th>decision</th><th>status</th></tr>{dec_rows}</table></div>
<h2>Open (every checklist item below 2 points)</h2><div class="card"><ul>{open_rows}</ul></div>
<h2>Open stack (docs/OPEN.md, all {len(stack)} rows)</h2><div class="card"><table><tr><th>id</th><th>status</th><th>what is open</th><th>decides</th></tr>{stack_rows}</table></div>
<h2>Results inventory (all results/*.json)</h2><div class="card"><p class="mut">{len(inv)} files; {sum(1 for r in inv if r["verdict"] != "-")} carry a verdict or status field; {sum(1 for r in inv if r["sealed"])} sealed; {sum(1 for r in inv if r["encoding"]=="cp1252")} decode as cp1252 only.</p><table><tr><th>file</th><th>bytes</th><th>encoding</th><th>verdict/status</th><th>sealed</th></tr>{inv_rows}</table></div>
<h2>Figures (docs/_static)</h2><div class="card"><table><tr><th>file</th><th>bytes</th></tr>{fig_rows}</table></div>
</body></html>
"""
    (ROOT / "docs/DASHBOARD.html").write_bytes(page.encode("utf-8"))
    print("wrote docs/DASHBOARD.html", f"checklist total {total:.1f}")


if __name__ == "__main__":
    main()

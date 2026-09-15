"""Jomission Pages atlas generator (workstream B, read-only publication layer).

Reads: site-src/ (manifest, templates, pages, style.css),
       results/<allowlisted>.json, manifests/<allowlisted>.json,
       docs/_static/plotly/<allowlisted>.html.
Writes: ONLY into --out (default site/).
Never runs simulations. Never writes outside --out.
Deterministic: sorted keys, fixed number formatting, no timestamps.
Commit identity via git rev-parse (override JOMISSION_PAGES_COMMIT for tests).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "site-src")

STATUS_COLOR = {
    "PASS": "#4ade80", "FAIL": "#f43f5e", "LOCKED": "#8b949e",
    "SUPERSEDED": "#94a3b8",
}

EVIDENCE_BADGE = ("<span class=\"badge badge-{e}\">{e}</span>")


def esc(x):
    return html.escape(str(x), quote=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def repo_commit():
    override = os.environ.get("JOMISSION_PAGES_COMMIT")
    if override:
        return override
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "not recorded"


def fmt(x, digits=4):
    if x is None:
        return "not recorded"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return esc(x)


def svg_open(w, h, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">'
        f"<title>{esc(title)}</title>"
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#0d1117"/>'
    )


def svg_line_chart(series, w=560, h=300, title="", xlabel="", ylabel="",
                   xlog=False, colors=None):
    """series: list of (label, xs, ys). Hand-rolled deterministic SVG."""
    colors = colors or ["#38bdf8", "#4ade80", "#fb923c", "#c084fc", "#f43f5e"]
    pad_l, pad_r, pad_t, pad_b = 64, 16, 14, 40
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    allx = [x for _, xs, _ in series for x in xs]
    ally = [y for _, _, ys in series for y in ys]
    if not allx or not ally:
        return svg_open(w, h, title) + "</svg>"
    import math
    if xlog:
        allx = [max(x, 1e-12) for x in allx]
        lx0, lx1 = math.log10(min(allx)), math.log10(max(allx))
        span = (lx1 - lx0) or 1.0

        def sx(x):
            return pad_l + (math.log10(max(x, 1e-12)) - lx0) / span * iw
    else:
        x0, x1 = min(allx), max(allx)
        span = (x1 - x0) or 1.0

        def sx(x):
            return pad_l + (x - x0) / span * iw
    y0, y1 = min(0.0, min(ally)), max(ally)
    yspan = (y1 - y0) or 1.0

    def sy(y):
        return pad_t + ih - (y - y0) / yspan * ih

    s = [svg_open(w, h, title)]
    desc = "; ".join(
        f"{label}: " + ",".join(f"({x:.4g},{y:.4g})" for x, y in zip(xs, ys))
        for label, xs, ys in series)
    s.append(f"<desc>source values: {esc(desc)}</desc>")
    s.append(f'<text x="{pad_l}" y="{h - 8}" fill="#8b949e" font-size="12">{esc(xlabel)}</text>')
    s.append(f'<text x="10" y="{pad_t + 10}" fill="#8b949e" font-size="12">{esc(ylabel)}</text>')
    for (label, xs, ys), c in zip(series, colors):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{c}" '
                     f'data-x="{x:.4g}" data-y="{y:.4g}"/>')
    lx = pad_l
    for (label, _, _), c in zip(series, colors):
        s.append(f'<circle cx="{lx}" cy="18" r="4" fill="{c}"/>'
                 f'<text x="{lx + 8}" y="22" fill="#c9d1d9" font-size="12">{esc(label)}</text>')
        lx += 8 + len(label) * 7 + 18
    return "".join(s) + "</svg>"


def svg_gate_graph(gates, historical, w=900, h=None):
    n = len(gates)
    bw, bh, gap, pad = 150, 54, 26, 20
    h = pad * 2 + bh + (60 if historical else 0)
    s = [svg_open(w, h, "gate graph")]
    x = pad
    for g in gates:
        color = STATUS_COLOR.get(g["status"], "#8b949e")
        s.append(f'<rect x="{x}" y="{pad}" width="{bw}" height="{bh}" rx="6" '
                 f'fill="#161b22" stroke="{color}" stroke-width="2"/>')
        s.append(f'<text x="{x + bw / 2}" y="{pad + 22}" fill="#c9d1d9" font-size="11" '
                 f'text-anchor="middle">{esc(g["label"])}</text>')
        s.append(f'<text x="{x + bw / 2}" y="{pad + 42}" fill="{color}" font-size="12" '
                 f'text-anchor="middle" font-weight="bold">{esc(g["status"])}</text>')
        x += bw + gap
    if historical:
        gh = historical[0]
        y0 = pad + bh + 24
        s.append(f'<text x="{pad}" y="{y0}" fill="#94a3b8" font-size="12">Historical (not current): '
                 f'{esc(gh["label"])} — {esc(gh["status"])}. {esc(gh.get("note", ""))}</text>')
    return "".join(s) + "</svg>"


def render_gate_graph(ctx):
    svg = svg_gate_graph(ctx["manifest"]["gates"], ctx["manifest"].get("gates_historical", []))
    return "assets/svg/gate_graph.svg", svg


def _probe_rows(nt):
    order = ["rest", "below", "transition", "active", "above"]
    rows = []
    for name in order:
        r = nt.get(name)
        if isinstance(r, dict):
            rows.append((name, r))
    return rows


def render_g_curve(ctx):
    nt = load_json(os.path.join(REPO, "results", "native_transformation.json"))
    rows = _probe_rows(nt)
    xs = [r["rate"] for _, r in rows]
    ws = [r["w"] for _, r in rows]
    svg = svg_line_chart([("w_EE(rate)", xs, ws)], title="native EE efficacy vs rate",
                         xlabel="driver rate (Hz)", ylabel="w_EE (native units)")
    return "assets/svg/g_curve.svg", svg


def render_i_curve(ctx):
    nt = load_json(os.path.join(REPO, "results", "native_transformation.json"))
    rows = _probe_rows(nt)
    xs = [r["rate"] for _, r in rows]
    cur = [r["I_exact"] for _, r in rows]
    svg = svg_line_chart([("I_EE(rate)", xs, cur)], title="realized EE current vs rate",
                         xlabel="driver rate (Hz)", ylabel="I_EE (exact offline, a.u.)",
                         colors=["#4ade80"])
    return "assets/svg/i_curve.svg", svg


def render_f_curves(ctx):
    b1 = load_json(os.path.join(REPO, "results", "battery_b1.json"))
    series = []
    for c in ("E", "PV", "SST", "VIP"):
        f = b1["F"][c]
        series.append((c, f["I"], f["r"]))
    svg = svg_line_chart(series, title="single-cell F-I per class (fresh)",
                         xlabel="input current (a.u.)", ylabel="rate (Hz)")
    return "assets/svg/f_curves.svg", svg


def render_susceptibility(ctx):
    su = load_json(os.path.join(REPO, "results", "susceptibility.json"))
    curve = su["curve"]
    xs = sorted(float(k) for k in curve)
    ys = [curve[str(x) if str(x) in curve else x] for x in xs]
    svg = svg_line_chart([("V4 rate", xs, ys)], title="V4 dose-response (direct current)",
                         xlabel="added current (units)", ylabel="V4 rate (Hz)",
                         colors=["#fb923c"])
    return "assets/svg/susceptibility.svg", svg


def render_attractor_diagram(ctx):
    at = load_json(os.path.join(REPO, "results", "attractor_solve.json"))
    fp = at["solution"]["fp"]
    w, h = 560, 320
    pad = 56
    x0, x1 = 0.0, 40.0

    def sx(r):
        return pad + (r - x0) / (x1 - x0) * (w - pad - 20)

    s = [svg_open(w, h, "reduced attractor schematic (derived diagram)")]
    s.append(f'<text x="{pad}" y="{h - 12}" fill="#8b949e" font-size="12">rE (Hz)</text>')
    for lo, hi, label in [(2, 4, "lower separatrix"), (25, 35, "upper separatrix")]:
        s.append(f'<rect x="{sx(lo):.1f}" y="30" width="{sx(hi) - sx(lo):.1f}" height="{h - 80}" '
                 f'fill="#fb923c" opacity="0.18"/>')
        s.append(f'<text x="{(sx(lo) + sx(hi)) / 2:.1f}" y="24" fill="#fb923c" font-size="11" '
                 f'text-anchor="middle">{label}</text>')
    s.append(f'<circle cx="{sx(0):.1f}" cy="{h - 60:.1f}" r="6" fill="#8b949e"/>')
    s.append(f'<text x="{sx(0):.1f}" y="{h - 44:.1f}" fill="#8b949e" font-size="11" '
             f'text-anchor="middle">silent co-attractor</text>')
    s.append(f'<circle cx="{sx(fp[0]):.1f}" cy="120" r="7" fill="#4ade80"/>')
    s.append(f'<text x="{sx(fp[0]):.1f}" y="104" fill="#4ade80" font-size="12" '
             f'text-anchor="middle">active fp ({fp[0]}, {fp[1]})</text>')
    s.append(f'<text x="{sx(fp[0]):.1f}" y="140" fill="#8b949e" font-size="11" '
             f'text-anchor="middle">maxRe {at["solution"]["maxRe"]}, capture {at["solution"]["capture"]}</text>')
    return "assets/svg/attractor.svg", "".join(s) + "</svg>"


def render_prep_transition(ctx):
    series = []
    for tag, s in (("s50p0", 50), ("s54p0", 54), ("s57p0", 57), ("s61p0", 61)):
        try:
            d = load_json(os.path.join(REPO, "results", f"s9_init_{tag}_P1.json"))
        except FileNotFoundError:
            continue
        scan = d.get("prep_scan", [])
        xs = [r["amp"] for r in scan]
        ys = [r["rate"] for r in scan]
        series.append((f"s_E={s}", xs, ys))
    svg = svg_line_chart(series, title="prep steering: settled E rate vs drive amp",
                         xlabel="prep drive amp (E cells)", ylabel="settled E rate (Hz)")
    return "assets/svg/prep_transition.svg", svg


def render_ladder_outcome(ctx):
    series = []
    for tag, s in (("s50p0", 50), ("s54p0", 54), ("s57p0", 57), ("s61p0", 61)):
        xs, ys = [], []
        for T in (0.5, 1.0, 2.0, 4.0, 8.0):
            try:
                d = load_json(os.path.join(
                    REPO, "results", f"b_ladder_{tag}_{T}.json"))
            except FileNotFoundError:
                continue
            xs.append(T)
            ys.append(d.get("rE_release", 0.0))
        series.append((f"s_E={s}", xs, ys))
    svg = svg_line_chart(series, title="duration ladder: release E rate vs drive duration",
                         xlabel="prep duration T (s)", ylabel="release E rate (Hz)")
    return "assets/svg/ladder_outcome.svg", svg


RENDERERS = {
    "gate_graph": render_gate_graph,
    "g_curve": render_g_curve,
    "i_curve": render_i_curve,
    "f_curves": render_f_curves,
    "susceptibility": render_susceptibility,
    "attractor_diagram": render_attractor_diagram,
    "prep_transition": render_prep_transition,
    "ladder_outcome": render_ladder_outcome,
}


def panel_html(panel, inner, commit):
    ev = panel.get("evidence", "not recorded")
    src = panel.get("source") or "manifest state specification"
    status = panel.get("status")
    badge = EVIDENCE_BADGE.format(e=esc(ev))
    title = esc(panel.get("title", panel.get("id", "panel")))
    foot = f"Evidence: {esc(ev)} · Source: <code>{esc(src)}</code> · commit <code>{esc(commit)}</code>"
    if status:
        foot += f" · Status: {esc(status)}"
    return (f'<section class="panel" id="{esc(panel.get("id", ""))}">'
            f"<h3>{badge}{title}</h3>{inner}"
            f'<footer class="panel-foot">{foot}</footer></section>')


def render_table(panel, ctx):
    src = panel.get("source")
    data = load_json(os.path.join(REPO, src)) if src else {}
    keys = panel.get("keys", [])
    # List-of-dicts value renders as a multi-row table.
    for k in keys:
        v = data
        for part in k.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
            cols = sorted({c for r in v for c in r})
            th = "".join(f"<th>{esc(c)}</th>" for c in cols)
            trs = "".join("<tr>" + "".join(
                f"<td>{esc(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in v)
            return (f'<table class="data"><thead><tr>{th}</tr></thead>'
                    f"<tbody>{trs}</tbody></table>")
    rows = []
    for k in keys:
        v = data
        for part in k.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if v is None:
            raise ValueError(f"panel {panel.get('id')}: key {k!r} missing in {src}")
        rows.append((k, v))
    trs = "".join(
        f"<tr><td><code>{esc(k)}</code></td><td>{fmt(v) if not isinstance(v, (dict, list)) else esc(json.dumps(v, sort_keys=True))}</td></tr>"
        for k, v in rows)
    return f'<table class="data"><thead><tr><th>quantity</th><th>value</th></tr></thead><tbody>{trs}</tbody></table>'


def render_lineage(ctx):
    edges = ctx["manifest"].get("lineage_edges", [])
    rows = []
    for e in edges:
        c = e["commit"]
        r = subprocess.run(["git", "cat-file", "-e", c], cwd=REPO,
                           capture_output=True)
        if r.returncode != 0:
            raise ValueError(f"lineage edge references unknown commit {c!r}")
        subject = subprocess.run(
            ["git", "log", "-1", "--format=%s", c], cwd=REPO,
            capture_output=True, text=True, check=True).stdout.strip()
        rows.append((c, subject, e.get("gate", ""), e.get("verdict", ""),
                     e.get("evidence", "")))
    trs = "".join(
        f"<tr><td><code>{esc(c)}</code></td><td>{esc(s)}</td>"
        f"<td>{esc(g)}</td><td>{esc(v)}</td><td><code>{esc(e)}</code></td></tr>"
        for c, s, g, v, e in rows)
    return (f'<table class="data"><thead><tr><th>commit</th><th>subject</th>'
            f"<th>gate</th><th>verdict</th><th>evidence</th></tr></thead>"
            f"<tbody>{trs}</tbody></table>")


def build(out_dir, commit=None):
    commit = commit or repo_commit()
    manifest = load_json(os.path.join(SRC, "manifest.json"))
    # Verify allowlisted artifacts exist and parse.
    for fn in manifest["results_allowlist"]:
        load_json(os.path.join(REPO, "results", fn))
    for fn in manifest["results_manifests_allowlist"]:
        load_json(os.path.join(REPO, "manifests", fn))
    for fn in manifest["plotly_allowlist"]:
        p = os.path.join(REPO, "docs", "_static", "plotly", fn)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"plotly allowlist missing: {fn}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    with open(os.path.join(SRC, "templates", "shell.html"), encoding="utf-8") as f:
        shell = f.read()
    shutil.copy(os.path.join(SRC, "style.css"), os.path.join(out_dir, "style.css"))
    # Copy allowlisted data + plotly.
    for fn in manifest["results_allowlist"]:
        d = os.path.join(out_dir, manifest["build"]["results_subdir"])
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(REPO, "results", fn), os.path.join(d, fn))
    for fn in manifest["results_manifests_allowlist"]:
        d = os.path.join(out_dir, manifest["build"]["manifests_subdir"])
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(REPO, "manifests", fn), os.path.join(d, fn))
    for fn in manifest["plotly_allowlist"]:
        d = os.path.join(out_dir, manifest["build"]["plotly_subdir"])
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(REPO, "docs", "_static", "plotly", fn), os.path.join(d, fn))
    os.makedirs(os.path.join(out_dir, manifest["build"]["svg_subdir"]), exist_ok=True)
    pages = manifest["pages"]
    titles = {}
    for p in pages:
        with open(os.path.join(SRC, "pages", p + ".html"), encoding="utf-8") as f:
            frag = f.read()
        m = re.search(r"<h1>(.*?)</h1>", frag, re.S)
        titles[p] = (m.group(1).strip() if m else p)
    ctx = {"manifest": manifest, "commit": commit, "_out": out_dir}
    rendered_panels = set()
    for p in pages:
        with open(os.path.join(SRC, "pages", p + ".html"), encoding="utf-8") as f:
            frag = f.read()

        def slot(m):
            pid = m.group(1)
            panel = manifest.get("panels", {}).get(pid)
            if panel is None:
                raise ValueError(f"page {p}: panel {pid!r} not in manifest registry")
            if panel.get("page", p) != p:
                raise ValueError(f"panel {pid!r} registered to different page")
            rendered_panels.add(pid)
            return render_panel(panel, ctx)

        frag2, n = re.subn(r'<div class="panel-slot" data-panel="([\w\-]+)"></div>', slot, frag)
        page_nav = "".join(
            f'<a href="{q}.html"{">" if q != p else " class=\"active\">"}{esc(titles[q])}</a> '
            for q in pages)
        page = shell.replace("{{title}}", esc(titles[p])).replace(
            "{{root}}", "").replace("{{nav}}", page_nav).replace(
            "{{content}}", frag2).replace("{{commit}}", esc(commit))
        check_output(page, manifest, out_dir)
        with open(os.path.join(out_dir, p + ".html"), "w", encoding="utf-8") as f:
            f.write(page)
    # Every registered panel must be used.
    unused = set(manifest.get("panels", {})) - rendered_panels
    if unused:
        raise ValueError(f"manifest panels never rendered: {sorted(unused)}")
    return out_dir


def render_panel(panel, ctx):
    out = ctx["_out"]
    ptype = panel.get("type", "figure")
    if ptype == "figure":
        renderer = RENDERERS.get(panel.get("renderer", ""))
        if renderer is None:
            raise ValueError(f"panel {panel.get('id')}: unknown renderer")
        rel, svg = renderer(ctx)
        with open(os.path.join(out, rel), "w", encoding="utf-8") as f:
            f.write(svg)
        inner = f"<figure><img src=\"{rel}\" alt=\"{esc(panel.get('title', ''))}\"></figure>"
        if panel.get("note"):
            inner += f'<div class="status-note">{esc(panel["note"])}</div>'
        return panel_html(panel, inner, ctx["commit"])
    if ptype == "table":
        return panel_html(panel, render_table(panel, ctx), ctx["commit"])
    if ptype == "lineage":
        return panel_html(panel, render_lineage(ctx), ctx["commit"])
    if ptype == "embed":
        fn = panel["plotly"]
        if fn not in ctx["manifest"]["plotly_allowlist"]:
            raise ValueError(f"embed {fn} not allowlisted")
        rel = ctx["manifest"]["build"]["plotly_subdir"] + "/" + fn
        note = panel.get("note", "")
        inner = (f'<iframe src="{rel}" loading="lazy"></iframe>'
                 + (f'<div class="status-note">{esc(note)}</div>' if note else ""))
        return panel_html(panel, inner, ctx["commit"])
    raise ValueError(f"panel {panel.get('id')}: unknown type {ptype!r}")


ABS_PATH_RE = re.compile(r"[A-Z]:[\\/]|/home/|/mnt/|/tmp/|\\\\")


def check_output(page, manifest, out_dir):
    for bad in manifest.get("prohibited_strings", []):
        if bad in page:
            raise ValueError(f"prohibited pre-freeze string in output: {bad!r}")
    if ABS_PATH_RE.search(page):
        m = ABS_PATH_RE.search(page)
        raise ValueError(f"absolute path leak in output near: {page[m.start():m.start()+80]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--commit", default=None)
    args = ap.parse_args()
    out = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    build(out, commit=args.commit)
    print(f"site built: {out}")


if __name__ == "__main__":
    main()

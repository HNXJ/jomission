"""Build a multi-page project dashboard (one self-contained index.html, tabs as pages).

Pages: overview, omission paradigm, LANSA and laminar motif, figures, gates and lineages, todo and open.
Every value is read from a repo file by this script; nothing is typed into the page. Text from
data enters the DOM through textContent. Style follows the omission Draft-S dashboard: colors and
fonts from host CSS variables with fallbacks, light and dark, ids in monospace.

Omission firewall: the page shows the paradigm definition and the LOCKED status of blind
omission only. The build fails if any `prohibited_strings` entry of site-src/manifest.json
appears in the output.

Usage: python scripts/build_series_dashboard.py --out <dir>      (writes <dir>/index.html)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def open_rows() -> list[dict]:
    """Rows of docs/OPEN.md: | OPEN-n | kind | text | ... |."""
    rows = []
    for line in (ROOT / "docs/OPEN.md").read_text(encoding="utf-8").splitlines():
        if re.match(r"^\| OPEN-\d+ \|", line):
            c = _cells(line)
            rows.append({"id": c[0], "kind": c[1], "text": c[2], "owner": c[-2] if len(c) > 5 else ""})
    return rows


def plan_rows() -> list[dict]:
    """Checklist rows of docs/PLAN_LAMINAR_MOTIF.md: | # | Item | Closing check | Builder | Rev A | Rev B |."""
    rows = []
    for line in (ROOT / "docs/PLAN_LAMINAR_MOTIF.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| (\d+) \| (.*) \| (\d+) \| (\d+) \| (\d+) \|$", line)
        if m:
            c = _cells(line)
            rows.append({"n": int(c[0]), "item": c[1], "check": c[2],
                         "builder": int(c[3]), "revA": int(c[4]), "revB": int(c[5])})
    return rows


def firewall_rule() -> str:
    for line in (ROOT / "AGENTS.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("- Omission firewall:"):
            return line[2:]
    raise SystemExit("AGENTS.md has no 'Omission firewall' line")


def head() -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or "unknown"


def collect() -> dict:
    cs = _json("manifests/current_state.json")
    td = _json("manifests/todo.json")
    lr = _json("manifests/lineage_registry.json")
    sm = _json("site-src/manifest.json")
    par = _json("manifests/milestone_01_short_trajectory.json")["paradigm_exact"]
    plan = plan_rows()
    opens = open_rows()
    statuses: dict[str, int] = {}
    for it in td["items"]:
        statuses[it["status"]] = statuses.get(it["status"], 0) + 1
    seal = (ROOT / "docs/laminar_motif/ITEM03_SEAL_PROPOSAL.md").read_text(encoding="utf-8")
    return {
        "stamp": {"head": head(), "state_updated": cs["updated"]},
        "overview": {
            "jaxfne_version": cs["jaxfne_version"],
            "next_authorized_task": cs["next_authorized_task"],
            "current_gate": cs["current_gate"],
            "todo_by_status": statuses,
            "open_rows": len(opens),
            "lineages": len(lr["lineages"]),
            "builder_score": sum(r["builder"] for r in plan),
            "plan_items": len(plan),
        },
        "omission": {
            "paradigm": par["checks"]["paradigm_name"],
            "n_conditions": par["checks"]["n_conditions"],
            "n_epochs": par["checks"]["n_epochs"],
            "unresolved_fields": par["checks"]["unresolved_fields"],
            "paradigm_note": par["paradigm_exact"],
            "blind_omission": [g for g in sm["gates"] if g["id"] == "omission"],
            "firewall": firewall_rule(),
            "prohibited_count": len(sm["prohibited_strings"]),
        },
        "laminar": {
            "plan": plan,
            "scores": {"builder": sum(r["builder"] for r in plan),
                       "revA": sum(r["revA"] for r in plan),
                       "revB": sum(r["revB"] for r in plan)},
            "unset_mentions": len(re.findall(r"\bUNSET\b", seal)),
            "open_rows": [r for r in opens if r["id"] in ("OPEN-36", "OPEN-37", "OPEN-38", "OPEN-39", "OPEN-40", "OPEN-41")],
            "minimal": _json("docs/laminar_motif/MINIMAL_MODEL_STATUS.json"),
        },
        "gates": {
            "verdicts": [{"id": k, **v} for k, v in cs["latest_verdict"].items()],
            "lineages": [{"id": x["lineage_id"], "parent": x.get("parent", ""),
                          "delta": x.get("principal_delta", "")} for x in lr["lineages"]],
        },
        "todo": {
            "items": [{"id": i["id"], "status": i["status"], "goal": i.get("goal", "")} for i in td["items"]],
            "open": opens,
        },
        "figures": figures(),
    }


def figures() -> dict:
    """Figure plan (docs/FIGURE_PLAN.json): per series, the figures with status and blockers."""
    fp = _json("docs/FIGURE_PLAN.json")
    out = {"rule": fp["rule"], "owner_questions": fp["owner_questions"], "conventions": fp["conventions"], "series": []}
    for sid, s in fp["series"].items():
        figs = s["figures"]
        by_status: dict[str, int] = {}
        for f in figs:
            by_status[f["status"]] = by_status.get(f["status"], 0) + 1
        out["series"].append({
            "id": sid, "title": s["title"],
            "main": sum(1 for f in figs if f["kind"] == "main"),
            "supplements": sum(1 for f in figs if f["kind"] == "supplement"),
            "by_status": by_status,
            "figures": [{**{k: f[k] for k in ("id", "kind", "title", "owner_status", "status", "content", "blocked_by")},
                         "proposed_mirror": f.get("proposed_mirror", "")} for f in figs],
        })
    return out


def prohibited() -> list[str]:
    return _json("site-src/manifest.json")["prohibited_strings"]


PAGE = r"""<title>Jomission Series</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
:root{--bg:#faf9f5;--panel:#f0eee6;--fg:#1f1e1d;--muted:#5e5d59;--line:#dedcd1;--good:#2e7d32;--warn:#a15c00;--bad:#b3261e;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1f1e1d;--panel:#2b2a27;--fg:#f0eee6;--muted:#b0aea5;--line:#3d3c38;--good:#81c784;--warn:#ffb74d;--bad:#ef9a9a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1f1e1d;--panel:#2b2a27;--fg:#f0eee6;--muted:#b0aea5;--line:#3d3c38;--good:#81c784;--warn:#ffb74d;--bad:#ef9a9a;color-scheme:dark}
html,body{margin:0;background:var(--bg);color:var(--fg)}
.fg{--sans:var(--font-anthropic-sans,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif);font:15px/1.5 var(--sans);max-width:1180px;margin:0 auto;padding-block:20px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.fg h1{font-size:1.25rem;font-weight:500;margin:0 0 2px}
.fg .stamp{color:var(--muted);font-size:.8125rem;margin-bottom:14px}
.fg nav{display:flex;flex-wrap:wrap;gap:6px;border-bottom:1px solid var(--line);margin-bottom:16px}
.fg nav a{padding:7px 12px;border-radius:8px 8px 0 0;color:var(--muted);text-decoration:none;font-size:.9rem}
.fg nav a[aria-current="page"]{background:var(--panel);color:var(--fg);font-weight:500}
.fg section{display:none}.fg section.on{display:block}
.fg .kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.fg .kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px}
.fg .kpi{min-width:0}
.fg .kpi b{display:block;font-size:1.15rem;font-weight:500;overflow-wrap:anywhere}
.fg .kpi span{color:var(--muted);font-size:.8125rem}
.fg h2{font-size:1.05rem;font-weight:500;margin:20px 0 8px}
.fg p{margin:6px 0;max-width:80ch}
.fg .tw{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
.fg table{border-collapse:collapse;width:100%;font-size:.875rem}
.fg th,.fg td{text-align:left;vertical-align:top;padding:6px 10px;border-bottom:1px solid var(--line)}
.fg th{background:var(--panel);font-weight:500;white-space:nowrap}
.fg td.id{white-space:nowrap}
.fg td.id,.fg .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.8125rem}
.fg .pill{display:inline-block;padding:0 8px;border-radius:99px;border:1px solid var(--line);font-size:.75rem}
.fg .PASS,.fg .DONE{color:var(--good)}.fg .FAIL,.fg .INFEASIBLE,.fg .BLOCKED{color:var(--bad)}
.fg .PARTIAL,.fg .OPEN,.fg .LOCKED,.fg .INSUFFICIENT_ALONE,.fg .TBD{color:var(--warn)}
.fg .PLANNED,.fg .SPECIFIED,.fg .QUEUED{color:var(--muted)}
.fg .SEALED{color:var(--good)}
.fg .note{color:var(--muted);font-size:.8125rem}
</style>
<div class="fg" id="root"><h1>Jomission series</h1><div class="stamp" id="stamp"></div>
<nav id="nav"></nav><div id="pages"></div></div>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById('data').textContent);
const PAGES = [['overview','Overview'],['omission','Omission paradigm'],['laminar','LANSA and laminar motif'],['figures','Figures'],['gates','Gates and lineages'],['todo','Todo and open']];
function el(tag, cls, text){const e=document.createElement(tag); if(cls) e.className=cls; if(text!==undefined) e.textContent=String(text); return e;}
function kpi(label, value){const k=el('div','kpi'); k.append(el('b','',value), el('span','',label)); return k;}
function pill(s){return el('span','pill '+String(s).replace(/[^A-Za-z_]/g,''), s);}
function table(cols, rows, render){
  const w=el('div','tw'), t=el('table'), h=el('tr');
  cols.forEach(c=>h.append(el('th','',c))); t.append(h);
  rows.forEach(r=>{const tr=el('tr'); render(r).forEach(c=>{const td=el('td'); if(c instanceof Node) td.append(c); else if(c&&c.id!==undefined){td.className='id'; td.textContent=c.id;} else td.textContent=c; tr.append(td);}); t.append(tr);});
  w.append(t); return w;}
function cut(s,n){s=String(s); return s.length>n? s.slice(0,n-1)+'…' : s;}
const P = {};
P.overview = s => {
  const o=D.overview, k=el('div','kpis');
  k.append(kpi('current gate', o.current_gate.id+' '+o.current_gate.status), kpi('next authorized task', o.next_authorized_task),
    kpi('JaxFNE (execution authority)', o.jaxfne_version), kpi('lineages registered', o.lineages), kpi('open rows', o.open_rows),
    kpi('laminar plan, builder score', o.builder_score+' / '+(o.plan_items*4)));
  s.append(k, el('h2','','Todo items by status'));
  s.append(table(['status','count'], Object.entries(o.todo_by_status), r=>[pill(r[0]), r[1]]));
  s.append(el('p','note','Pages: omission paradigm, LANSA and laminar motif, figures, gates and lineages, todo and open. Source files are named on each page.'));
};
P.omission = s => {
  const m=D.omission;
  const k=el('div','kpis'); k.append(kpi('paradigm', m.paradigm), kpi('conditions', m.n_conditions), kpi('epochs', m.n_epochs), kpi('unresolved fields', m.unresolved_fields.length));
  s.append(k, el('h2','','Blind omission'));
  s.append(table(['stage','status'], m.blind_omission, g=>[g.label, pill(g.status)]));
  s.append(el('p','',m.firewall), el('p','note',m.prohibited_count+' prohibited strings are checked at build time. No omission outcome is shown on this dashboard.'));
  s.append(el('h2','','Paradigm definition'), el('p','',m.paradigm_note));
  s.append(el('h2','','Unresolved paradigm fields'), table(['field'], m.unresolved_fields, f=>[f]));
  s.append(el('p','note','Source: manifests/milestone_01_short_trajectory.json (paradigm_exact), site-src/manifest.json, AGENTS.md.'));
};
P.laminar = s => {
  const l=D.laminar, k=el('div','kpis');
  k.append(kpi('builder', l.scores.builder+' / 100'), kpi('reviewer A', l.scores.revA+' / 100'), kpi('reviewer B', l.scores.revB+' / 100'), kpi('UNSET mentions in seal proposal', l.unset_mentions));
  s.append(k, el('p','','Goal: a four-type (E, PV, SST, VIP) Izhikevich-TFNE column in JaxFNE whose simulated LFP reproduces the spectrolaminar motif from cell densities, then the building block of the omission paradigm. Score is 4 points per item, no partial credit.'));
  const mm=l.minimal;
  s.append(el('h2','','Minimal model '+mm.model));
  s.append(table(['decision'], mm.decisions, d=>[d]));
  s.append(table(['id','manipulation','status','run authorized'], mm.experiments, r=>[{id:r.id}, r.manipulation, pill(r.status), r.authorized?'yes':'no']));
  s.append(el('h2','','E0 gates (sealed pre-run checks)'));
  s.append(table(['gate','check','state','evidence'], mm.e0_gates, g=>[String(g.id), g.name, pill(g.state), g.evidence]));
  s.append(el('h2','','Checklist'));
  s.append(table(['#','item','closing check','builder','rev A','rev B'], l.plan, r=>[String(r.n), cut(r.item,140), cut(r.check,160), r.builder, r.revA, r.revB]));
  s.append(el('h2','','Owner decisions open'));
  s.append(table(['id','kind','text'], l.open_rows, r=>[{id:r.id}, r.kind, cut(r.text,420)]));
  s.append(el('p','note','Source: docs/PLAN_LAMINAR_MOTIF.md, docs/laminar_motif/ITEM03_SEAL_PROPOSAL.md, docs/OPEN.md. LANSA densities are not in the repository (OPEN-38); no density value is shown.'));
};
P.figures = s => {
  const f=D.figures;
  s.append(el('p','',f.rule));
  f.series.forEach(se=>{
    const k=el('div','kpis'); k.append(kpi('main figures', se.main), kpi('supplements', se.supplements));
    Object.entries(se.by_status).forEach(([st,n])=>k.append(kpi(st, n)));
    s.append(el('h2','',se.title), k);
    s.append(table(['id','kind','title','status','who','needs','proposed mirror (not owner-set)'], se.figures,
      r=>[{id:r.id}, r.kind, r.title, pill(r.status), r.owner_status, cut(r.blocked_by.join('; '),160), r.proposed_mirror]));
  });
  s.append(el('h2','','Matching conventions (Draft-S)'));
  s.append(table(['rule'], f.conventions.match, c=>[c]));
  s.append(el('p','note',f.conventions.constraint));
  s.append(el('h2','','Questions for the owner'), table(['question'], f.owner_questions, q=>[q]));
  s.append(el('p','note','Source: docs/FIGURE_PLAN.json. A figure shows a result only after the checklist item that produces it has closed.'));
};
P.gates = s => {
  const g=D.gates;
  s.append(el('h2','','Latest verdicts'));
  s.append(table(['id','status','scope','evidence'], g.verdicts, r=>[{id:r.id}, pill(r.status), r.scope||'', r.evidence||'']));
  s.append(el('h2','','Lineages ('+g.lineages.length+')'));
  s.append(table(['lineage','parent','principal delta'], g.lineages, r=>[{id:r.id}, r.parent, cut(r.delta,200)]));
  s.append(el('p','note','Source: manifests/current_state.json, manifests/lineage_registry.json.'));
};
P.todo = s => {
  const t=D.todo;
  s.append(el('h2','','Todo ('+t.items.length+')'));
  s.append(table(['id','status','goal'], t.items, r=>[{id:r.id}, pill(r.status), cut(r.goal,220)]));
  s.append(el('h2','','Open ('+t.open.length+')'));
  s.append(table(['id','kind','text'], t.open, r=>[{id:r.id}, r.kind, cut(r.text,300)]));
  s.append(el('p','note','Source: manifests/todo.json, docs/OPEN.md.'));
};
const nav=document.getElementById('nav'), pages=document.getElementById('pages');
document.getElementById('stamp').textContent='commit '+D.stamp.head+' · state updated '+D.stamp.state_updated;
PAGES.forEach(([id,label])=>{
  const a=el('a','',label); a.href='#'+id; a.dataset.p=id; nav.append(a);
  const sec=el('section'); sec.id='p-'+id; P[id](sec); pages.append(sec);
});
function route(){
  const id=(location.hash||'#overview').slice(1); const cur=PAGES.some(p=>p[0]===id)?id:'overview';
  document.querySelectorAll('#nav a').forEach(a=>{ if(a.dataset.p===cur) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current'); });
  document.querySelectorAll('#pages section').forEach(x=>x.classList.toggle('on', x.id==='p-'+cur));
}
window.addEventListener('hashchange', route); route();
</script>
"""

WRAP = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>{}</body></html>\n')


def build(out: Path, standalone: bool = False) -> Path:
    """Write index.html. Default is the fragment the Artifact publish step wraps; --standalone adds the
    document skeleton for opening the file directly."""
    data = collect()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.replace("__DATA__", blob)
    if standalone:
        html = WRAP.format(html)
    for bad in prohibited():
        if bad in html:
            raise SystemExit(f"firewall: prohibited string in output: {bad!r}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--standalone", action="store_true", help="wrap in a document skeleton for local viewing")
    a = ap.parse_args()
    p = build(a.out, a.standalone)
    d = collect()
    print(f"wrote {p} ({p.stat().st_size} bytes); todo {len(d['todo']['items'])}, open {len(d['todo']['open'])}, "
          f"lineages {len(d['gates']['lineages'])}, plan items {len(d['laminar']['plan'])}")


if __name__ == "__main__":
    main()

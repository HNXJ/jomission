"""OPEN-38 status source for dashboards (owner 2026-11-17): derived only from the repository's own receipts.

Usage: python scripts/open38_status.py OUT_JSON [--repo D] [--run-tests] [--stale-min 30]

Owner of the scientific content: main-jomission-lansa. Dashboard and Labyrinth integration read OUT_JSON and never
edit the receipts. No status here is typed by hand: every field is read from a log line, a checker JSON or git.

Statuses: PASS, FAIL, RUNNING, BLOCKED, UNKNOWN.
  PASS     only when the stage's driver logged DONE and every checker receipt it names reports pass
  FAIL     a checker receipt reports fail, or a run exited non-zero
  RUNNING  a driver segment started and has neither DONE nor STOP, and its log changed within --stale-min minutes
  BLOCKED  the driver logged STOP, or a phase passed and the next is not launched (reason given)
  UNKNOWN  a needed receipt is missing or unreadable, or a running segment's log went stale
A launched simulation is never PASS: a run counts only with its "TAG rc=0" line, and a stage only with DONE and its checks.

Cross-commit consistency (owner 2026-11-17): a PASS never combines receipts from different source revisions. Every
revision a stage's receipts carry (driver log HEAD, run git_sha, integrity "head") must be one SHA. A receipt written
at another revision counts only if RESULTS/o38s_receipt_compat.json names it, with its source SHA and the paths that
must be identical to the execution SHA, and `git diff` confirms those paths are identical. Otherwise the stage is
UNKNOWN. Receipts written by the driver inside its own logged segment (construction, freeze) carry the segment's HEAD.

Public view (owner 2026-11-17): `public_status` equals `status` only when every receipt the stage reads is committed
and unmodified at the source SHA; otherwise it is UNKNOWN. The live (machine-local) state stays in `status`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

R = "results/laminar_e2"
DOCS = "docs/laminar_motif"
REPO_URL = "https://github.com/HNXJ/jomission"
TS = r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)"
PROJECT = {"name": "Jomission / LANSA (OPEN-38)",
           "objective": "Test, in the LANSA E-PV-SOM model, whether the laminar spectral motif depends on the SST/PV "
                        "composition profile when composition is set to the profile observed in monkey #1 "
                        "(spec SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md, H-mech)"}
EVIDENCE = {"spec": f"{DOCS}/SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md",
            "stage1_receipt": f"{DOCS}/STAGE1_O38S_PC_2026-11-17.md",
            "stage2_receipt": f"{DOCS}/STAGE2_O38S_2026-11-17.md",
            "gate_ruling": f"{DOCS}/GATE_RULING_REQUEST_O38S_S2_2026-11-17.md"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _verdict(rec) -> str:
    return "UNKNOWN" if rec is None or not isinstance(rec.get("pass"), bool) else ("PASS" if rec["pass"] else "FAIL")


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def parse_segments(text: str) -> list[dict]:
    """Split a driver log into segments; stage-1 logs have no 'mode' line and form one segment."""
    segs, cur = [], None
    for line in text.splitlines():
        m = re.match(TS + r" (.*)$", line)
        if not m:
            continue
        t, msg = m.groups()
        if msg.startswith("mode ") or cur is None:
            cur = {"mode": msg[5:] if msg.startswith("mode ") else "stage", "start": t, "head": None, "construction": None,
                   "started": [], "rc": {}, "integrity": None, "stage_check": None, "done": False, "stop": None, "last": t}
            segs.append(cur)
        cur["last"] = t
        if msg.startswith("HEAD "):
            cur["head"] = msg[5:]
        elif m2 := re.match(r"construction invariance: (PASS|FAIL)", msg):
            cur["construction"] = m2.group(1)
        elif m2 := re.match(r"start (\S+) ", msg):
            cur["started"].append(m2.group(1))
        elif m2 := re.fullmatch(r"(\S+) rc=(\d+)", msg):
            cur["rc"][m2.group(1)] = int(m2.group(2))
        elif m2 := re.match(r"(?:smoke |stage-2 )?integrity: (PASS|FAIL)", msg):
            cur["integrity"] = m2.group(1)
        elif m2 := re.match(r"stage check: (PASS|FAIL)", msg):
            cur["stage_check"] = m2.group(1)
        elif msg.startswith("DONE"):
            cur["done"] = True
        elif msg.startswith("STOP"):
            cur["stop"] = msg
    return segs


def segment_status(seg: dict, log_age_min: float, stale_min: float, needs: tuple[str, ...]) -> tuple[str, str]:
    if seg["stop"]:
        return "BLOCKED", seg["stop"]
    if seg["construction"] == "FAIL" or any(v != 0 for v in seg["rc"].values()):
        return "FAIL", "construction FAIL or a run exited non-zero"
    if not seg["done"]:
        if log_age_min > stale_min:
            return "UNKNOWN", f"no DONE or STOP and log unchanged for {log_age_min:.0f} min"
        return "RUNNING", f"{len(seg['rc'])} of {len(seg['started'])} started runs finished"
    checks = [seg[k] for k in needs]
    if any(c == "FAIL" for c in checks):
        return "FAIL", "a stage checker failed"
    if any(c is None for c in checks):
        return "UNKNOWN", "DONE without a checker line"
    missing = [t for t in seg["started"] if t not in seg["rc"]]
    if missing:
        return "UNKNOWN", f"started without an exit line: {missing}"
    return "PASS", "DONE; checks PASS"


def _log(repo: Path, name: str) -> tuple[list[dict], float | None, str | None]:
    p = repo / R / name
    if not p.is_file():
        return [], None, None
    age = (dt.datetime.now().timestamp() - p.stat().st_mtime) / 60.0
    return parse_segments(p.read_text(encoding="utf-8")), age, _sha(p)


def _git(repo: Path, *a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, stdin=subprocess.DEVNULL)


def committed(repo: Path, rels: list[str]) -> bool | None:
    """True if every path is tracked and unmodified at HEAD; None without git."""
    if _git(repo, "rev-parse", "HEAD").returncode != 0:
        return None
    for rel in rels:
        if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
            return False
        if _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
            return False
    return True


def compat(repo: Path, rel: str, execution_sha: str) -> tuple[bool | None, str]:
    """Is a receipt written at another revision declared and verified compatible with execution_sha?"""
    man = _json(repo / R / "o38s_receipt_compat.json") or {}
    ent = next((e for e in man.get("entries", []) if e.get("receipt") == rel), None)
    if ent is None:
        return False, f"{rel}: no compatibility entry"
    if not str(ent.get("execution_sha", "")).startswith(execution_sha[:7]) or not ent.get("identical_paths"):
        return False, f"{rel}: entry names execution {ent.get('execution_sha')} or no identical_paths"
    r = _git(repo, "diff", "--quiet", ent["execution_sha"], ent["source_sha"], "--", *ent["identical_paths"])
    if r.returncode == 1:
        return False, f"{rel}: declared paths differ between {ent['execution_sha'][:7]} and {ent['source_sha'][:7]}"
    if r.returncode != 0:
        return None, f"{rel}: git could not verify ({r.stderr.strip()[:80]})"
    # identical code is not experimental equivalence (owner 2026-11-17): the receipt must also reproduce the
    # execution-SHA receipt it re-checks, field by field, except the fields the entry declares as changed
    ref_rel, ign = ent.get("equivalent_to"), set(ent.get("ignore_keys", []))
    a, b = _json(repo / rel), _json(repo / ref_rel) if ref_rel else None
    if a is None or b is None:
        return False, f"{rel}: no readable equivalent_to receipt ({ref_rel})"
    if _strip(a, ign) != _strip(b, ign):
        return False, f"{rel}: differs from {ref_rel} outside {sorted(ign)}"
    if ent.get("rule") == "ruling_b_construction":  # the ignored fields are re-derived, not trusted
        stray = [p for x in (a, b) for p in _key_paths(x, ign) if not _ruling_b_slot(p)]
        try:
            probs = [f"{'/'.join(map(str, p))}: ignored key outside its declared slot" for p in stray] \
                or verify_ruling_b(a, b)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError) as e:  # fail closed
            probs = [f"malformed receipt: {type(e).__name__} {e}"]
        if probs:
            return False, f"{rel}: ruling-B re-derivation failed: {probs[:3]}"
    elif ign:
        return False, f"{rel}: ignore_keys {sorted(ign)} without a re-derivation rule"
    return True, (f"{rel}: {ent['source_sha'][:7]} ~ {execution_sha[:7]}: identical {ent['identical_paths']}; "
                  f"equal to {ref_rel} except {sorted(ign)}")


RULING_B_ALPHA, Z_FROZEN_LEVEL, WIDTH_TOL, CENTRE_TOL = 0.05, 0.999, 0.01, 2.0


def verify_ruling_b(new: dict, old: dict) -> list[str]:
    """Re-derive every field a ruling-B construction re-check may change, from the frozen receipt of the same builds.
    family: alpha 0.05, n_tests = PV/SOM rules in the receipt, level 1 - alpha/n_tests. Each interval: centre equal to
    the frozen one (same p*N) within CENTRE_TOL edges, width = frozen width * z_B / z_0.999 within WIDTH_TOL (binomial
    discreteness; observed <= 0.7 %). Each ok and pass: recomputed from the counts and the re-derived intervals."""
    from statistics import NormalDist
    probs, fam = [], new.get("family") or {}
    n = sum(len(r["inh_rules"]["rules"]) for r in new.get("results", {}).values())
    if fam.get("alpha") != RULING_B_ALPHA or fam.get("n_tests") != n or n == 0:
        return [f"family {fam} vs {n} PV/SOM tests"]
    if abs(fam.get("per_test_level", 0) - (1 - RULING_B_ALPHA / n)) > 1e-12:
        return [f"per_test_level {fam.get('per_test_level')} != {1 - RULING_B_ALPHA / n}"]
    nd = NormalDist()
    scale = nd.inv_cdf(1 - RULING_B_ALPHA / n / 2) / nd.inv_cdf(1 - (1 - Z_FROZEN_LEVEL) / 2)
    all_ok = True
    for key, r in new["results"].items():
        pair_ok = True
        for rule, v in r["inh_rules"]["rules"].items():
            lo0, hi0 = old["results"][key]["inh_rules"]["rules"][rule]["interval"]
            lo, hi = v["interval"]
            if abs((lo + hi) / 2 - (lo0 + hi0) / 2) > CENTRE_TOL:
                probs.append(f"{key} {rule}: centre moved")
            if abs((hi - lo) / ((hi0 - lo0) * scale) - 1) > WIDTH_TOL:
                probs.append(f"{key} {rule}: width {(hi - lo):.0f} not ruling-B ({(hi0 - lo0) * scale:.0f})")
            ok = lo <= v["count"] <= hi and v["params_equal"]
            if v["ok"] != ok:
                probs.append(f"{key} {rule}: ok {v['ok']} != recomputed {ok}")
            pair_ok &= ok
        if r["inh_rules"]["ok"] and not pair_ok:
            probs.append(f"{key}: inh_rules ok while a rule fails")
        for g, val in r.items():  # every other gate does not depend on the level: its ok must be the frozen one
            if g != "inh_rules" and val.get("ok") != old["results"][key][g].get("ok"):
                probs.append(f"{key} {g}: ok changed from the frozen receipt")
        gates_ok = all(g["ok"] for g in r.values())
        all_ok &= gates_ok
    if new.get("pass") is not all_ok:
        probs.append(f"pass {new.get('pass')} != all gates {all_ok}")
    return probs


def _key_paths(x, ign: set, path=()) -> list[tuple]:
    """Every path at which an ignored key occurs."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            out += [path + (k,)] if k in ign else _key_paths(v, ign, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += _key_paths(v, ign, path + (i,))
    return out


def _ruling_b_slot(p: tuple) -> bool:
    """The only positions ruling B may change: top-level family and pass, each gate's ok, each PV/SOM rule's
    interval and ok."""
    return (p in (("family",), ("pass",))
            or (len(p) == 4 and p[0] == "results" and p[3] == "ok")
            or (len(p) == 6 and p[0] == "results" and p[2:4] == ("inh_rules", "rules") and p[5] in ("interval", "ok")))


def _strip(x, ign: set):
    if isinstance(x, dict):
        return {k: _strip(v, ign) for k, v in x.items() if k not in ign}
    if isinstance(x, list):
        return [_strip(v, ign) for v in x]
    return x


def consistency(repo: Path, heads: dict[str, set], foreign: list[str]) -> dict:
    """One execution SHA across all revision-bearing receipts; foreign receipts need a verified compat entry."""
    seen = set().union(*heads.values()) if heads else set()
    probs = [f"revisions differ: { {k: sorted(v) for k, v in heads.items()} }"] if len(seen) != 1 or None in seen else []
    exe = next(iter(seen)) if len(seen) == 1 else None
    notes = []
    for rel in foreign if exe else []:
        ok, why = compat(repo, rel, exe)
        (notes if ok else probs).append(why)  # ok None (git unavailable) counts as unverified
    return {"ok": not probs, "execution_sha": exe, "problems": probs, "compatible": notes}


def _apply(st: str, why: str, cons: dict) -> tuple[str, str]:
    if st == "PASS" and not cons["ok"]:
        return "UNKNOWN", "cross-commit receipts: " + "; ".join(cons["problems"])
    return st, why


def stage1(repo: Path, stale: float) -> dict:
    segs, age, lsha = _log(repo, "o38s_pc_stage.log")
    files = {"stage_check": "o38s_pc_stage1_check.json", "integrity": "o38s_pc_integrity.json",
             "construction_frozen": "o38s_pc_construction.json", "construction_ruling_b": "o38s_pc_construction_rulingB.json"}
    js = {k: _json(repo / R / f) for k, f in files.items()}
    rec = {k: _verdict(v) for k, v in js.items()}
    if not segs:
        return {"status": "UNKNOWN", "public_status": "UNKNOWN", "reason": "no stage log", "checks": rec}
    st, why = segment_status(segs[-1], age, stale, ("stage_check",))
    if st == "PASS" and any(v != "PASS" for v in rec.values()):
        st, why = ("FAIL" if "FAIL" in rec.values() else "UNKNOWN"), f"checker receipts: {rec}"
    s = segs[-1]
    runs = (js["stage_check"] or {}).get("runs", {})
    heads = {"log": {s["head"]}, "integrity": {(js["integrity"] or {}).get("head")},
             "run_git_sha": {r.get("git_sha") for a in runs.values() for r in a.values()} or {None}}
    cons = consistency(repo, heads, [f"{R}/{files['construction_ruling_b']}"])
    st, why = _apply(st, why, cons)
    pub = committed(repo, [f"{R}/o38s_pc_stage.log", f"{R}/o38s_receipt_compat.json",
                           *(f"{R}/{f}" for f in files.values())])
    return {"status": st, "public_status": st if pub else "UNKNOWN", "committed": pub, "reason": why,
            "head": s["head"], "runs_exit0": sum(v == 0 for v in s["rc"].values()), "runs_expected": 15,
            "checks": rec, "consistency": cons, "log_sha256": lsha}


def stage2(repo: Path, stale: float) -> dict:
    segs, age, lsha = _log(repo, "o38s_s2_stage.log")
    att1, _, _ = _log(repo, "o38s_s2_attempt1_stage.log")
    cjs = _json(repo / R / "o38s_s2_construction.json")
    out = {"runs_expected": 30, "log_sha256": lsha,
           "attempts": [{"attempt": 1, "mode": s["mode"], "head": s["head"], "construction": s["construction"],
                         "stop": s["stop"]} for s in att1],
           "construction_gate": {"verdict": _verdict(cjs) if cjs is not None else
                                 ("RUNNING" if segs and segs[-1]["construction"] is None and not segs[-1]["stop"]
                                  and (age or 1e9) <= stale else "UNKNOWN"),
                                 "family": (cjs or {}).get("family")}}
    if not segs:
        return {**out, "status": "UNKNOWN", "public_status": "UNKNOWN", "reason": "no stage-2 log", "phases": []}
    phases = []
    for s in segs:
        st, why = segment_status(s, age if s is segs[-1] else 0.0, stale, ("integrity",))
        phases.append({"mode": s["mode"], "status": st, "reason": why, "head": s["head"], "started": len(s["started"]),
                       "exit0": sum(v == 0 for v in s["rc"].values()), "construction": s["construction"],
                       "integrity": s["integrity"], "last_line_utc": s["last"]})
    last = phases[-1]
    integ = _verdict(_json(repo / R / "o38s_s2_integrity.json"))
    if any(p["status"] == "FAIL" for p in phases):
        st, why = "FAIL", "a phase failed"
    elif last["mode"] == "rest" and last["status"] == "PASS":
        st, why = ("PASS", "rest DONE; integrity PASS") if integ == "PASS" else ("UNKNOWN", f"integrity receipt {integ}")
    elif last["mode"] == "smoke" and last["status"] == "PASS":
        st, why = "BLOCKED", "smoke PASS; rest phase not launched"
    else:
        st, why = last["status"], last["reason"]
    runs = sum(p["exit0"] for p in phases)
    integ_files = [f for f in ("o38s_s2_smoke_integrity.json", "o38s_s2_integrity.json") if (repo / R / f).is_file()]
    heads = {"log": {s["head"] for s in segs}, **{f: {(_json(repo / R / f) or {}).get("head")} for f in integ_files}}
    cons = consistency(repo, heads, [])  # the construction receipt is written inside the logged smoke segment
    st, why = _apply(st, why, cons)
    pub = committed(repo, [f"{R}/o38s_s2_stage.log", f"{R}/o38s_s2_construction.json", *(f"{R}/{f}" for f in integ_files)])
    return {**out, "status": st, "public_status": st if pub else "UNKNOWN", "committed": pub, "reason": why,
            "phases": phases, "runs_exit0": runs, "consistency": cons}


def tests(repo: Path, run: bool) -> dict:
    files = sorted((repo / "tests").glob("test_open38_*.py"))
    defined = sum(len(re.findall(r"^def test_", f.read_text(encoding="utf-8"), re.M)) for f in files)
    out = {"files": len(files), "defined": defined, "status": "UNKNOWN", "reason": "not run (--run-tests not given)"}
    if run:
        r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *map(str, files)],
                           cwd=repo, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        m = dict((k, int(v)) for v, k in re.findall(r"(\d+) (passed|failed|error)", tail))
        out.update(passed=m.get("passed", 0), failed=m.get("failed", 0) + m.get("error", 0), summary=tail,
                   status="PASS" if r.returncode == 0 and m.get("passed") else "FAIL", reason=f"pytest rc {r.returncode}")
    return out


def git(repo: Path) -> dict:
    def g(*a):
        r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"sha": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked": bool(g("status", "--porcelain", "--untracked-files=no"))}


def main(out: str, repo: Path, run_tests: bool, stale: float) -> int:
    s1, s2 = stage1(repo, stale), stage2(repo, stale)
    current = ("stage 2" if s2["status"] != "UNKNOWN" or s2.get("phases") else "stage 1")
    milestone = None  # from committed receipts only (public_status), never from a machine-local state
    if s1["public_status"] == "PASS":
        milestone = {"what": "stage 1 positive controls: 15/15 runs, frozen checker, integrity and construction "
                             "(frozen and ruling-B) receipts PASS", "evidence": EVIDENCE["stage1_receipt"]}
    if s2["public_status"] == "PASS":
        milestone = {"what": "stage 2: 30/30 runs, integrity PASS", "evidence": EVIDENCE["stage2_receipt"]}
    blockers = [f"{k}: {v['reason']}" for k, v in (("stage 2", s2), ("stage 1", s1)) if v["status"] in ("BLOCKED", "FAIL")]
    rep = {"schema": "o38s_status/1", "generated_utc": _now(), "source": {**git(repo), "repo_url": REPO_URL},
           "project": PROJECT, "current_stage": current, "stage1": s1, "stage2": s2, "tests": tests(repo, run_tests),
           "latest_verified_milestone": milestone, "blockers": blockers,
           "evidence": {k: {"path": v, "sha256": _sha(repo / v)} for k, v in EVIDENCE.items()},
           "notes": ["milestones are receipt-verified by checkers; owner acceptance and independent review are "
                     "recorded in the receipts, not here",
                     "a running stage's log is untracked until the stage closes"]}
    Path(out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"stage1": s1["status"], "stage1_public": s1["public_status"], "stage2": s2["status"],
                      "stage2_public": s2["public_status"], "gate": s2["construction_gate"]["verdict"],
                      "sha": rep["source"]["sha"]}))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--stale-min", type=float, default=30.0)
    a = ap.parse_args()
    sys.exit(main(a.out, Path(a.repo), a.run_tests, a.stale_min))

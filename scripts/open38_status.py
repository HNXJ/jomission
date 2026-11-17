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
The live log of a running stage is untracked until the stage closes; a refresh from a clone shows the last commit.
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


def stage1(repo: Path, stale: float) -> dict:
    segs, age, lsha = _log(repo, "o38s_pc_stage.log")
    rec = {k: _verdict(_json(repo / R / f)) for k, f in (
        ("stage_check", "o38s_pc_stage1_check.json"), ("integrity", "o38s_pc_integrity.json"),
        ("construction_frozen", "o38s_pc_construction.json"), ("construction_ruling_b", "o38s_pc_construction_rulingB.json"))}
    if not segs:
        return {"status": "UNKNOWN", "reason": "no stage log", "checks": rec}
    st, why = segment_status(segs[-1], age, stale, ("stage_check",))
    if st == "PASS" and any(v != "PASS" for v in rec.values()):
        st, why = ("FAIL" if "FAIL" in rec.values() else "UNKNOWN"), f"checker receipts: {rec}"
    s = segs[-1]
    return {"status": st, "reason": why, "head": s["head"], "runs_exit0": sum(v == 0 for v in s["rc"].values()),
            "runs_expected": 15, "checks": rec, "log_sha256": lsha}


def stage2(repo: Path, stale: float) -> dict:
    segs, age, lsha = _log(repo, "o38s_s2_stage.log")
    att1, _, _ = _log(repo, "o38s_s2_attempt1_stage.log")
    cons = _json(repo / R / "o38s_s2_construction.json")
    out = {"runs_expected": 30, "log_sha256": lsha,
           "attempts": [{"attempt": 1, "mode": s["mode"], "head": s["head"], "construction": s["construction"],
                         "stop": s["stop"]} for s in att1],
           "construction_gate": {"verdict": _verdict(cons) if cons is not None else
                                 ("RUNNING" if segs and segs[-1]["construction"] is None and not segs[-1]["stop"]
                                  and (age or 1e9) <= stale else "UNKNOWN"),
                                 "family": (cons or {}).get("family")}}
    if not segs:
        return {**out, "status": "UNKNOWN", "reason": "no stage-2 log", "phases": []}
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
    return {**out, "status": st, "reason": why, "phases": phases, "runs_exit0": runs}


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
    milestone = None
    if s1["status"] == "PASS":
        milestone = {"what": "stage 1 positive controls: 15/15 runs, frozen checker, integrity and construction "
                             "(frozen and ruling-B) receipts PASS", "evidence": EVIDENCE["stage1_receipt"]}
    if s2["status"] == "PASS":
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
    print(json.dumps({"stage1": s1["status"], "stage2": s2["status"], "gate": s2["construction_gate"]["verdict"],
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

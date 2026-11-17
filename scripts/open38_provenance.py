"""OPEN-38 file-level provenance for the tracked results/laminar_e2/o38s_*.json files.

Registration, not validation and not sealing. Each file gets its group, the commit that last
changed it (evidence commit), the execution SHA its own content records (or UNRECORDED), its git
blob and the SHA-256 of its committed bytes. Standard library only.

  python -B scripts/open38_provenance.py           write manifests/o38s_provenance.json at HEAD
  python -B scripts/open38_provenance.py --check   recompute and compare; exit 1 on any difference
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = "manifests/o38s_provenance.json"
PATTERN = "results/laminar_e2/o38s_*.json"
S2_ARMS = ("v1", "mt", "mst", "teo", "chc", "shape")

# group -> (execution SHA declared by its closure record, closure record). The SHA is checked
# against every git_sha / head field inside the group's files.
GROUPS = {
    "smoke": ("3294090b6b00375591eb5fcd9d47fb9238b458e7", "docs/laminar_motif/SMOKE_O38S_2026-11-16.md"),
    "stage1": ("53f8b89fdb1ec4d67e3c972dd24ee4c899a680cc", "docs/laminar_motif/STAGE1_O38S_PC_2026-11-17.md"),
    "stage1_ruling_b_recheck": ("8fed7f20ce4adb5669b8387aa84161b5fe21042c",
                                "docs/laminar_motif/STAGE2_O38S_2026-11-17.md (checker at 8fed7f2; SHA not in the file)"),
    "stage2_attempt1_stop": (None, "docs/laminar_motif/STAGE2_O38S_2026-11-17.md (attempt-1 STOP, files kept)"),
    "stage2": ("0d7162210553127482659d758b102f3b8f5ea71c", "docs/laminar_motif/STAGE2_O38S_2026-11-17.md"),
    "section6": (None, "docs/laminar_motif/S6_O38S_2026-11-17.md"),
    "status_config": (None, "docs/laminar_motif/STATUS_SOURCE_O38S.md (status-generator input, not a run output)"),
}


def git(*a, binary=False):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode().strip()


def group(name: str) -> str:
    if name.startswith("o38s_s6_"):
        return "section6"
    if name == "o38s_receipt_compat.json":
        return "status_config"
    if name == "o38s_pc_construction_rulingB.json":
        return "stage1_ruling_b_recheck"
    if name.startswith("o38s_s2_attempt1_"):
        return "stage2_attempt1_stop"
    if name.startswith("o38s_pc_"):
        return "stage1"
    if name.startswith("o38s_s2_") or re.match(rf"o38s_({'|'.join(S2_ARMS)})_n4000_s\d_T8", name):
        return "stage2"
    if re.match(r"o38s_(nc|ref)_n4000_s1_T8", name) or name == "o38s_smoke_check.json":
        return "smoke"
    raise SystemExit(f"no provenance group for {name}")


def recorded_heads(o):
    if isinstance(o, dict):
        for k, x in o.items():
            if k in ("git_sha", "head") and isinstance(x, str):
                yield x
            else:
                yield from recorded_heads(x)
    elif isinstance(o, list):
        for x in o:
            yield from recorded_heads(x)


def build(rev: str = "HEAD") -> dict:
    rev = git("rev-parse", rev)
    blobs = {}
    for line in git("ls-tree", "-r", rev, "--", "results/laminar_e2/").splitlines():
        meta, p = line.split("\t", 1)
        if re.fullmatch(r"results/laminar_e2/o38s_[^/]*\.json", p):
            blobs[p] = meta.split()[2]
    paths = sorted(blobs)
    last = {}  # newest commit touching each path: first hit in one history walk
    commit = None
    for line in git("log", "--format=@%H", "--name-only", rev, "--", "results/laminar_e2/").splitlines():
        if line.startswith("@"):
            commit = line[1:]
        elif line in blobs:
            last.setdefault(line, commit)
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, capture_output=True, check=True,
                         input="".join(f"{blobs[p]}\n" for p in paths).encode()).stdout
    content, i = {}, 0
    for p in paths:  # each record: "<sha> blob <size>\n<bytes>\n"
        j = out.index(b"\n", i)
        size = int(out[i:j].split()[2])
        content[p] = out[j + 1:j + 1 + size]
        i = j + 2 + size
    files, problems = [], []
    for p in paths:
        data = content[p]
        g = group(p.rsplit("/", 1)[1])
        exe, _ = GROUPS[g]
        heads = sorted(set(recorded_heads(json.loads(data))))
        if exe is not None and any(h != exe for h in heads):
            problems.append(f"{p}: recorded heads {heads} != group execution SHA {exe}")
        files.append({
            "path": p,
            "group": g,
            "sha256": hashlib.sha256(data).hexdigest(),
            "blob": blobs[p],
            "evidence_commit": last[p],
            "execution_sha": exe or "UNRECORDED",
            "execution_sha_in_file": heads[0] if len(heads) == 1 else ("MIXED" if heads else "NONE"),
        })
    if problems:
        raise SystemExit("\n".join(problems))
    counts = {g: sum(f["group"] == g for f in files) for g in GROUPS}
    return {
        "schema": "jomission.o38s_provenance.v1",
        "semantics": "provenance registration only: each file is generated evidence. Registration does not "
                     "validate a result or seal a file. Validation status lives in each group's closure record; "
                     "seals live only in manifests/sealed_artifacts.json.",
        "generator": "scripts/open38_provenance.py",
        "source_rev": rev,
        "pattern": PATTERN,
        "n_files": len(files),
        "groups": {g: {"execution_sha": e or "UNRECORDED", "closure_record": c, "n_files": counts[g]}
                   for g, (e, c) in GROUPS.items()},
        "files": files,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rev", default="HEAD")
    a = ap.parse_args()
    if a.check:
        held = json.loads((ROOT / OUT).read_text(encoding="utf-8"))
        now = build(held["source_rev"])
        live = {f["path"] for f in build(a.rev)["files"]}
        diffs = [k for k in now if now[k] != held.get(k)]
        missing = sorted(live - {f["path"] for f in held["files"]})
        for f in held["files"]:  # the working tree must still hold the registered bytes
            disk = ROOT / f["path"]
            if not disk.is_file() or hashlib.sha256(disk.read_bytes()).hexdigest() != f["sha256"]:
                diffs.append(f"working tree differs: {f['path']}")
        print("O38S_PROVENANCE " + ("PASS" if not diffs and not missing else f"FAIL {diffs} {missing}"))
        return 1 if diffs or missing else 0
    (ROOT / OUT).write_bytes((json.dumps(build(a.rev), indent=1) + "\n").encode())
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

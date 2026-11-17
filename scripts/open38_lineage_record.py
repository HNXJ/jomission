"""Append the OPEN-38 lineage record to manifests/lineage_registry.json (one-shot; refuses if present).

  python -B scripts/open38_lineage_record.py VIZ_COMMIT

The record is inserted as text before the registry's closing bracket, so no other record is re-serialized.
Artifacts: the 234 files of manifests/o38s_provenance.json, the V0-V3 tree and the closure records.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "manifests" / "lineage_registry.json"
LID = "SCI-OPEN38-O38S"
VIZ = f"results/viz/{LID}"
DOCS = ["docs/laminar_motif/SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md",
        "docs/laminar_motif/SMOKE_O38S_2026-11-16.md", "docs/laminar_motif/STAGE1_O38S_PC_2026-11-17.md",
        "docs/laminar_motif/STAGE2_O38S_2026-11-17.md", "docs/laminar_motif/S6_O38S_2026-11-17.md",
        "docs/laminar_motif/PROVENANCE_O38S_2026-11-17.md", "manifests/o38s_provenance.json"]
S6 = "results/laminar_e2/o38s_s6_analysis.json"


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def main(viz_commit: str) -> None:
    viz_commit = git("rev-parse", viz_commit).strip()
    text = REG.read_bytes().decode("utf-8")
    if f'"lineage_id": "{LID}"' in text:
        raise SystemExit(f"{LID} already registered")
    prov = json.loads((ROOT / "manifests/o38s_provenance.json").read_text(encoding="utf-8"))
    viz_files = sorted(git("ls-tree", "-r", "--name-only", viz_commit, "--", VIZ).split())
    s6 = json.loads((ROOT / S6).read_text(encoding="utf-8"))
    v1 = s6["arms"]["v1"]["outcomes"]["r_ab"]
    rec = {
        "lineage_id": LID,
        "parent": "SCI-LAMINAR-MOTIF-1",
        "principal_delta": "OPEN-38 spectrum sensitivity: SOM/PV composition of the S/M/D CAMINOS stack (n4000, T8 "
                           "readout) changed from the prescribed ramp (S 0.25, M 1, D 2) to flat and reversed "
                           "controls (stage 1) and to four monkey-#1 area profiles, PV_CHC and shape-only (stage 2), "
                           "seeds 1-5; spec section 6 paired analysis",
        "preexecution_spec": {"path": DOCS[0], "commit": "4478e9a04f5c0b21514e6648fcb5a3a51343809e"},
        "spec_amendments": [
            {"commit": "462c07897eff948102aba8649f9aa750e7142822", "before": "stage 1 (53f8b89)"},
            {"commit": "584922c80baa26a573ef90e3186e156b411b05d7", "before": "stage-2 attempt 1 (584922c)"},
            {"commit": "8fed7f20ce4adb5669b8387aa84161b5fe21042c",
             "before": "stage 2 (0d71622)", "note": "owner ruling B, a post-hoc amendment of the construction gate"},
        ],
        "authority": "owner 2026-11-16/17: spec acceptance, ruling B, stage-2 and section-6 acceptance "
                     "(STAGE2_O38S_2026-11-17.md, S6_O38S_2026-11-17.md); provenance registration and the "
                     "retrospective V0-V3 authorized 2026-11-17",
        "inputs": [DOCS[0], "scripts/run_e2_stack.py", "jomission/laminar/caminos_stack.py"],
        "frozen": ["spec SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md (amendments listed in spec_amendments)",
                   "JaxFNE 351f6048 export (0.5.9), edge-list event runtime, float32",
                   "T8 scorer and readout (e2_scorer, e2_motif_figure)"],
        "executed": True,
        "executions": {"smoke": "3294090b6b00375591eb5fcd9d47fb9238b458e7",
                       "stage1": "53f8b89fdb1ec4d67e3c972dd24ee4c899a680cc",
                       "stage2": "0d7162210553127482659d758b102f3b8f5ea71c"},
        "observations": [
            {"claim": f"v1 arm: mean paired d = r_ab(ref) - r_ab(v1) {v1['mean']:.4f} over seeds 1-5",
             "evidence_class": "OBSERVED", "receipt": S6,
             "check": {"path": S6, "json_pointer": "/arms/v1/outcomes/r_ab/mean", "expected": v1["mean"],
                       "tol": 1e-9}},
            {"claim": "Holm-adjusted paired p for every arm 7.61e-8 (8 arms)", "evidence_class": "OBSERVED",
             "receipt": S6,
             "check": {"path": S6, "json_pointer": "/arms/v1/outcomes/r_ab/p_holm", "expected": v1["p_holm"],
                       "tol": 1e-15}},
            {"claim": f"ref r_ab mean over seeds 1-5 {s6['noise_floor']['mean']:.4f} (noise floor SD "
                      f"{s6['noise_floor']['sd']:.4f})", "evidence_class": "OBSERVED", "receipt": S6,
             "check": {"path": S6, "json_pointer": "/noise_floor/mean", "expected": s6["noise_floor"]["mean"],
                       "tol": 1e-12}},
        ],
        "derived": [{"claim": "section 6 verdict 'ref higher' on r_ab in all 8 arms (Holm over 8, every seed d > 0); "
                              "no arm equivalent within +-0.1", "evidence_class": "DERIVED"}],
        "inferred": [{"claim": "r_ab takes two values across runs (ref about +0.85, every arm about -0.73): the "
                               "readout behaves like a switch between two motif states; graded compositions were "
                               "not sampled", "evidence_class": "INFERRED"}],
        "verdict": {"label": "O38S_GATES_PASS_S6_REF_HIGHER_ALL_ARMS", "status": "PASS",
                    "status_note": "construction-invariance (ruling B) and integrity gates PASS in stages 1 and 2; "
                                   "section 6 'ref higher' in 8/8 arms; owner accepted stage 2 and section 6 on "
                                   "2026-11-17. This record registers provenance; it adds no validation and seals "
                                   "nothing (no file here is in manifests/sealed_artifacts.json)."},
        "acceptance": "spec section 6 rule as frozen; owner acceptance recorded in S6_O38S_2026-11-17.md",
        "fail_boundary": "a construction-invariance or integrity FAIL, a run not evaluable, or a section 6 verdict "
                         "other than 'ref higher'",
        "supersedes": [],
        "retired": [],
        "file_semantics": "every listed results file is generated evidence; validation status per group is in its "
                          "closure record (manifests/o38s_provenance.json groups); sealing only via "
                          "manifests/sealed_artifacts.json",
        "provenance_manifest": "manifests/o38s_provenance.json",
        "artifacts": DOCS + [f["path"] for f in prov["files"]] + viz_files,
        "tests": ["tests/test_open38_provenance.py", "tests/test_open38_section6.py", "tests/test_open38_status.py",
                  "tests/test_open38_construction_check.py"],
        "commit": viz_commit,
        "visualization": {"V0": f"{VIZ}/jaxfne_network_hspice.png", "V1": f"{VIZ}/raster_initial_1000ms.png",
                          "V2": f"{VIZ}/raster_final.png", "V3": f"{VIZ}/atlas/index.html"},
        "visualization_note": "retrospective: drawn 2026-11-17 by scripts/open38_viz.py from recorded outputs, after "
                              "interpretation; no simulation was run to make them; sources and hashes in "
                              f"{VIZ}/provenance.json",
        "driver_provenance": {"path": "scripts/run_e2_stack.py"},
        "harness_changes": [],
        "introduces_mechanism": False,
        "qualifies": [],
        "reports": [{"id": "OPEN38-H-MECH", "finding": "In this model deep alpha-beta placement (r_ab) holds under "
                     "the prescribed ramp and fails under every tested composition, d 1.563-1.587, Holm p 7.6e-8."}],
    }
    for p in rec["artifacts"]:
        subprocess.run(["git", "cat-file", "-e", f"{viz_commit}:{p}"], cwd=ROOT, check=True)
    body = "\n".join("  " + line for line in json.dumps(rec, indent=1, ensure_ascii=False).splitlines())
    tail = "\n  }\n ]\n}\n"
    if not text.endswith(tail):
        raise SystemExit("registry tail is not the expected '  }\\n ]\\n}\\n'")
    REG.write_bytes((text[: -len(tail)] + "\n  },\n" + body + "\n ]\n}\n").encode("utf-8"))
    print(f"{LID}: {len(rec['artifacts'])} artifacts at {viz_commit[:7]}")


if __name__ == "__main__":
    main(sys.argv[1])

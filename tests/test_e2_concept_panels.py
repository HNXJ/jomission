"""Concept panels and Pages generator: older records with scalar per-unit fields; manifest byte form preserved."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_concept_panels as cp  # noqa: E402
import e2_pages_spectrolaminar as pg  # noqa: E402

V = {"r_proto": 0.1, "r_ab": 0.2, "r_gm": 0.3, "crossing_2se": True, "perm_rank": 1, "smooth_p95": 0.38, "pass": False}
SIZES = {"S": {"E": 8, "PV": 1, "SOM": 1}, "D": {"E": 4, "PV": 1, "SOM": 2}}


def test_config_row_scalar_and_dict_fields():
    old = cp.config_row("a", {"sizes": SIZES, "tau_sst_e_ms": 15.0, "p_local": 0.1}, V)
    new = cp.config_row("b", {"sizes": SIZES, "tau_sst_e_ms": {"S": 15.0, "D": 21.0}, "p_local": {"S": 0.1, "D": 0.2},
                              "modifiers": {"w": {"e_e:S": 2.0}}}, V)
    assert old["tau_sst_ms"] == "all:15" and old["p_local"] == "all:0.100"
    assert new["tau_sst_ms"] == "S:15 D:21" and new["w_mods"] == "e_e:Sx2" and new["N"] == 17


def test_manifest_dump_keeps_crlf_and_no_trailing_newline():
    obj = {"b": [1, 2], "a": {"x": "y"}}
    raw = json.dumps(obj, indent=2).encode().replace(b"\n", b"\r\n")
    assert b"\r\n" in raw and not raw.endswith(b"\n")
    assert pg.dump(json.loads(raw), {"indent": 2}, b"\r\n", b"") == raw

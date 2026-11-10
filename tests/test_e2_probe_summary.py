"""scripts/e2_probe_summary.py: seed mean and SD per geometry; a missing value in any seed gives None."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_ladder_rule as lr  # noqa: E402
import e2_probe_summary as ps  # noqa: E402


def test_mean_sd_and_missing(tmp_path):
    for s, (m, dr) in zip(lr.SEEDS, [(1.0, 0.1), (2.0, 0.2), (3.0, None)]):
        rows = [{"theta_deg": 0.0, "offset": 0.0, "M_arc": m, "delta_r": dr}]
        (tmp_path / f"{lr.tag(1000, s)}_probe_geometry.json").write_text(json.dumps({"rows": rows}))
    (row,) = ps.summarise(ps.load(1000, tmp_path))
    assert row["M_arc"] == {"mean": 2.0, "sd": 1.0, "n": 3}
    assert row["delta_r"]["mean"] is None and row["delta_r"]["n"] == 3

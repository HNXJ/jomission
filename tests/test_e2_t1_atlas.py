"""scripts/e2_t1_atlas.py: a level with no finished seed yields no maps (drawn empty), never an error."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_t1_atlas as at  # noqa: E402


def test_missing_level_is_empty(tmp_path):
    assert at.level_maps("kd", 4000, "0p5", tmp_path) == ([], [], [])


def test_partial_level_uses_only_finished_seeds(tmp_path, monkeypatch):
    import json
    import numpy as np
    for s in (1, 3):  # seed 2 not finished
        tag = f"t1_kd0p5_n4000_s{s}_T8"
        (tmp_path / f"{tag}_verdict.json").write_text(json.dumps({"r_proto": 0.1 * s}))
        (tmp_path / f"{tag}_motif.json").write_text(json.dumps({"discard_s": 2.0}))
    S = np.ones((1, 151, 3, 3))
    K = np.ones((21, 3))
    monkeypatch.setattr(at.sc, "unit_spectra", lambda tag, ds, d: (None, S, K))
    seeds, maps, vs = at.level_maps("kd", 4000, "0p5", tmp_path)
    assert seeds == [1, 3] and len(maps) == 2 and [v["r_proto"] for v in vs] == [0.1, 0.30000000000000004]

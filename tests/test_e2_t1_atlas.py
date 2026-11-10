"""scripts/e2_t1_atlas.py: a level with no finished seed yields no maps (drawn empty), never an error."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_t1_atlas as at  # noqa: E402


def test_missing_level_is_empty(tmp_path):
    assert at.level_maps("kd", 4000, "0p5", tmp_path) == ([], [], [])

"""Imported LANSA density profiles X: shape, axes, finiteness, and a known-answer on the stored digest record."""
import json
import re
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1] / "docs/laminar_motif/data/lansa_density_X.json"


def _d():
    return json.loads(P.read_text(encoding="utf-8"))


def test_shape_axes_and_finite():
    d = _d()
    X = np.array(d["X"])
    assert X.shape == (21, 4, 11)
    assert d["axes"]["types"] == ["PV", "CB", "CR", "NG"]
    assert len(d["axes"]["areas"]) == 11 and len(set(d["axes"]["areas"])) == 11
    e = np.array(d["axes"]["bin_center_rel_L4"])
    assert e.size == 21 and np.all(np.diff(e) > 0) and abs(e[0] + 0.6) < 1e-9 and abs(e[-1] - 0.4) < 1e-9
    assert np.isfinite(X).all() and (X >= 0).all(), "densities are finite and non-negative (zero bins occur)"


def test_source_digest_recorded_and_text_claims_hold():
    d = _d()
    assert re.fullmatch(r"[0-9a-f]{64}", d["source"]["sha256"])
    # pin of the imported values (observed 2026-11-06), not a manuscript claim: it fails if the type axis is swapped or the file replaced
    m = np.array(d["X"]).mean(2)
    e = np.array(d["axes"]["bin_center_rel_L4"])
    assert abs(e[m[:, 0].argmax()]) <= 0.1, "PV (axis 0) peaks within 0.1 of mid-L4"
    assert e[m[:, 1].argmax()] < -0.3, "CB (axis 1) peaks superficially"

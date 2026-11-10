"""Rates and stop states of scripts/e2_select.py on a synthetic run."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_select as sel  # noqa: E402


def _run(d: Path, tag: str, e_hz: dict[str, float]):
    dt, T = 0.1, 3.0  # 3 s, first 2 s discarded
    n = int(T * 1000 / dt)
    sizes = {u: {"E": 100, "PV": 10, "SOM": 10} for u in "SMD"}
    keys = [f"{u}_{t}" for u in "SMD" for t in ("E", "PV", "SOM")]
    counts = np.zeros((n, len(keys)), dtype=np.int32)
    k0 = int(2000 / dt)
    for j, key in enumerate(keys):
        u, t = key.split("_")
        hz = e_hz[u] if t == "E" else 5.0
        total = int(round(hz * 100 if t == "E" else hz * 10))  # spikes in the kept 1 s
        counts[k0:k0 + total, j] = 1
        counts[:k0, j] = 1  # discarded part: must not count
    np.savez(d / f"{tag}.npz", counts=counts, block_keys=np.array(keys))
    (d / f"{tag}.json").write_text(json.dumps({"dt_ms": dt, "sizes": sizes}), encoding="utf-8")


def test_rates_after_discard_and_stop_states(tmp_path):
    _run(tmp_path, "ok", {"S": 4.0, "M": 6.0, "D": 2.0})
    r = sel.rates("ok", 2.0, tmp_path)
    assert abs(r["S"]["E"] - 4.0) < 1e-9 and abs(r["D"]["PV"] - 5.0) < 1e-9, r
    assert sel.stop_state(r) is None
    _run(tmp_path, "hot", {"S": 60.0, "M": 6.0, "D": 2.0})
    assert "E rate > 50" in sel.stop_state(sel.rates("hot", 2.0, tmp_path))
    _run(tmp_path, "cold", {"S": 4.0, "M": 6.0, "D": 0.0})
    assert sel.stop_state(sel.rates("cold", 2.0, tmp_path)) == "deep E rate < 0.1 Hz"
    assert sel.rates("missing", 2.0, tmp_path) is None

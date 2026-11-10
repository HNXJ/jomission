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


def _verdict(d: Path, tag: str, r: float, ok: bool):
    (d / f"{tag}_verdict.json").write_text(json.dumps({"r_proto": r, "pass": ok}), encoding="utf-8")


def test_select_rules(tmp_path):
    good = {"S": 4.0, "M": 6.0, "D": 2.0}
    plan = {  # level: [(E rates, r_proto, pass) per seed]; None = seed missing
        "0p5": [(good, 0.40, True)] * 3,                                            # counted
        "0p71": [(good, 0.60, True), (good, 0.60, True), ({**good, "S": 60.0}, 0.60, True)],  # hot seed: dropped
        "1": [(good, 0.40, False), (good, 0.40, True), (good, 0.40, False)],        # ties 0p5; reference level
        "1p41": [(good, 0.90, True), (good, 0.90, True), None],                     # 2 seeds only: dropped
        "2": [(good, 0.90, True), (good, float("nan"), True), (good, 0.90, True)],  # NaN r_proto: dropped
    }
    for lv, seeds in plan.items():
        for s, x in enumerate(seeds, 1):
            if x is None:
                continue
            tag = f"t1_kd{lv}_n1000_s{s}_T8"
            _run(tmp_path, tag, x[0])
            _verdict(tmp_path, tag, x[1], x[2])
    o = sel.select("kd", 1000, "1", tmp_path)
    counted = {r["level"]: r["counts"] for r in o["levels"]}
    assert counted == {"0p25": False, "0p35": False, "0p5": True, "0p71": False, "1": True, "1p41": False,
                       "2": False}, counted
    assert o["winner"] == "1" and o["winner_passes_2of3"] is False, o
    assert (tmp_path / "t1_kd_n1000_selection.json").exists()


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

"""SEAL_LFS rule of scripts/e2_ladder_rule.py on synthetic ladder files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_ladder_rule as lr  # noqa: E402


def _write(d: Path, plan: dict):
    for n, seeds in plan.items():
        for s, (r, peaks) in zip(lr.SEEDS, seeds):
            t = lr.tag(n, s)
            (d / f"{t}_verdict.json").write_text(json.dumps({"r_proto": r}), encoding="utf-8")
            (d / f"{t}_motif.json").write_text(json.dumps({"sum_over_cells": {"peak_hz_unit": peaks}}),
                                               encoding="utf-8")


REF = [(0.30, {"S": 40, "M": 10, "D": 12}), (0.40, {"S": 42, "M": 11, "D": 12}), (0.35, {"S": 44, "M": 12, "D": 12})]


def test_tag_seed1_n4000_is_earlier_run():
    assert lr.tag(4000, 1) == "sw_n4000_dens_k1_kd1_sst15_s1_T8"
    assert lr.tag(2000, 1) == "lfs_n2000_s1_T8"


def test_smallest_passing_n_wins(tmp_path):
    close = [(r + 0.01, {u: p + 1 for u, p in pk.items()}) for r, pk in REF]  # within 2 SD (0.10) and 3 Hz
    _write(tmp_path, {1000: close, 2000: close, 4000: REF})
    o = lr.rule(tmp_path)
    assert o["tests"]["1000"]["accept"] and o["tests"]["2000"]["accept"]
    assert o["t1_size"] == 1000
    assert (tmp_path / "lfs_rule.json").exists()


def test_peak_off_by_more_than_3hz_rejects(tmp_path):
    close = [(r, dict(pk)) for r, pk in REF]
    s_off = [(r, {**pk, "S": pk["S"] - 3.5}) for r, pk in REF]  # S mean off by 3.5 Hz; r identical
    _write(tmp_path, {1000: s_off, 2000: close, 4000: REF})
    o = lr.rule(tmp_path)
    t = o["tests"]["1000"]
    assert t["r_ok"] and not t["peak_ok"] and not t["accept"]
    assert o["t1_size"] == 2000


def test_sd_is_n4000_seed_sd_ddof1(tmp_path):
    # n4000 r = 0.30, 0.40, 0.35: SD ddof=1 0.05 (2 SD 0.10), ddof=0 0.0408 (2 SD 0.0816).
    # n1000: mean 0.50 (d 0.15) with its own SD 0.40; its own 2 SD would accept, n4000's rejects.
    wide = [(0.10, dict(REF[0][1])), (0.50, dict(REF[1][1])), (0.90, dict(REF[2][1]))]
    # n2000: d 0.09; accepted with ddof=1, rejected with ddof=0.
    near = [(r + 0.09, dict(pk)) for r, pk in REF]
    _write(tmp_path, {1000: wide, 2000: near, 4000: REF})
    o = lr.rule(tmp_path)
    assert abs(o["tests"]["1000"]["two_sd"] - 0.10) < 1e-12
    assert not o["tests"]["1000"]["r_ok"]
    assert o["tests"]["2000"]["r_ok"]
    assert o["t1_size"] == 2000


def test_r_off_and_none_pass_gives_n4000(tmp_path):
    far = [(r - 0.25, dict(pk)) for r, pk in REF]  # 0.25 > 2 SD = 0.10; peaks identical
    _write(tmp_path, {1000: far, 2000: far, 4000: REF})
    o = lr.rule(tmp_path)
    assert o["tests"]["2000"]["peak_ok"] and not o["tests"]["2000"]["r_ok"]
    assert o["t1_size"] == 4000

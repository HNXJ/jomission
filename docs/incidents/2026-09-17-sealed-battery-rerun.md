# 2026-09-17: a bare full-suite pytest overwrote 55 sealed result artifacts

## What happened

While locating a failing tier-0 test, the agent ran `python -m pytest -q tests/ -m ""`. That runs
the tier-3 batteries, and those batteries rewrite `results/*.json` when they run. 55 registered
sealed artifacts were overwritten in one command: recomputed floats and reordered class keys, for
example `battery_b1.json` settle `8.67e-05` -> `9.70e-05`. Nothing in the repository prevented it.
`manifests/sealed_artifacts.json` already carried the rule -- "Slow batteries rewrite results/*.json
when run: rerun only under an authorized lineage" -- as prose, enforced by nothing.

No scientific conclusion was drawn from the overwritten files, and no lineage of that session read
them. The V2.1 PV results of the session were committed before the overwrite surfaced.

## Recovery

`git stash push -- results/` preserved the unauthorized output rather than discarding it, restoring
every file to its sealed blob. `python scripts/project_check.py` then passed, including the sealed
artifacts check. The stash commit was `4d31ae60f0fadc25c0cbbae5e602732a4d831bba`; it held exactly 55 files, all of them
`results/*.json` and all registered in `manifests/sealed_artifacts.json`. It was dropped after this
record was written; it remains reachable through the reflog until git expires it.

## Permanent repair

`jomission/harness/seals.py` denies any write to a path registered in
`manifests/sealed_artifacts.json`, raising before the file is touched. `tests/conftest.py` installs
it for every pytest session, so the exact command that caused this incident now fails instead of
mutating evidence. An authorized lineage names the paths it may rewrite in
`JOMISSION_ALLOW_SEAL_WRITE`. `tests/test_seal_write_guard.py` (tier 0) and a cheap non-mutating
probe inside `scripts/project_check.py` keep the guard honest.

## Files affected

- `results/battery_b1.json`
- `results/battery_b2.json`
- `results/battery_b3.json`
- `results/hdp_scale_battery_b1_s18p0.json`
- `results/hdp_scale_battery_b1_s2p0.json`
- `results/hdp_scale_battery_b1_s36p0.json`
- `results/hdp_scale_battery_b1_s4p5.json`
- `results/hdp_scale_battery_b1_s9p0.json`
- `results/hdp_scale_battery_b2_s18p0.json`
- `results/hdp_scale_battery_b2_s2p0.json`
- `results/hdp_scale_battery_b2_s36p0.json`
- `results/hdp_scale_battery_b2_s9p0.json`
- `results/hdp_scale_battery_b3_s18p0.json`
- `results/hdp_scale_battery_b3_s2p0.json`
- `results/hdp_scale_battery_b3_s36p0.json`
- `results/hdp_scale_battery_b3_s4p5.json`
- `results/hdp_scale_battery_b3_s9p0.json`
- `results/hdp_scale_geometry_s18p0.json`
- `results/hdp_scale_geometry_s36p0.json`
- `results/hdp_scale_geometry_s4p5.json`
- `results/hdp_scale_geometry_s9p0.json`
- `results/n_traj_s50p0_hi.json`
- `results/n_traj_s50p0_lo.json`
- `results/n_traj_s50p0_mid.json`
- `results/n_traj_s54p0_hi.json`
- `results/n_traj_s54p0_lo.json`
- `results/n_traj_s54p0_mid.json`
- `results/n_traj_s57p0_hi.json`
- `results/n_traj_s57p0_lo.json`
- `results/n_traj_s57p0_mid.json`
- `results/n_traj_s61p0_hi.json`
- `results/n_traj_s61p0_lo.json`
- `results/n_traj_s61p0_mid.json`
- `results/s7_battery_b1_s50p0.json`
- `results/s7_battery_b1_s54p0.json`
- `results/s7_battery_b1_s57p0.json`
- `results/s7_battery_b1_s61p0.json`
- `results/s7_battery_b1_s65p0.json`
- `results/s7_battery_b3_s50p0.json`
- `results/s7_battery_b3_s54p0.json`
- `results/s7_battery_b3_s57p0.json`
- `results/s7_battery_b3_s61p0.json`
- `results/s7_battery_b3_s65p0.json`
- `results/s7_geometry_s50p0.json`
- `results/s7_geometry_s54p0.json`
- `results/s7_geometry_s57p0.json`
- `results/s7_geometry_s61p0.json`
- `results/s7_geometry_s65p0.json`
- `results/s8_candidate_s50p0.json`
- `results/s8_candidate_s54p0.json`
- `results/s8_candidate_s57p0.json`
- `results/s8_candidate_s61p0.json`
- `results/s9_init_s50p0_P0.json`
- `results/s9_init_s50p0_P4.json`
- `results/s9_init_s50p0_P7.json`

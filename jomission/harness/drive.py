"""Drive additivity guard: executed input = emitter tonic + schedule.

Schedules add to emitter drive_per_neuron inside the engine, so a schedule
built from the tonic (e.g. np.tile(emitter_drive)) doubles it
(results/duration_postmortem.json). Call before any run that passes a
non-zero schedule.
"""

from __future__ import annotations

import numpy as np


class DriveAdditivityError(AssertionError):
    pass


def check_additive_schedule(emitter_drive, schedule, cls, expected_schedule_mean=None,
                            rel_tol=0.05, abs_tol=1e-3):
    """Validate a (steps, n) schedule against (n,) emitter drive.

    Raises DriveAdditivityError if, for a class with non-zero tonic, the schedule
    class mean sits within rel_tol of the tonic (tonic duplicated), or if it misses
    expected_schedule_mean[class] by more than max(abs_tol, rel_tol*|tonic|).
    Returns {class: {"tonic", "schedule_mean", "executed_mean"}} (native units).
    """
    drive = np.asarray(emitter_drive, dtype=np.float64)
    sched = np.asarray(schedule, dtype=np.float64)
    cls = np.asarray(cls)
    if sched.ndim != 2 or sched.shape[1] != drive.shape[0]:
        raise DriveAdditivityError(f"schedule shape {sched.shape} vs drive {drive.shape}")
    col = sched.mean(axis=0)
    out = {}
    for c in np.unique(cls):
        m = cls == c
        tonic = float(drive[m].mean())
        smean = float(col[m].mean())
        if tonic != 0.0 and abs(smean - tonic) <= rel_tol * abs(tonic):
            raise DriveAdditivityError(f"class {c}: schedule mean {smean:.4g} duplicates tonic {tonic:.4g}")
        if expected_schedule_mean is not None and c in expected_schedule_mean:
            tol = max(abs_tol, rel_tol * abs(tonic))
            if abs(smean - expected_schedule_mean[c]) > tol:
                raise DriveAdditivityError(
                    f"class {c}: schedule mean {smean:.4g} != expected {expected_schedule_mean[c]:.4g} (tol {tol:.3g})")
        out[str(c)] = {"tonic": tonic, "schedule_mean": smean, "executed_mean": tonic + smean}
    return out

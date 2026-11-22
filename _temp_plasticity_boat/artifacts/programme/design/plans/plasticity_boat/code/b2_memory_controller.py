"""B2 explicit-memory reference controller (D3.6 positive control; not learned, not a LIF claim).

Latches the target at cue onset into a register, holds it through DELAY and GO, and acts in GO by the
PD law toward the held target. Reads only the observation returned by DMSEnv.observe(); requires
proprioception (TaskConfig.proprioception=True).

Modes (positive-control arms for T11):
    intact    latch the true cue: q_hold = p_onset + s, equal to q
    absent    never latch; register stays empty; zero action in GO
    shuffled  latch a supplied target instead of the true one (cue from another trial)
"""

from __future__ import annotations

import numpy as np

from dms_task import IDX_B, IDX_CUE_VALID, IDX_GO, IDX_S, KD, KP

MODES = ("intact", "absent", "shuffled")


class ExplicitMemoryController:
    def __init__(self, mode: str = "intact", shuffled_targets=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "shuffled" and shuffled_targets is None:
            raise ValueError("shuffled mode needs shuffled_targets")
        self.mode = mode
        self._targets = None if shuffled_targets is None else [np.asarray(t, float) for t in shuffled_targets]
        self._prev_cue = False
        self._n_latched = 0
        self.q_hold: np.ndarray | None = None  # register; None = empty

    def _latch(self, obs: np.ndarray) -> None:
        if self.mode == "absent":
            return
        if self.mode == "intact":
            self.q_hold = obs[IDX_B][:2] + obs[IDX_S]  # p_onset + (q - p_onset)
        else:
            self.q_hold = self._targets[self._n_latched]
        self._n_latched += 1

    def act(self, obs: np.ndarray) -> np.ndarray:
        cue = bool(obs[IDX_CUE_VALID] == 1.0)
        if cue and not self._prev_cue:
            self._latch(obs)  # first CUE step: p equals p_onset, so the latch is exact
        self._prev_cue = cue
        if obs[IDX_GO] != 1.0 or self.q_hold is None:
            return np.zeros(2)
        p, v = obs[IDX_B][:2], obs[IDX_B][2:]
        return KP * (self.q_hold - p) - KD * v

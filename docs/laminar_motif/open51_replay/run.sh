#!/usr/bin/env bash
# OPEN-51 diagnostic replay: control (R1 entry) then replay (original entry). Replay runs after any control
# outcome except a setup failure (exit 2, 5, 6, 7), per the reviewed stop rule.
PY=D:/venvs/_temp_o51_rerun/Scripts/python.exe
DRV=D:/jomission/scripts/open51_replay.py
WT=D:/repos/_work/_temp_o51_wt_e0
DIAG=D:/jomission/docs/laminar_motif/open51_diag
OUT=D:/tmp/_temp_o51_replay
LOCK=$HOME/.claude/bin/suite_lock.py
mkdir -p $OUT
cd /d/tmp
python $LOCK --wait-s 14400 --label jomission-open51-replay-ctrl -- $PY -B $DRV --worktree $WT \
  --entry $DIAG/r1_e0s1_jit_scan.cache --expect-ynn 0 --label control --out $OUT > $OUT/control.stdout.txt 2>&1
rc=$?
echo "control rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $OUT/run.log
case $rc in 2|5|6|7) echo "stop: control setup failure" >> $OUT/run.log; exit $rc;; esac
python $LOCK --wait-s 14400 --label jomission-open51-replay-orig -- $PY -B $DRV --worktree $WT \
  --entry $DIAG/original_e0s1_jit_scan.cache --expect-ynn 7 --label replay --out $OUT > $OUT/replay.stdout.txt 2>&1
rc=$?
echo "replay rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $OUT/run.log
exit $rc

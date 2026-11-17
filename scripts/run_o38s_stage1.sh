#!/usr/bin/env bash
# OPEN-38 positive-control stage (owner 2026-11-16): ref, flat, rev x seeds 1-5, serial, frozen.
# Usage: bash scripts/run_o38s_stage1.sh      (launched detached; everything it does is logged)
# Order: clean-tree check -> freeze hashes -> export manifest check (pre) -> construction invariance (stops on
# failure) -> 15 runs (failures kept, never retried) -> export manifest check (post) -> stage check.
set -u
cd "$(dirname "$0")/.." || exit 1
export O38S_PY=D:/venvs/o38s_jaxfne059/Scripts/python.exe
export JAXFNE_SITE=D:/repos/_work/_temp_jaxfne_351f6048 JAXFNE_EDGE_DELIVERY=event PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$JAXFNE_SITE
R=results/laminar_e2 MAN=D:/repos/_work/_temp_jaxfne_351f6048.sha256 L=e2_3unit_n4000_dens_k1_kd1_sst15
LOG=$R/o38s_pc_stage.log
mkdir -p "$R"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

exec </dev/null  # detached launch has no stdin; sha256sum and git need a valid fd 0
command -v git >/dev/null || { say "STOP: git not on PATH"; exit 2; }
dirty=$(git status --porcelain --untracked-files=no) || { say "STOP: git status failed"; exit 2; }
[[ -z "$dirty" ]] || { say "STOP: execution tree not clean: $dirty"; exit 2; }
head=$(git rev-parse HEAD) && [[ -n "$head" ]] || { say "STOP: no HEAD"; exit 2; }
say "HEAD $head"
say "env JAXFNE_SITE=$JAXFNE_SITE PYTHONPATH=$PYTHONPATH JAXFNE_EDGE_DELIVERY=$JAXFNE_EDGE_DELIVERY py=$("$O38S_PY" --version 2>&1)"
sha256sum scripts/run_o38s_scored.sh scripts/run_o38s_stage1.sh scripts/run_e2_stack.py scripts/e2_motif_figure.py \
  scripts/e2_motif_similarity.py scripts/e2_scorer.py jomission/laminar/caminos_stack.py jomission/laminar/caminos_e0.py \
  scripts/open38_smoke_check.py scripts/open38_stage1_check.py scripts/open38_construction_check.py \
  docs/laminar_motif/SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md "$MAN" > "$R/o38s_pc_freeze.sha256"
"$O38S_PY" -m pip freeze > "$R/o38s_pc_pip_freeze.txt"
say "freeze written ($(wc -l < "$R/o38s_pc_freeze.sha256") files)"

( cd "$JAXFNE_SITE" && sha256sum -c --quiet "$MAN" ) && pre=PASS || pre=FAIL
say "export manifest pre: $pre; files $(find "$JAXFNE_SITE" -type f | wc -l); pycache $(find "$JAXFNE_SITE" -name __pycache__ | wc -l)"
[[ $pre == PASS ]] || { say "STOP: export manifest failed before runs"; exit 3; }

"$O38S_PY" -B scripts/open38_construction_check.py "$R/o38s_pc_construction.json" $L ${L}_rhof0p794 ${L}_rhor \
  --seeds 1 2 3 4 5 >> "$LOG" 2>&1 && cc=PASS || cc=FAIL
say "construction invariance: $cc"
[[ $cc == PASS ]] || { say "STOP: construction invariance failed; no run started"; exit 4; }

for s in 1 2 3 4 5; do
  for arm in ref flat rev; do
    case $arm in ref) lay=$L ;; flat) lay=${L}_rhof0p794 ;; rev) lay=${L}_rhor ;; esac
    say "start o38s_pc_${arm}_n4000_s${s}_T8 $lay"
    say "$(bash scripts/run_o38s_scored.sh "$s" 8 "o38s_pc_${arm}_n4000_s${s}_T8" "$lay")"
  done
done

( cd "$JAXFNE_SITE" && sha256sum -c --quiet "$MAN" ) && post=PASS || post=FAIL
say "export manifest post: $post; files $(find "$JAXFNE_SITE" -type f | wc -l); pycache $(find "$JAXFNE_SITE" -name __pycache__ | wc -l)"

"$O38S_PY" -B scripts/open38_stage1_check.py "$R/o38s_pc_stage1_check.json" >> "$LOG" 2>&1 && st=PASS || st=FAIL
say "stage check: $st"
say "DONE"

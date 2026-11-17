#!/usr/bin/env bash
# OPEN-38 stage 2 (owner 2026-11-17): v1, mt, mst, teo, chc, shape x seeds 1-5, serial, frozen. Twin of
# run_o38s_stage1.sh (same preflight, chain, no retry). Additions: environment check, one execution manifest per run.
# Usage: bash scripts/run_o38s_stage2.sh smoke   (v1 seed 1 only; verified before the rest)
#        bash scripts/run_o38s_stage2.sh rest    (the other 29; refuses unless smoke passed at the same freeze)
# Order: clean tree -> freeze -> env check -> export manifest (pre) -> construction invariance (smoke: all 30
# arm-seed pairs; rest: requires the smoke's PASS at an identical freeze) -> runs -> export manifest (post) -> integrity.
set -u
cd "$(dirname "$0")/.." || exit 1
mode=${1:-}
[[ $mode == smoke || $mode == rest ]] || { echo "usage: run_o38s_stage2.sh smoke|rest"; exit 1; }
export O38S_PY=D:/venvs/o38s_jaxfne059/Scripts/python.exe
export JAXFNE_SITE=D:/repos/_work/_temp_jaxfne_351f6048 JAXFNE_EDGE_DELIVERY=event PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$JAXFNE_SITE
R=results/laminar_e2 MAN=D:/repos/_work/_temp_jaxfne_351f6048.sha256 B=e2_3unit_n4000_dens_k1_kd1_sst15
LOG=$R/o38s_s2_stage.log
ARMS=(v1 mt mst teo chc shape)
declare -A LAY=([v1]=${B}_rhouS1p688_rhouM0p558_rhouD1p584 [mt]=${B}_rhouS1p280_rhouM0p599_rhouD1p014
  [mst]=${B}_rhouS1p220_rhouM0p615_rhouD1p103 [teo]=${B}_rhouS1p409_rhouM0p695_rhouD1p153
  [chc]=${B}_rhouS1p345_rhouM0p595_rhouD1p178 [shape]=${B}_rhouS1p095_rhouM0p485_rhouD0p942)
mkdir -p "$R"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

exec </dev/null  # detached launch has no stdin; sha256sum and git need a valid fd 0
say "mode $mode"
command -v git >/dev/null || { say "STOP: git not on PATH"; exit 2; }
dirty=$(git status --porcelain --untracked-files=no) || { say "STOP: git status failed"; exit 2; }
[[ -z "$dirty" ]] || { say "STOP: execution tree not clean: $dirty"; exit 2; }
head=$(git rev-parse HEAD) && [[ -n "$head" ]] || { say "STOP: no HEAD"; exit 2; }
say "HEAD $head"
say "env JAXFNE_SITE=$JAXFNE_SITE PYTHONPATH=$PYTHONPATH JAXFNE_EDGE_DELIVERY=$JAXFNE_EDGE_DELIVERY py=$("$O38S_PY" --version 2>&1)"
FZ=$R/o38s_s2_freeze_$mode.sha256
sha256sum scripts/run_o38s_scored.sh scripts/run_o38s_stage2.sh scripts/run_e2_stack.py scripts/e2_motif_figure.py \
  scripts/e2_motif_similarity.py scripts/e2_scorer.py jomission/laminar/caminos_stack.py jomission/laminar/caminos_e0.py \
  scripts/open38_smoke_check.py scripts/open38_stage1_check.py scripts/open38_construction_check.py \
  scripts/open38_stage1_integrity.py scripts/open38_exec_manifest.py scripts/open38_env_check.py \
  docs/laminar_motif/SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md "$MAN" > "$FZ" || { say "STOP: freeze failed"; exit 2; }
"$O38S_PY" -m pip freeze > "$R/o38s_s2_pip_freeze_$mode.txt"
say "freeze written ($(wc -l < "$FZ") files)"
if [[ $mode == smoke ]]; then
  echo "$head" > "$R/o38s_s2_head.txt"
else
  [[ "$(cat "$R/o38s_s2_head.txt" 2>/dev/null)" == "$head" ]] || { say "STOP: HEAD differs from the smoke's"; exit 2; }
  cmp -s "$FZ" "$R/o38s_s2_freeze_smoke.sha256" || { say "STOP: freeze differs from the smoke's"; exit 2; }
  grep -q "^[0-9TZ:-]* smoke integrity: PASS$" "$LOG" || { say "STOP: no smoke integrity PASS in the log"; exit 2; }
fi

"$O38S_PY" -B scripts/open38_env_check.py "$R/o38s_s2_env_$mode.json" >> "$LOG" 2>&1 && ev=PASS || ev=FAIL
say "environment check: $ev"
[[ $ev == PASS ]] || { say "STOP: environment check failed"; exit 3; }
( cd "$JAXFNE_SITE" && sha256sum -c --quiet "$MAN" ) && pre=PASS || pre=FAIL
say "export manifest pre: $pre; files $(find "$JAXFNE_SITE" -type f | wc -l); pycache $(find "$JAXFNE_SITE" -name __pycache__ | wc -l)"
[[ $pre == PASS ]] || { say "STOP: export manifest failed before runs"; exit 3; }

if [[ $mode == smoke ]]; then
  lays=(); for a in "${ARMS[@]}"; do lays+=("${LAY[$a]}"); done
  "$O38S_PY" -B scripts/open38_construction_check.py "$R/o38s_s2_construction.json" $B "${lays[@]}" \
    --seeds 1 2 3 4 5 >> "$LOG" 2>&1 && cc=PASS || cc=FAIL
  say "construction invariance: $cc"
  [[ $cc == PASS ]] || { say "STOP: construction invariance failed; no run started"; exit 4; }
  runs=("v1 1")
else
  grep -q '"pass": true' "$R/o38s_s2_construction.json" || { say "STOP: no construction PASS from the smoke"; exit 4; }
  say "construction invariance: PASS at the smoke (identical freeze)"
  runs=(); for s in 1 2 3 4 5; do for a in "${ARMS[@]}"; do [[ "$a $s" == "v1 1" ]] || runs+=("$a $s"); done; done
fi
for r in "${runs[@]}"; do  # refuse to reuse any receipt: every run here is new
  set -- $r; t=o38s_${1}_n4000_s${2}_T8
  for f in "$R/$t.json" "$R/${t}_verdict.json" "$R/${t}_exec.json"; do
    [[ -e $f ]] && { say "STOP: $f exists before its run"; exit 5; }
  done
done

for r in "${runs[@]}"; do
  set -- $r; a=$1 s=$2 t=o38s_${1}_n4000_s${2}_T8
  id=$("$O38S_PY" -B scripts/open38_exec_manifest.py begin "$t" --head "$head") || { say "STOP: manifest begin failed $t"; exit 6; }
  say "start $t ${LAY[$a]} exec=$id"
  out=$(bash scripts/run_o38s_scored.sh "$s" 8 "$t" "${LAY[$a]}")
  say "$out"
  rc=${out##*rc=}; [[ $rc =~ ^[0-9]+$ ]] || rc=99
  "$O38S_PY" -B scripts/open38_exec_manifest.py end "$t" "$rc" || { say "STOP: manifest end failed $t"; exit 6; }
done

( cd "$JAXFNE_SITE" && sha256sum -c --quiet "$MAN" ) && post=PASS || post=FAIL
say "export manifest post: $post; files $(find "$JAXFNE_SITE" -type f | wc -l); pycache $(find "$JAXFNE_SITE" -name __pycache__ | wc -l)"

if [[ $mode == smoke ]]; then
  "$O38S_PY" -B scripts/open38_stage1_integrity.py "$R/o38s_s2_smoke_integrity.json" --head "$head" --stage 2 \
    --runs v1:1 --stage-log "$LOG" >> "$LOG" 2>&1 && ig=PASS || ig=FAIL
  [[ $post == PASS ]] || ig=FAIL
  say "smoke integrity: $ig"
else
  "$O38S_PY" -B scripts/open38_stage1_integrity.py "$R/o38s_s2_integrity.json" --head "$head" --stage 2 \
    --stage-log "$LOG" >> "$LOG" 2>&1 && ig=PASS || ig=FAIL
  say "stage-2 integrity: $ig"
fi
say "DONE $mode"

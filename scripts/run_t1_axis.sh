#!/bin/zsh
# T1 one axis (docs/laminar_motif/SEAL_T1.md): 7 levels x seeds 1-3, 8 s, then the W2 scorer per run.
# BASE is the layout with the active axis's token removed (e.g. e2_3unit_n1000_dens_k1_sst15 for axis kd).
# Usage: zsh scripts/run_t1_axis.sh AXIS BASE N [P]     AXIS in kd | sst | sstd
cd "${0:A:h}/.." || exit 1
zmodload zsh/parameter
axis=$1 base=$2 N=$3 P=${4:-2}
case $axis in
  kd)   levels=(0p25 0p35 0p5 0p71 1 1p41 2) ;;
  sst)  levels=(5p3 7p5 10p6 15 21 30 42) ;;
  sstd) levels=(7p5 10p6 15 21 30 42 60) ;;
  *) echo "unknown axis $axis"; exit 2 ;;
esac
run_one() {  # level seed
  local tag=t1_${axis}${1}_n${N}_s${2}_T8 lay=${base}_${axis}${1}
  [[ -f results/laminar_e2/${tag}_verdict.json ]] && { echo "${tag} done"; return; }
  /usr/local/bin/python3 -u scripts/run_e2_stack.py "$2" 8 "$tag" "$lay" > "results/laminar_e2/${tag}.log" 2>&1 \
    && /usr/local/bin/python3 scripts/e2_motif_figure.py "$tag" 2 >> "results/laminar_e2/${tag}.log" 2>&1 \
    && /usr/local/bin/python3 scripts/e2_motif_similarity.py "$tag" >> "results/laminar_e2/${tag}.log" 2>&1 \
    && /usr/local/bin/python3 scripts/e2_scorer.py "$tag" >> "results/laminar_e2/${tag}.log" 2>&1
  echo "${tag} rc=$?"
}
for l in $levels; do
  for s in 1 2 3; do
    while (( ${#jobstates} >= P )); do sleep 10; done
    run_one "$l" "$s" &
  done
done
wait

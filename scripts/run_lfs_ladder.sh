#!/bin/zsh
# L-FS finite-size ladder (docs/laminar_motif/SEAL_LFS.md): kd1_sst15 at N 1000/2000/4000 E per unit, seeds 1-3,
# 8 s, JaxFNE 0.5.8 authority (no JAXFNE_SITE). n4000 seed 1 exists. At most P runs at once (default 2).
# Usage: zsh scripts/run_lfs_ladder.sh [P]
cd "${0:A:h}/.." || exit 1
zmodload zsh/parameter
P=${1:-2}
jobs_list=(1000:1 1000:2 1000:3 2000:1 2000:2 2000:3 4000:2 4000:3)
run_one() {  # N seed
  local tag=lfs_n${1}_s${2}_T8 lay=e2_3unit_n${1}_dens_k1_kd1_sst15
  /usr/local/bin/python3 -u scripts/run_e2_stack.py "$2" 8 "$tag" "$lay" > "results/laminar_e2/${tag}.log" 2>&1 \
    && /usr/local/bin/python3 scripts/e2_motif_figure.py "$tag" 2 >> "results/laminar_e2/${tag}.log" 2>&1 \
    && /usr/local/bin/python3 scripts/e2_motif_similarity.py "$tag" >> "results/laminar_e2/${tag}.log" 2>&1
  echo "${tag} rc=$?"
}
for j in $jobs_list; do
  while (( ${#jobstates} >= P )); do sleep 10; done  # $(jobs) runs in a subshell and sees no jobs
  run_one "${j%%:*}" "${j##*:}" &
done
wait

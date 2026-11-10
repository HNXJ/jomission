#!/bin/zsh
# E2 log sweeps (owner 2026-11-09): inter-unit coupling k and deep E-cell size dsz on the LANSA-density
# 3-unit stack (mean 4000 E per unit), seed 1, 8 s, scored with e2_motif_figure.py (2 s discarded).
# Usage: JAXFNE_SITE=<site> zsh scripts/run_e2_sweep.sh [PARALLEL]   (default 3 runs at once)
# Each run writes results/laminar_e2/sw_<variant>_s1_T8.{npz,json,log} and _motif.{png,json}.
cd "${0:A:h}/.." || exit 1
zmodload zsh/parameter
P=${1:-3}
variants=(n4000_dens_k0p1 n4000_dens_k0p3 n4000_dens_k1 n4000_dens_k3 n4000_dens_k10
          n4000_dens_k1_dsz0p5 n4000_dens_k1_dsz2 n4000_dens_k1_dsz4)
run_one() {
  local v=$1 tag=sw_${1}_s1_T8
  /usr/local/bin/python3 -u scripts/run_e2_stack.py 1 8 "$tag" "e2_3unit_${v}" > "results/laminar_e2/${tag}.log" 2>&1 \
    && /usr/local/bin/python3 scripts/e2_motif_figure.py "$tag" 2 >> "results/laminar_e2/${tag}.log" 2>&1 \
    && /usr/local/bin/python3 scripts/e2_gallery.py >> "results/laminar_e2/${tag}.log" 2>&1
  echo "${tag} rc=$?"
}
for v in $variants; do
  while (( ${#jobstates} >= P )); do sleep 5; done  # $(jobs) runs in a subshell and sees no jobs
  run_one "$v" &
done
wait

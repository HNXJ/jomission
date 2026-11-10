#!/bin/zsh
# T8 density and SOM/PV-ratio controls (docs/laminar_motif/SEAL_T8.md): 5 arms x seeds 1-3, 8 s, scored.
# Usage: JAXFNE_SITE=<JaxFNE 0.5.9 at 351f6048> zsh scripts/run_t8.sh [P]   (P runs at once, default 2)
cd "${0:A:h}/.." || exit 1
[[ -n "$JAXFNE_SITE" ]] || { echo "set JAXFNE_SITE to JaxFNE 0.5.9 at 351f6048"; exit 1; }
export JAXFNE_EDGE_DELIVERY=event
P=${1:-2}
typeset -A LAY
LAY=(ref e2_3unit_n4000_dens_k1_kd1_sst15
     dflat e2_3unit_n4000_k1_sst15
     drev e2_3unit_n4000_densr_k1_kd1_sst15
     rflat e2_3unit_n4000_dens_k1_kd1_sst15_rhof0p794
     rrev e2_3unit_n4000_dens_k1_kd1_sst15_rhor)
jobs_list=()
for s in 1 2 3; do
  for arm in ref dflat drev rflat rrev; do
    jobs_list+=("$s t8_${arm}_n4000_s${s}_T8 ${LAY[$arm]}")
  done
done
for j in "${jobs_list[@]}"; do
  set -- ${=j}
  while (( $(jobs -r | wc -l) >= P )); do sleep 5; done
  zsh scripts/run_e2_scored.sh "$1" 8 "$2" "$3" &
done
wait
echo "t8 done"

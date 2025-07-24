#!/bin/bash

source ../config.sh 
motif_counts=(5 10 15 2)
width_windows=("5 10" "5 15" "5 20" "5 25" "5 30")
for n_motifs in "${motif_counts[@]}"; do
  for width_range in "${width_windows[@]}"; do
    read minw maxw <<< "$width_range"
    sbatch --export=ALL,N_MOTIFS=$n_motifs,MIN_MOTIF_W=$minw,MAX_MOTIF_W=$maxw --account=$ACCOUNT_STRING de_novo.sh
  done
done
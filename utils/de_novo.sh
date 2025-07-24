#!/bin/bash

#SBATCH --job-name=motif
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/motif.log

module load bedtools

source ../config.sh

cd "$DATA_PROCESSING_DIR"
mkdir -p meme_out
bedtools getfasta -fi "$REF_FA" -bed union_peaks.bed -fo union_peaks.fasta
"$MEME" union_peaks.fasta \
  -oc meme_out/motifs_${N_MOTIFS}_w${MIN_MOTIF_W}-${MAX_MOTIF_W} \
  -dna \
  -mod zoops \
  -nmotifs "$N_MOTIFS" \
  -minw "$MIN_MOTIF_W" \
  -maxw "$MAX_MOTIF_W" \
  -p "$N_THREADS"
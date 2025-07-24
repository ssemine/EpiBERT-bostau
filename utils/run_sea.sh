#!/bin/bash


source ../config.sh
MOTIF_FILE="$HOME/EpiBERT-bostau/example_usage/consensus_pwms.meme"
OUTPUT_BASE="/home/sstyopa/data/bovineATAC/data_processing/sea_out/consensus"
mkdir -p OUTPUT_BASE


for fasta_file in "$FILTERED_FASTA_DIR"/*.top50000.peaks.filtered.fasta; do
  sample=$(basename "$fasta_file" .top50000.peaks.filtered.fasta)
  
  echo "Processing sample: $sample"
  
  mkdir -p "$OUTPUT_BASE/$sample"
  
  echo "Starting SEA for sample: $sample"
  sea --p "$fasta_file" \
      --m "$MOTIF_FILE" \
      --n "$FILTERED_FASTA_DIR/${sample}_bg.fasta" \
      --thresh 1.0 \
      --verbosity 1 \
      -oc "$OUTPUT_BASE/$sample"
  
  echo "Finished SEA for sample: $sample"
done

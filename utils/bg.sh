#!/bin/bash

source ../config.sh
sample="X438.Placental.Cotyledon"
cd "$DATA_PROCESSING_DIR"

awk '$1 ~ /^chr([1-9]$|1[0-9]$|2[0-9]$|X$|Y$)/' union_peaks.bed > union_peaks.filtered.bed

bedtools subtract -a union_peaks.filtered.bed -b $OUTPUT_DIR/unfiltered/$sample.top50000.centered.bed > "${sample}_bg.bed"
bedtools getfasta -fi $REF_FA -bed "${sample}_bg.bed" -fo "${sample}_bg.fasta"



# $HOME/EpiBERT-bostau/example_usage/consensus_pwms.meme
sea \
  --p "$OUTPUT_DIR/filtered/$sample.top50000.peaks.filtered.fasta" \
  --m "$DATA_PROCESSING_DIR/meme_out/meme.txt" \
  --n "${sample}_bg.fasta" \
  --thresh 1.0 \
  --verbosity 1 \
  -oc "$DATA_PROCESSING_DIR/sea_out/de_novo"
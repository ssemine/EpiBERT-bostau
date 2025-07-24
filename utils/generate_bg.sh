#!/bin/bash

source ../config.sh
cd "$DATA_PROCESSING_DIR"

awk '$1 ~ /^chr([1-9]$|1[0-9]$|2[0-9]$|X$|Y$)/' union_peaks.bed > union_peaks.filtered.bed

for bfile in "$OUTPUT_DIR/unfiltered/"*.top50000.centered.bed; do
    sample=$(basename "$bfile" .top50000.centered.bed)

    echo "Processing sample: $sample"
    
    echo "Filtering union peaks for $sample..."
    bedtools subtract -a union_peaks.filtered.bed -b "$bfile" > "$FILTERED_BED_DIR/${sample}_bg.bed"
    echo "Filtering union peaks for $sample..."
    bedtools getfasta -fi "$REF_FA" -bed "$FILTERED_BED_DIR/${sample}_bg.bed" -fo "$FILTERED_FASTA_DIR/${sample}_bg.fasta"
done
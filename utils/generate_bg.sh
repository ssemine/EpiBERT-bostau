#!/bin/bash

source ../config.sh
cd "$DATA_PROCESSING_DIR"

for bfile in "$OUTPUT_DIR/unfiltered/"*.top50000.centered.bed; do
    sample=$(basename "$bfile" .top50000.centered.bed)

    echo "Processing sample: $sample"
    
    echo "Filtering union peaks for $sample..."
    bedtools subtract -a union_peaks.filtered.bed -b "$bfile" > "$FILTERED_BED_DIR/${sample}_bg.bed"
    echo "Filtering union peaks for $sample..."
    bedtools getfasta -fi "$REF_FA" -bed "$FILTERED_BED_DIR/${sample}_bg.bed" -fo "$FILTERED_FASTA_DIR/${sample}_bg.fasta"
done

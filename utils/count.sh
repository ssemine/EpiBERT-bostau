#!/bin/bash



source ../config.sh

cd "$DATA_PROCESSING_DIR"

mkdir -p counts

for bam in "$OUTPUT_DIR/unfiltered"/*.dedup.bam; do
    [ -e "$bam" ] || continue
    sample=$(basename "$bam" .dedup.bam)
    echo "Processing $sample..."
    bedtools coverage -a union_peaks.filtered.bed -b "$bam" -counts | cut -f4 > "counts/${sample}.txt"
done

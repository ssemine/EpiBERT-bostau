#!/bin/bash

source ../config.sh

cd "$OUTPUT_DIR"

# cat *.narrowPeak | sort -k1,1 -k2,2n | bedtools merge > $DATA_PROCESSING_DIR/union_peaks.bed
cat *.top50000.centered.bed | sort -k1,1 -k2,2n | bedtools merge > $DATA_PROCESSING_DIR/union_peaks.bed

cd "$DATA_PROCESSING_DIR"

mkdir -p counts


export UNION_PEAKS=union_peaks.bed

parallel --jobs 8 '
    sample=$(basename {} .dedup.bam);
    bedtools coverage -a "$UNION_PEAKS" -b {} -counts | cut -f4 > "counts/${sample}.txt"
' ::: "$OUTPUT_DIR"/*.dedup.bam






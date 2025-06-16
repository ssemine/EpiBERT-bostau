#!/bin/bash

source ../config.sh

for R1 in "$ARRAY_EXPRESS_DIR"/*_R1.fastq.gz; do
    base=$(basename "$R1" _R1.fastq.gz)
    R2="$ARRAY_EXPRESS_DIR/${base}_R2.fastq.gz"

    if [[ -f "$R2" ]]; then
        echo "Found pair: $base"
        export TESTER_R1="${base}_R1.fastq.gz"
        export TESTER_R2="${base}_R2.fastq.gz"
        ./local_tester.sh
    else
        echo "Warning: Missing R2 for $base"
    fi
done

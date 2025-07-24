#!/bin/bash

source ../config.sh

count=1

for R1 in "$ARRAY_EXPRESS_DIR"/*_R1.fastq.gz; do
    base=$(basename "$R1" _R1.fastq.gz)
    R2="$ARRAY_EXPRESS_DIR/${base}_R2.fastq.gz"

    if [[ -f "$R2" ]]; then
        echo "Found pair: $base"
        export TESTER_R1="${base}_R1.fastq.gz"
        export TESTER_R2="${base}_R2.fastq.gz"

        sbatch --account="$ACCOUNT_STRING" \
               --job-name="local_tester_${count}" \
               --export=TESTER_R1="$TESTER_R1",TESTER_R2="$TESTER_R2" \
               local_tester.sh

        ((count++))
    else
        echo "Warning: Missing R2 for $base"
    fi
done


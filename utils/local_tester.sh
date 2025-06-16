#!/bin/bash


mkdir -p logs
source ../config.sh
mkdir -p "$OUTPUT_DIR"

# sourced from config.sh
R1="$ARRAY_EXPRESS_DIR/$TESTER_R1"
R2="$ARRAY_EXPRESS_DIR/$TESTER_R2"

sample=$(basename "$R1" _R1.fastq.gz)

echo "Processing $sample..."

# trimming
"$TRIM_GALORE" --paired -o "$OUTPUT_DIR" "$R1" "$R2"
R1_TRIM="$OUTPUT_DIR/${sample}_R1_val_1.fq.gz"
R2_TRIM="$OUTPUT_DIR/${sample}_R2_val_2.fq.gz"

# alignment
bowtie2 -x "$BOWTIE2_IDX" -1 "$R1_TRIM" -2 "$R2_TRIM" -S "$OUTPUT_DIR/$sample.sam" -p "$N_THREADS" \
    2> "$OUTPUT_DIR/$sample.bowtie2.log"
samtools view -@ "$N_THREADS" -bS "$OUTPUT_DIR/$sample.sam" > "$OUTPUT_DIR/$sample.bam"
samtools sort -@ "$N_THREADS" -n "$OUTPUT_DIR/$sample.bam" -o "$OUTPUT_DIR/$sample.namesorted.bam"
samtools fixmate -@ "$N_THREADS" -m "$OUTPUT_DIR/$sample.namesorted.bam" "$OUTPUT_DIR/$sample.fixmate.bam"
samtools sort -@ "$N_THREADS" "$OUTPUT_DIR/$sample.fixmate.bam" -o "$OUTPUT_DIR/$sample.sorted.bam"
samtools markdup -@ "$N_THREADS" -r "$OUTPUT_DIR/$sample.sorted.bam" "$OUTPUT_DIR/$sample.dedup.bam"

# cleaning up intermediate alignment files
rm "$OUTPUT_DIR/$sample.sam" \
    "$OUTPUT_DIR/$sample.bam" \
    "$OUTPUT_DIR/$sample.namesorted.bam" \
    "$OUTPUT_DIR/$sample.fixmate.bam" \
    "$OUTPUT_DIR/$sample.sorted.bam"

# peak call
macs2 callpeak -t "$OUTPUT_DIR/$sample.dedup.bam" -f BAMPE -g "$GENOME_SIZE" -n "$sample" \
    --outdir "$OUTPUT_DIR" --keep-dup all -B --SPMR

# top 50k peaks
cat "$OUTPUT_DIR/${sample}_peaks.narrowPeak" | \
    sort -k5,5nr | head -n "$N_PEAKS" | \
    awk -v win="$HALF_WINDOW_SIZE" 'OFS="\t" {center = int(($2 + $3)/2); print $1, center - win, center + win}' | \
    sort -k1,1 -k2,2n > "$OUTPUT_DIR/$sample.top50000.centered.bed"
bedtools getfasta -fi "$REF_FA" -bed "$OUTPUT_DIR/$sample.top50000.centered.bed" \
    -fo "$OUTPUT_DIR/$sample.top50000.peaks.fasta"

# final cleanup
rm "$R1_TRIM" "$R2_TRIM"
rm "$OUTPUT_DIR/${sample}_R*fastq.gz_trimming_report.txt"

echo "Done $sample"
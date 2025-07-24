#!/bin/bash

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --export=ALL

# Load required modules
module load bowtie2
module load samtools
module load macs2
module load bedtools

# Create log and output directories
mkdir -p logs
source ../config.sh
mkdir -p "$OUTPUT_DIR"

# Use TESTER_R1 and TESTER_R2 passed via sbatch
R1="$ARRAY_EXPRESS_DIR/$TESTER_R1"
R2="$ARRAY_EXPRESS_DIR/$TESTER_R2"
sample=$(basename "$R1" _R1.fastq.gz)

echo "Processing $sample..."

# Trimming
"$TRIM_GALORE" --paired -o "$OUTPUT_DIR" "$R1" "$R2"
R1_TRIM="$OUTPUT_DIR/${sample}_R1_val_1.fq.gz"
R2_TRIM="$OUTPUT_DIR/${sample}_R2_val_2.fq.gz"

# Alignment and BAM cleanup
bowtie2 -x "$BOWTIE2_IDX" -1 "$R1_TRIM" -2 "$R2_TRIM" -S "$OUTPUT_DIR/$sample.sam" -p "$N_THREADS" \
    2> "$OUTPUT_DIR/$sample.bowtie2.log"

samtools view -@ "$N_THREADS" -bS "$OUTPUT_DIR/$sample.sam" > "$OUTPUT_DIR/$sample.bam"
samtools sort -@ "$N_THREADS" -n "$OUTPUT_DIR/$sample.bam" -o "$OUTPUT_DIR/$sample.namesorted.bam"
samtools fixmate -@ "$N_THREADS" -m "$OUTPUT_DIR/$sample.namesorted.bam" "$OUTPUT_DIR/$sample.fixmate.bam"
samtools sort -@ "$N_THREADS" "$OUTPUT_DIR/$sample.fixmate.bam" -o "$OUTPUT_DIR/$sample.sorted.bam"
samtools markdup -@ "$N_THREADS" -r "$OUTPUT_DIR/$sample.sorted.bam" "$OUTPUT_DIR/$sample.dedup.bam"

# Cleanup intermediate BAMs
rm "$OUTPUT_DIR/$sample.sam" \
   "$OUTPUT_DIR/$sample.bam" \
   "$OUTPUT_DIR/$sample.namesorted.bam" \
   "$OUTPUT_DIR/$sample.fixmate.bam" \
   "$OUTPUT_DIR/$sample.sorted.bam"

# Peak calling
macs2 callpeak -t "$OUTPUT_DIR/$sample.dedup.bam" -f BAMPE -g "$GENOME_SIZE" -n "$sample" \
    --outdir "$OUTPUT_DIR" --keep-dup all -B --SPMR

# Extract top peaks
cat "$OUTPUT_DIR/${sample}_peaks.narrowPeak" | \
    sort -k5,5nr | head -n "$N_PEAKS" | \
    awk -v win="$HALF_WINDOW_SIZE" 'OFS="\t" {center = int(($2 + $3)/2); print $1, center - win, center + win}' | \
    sort -k1,1 -k2,2n > "$OUTPUT_DIR/$sample.top${N_PEAKS}.centered.bed"

# Get FASTA
bedtools getfasta -fi "$REF_FA" -bed "$OUTPUT_DIR/$sample.top${N_PEAKS}.centered.bed" \
    -fo "$OUTPUT_DIR/$sample.top${N_PEAKS}.peaks.fasta"

# Final cleanup
rm "$R1_TRIM" "$R2_TRIM"
rm "$OUTPUT_DIR/${sample}_R"*fastq.gz_trimming_report.txt

echo "Done $sample"

#!/bin/bash

#SBATCH --job-name=atac_paired
#SBATCH --output=logs/atac_paired_%A_%a.out
#SBATCH --error=logs/atac_paired_%A_%a.err
#SBATCH --array=0-207
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=general

module load bowtie2
module load samtools
module load macs2
module load bedtools

mkdir -p logs

source ../config.sh

GENOME_DIR="$SCRATCH_DIR/EpiBERT-bostau_data/refGenomes/bovine"
ATAC_DIR="$SCRATCH_DIR/EpiBERT-bostau_data/bovineATAC/arrayExpress"
REF_FA="$GENOME_DIR/bosTau9.fa"
BOWTIE2_IDX="$GENOME_DIR/bosTau9"
GENOME_SIZE="2.7e9"
INPUT_DIR="$ATAC_DIR"
OUTPUT_DIR="$ATAC_DIR/output"

TRIM_GALORE="/home/s4693165/.local/bin/TrimGalore-0.6.10/trim_galore"

mkdir -p "$OUTPUT_DIR"

R1_FILES=($(find "$INPUT_DIR" -name "*_R1.fastq.gz" | sort))
R2_FILES=($(find "$INPUT_DIR" -name "*_R2.fastq.gz" | sort))
R1="${R1_FILES[$SLURM_ARRAY_TASK_ID]}"
R2="${R2_FILES[$SLURM_ARRAY_TASK_ID]}"
sample=$(basename "$R1" _R1.fastq.gz)

echo "Processing $sample..."

# trimming

"$TRIM_GALORE" --paired -o "$OUTPUT_DIR" "$R1" "$R2"

R1_TRIM="$OUTPUT_DIR/${sample}_R1_val_1.fq.gz"
R2_TRIM="$OUTPUT_DIR/${sample}_R2_val_2.fq.gz"

# align
bowtie2 -x "$BOWTIE2_IDX" -1 "$R1_TRIM" -2 "$R2_TRIM" -S "$OUTPUT_DIR/$sample.sam" -p 8 \
    2> "$OUTPUT_DIR/$sample.bowtie2.log"

# sam bam fix this
samtools view -@ 8 -bS "$OUTPUT_DIR/$sample.sam" | \
samtools sort -@ 8 -o "$OUTPUT_DIR/$sample.sorted.bam"
rm "$OUTPUT_DIR/$sample.sam"

samtools markdup -r -@ 8 "$OUTPUT_DIR/$sample.sorted.bam" "$OUTPUT_DIR/$sample.dedup.bam"

samtools index "$OUTPUT_DIR/$sample.dedup.bam"

rm "$OUTPUT_DIR/$sample.sorted.bam"

# peak call
macs2 callpeak -t "$OUTPUT_DIR/$sample.dedup.bam" -f BAMPE -g "$GENOME_SIZE" -n "$sample" \
    --outdir "$OUTPUT_DIR" --keep-dup all -B --SPMR

# top 50k
zcat "$OUTPUT_DIR/${sample}_peaks.narrowPeak.gz" | \
    sort -k5,5nr | head -n 50000 | \
    awk 'OFS="\t" {center = int(($2 + $3)/2); print $1, center - 64, center + 64}' | \
    sort -k1,1 -k2,2n > "$OUTPUT_DIR/$sample.top50000.centered.bed"

# get fasta
bedtools getfasta -fi "$REF_FA" -bed "$OUTPUT_DIR/$sample.top50000.centered.bed" \
    -fo "$OUTPUT_DIR/$sample.top50000.peaks.fasta"

echo "Done $sample"


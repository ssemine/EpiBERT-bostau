#!/bin/bash

source ../config.sh
cd "$DATA_PROCESSING_DIR"
module load bedtools

cat $UNFILTERED_DIR/*.top50000.centered.bed > all_peaks.bed
awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-9]|X|Y)$/' all_peaks.bed > all_peaks.filtered.bed
sort -k1,1 -k2,2n all_peaks.filtered.bed > all_peaks.sorted.bed
bedtools merge -i all_peaks.sorted.bed > union_peaks.bed
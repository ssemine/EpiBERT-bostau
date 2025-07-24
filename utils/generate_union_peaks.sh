#!/bin/bash

source ../config.sh
cd "$DATA_PROCESSING_DIR"

cat $UNFILTERED_DIR/*.bed > all_peaks.bed
sort -k1,1 -k2,2n all_peaks.bed > all_peaks.sorted.bed
bedtools merge -i all_peaks.sorted.bed > union_peaks.bed


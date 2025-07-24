#!/bin/bash

source ../config.sh

cd "$DATA_PROCESSING_DIR"
bedtools getfasta -fi "$REF_FA" -bed union_peaks.filtered.bed -fo union_peaks.filtered.fasta
meme union_peaks.filtered.fasta -oc meme_out -dna -mod zoops -nmotifs 10 -minw 6 -maxw 20


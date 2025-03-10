## EpiBERT
<img src="docs/graphical_abstract.png" width="400">

**EpiBERT** learns representations of accessible sequence that generalize across cell types via "masked accessibility modeling" which can be used for downstream fine-tuning tasks, such as caQTL predictions and RAMPAGE-seq prediction in new cell types. [Link to manuscript](https://www.cell.com/cell-genomics/fulltext/S2666-979X(25)00018-7)

### Installation
This repository depends on Python 3 and Tensorflow. See all requirements which you can see and install via requirements.txt for pip. For ease you can clone this repository and run set pip install -e .

### Model Usage
These notebooks are a work in progress. I will be adding notebooks for variant scoring w/ null distribution, motif generation, and RAMPAGE-seq/QTL prediction shortly. If there is a task in particular you would like please post it as an issue. 
- [Data preprocessing for model usage](example_usage/data_processing.ipynb)
- [caQTL scoring and plotting ](example_usage/caqTL_predict.ipynb)

### dataset processing
Scripts and code for dataset processing can be found in the data_processing directory.
 * alignment_and_peak_call : ATAC-seq alignment and peak calling
 * create_signal_tracks : 
    * bam_to_bed_RAMPAGE_5PRIME.wdl - convert RAMPAGE-seq bam to fragment bed file containing locations of mapped TSS, and then convert to scaled bedGraph signal file
    * bam_to_bed_ATAC.wdl - convert ATAC-seq bam to fragments file
    * liftover_fragments.wdl - convert hg19 fragments files from CATLAS to hg38 given the required input chain file
    * fragments_to_bed_scores.wdl - convert input fragments files corresponding to tn5 cut sites to scaled bedGraph signal file

 * downloading utilities : download fastqs from GEO or processed bams from ENCODE
 * motif_enrichment : compute motif enrichments of consensus motifs from Vierstra et. al 2020 using MEME simple enrichment analysis (SEA)
 * write_TF_records : convert input ATAC and RAMPAGE profiles, and motif enrichments to tfrecords. Also contains all the train/validation/test sequence splits in sequence_splits subdirectory. 


### dataset inputs
Training and evaluation data are available on google cloud at gs://epibert/data. You will need google cloud access with a valid billing account to access/read the data. Datasets are in tensorflow record format - functions for reading/deserializing/decoding this data are available in the scripts described in 'main_files' directory below. 

Pre-training data
 * training data: epibert/data/atac_pretraining/train
 * 18 validation cell-types + 2160 validation sequences: epibert/data/atac_pretraining/valid
 * evaluation over testing regions (~40k centered windows on peaks + background negative regions in the 1840 testing sequences):
  * 16 testing cell-types + testing regions: epibert/data/atac_pretraining/test/testing_celltypes
  * 15 training cell-types + testing regions: epibert/data/atac_pretraining/test/training_celltypes
 
 Fine-tuning data
 * training data (50 paired ATAC-RAMPAGE datasets over 34021 training sequences): epibert/data/rampage_fine_tuning/train
 * validation data (50 paired ATAC-RAMPAGE datasets over )

Processed finetuning data (paired ATAC/RAMPAGE datasets) is available at gs://epibert/data/rampage_fine_tuning

### main files
For pre-training(masked atac prediction, _atac suffix files):
 * execute_pretraining.sh - training bash script where you can define dataset locations, weights and biases username and project id, and model hyperparameters
 * training_utils_atac_pretrain.py - define functions for train and validation steps, data loading and augmentation, masking, early stopping, model saving
 * train_model_atac_pretrain.py - define main training loop, argument parsing, wandb initialization code, TPU initialization code
 * src/models/epibert_atac_pretrain.py - main model file
 * src/layers/layers.py - all custom layers
 * src/layers/snnk_attention.py - linear attention code with rotary positional encodings

Files for fine-tuning for RAMPAGE prediction follow a similar structure

### model weights
 * Two pre-trained models are available at gs://epibert/models/pretrained. Both models are used for caQTL analyses. Only model1 was used for downstream fine-tuning.

 * The fine-tuned model for RAMPAGE-seq prediction is available at gs://epibert/models/fine_tuned







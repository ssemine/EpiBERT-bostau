import argparse
import collections
import gzip
import math
import os
import random
import shutil
import subprocess
import sys
import time
sys.path.insert(0, '/home/jupyter/repos/genformer')
import numpy as np
import pandas as pd
import tensorflow as tf


import pybedtools as pybt
import tabix as tb

from datetime import datetime
from tensorflow import strings as tfs
from tensorflow.keras import initializers as inits
from scipy import stats
from scipy.signal import find_peaks

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
from kipoiseq import Interval
import pyfaidx
import kipoiseq

import src.models.aformer_atac as genformer
import logomaker

def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=4)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters),
                      depth = 5,
                      dtype=tf.float32)[:, :4]
    return out
    
    
 ################################################################################
def process_bedgraph(interval, df):
    '''
    Extract coverage info for input pybedtools interval object

    Return:
        numpy array of interval length where each pos is 5' read coverage
    '''
    
    df['start'] = df['start'].astype('int64') - int(interval.start)
    df['end'] = df['end'].astype('int64') - int(interval.start)
    
    ## ensure within bounds of interval
    df.loc[df.start < 0, 'start'] = 0
    df.loc[df.end > len(interval), 'end'] = len(interval)
    
    per_base = np.zeros(len(interval), dtype=np.float64)
    
    num_intervals = df['start'].to_numpy().shape[0]

    output = np.asarray(get_per_base_score_f(
                    df['start'].to_numpy().astype(np.int_),
                    df['end'].to_numpy().astype(np.int_),
                    df['score'].to_numpy().astype(np.float64),
                    per_base), dtype = np.float64)

    ### bin at the desired output res
    return output


def get_per_base_score_f(start, end, score, base):
    num_bins = start.shape[0]
    for k in range(num_bins):
        base[start[k]:end[k]] = score[k]
    return base


def return_bg_interval(atac_bedgraph,
                         chrom,interval_start,interval_end,num_bins,resolution):
    
    interval_str = '\t'.join([chrom, 
                              str(interval_start),
                              str(interval_end)])
    interval_bed = pybt.BedTool(interval_str, from_string=True)
    interval = interval_bed[0]
    
    atac_bedgraph_bed = tb.open(atac_bedgraph)
    ### atac processing ######################################-
    atac_subints= atac_bedgraph_bed.query(chrom,
                                          interval_start,
                                          interval_end)
    atac_subints_df = pd.DataFrame([rec for rec in atac_subints])


    # if malformed line without score then disard
    if (len(atac_subints_df.index) == 0):
        atac_bedgraph_out = np.array([0.0] * (target_length))
    else:
        atac_subints_df.columns = ['chrom', 'start', 'end', 'score']
        atac_bedgraph_out = process_bedgraph(
            interval, atac_subints_df)
        
    atac_processed = atac_bedgraph_out
    atac_processed = np.reshape(atac_processed, [num_bins,resolution])
    atac_processed = np.sum(atac_processed,axis=1,keepdims=True)
    
    atac_processed = tf.constant(atac_processed,dtype=tf.float32)
    
    return atac_processed


def resize_interval(interval_str,size):
    
    chrom = interval_str.split(':')[0]
    start=int(interval_str.split(':')[1].split('-')[0].replace(',',''))
    stop=int(interval_str.split(':')[1].split('-')[1].replace(',',''))


    new_start = (int(start)+int(stop))//2 - int(size)//2
    new_stop = (int(start)+int(stop))//2 + int(size)//2

    return chrom,new_start,new_stop

def plot_tracks(tracks,start, end, y_lim, height=1.5):
    fig, axes = plt.subplots(len(tracks)+1, 1, figsize=(24, height * (len(tracks)+1)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(start, end, num=len(y[0])), y[0],color=y[1])
        ax.set_title(title)
        ax.set_ylim((0,y_lim))
    plt.tight_layout()


class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


class genformer_model:
    def __init__(self, strategy, model, model_checkpoint):
        self.model = model
        
        dummy_seq = tf.data.Dataset.from_tensor_slices([tf.ones((524288,4),dtype=tf.float32)] *6)
        dummy_atac = tf.data.Dataset.from_tensor_slices([tf.ones((131072,1),dtype=tf.float32)]*6)
        dummy_motif = tf.data.Dataset.from_tensor_slices([tf.ones((1,693),dtype=tf.float32)]*6)
        combined_dataset = tf.data.Dataset.zip((dummy_seq, dummy_atac, dummy_motif))
        batched_dataset = combined_dataset.batch(6)
        dist = strategy.experimental_distribute_dataset(batched_dataset)
        dist_it = iter(dist)
        print('loading')
        @tf.function
        def build(input_dummy):
            self.model(input_dummy,training=False)
        strategy.run(build, args=(next(dist_it),))
        
        print('ran test input')
        ckpt = tf.train.Checkpoint(model=self.model)
        status = ckpt.restore(model_checkpoint)
        status.assert_existing_objects_matched()
        print('loaded weights')
        
    
    def predict_on_batch_dist(self, strategy, inputs):
        
        @tf.function
        def run_model(inputs):
            output,att_matrices,att_matrices_norm,out_performer,other_data = self.model.predict_on_batch(inputs)
            
            return output,att_matrices,att_matrices_norm,out_performer,other_data
            
        outputs,att_matrices,att_matrices_norm,out_performer,other_data = strategy.run(run_model, args=(next(inputs),))
        
        return outputs,att_matrices,att_matrices_norm,out_performer,other_data

    def contribution_input_grad_dist(self, strategy, model_inputs, gradient_mask):
        @tf.function
        def contribution_input_grad_dist(model_inputs,gradient_mask):
            seq,atac,tf_activity = model_inputs
            gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
            gradient_mask_mass = tf.reduce_sum(gradient_mask)

            with tf.GradientTape() as input_grad_tape:
                input_grad_tape.watch(seq)
                input_grad_tape.watch(atac)
                prediction,att_matrices,att_matrices_norm,out_performer,other_data = self.model.predict_on_batch(model_inputs)
                
                prediction = tf.cast(prediction,dtype=tf.float32)
                gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)

                prediction_mask = tf.reduce_sum(gradient_mask *
                                                prediction) / gradient_mask_mass
                
            input_grads = input_grad_tape.gradient(prediction_mask, model_inputs)

            input_grads_seq = input_grads[0] 
            input_grads_atac = input_grads[1]

            atac_grads = input_grads_atac[0,:,] * atac[0,:,]

            return seq, input_grads_seq, atac_grads, prediction, att_matrices
        
        seq, seq_grads_orig, atac_grads, prediction, att_matrices = \
            strategy.run(contribution_input_grad_dist, args = (next(model_inputs),gradient_mask))
        
        ## adjust the seq_grads, just make them all to length 524287
        seq_grads_min1 = seq_grads_orig.values[0][0,2:,:]
        seq_grads = seq_grads_orig.values[1][0,1:-1,:]
        seq_grads_max1 = seq_grads_orig.values[2][0,:-2,:]
        
        seq_grads_min1_r = tf.reverse(tf.gather(seq_grads_orig.values[3][0,:-2,:],
                                        [3,2,1,0],axis=-1),axis=[0])
        seq_grads_r = tf.reverse(tf.gather(seq_grads_orig.values[4][0,1:-1,:],
                                        [3,2,1,0],axis=-1),axis=[0])
        seq_grads_min1_r = tf.reverse(tf.gather(seq_grads_orig.values[5][0,2:,:],
                                        [3,2,1,0],axis=-1),axis=[0])
        
        seq_grads_all = [seq_grads_min1, seq_grads, seq_grads_max1,
                         seq_grads_min1_r, seq_grads_r,seq_grads_min1_r]
        
        return seq, seq_grads_all, atac_grads, prediction, att_matrices
    
    

class genformer_model_nostrat:
    def __init__(self, model, model_checkpoint):
        self.model = model
        
        dummy_seq = tf.ones((1,524288,4),dtype=tf.float32)
        dummy_atac = tf.ones((1,131072,1),dtype=tf.float32)
        dummy_motif = tf.ones((1,1,693),dtype=tf.float32)
        inputs = dummy_seq,dummy_atac,dummy_motif
        print('loading')

        self.model(inputs,training=False)

        print('ran test input')
        ckpt = tf.train.Checkpoint(model=self.model)
        status = ckpt.restore(model_checkpoint)
        status.assert_existing_objects_matched()
        print('loaded weights')
        
    
    def predict_on_batch_dist(self, inputs):
        
        output,att_matrices,att_matrices_norm,out_performer,other_data = self.model.predict_on_batch(inputs)

        return output,att_matrices,att_matrices_norm,out_performer,other_data

    def contribution_input_grad_dist(self, model_inputs, gradient_mask):
                             
        @tf.function
        def contribution_input_grad_dist(model_inputs,gradient_mask):
            seq,atac,tf_activity = model_inputs
            gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
            gradient_mask_mass = tf.reduce_sum(gradient_mask)

            with tf.GradientTape() as input_grad_tape:
                input_grad_tape.watch(seq)
                input_grad_tape.watch(atac)
                prediction,att_matrices,att_matrices_norm,out_performer,other_data = self.model.predict_on_batch(model_inputs)
                prediction = tf.cast(prediction,dtype=tf.float32)
                gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
                prediction_mask = tf.reduce_sum(gradient_mask * prediction) / gradient_mask_mass
                
            input_grads = input_grad_tape.gradient(prediction_mask, model_inputs)

            input_grads_seq = input_grads[0] 
            input_grads_atac = input_grads[1]

            atac_grads = input_grads_atac[0,:,] * atac[0,:,]

            return seq, input_grads_seq, atac_grads, prediction, att_matrices
        
        seq, seq_grads_orig, atac_grads, prediction, att_matrices = contribution_input_grad_dist(model_inputs,gradient_mask)

        return seq, seq_grads_orig, atac_grads, prediction, att_matrices

# Function to process the motif data and load it into a NumPy array
def process_and_load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter, process, and store data
    processed_data = []
    for line in lines:
        if 'consensus_pwms.meme' in line and 'AC' in line:
            fields = line.split('\t')
            try:
                # Store the 3rd field and the negated 15th field
                processed_data.append((str(fields[2]), -1 * float(fields[14])))
            except (IndexError, ValueError):
                # Skip lines with missing or non-numeric data
                continue
    # Sort the list based on the first field (equivalent to sort -k1,1)
    processed_data.sort(key=lambda x: x[0])

    # Extract the second element (negated 15th field) for each tuple
    motif_activity = [value for _, value in processed_data]
    #motif_activity = np.round(np.array(motif_activity),2)
    
    motif_activity=tf.constant(motif_activity,dtype=tf.float32)
    min_val = tf.reduce_min(motif_activity)
    max_val = tf.reduce_max(motif_activity)
    motif_activity = (motif_activity - min_val) / (max_val - min_val)
    return motif_activity

    
def return_all_inputs(interval, atac_dataset, SEQUENCE_LENGTH,
                      num_bins, resolution, motif_activity,crop_size,output_length,
                      fasta_extractor,mask_indices_list,strategy):

    chrom,start,stop = resize_interval(interval,SEQUENCE_LENGTH)
    atac_arr = return_bg_interval(atac_dataset,chrom,
                                    start,stop,num_bins,resolution)
    atac_arr = atac_arr + tf.math.abs(tf.random.normal(atac_arr.shape,mean=1.0e-04,stddev=1.0e-04,dtype=tf.float32))
    
    ## temporary
    
    motif_activity=process_and_load_data(motif_activity)
    motif_activity = tf.expand_dims(motif_activity,axis=0)

    ## get two around
    interval = kipoiseq.Interval(chrom, start-1, stop+1)
    sequence_one_hot_orig = tf.constant(one_hot(fasta_extractor.extract(interval)),dtype=tf.float32)

    mask = np.zeros((1,SEQUENCE_LENGTH//128,1))
    mask_centered = np.zeros((1,SEQUENCE_LENGTH//128,1))
    atac_mask = np.ones((SEQUENCE_LENGTH//128,1))
    for entry in mask_indices_list.split(','): 
        
        mask_start = int(entry.split('-')[0])
        mask_end = int(entry.split('-')[1])
        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start,mask_end):
                mask[0,k,0]=1
        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start+5,mask_end-3):
                mask_centered[0,k,0]=1
        for k in tf.range(mask_start,mask_end):
            atac_mask[k,0] = 0.0
            
    atac_mask = tf.constant(atac_mask,dtype=tf.float32)
    atac_mask = tf.reshape(tf.tile(atac_mask, [1,32]),[-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    

    masked_atac = atac_arr * atac_mask 
    
    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 150.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=150.0) + diff
    
    masked_atac_reshape = tf.reduce_sum(tf.reshape(masked_atac, [-1,32]),axis=1,keepdims=True)
    masked_atac_reshape = tf.slice(masked_atac_reshape,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    
    target_atac = tf.reduce_sum(tf.reshape(atac_arr, [-1,32]),axis=1,keepdims=True)

    diff = tf.math.sqrt(tf.nn.relu(target_atac - 2000.0 * tf.ones(target_atac.shape)))
    target_atac = tf.clip_by_value(target_atac, clip_value_min=0.0, clip_value_max=2000.0) + diff

    target_atac = tf.slice(target_atac,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    
    ### do the test time augmentation
    sequence_one_hot_min1 = tf.slice(sequence_one_hot_orig,[0,0],[output_length*128,-1])
    sequence_one_hot = tf.slice(sequence_one_hot_orig,[1,0],[output_length*128,-1])
    sequence_one_hot_max1 = tf.slice(sequence_one_hot_orig,[2,0],[output_length*128,-1])

    
    sequence_one_hot_min1_rev = tf.gather(sequence_one_hot_min1, [3, 2, 1, 0], axis=-1)
    sequence_one_hot_min1_rev = tf.reverse(sequence_one_hot_min1_rev, axis=[0])
    
    sequence_one_hot_rev = tf.gather(sequence_one_hot, [3, 2, 1, 0], axis=-1)
    sequence_one_hot_rev = tf.reverse(sequence_one_hot_rev, axis=[0])
    
    sequence_one_hot_max1_rev = tf.gather(sequence_one_hot_max1, [3, 2, 1, 0], axis=-1)
    sequence_one_hot_max1_rev = tf.reverse(sequence_one_hot_max1_rev, axis=[0])
    
    masked_atac_rev = tf.reverse(masked_atac,axis=[0])
    
    # Create a dataset for each tensor
    seqs = tf.data.Dataset.from_tensor_slices([sequence_one_hot_min1,sequence_one_hot,sequence_one_hot_max1,
                                               sequence_one_hot_min1_rev,sequence_one_hot_rev,sequence_one_hot_max1_rev])
    
    atacs = tf.data.Dataset.from_tensor_slices([masked_atac,masked_atac,masked_atac,
                                                masked_atac_rev,masked_atac_rev,masked_atac_rev])
    motifs = tf.data.Dataset.from_tensor_slices([motif_activity]*6)

    # Zip the datasets together
    combined_dataset = tf.data.Dataset.zip((seqs, atacs, motifs))

    batched_dataset = combined_dataset.batch(6).repeat(2)

    # Convert the batched dataset to an iterator
    dist = strategy.experimental_distribute_dataset(batched_dataset)
    dist_it = iter(dist)
    
    mask = tf.slice(mask, [0,crop_size,0],[-1,output_length-2*crop_size,-1])
    mask_centered = tf.slice(mask_centered, [0,crop_size,0],[-1,output_length-2*crop_size,-1])
    return dist_it,target_atac,masked_atac_reshape, mask, mask_centered


def plot_logo(matrix,y_min,y_max):
    
    df = pd.DataFrame(matrix, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'
    
    # create Logo object
    nn_logo = logomaker.Logo(df)

    # style using Logo methods
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=['left'], visible=True, bounds=[y_min, y_max])

    # style using Axes methods
    nn_logo.ax.set_ylim([y_min, y_max])
    nn_logo.ax.set_yticks([])
    nn_logo.ax.set_yticklabels([])
    nn_logo.ax.set_ylabel('saliency', labelpad=-1)
    


def write_bg(window,seq_length, crop,input_arr,output_res,out_file_name):
    
    chrom,start,stop = resize_interval(window,seq_length)
    start = start + crop * output_res
    
    out_file = open(out_file_name, 'w')
    for k, value in enumerate(input_arr):
        start_interval = k * output_res + start
        end_interval = (k+1) * output_res + start

        line = [str(chrom),
                str(start_interval), str(end_interval),
                str(value)]
        
        out_file.write('\t'.join(line) + '\n')
    out_file.close()
    
    

def return_all_inputs_no_strategy(interval, atac_dataset, SEQUENCE_LENGTH,
                      num_bins, resolution, motif_activity,crop_size,output_length,
                      fasta_extractor,mask_indices_list):

    chrom,start,stop = resize_interval(interval,SEQUENCE_LENGTH)
    atac_arr = return_bg_interval(atac_dataset,chrom,
                                    start,stop,num_bins,resolution)
    atac_arr = atac_arr + tf.math.abs(tf.random.normal(atac_arr.shape,mean=1.0e-04,stddev=1.0e-04,dtype=tf.float32))
    
    ## temporary
    
    motif_activity=process_and_load_data(motif_activity)
    motif_activity = tf.expand_dims(motif_activity,axis=0)

    ## get two around
    interval = kipoiseq.Interval(chrom, start-1, stop+1)
    
    sequence_one_hot_orig = tf.constant(one_hot(fasta_extractor.extract(interval)),dtype=tf.float32)
    
    sequence_one_hot_min1 = tf.slice(sequence_one_hot_orig,[0,0],[output_length*128,-1])
    sequence_one_hot = tf.slice(sequence_one_hot_orig,[1,0],[output_length*128,-1])
    sequence_one_hot_max1 = tf.slice(sequence_one_hot_orig,[2,0],[output_length*128,-1])

    sequence_one_hot_min1_rev = tf.gather(sequence_one_hot_min1, [3, 2, 1, 0], axis=-1)
    sequence_one_hot_min1_rev = tf.reverse(sequence_one_hot_min1_rev, axis=[0])
    
    sequence_one_hot_rev = tf.gather(sequence_one_hot, [3, 2, 1, 0], axis=-1)
    sequence_one_hot_rev = tf.reverse(sequence_one_hot_rev, axis=[0])
    
    sequence_one_hot_max1_rev = tf.gather(sequence_one_hot_max1, [3, 2, 1, 0], axis=-1)
    sequence_one_hot_max1_rev = tf.reverse(sequence_one_hot_max1_rev, axis=[0])
    
    sequences = sequence_one_hot_min1,sequence_one_hot, \
                    sequence_one_hot_max1, sequence_one_hot_min1_rev, \
                        sequence_one_hot_rev,sequence_one_hot_max1_rev  
    
    mask = np.zeros((1,SEQUENCE_LENGTH//128,1))
    mask_centered = np.zeros((1,SEQUENCE_LENGTH//128,1))
    atac_mask = np.ones((SEQUENCE_LENGTH//128,1))
    for entry in mask_indices_list.split(','): 
        
        mask_start = int(entry.split('-')[0])
        mask_end = int(entry.split('-')[1])
        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start,mask_end):
                mask[0,k,0]=1
        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start+5,mask_end-3):
                mask_centered[0,k,0]=1
        for k in tf.range(mask_start,mask_end):
            atac_mask[k,0] = 0.0
            
    atac_mask = tf.constant(atac_mask,dtype=tf.float32)
    atac_mask = tf.reshape(tf.tile(atac_mask, [1,32]),[-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    

    masked_atac = atac_arr * atac_mask 
    
    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 150.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=150.0) + diff
    
    masked_atac_rev = tf.reverse(masked_atac,axis=[0])
    
    masked_atac_reshape = tf.reduce_sum(tf.reshape(masked_atac, [-1,32]),axis=1,keepdims=True)
    masked_atac_reshape = tf.slice(masked_atac_reshape,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    
    target_atac = tf.reduce_sum(tf.reshape(atac_arr, [-1,32]),axis=1,keepdims=True)

    diff = tf.math.sqrt(tf.nn.relu(target_atac - 2000.0 * tf.ones(target_atac.shape)))
    target_atac = tf.clip_by_value(target_atac, clip_value_min=0.0, clip_value_max=2000.0) + diff

    target_atac = tf.slice(target_atac,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    
    mask = tf.slice(mask, [0,crop_size,0],[-1,output_length-2*crop_size,-1])
    mask_rev = tf.reverse(mask,axis=[0])
    mask_centered = tf.slice(mask_centered, [0,crop_size,0],[-1,output_length-2*crop_size,-1])
    mask_centered_rev = tf.reverse(mask_centered,axis=[0])
    
    masked_atacs = masked_atac, masked_atac_rev
    masks = mask, mask_rev
    masks_centered = mask_centered,mask_centered_rev
    
    return sequences, masked_atacs, motif_activity,target_atac, masked_atac_reshape, masks, masks_centered



def return_grads(model1, model2): 
    
    @tf.function
    def contribution_input_grad_dist(inputs, gradient_mask):
        sequence,rev_seq,atac,rev_atac,mask,mask_rev,target,target_rev,motif_activity,interval_id,cell_type=inputs
        gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
        gradient_mask_mass = tf.reduce_sum(gradient_mask)
        
        
        input_tuple = sequence,atac,motif_activity
        with tf.GradientTape() as input_grad_tape:
            input_grad_tape.watch(sequence)
            input_grad_tape.watch(atac)
            prediction,att_matrices,att_matrices_norm,out_performer,other_data = model1.predict_on_batch(input_tuple)
            prediction = tf.cast(prediction,dtype=tf.float32)
            gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
            prediction_mask = tf.reduce_sum(gradient_mask * prediction) / gradient_mask_mass

        input_grads = input_grad_tape.gradient(prediction_mask, input_tuple)

        input_grads_seq = input_grads[0]
        
        '__________________________________'
        input_tuple = rev_seq,rev_atac,motif_activity

        with tf.GradientTape() as input_grad_tape:
            input_grad_tape.watch(rev_seq)
            input_grad_tape.watch(atac)
            prediction,att_matrices,att_matrices_norm,out_performer,other_data = model1.predict_on_batch(input_tuple)
            prediction = tf.cast(prediction,dtype=tf.float32)
            gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
            prediction_mask = tf.reduce_sum(gradient_mask * prediction) / gradient_mask_mass

        input_grads_rev = input_grad_tape.gradient(prediction_mask, input_tuple)

        input_grads_seq_rev = input_grads_rev[0] 
        
        '------model2---------------'
        input_tuple = sequence,atac,motif_activity
        with tf.GradientTape() as input_grad_tape:
            input_grad_tape.watch(sequence)
            input_grad_tape.watch(atac)
            prediction,att_matrices,att_matrices_norm,out_performer,other_data = model2.predict_on_batch(input_tuple)
            prediction = tf.cast(prediction,dtype=tf.float32)
            gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
            prediction_mask = tf.reduce_sum(gradient_mask * prediction) / gradient_mask_mass

        input_grads2 = input_grad_tape.gradient(prediction_mask, input_tuple)

        input_grads_seq2 = input_grads2[0]
        
        '__________________________________'
        input_tuple = rev_seq,rev_atac,motif_activity

        with tf.GradientTape() as input_grad_tape:
            input_grad_tape.watch(rev_seq)
            input_grad_tape.watch(atac)
            prediction,att_matrices,att_matrices_norm,out_performer,other_data = model2.predict_on_batch(input_tuple)
            prediction = tf.cast(prediction,dtype=tf.float32)
            gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
            prediction_mask = tf.reduce_sum(gradient_mask * prediction) / gradient_mask_mass

        input_grads_rev2 = input_grad_tape.gradient(prediction_mask, input_tuple)

        input_grads_seq_rev2 = input_grads_rev2[0] 
        
        all_grads_out = input_grads_seq + \
                            tf.reverse(tf.gather(input_grads_seq_rev,[3,2,1,0],axis=-1),axis=[0]) + \
                                input_grads_seq2 + \
                                    tf.reverse(tf.gather(input_grads_seq_rev2,[3,2,1,0],axis=-1),axis=[0])
        
        all_grads_out_sub = all_grads_out[:,261888:262400,:] / 4.0
        all_grads_out_sub = all_grads_out_sub / tf.reduce_max(tf.math.abs(all_grads_out_sub))
        seq_out_sub = sequence[:,261888:262400,:]

        return seq_out_sub, all_grads_out_sub
    
    
    return contribution_input_grad_dist


def deserialize_test(serialized_example, g, use_motif_activity, mask, atac_mask,
                     input_length = 196608,
                   max_shift = 10, output_length_ATAC = 49152, output_length = 1536,
                   crop_size = 2, output_res = 128, 
                   mask_size = 1536, log_atac = False, use_atac = True, use_seq = True):
    """Deserialize bytes stored in TFRecordFile."""
    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'peaks': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'motif_activity': tf.io.FixedLenFeature([], tf.string),
        'interval': tf.io.FixedLenFeature([], tf.string),
        'cell_type':tf.io.FixedLenFeature([],tf.string)
    }
    ### stochastic sequence shift and gaussian noise
    seq_shift=5
    
    keep_length_atac = input_length // 4

    ## now parse out the actual data
    data = tf.io.parse_example(serialized_example, feature_map)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,524288))
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float16),
                           [output_length_ATAC,1])
    atac = tf.cast(atac,dtype=tf.float32)
    atac = atac + tf.math.abs(g.normal(atac.shape,mean=1.0e-04,stddev=1.0e-04,dtype=tf.float32))
    atac_target = atac ## store the target
    

    motif_activity = tf.ensure_shape(tf.io.parse_tensor(data['motif_activity'],
                                              out_type=tf.float16),
                           [693])
    motif_activity = tf.cast(motif_activity,dtype=tf.float32)
    min_val = tf.reduce_min(motif_activity)
    max_val = tf.reduce_max(motif_activity)
    motif_activity = (motif_activity - min_val) / (max_val - min_val)
    motif_activity = tf.expand_dims(motif_activity,axis=0)


    interval_id = tf.ensure_shape(tf.io.parse_tensor(data['interval'],
                                              out_type=tf.int32),
                           [])

    cell_type = tf.ensure_shape(tf.io.parse_tensor(data['cell_type'],
                                              out_type=tf.int32),
                           [])

    atac_mask = tf.constant(atac_mask,dtype=tf.float32)
    atac_mask = tf.reshape(tf.tile(atac_mask, [1,32]),[-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)

    masked_atac = atac * atac_mask
    
    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 150.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=150.0) + diff


    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1,32]),axis=1,keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(atac_out - 2000.0 * tf.ones(atac_out.shape)))
    atac_out = tf.clip_by_value(atac_out, clip_value_min=0.0, clip_value_max=2000.0) + diff
    atac_out = tf.slice(atac_out,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    
    mask = tf.slice(mask, [crop_size,0],[output_length-2*crop_size,-1])

    ### get rev 
    rev_seq = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
    rev_seq = tf.reverse(rev_seq, axis=[0])
    masked_atac_rev = tf.reverse(masked_atac,axis=[0])
    mask_rev = tf.reverse(mask,axis=[0])
    atac_out_rev = tf.reverse(atac_out,axis=[0])
    
    return tf.cast(tf.ensure_shape(sequence,[524288,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(rev_seq,[524288,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac_rev, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(mask, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(mask_rev, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(atac_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(atac_out_rev,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(motif_activity, [1,693]),dtype=tf.bfloat16), \
                tf.cast(interval_id,dtype=tf.int32),\
                tf.cast(cell_type,dtype=tf.int32)




def return_dataset(gcs_path, batch, input_length, output_length_ATAC,
                   output_length, crop_size, output_res, max_shift, options,
                   num_parallel, mask,atac_mask,
                   random_mask_size, log_atac, use_atac, use_seq, seed,
                   use_motif_activity, g):
    """
    return a tf dataset object for given gcs path
    """
    wc = "*.tfr"


    list_files = (tf.io.gfile.glob(gcs_path))

    print(os.path.join(gcs_path,wc))
    files = tf.data.Dataset.list_files(list_files,shuffle=False)
    dataset = tf.data.TFRecordDataset(files,
                                          compression_type='ZLIB',
                                          num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)
    dataset = dataset.map(lambda record: deserialize_test(record, g, use_motif_activity,mask,atac_mask,
                                                             input_length, max_shift,
                                                             output_length_ATAC, output_length,
                                                             crop_size, output_res,
                                                             random_mask_size,
                                                             log_atac, use_atac, use_seq),
                          
                          
                      deterministic=True,
                      num_parallel_calls=num_parallel)

    return dataset.repeat(2).batch(batch).prefetch(tf.data.AUTOTUNE)


def return_distributed_iterators(gcs_path, global_batch_size,
                                 input_length, max_shift, output_length_ATAC,
                                 output_length, crop_size, output_res,
                                 num_parallel_calls, strategy,
                                 options, random_mask_size, mask,atac_mask,
                                 log_atac, use_atac, use_seq, seed,
                                 use_motif_activity, g):

    test_data = return_dataset(gcs_path, global_batch_size, input_length,
                             output_length_ATAC, output_length, crop_size,
                             output_res, max_shift, options, num_parallel_calls,
                               mask,atac_mask, random_mask_size,
                             log_atac, use_atac, use_seq, seed, use_motif_activity, g)

    test_dist = strategy.experimental_distribute_dataset(test_data)
    test_data_it = iter(test_dist)

    return test_data_it

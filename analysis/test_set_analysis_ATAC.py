import time
import os
import subprocess
import sys
sys.path.insert(0, '/home/jupyter/repos/genformer')
import re
import argparse
import collections
import gzip
import math
import shutil

import numpy as np
import time
from datetime import datetime
import random

import seaborn as sns

import logging
os.environ['TPU_LOAD_LIBRARY']='0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf

import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
from scipy.stats.stats import pearsonr  
from scipy.stats.stats import spearmanr  
## custom modules
import src.models.aformer_atac_old as aformer
from src.layers.layers import *
import src.metrics as metrics
from src.optimizers import *
import src.schedulers as schedulers

import training_utils_atac as training_utils

from scipy import stats


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
    
    ablate_length = (524288 - input_length) // 2
    keep_length_atac = input_length // 4
    atac_ablate_length = ablate_length // 4

    ## now parse out the actual data
    data = tf.io.parse_example(serialized_example, feature_map)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,524288))
    
    if ablate_length > 0:
        edge_start = sequence[:ablate_length, :]
        edge_end = sequence[-ablate_length:, :]

        # Randomly permute these slices along axis 0
        permuted_indices_start = tf.random.shuffle(tf.range(ablate_length))
        permuted_indices_end = tf.random.shuffle(tf.range(ablate_length))

        permuted_edge_start = tf.gather(edge_start, permuted_indices_start)
        permuted_edge_end = tf.gather(edge_end, permuted_indices_end)

        sequence = tf.slice(sequence,[ablate_length,0],[input_length,-1])
        sequence = tf.concat([permuted_edge_start,
                              sequence,
                              permuted_edge_end],axis=0)

    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float16),
                           [output_length_ATAC,1])
    atac = tf.cast(atac,dtype=tf.float32)
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

    if atac_ablate_length > 0:
        edge_start = masked_atac[:atac_ablate_length, :]
        edge_end = masked_atac[-atac_ablate_length:, :]

        # Randomly permute these slices along axis 0
        permuted_indices_start = tf.random.shuffle(tf.range(atac_ablate_length))
        permuted_indices_end = tf.random.shuffle(tf.range(atac_ablate_length))

        permuted_edge_start = tf.gather(edge_start, permuted_indices_start)
        permuted_edge_end = tf.gather(edge_end, permuted_indices_end)

        masked_atac = tf.slice(masked_atac,[atac_ablate_length, 0], [keep_length_atac,-1])
        masked_atac = tf.concat([permuted_edge_start,masked_atac,permuted_edge_end],axis=0)
    

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

def return_test_build(model,strategy):
    
    @tf.function(reduce_retracing=True)
    def dist_test_step(inputs):
        sequence,rev_seq,atac,rev_atac,mask,mask_rev,target,target_rev,motif_activity,interval_id,cell_type=inputs

        input_tuple = sequence,atac,motif_activity

        output_profile = model(input_tuple, training=False)
        output_profile = tf.cast(output_profile,dtype=tf.float32) # ensure cast to float32
        
        mask_indices = tf.where(mask[0,:,0] == 1)[:,0]
        
        target_atac = tf.gather(target, mask_indices,axis=1)
        output_atac = tf.gather(output_profile, mask_indices,axis=1)
        
        '------------------------------------------'
        input_tuple_rev = rev_seq,rev_atac,motif_activity
        output_profile_rev = model(input_tuple_rev, training=False)
        output_profile_rev = tf.cast(output_profile_rev,dtype=tf.float32) # ensure cast to float32
        
        mask_indices_rev = tf.where(mask_rev[0,:,0] == 1)[:,0]
        target_atac_rev = tf.reverse(tf.gather(target_rev, mask_indices_rev,axis=1),axis=[1])
        output_atac_rev = tf.reverse(tf.gather(output_profile_rev, mask_indices_rev,axis=1),axis=[1])
        
        target_atac_mean = (target_atac + target_atac_rev) / 2.0
        output_atac_mean = (output_atac + output_atac_rev) / 2.0
       
        target_atac_mean_sum = tf.squeeze(tf.reduce_sum(target_atac_mean,axis=1))
        output_atac_mean_sum = tf.squeeze(tf.reduce_sum(output_atac_mean,axis=1)) 

        interval_id_single = interval_id
        interval_id = tf.tile(tf.expand_dims(tf.expand_dims(interval_id,axis=1),axis=2),
                                [1,12,1])
        cell_type_single = cell_type
        cell_type = tf.tile(tf.expand_dims(tf.expand_dims(cell_type,axis=1),axis=2),
                                [1,12,1])
                                 
        return target_atac_mean, output_atac_mean, interval_id,cell_type,target_atac_mean_sum,output_atac_mean_sum,interval_id_single,cell_type_single


    def build_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(reduce_retracing=True)
        def val_step(inputs):
            sequence,atac,motif_activity=inputs

            input_tuple = sequence,atac,motif_activity

            output_profile = model(input_tuple, training=False)

        #for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
        strategy.run(val_step, args=(next(iterator),))

    return dist_test_step, build_step


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



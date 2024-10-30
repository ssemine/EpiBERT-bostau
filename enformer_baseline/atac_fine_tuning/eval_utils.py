import time
import os
import subprocess
import sys
import re
import argparse
import collections
import gzip
import math 
import shutil
import matplotlib.pyplot as plt
import wandb
import numpy as np
from datetime import datetime
import random

import multiprocessing
#import logging
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf
import sonnet as snt
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision

import pandas as pd
import seaborn as sns

from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import linregress
from scipy import stats
import keras.backend as kb

import scipy.special
import scipy.stats
import scipy.ndimage

import metrics
from scipy.stats import zscore

tf.keras.backend.set_floatx('float32')

def tf_tpu_initialize(tpu_name):
    """Initialize TPU and return global batch size for loss calculation
    Args:
        tpu_name
    Returns:
        distributed strategy
    """
    
    try: 
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    except ValueError: # no TPU found, detect GPUs
        strategy = tf.distribute.get_strategy()

    return strategy


"""
having trouble w/ providing organism/step inputs to train/val steps w/o
triggering retracing/metadata resource exhausted errors, so defining 
them separately for hg, mm 
to do: simplify to two functions w/ organism + mini_batch_step_inputs
consolidate into single simpler function
"""


def return_test_build_functions(model,
                               strategy,
                               metric_dict):
    """Returns distributed train and validation functions for
    a given list of organisms
    Args:
        model: model object
        optimizer: optimizer object
        metric_dict: empty dictionary to populate with organism
                     specific metrics
        num_replicas: # replicas 
        gradient_clip: gradient clip value to be applied in case of adam/adamw optimizer
    Returns:
        distributed train function
        distributed val function
        metric_dict: dict of tr_loss,val_loss, correlation_stats metrics
                     for input organisms
    
    return distributed train and val step functions for given organism
    train_steps is the # steps in a single epoch
    val_steps is the # steps to fully iterate over validation set
    """

    @tf.function(jit_compile=True)
    def test_step(inputs):
        target=tf.cast(inputs['target'],
                       dtype = tf.float32)
        sequence=tf.cast(inputs['sequence'],
                         dtype=tf.float32)
        center_mask=tf.cast(inputs['center_mask'],
                         dtype=tf.float32)
        cell_types=tf.cast(inputs['cell_types'],
                         dtype=tf.float32)
        output = model(sequence, is_training=False)['human']

        pred = tf.reduce_sum(output*center_mask,axis=0)
        true = tf.reduce_sum(target*center_mask,axis=0)

        return pred, true, cell_types

    def build_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            target=tf.cast(inputs['target'],
                           dtype = tf.float32)
            sequence=tf.cast(inputs['sequence'],
                             dtype=tf.float32)
            output = model(sequence, is_training=False)['human']

        for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(val_step, args=(next(iterator),))
        
    return test_step, build_step, metric_dict

def deserialize_val(serialized_example,input_length=196608,out_length=1536,num_targets=50):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string)
    }
    
    shift = 5
    ### rev_comp
    #rev_comp = random.randrange(0,2)

    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (input_length + 10, 4))
    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.slice(sequence, [shift,0],[input_length,-1])
    
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (out_length, num_targets))
    target = tf.cast(target,dtype=tf.float32)
    diff = tf.math.sqrt(tf.nn.relu(target - 65536.0 * tf.ones(target.shape)))
    target = tf.clip_by_value(target, clip_value_min=0.0, clip_value_max=65536.0) + diff
    target = tf.slice(target,
                      [320,0],
                      [896,-1])
    
    top_zeros = tf.zeros((446, 34),dtype=tf.float32)
    center_ones = tf.ones((4, 34),dtype=tf.float32)
    # Concatenate along the 0th dimension
    center_mask = tf.concat([top_zeros, center_ones, top_zeros], axis=0)
    cell_types = tf.range(0,34)
    return {'sequence': tf.ensure_shape(sequence,
                                        [input_length,4]),
            'target': tf.ensure_shape(target,
                                      [896,num_targets]),
            'center_mask': tf.ensure_shape(center_mask,
                                      [896,num_targets]),
            'cell_types': tf.ensure_shape(cell_types,
                                           [num_targets,])}

def return_dataset(gcs_path,
                   batch,
                   input_length,
                   out_length,
                   num_targets,
                   options,
                   num_parallel):

    """
    return a tf dataset object for given gcs path
    """
    wc = "test*.tfr"
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                wc)))
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize_val(record,
                                                        input_length,
                                                        out_length,
                                                        num_targets),
                            deterministic=False,
                            num_parallel_calls=num_parallel)

    return dataset.batch(batch).prefetch(tf.data.AUTOTUNE).repeat(3)

def return_distributed_iterators(gcs_path,
                                 global_batch_size,
                                 input_length,
                                 num_targets,
                                 out_length,
                                 num_parallel_calls,
                                 strategy,
                                 options):
    """ 
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """
    with strategy.scope():
        test_data = return_dataset(gcs_path,
                                 global_batch_size,
                                 input_length,
                                 out_length,
                                 num_targets,
                                 options,
                                 num_parallel_calls)
        
            
        test_dist= strategy.experimental_distribute_dataset(test_data)
        test_data_it = iter(test_dist)

        build_data = return_dataset(gcs_path,
                                 global_batch_size,
                                 input_length,
                                 out_length,
                                 num_targets,
                                 options,
                                 num_parallel_calls)
            
        build_dist= strategy.experimental_distribute_dataset(build_data)
        build_data_it = iter(build_dist)

    return build_data_it,test_data_it


def parse_args(parser):
    """Loads in command line arguments
    """
        
    parser.add_argument('--tpu_name', dest = 'tpu_name',
                        help='tpu_name')
    parser.add_argument('--tpu_zone', dest = 'tpu_zone',
                        help='tpu_zone')
    parser.add_argument('--wandb_project', 
                        dest='wandb_project',
                        help ='wandb_project')
    parser.add_argument('--wandb_user',
                        dest='wandb_user',
                        help ='wandb_user')
    parser.add_argument('--wandb_sweep_name',
                        dest='wandb_sweep_name',
                        help ='wandb_sweep_name')
    parser.add_argument('--gcs_project', dest = 'gcs_project',
                        help='gcs_project')
    parser.add_argument('--gcs_path',
                        dest='gcs_path',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=tf.data.AUTOTUNE,
                        help='thread count for tensorflow record loading')
    parser.add_argument('--batch_size', dest = 'batch_size',
                        type=int, help='batch_size')
    parser.add_argument('--num_epochs', dest = 'num_epochs',
                        type=int, help='num_epochs')
    parser.add_argument('--input_length',
                        dest='input_length',
                        default=196608,
                        type=int,
                        help='input_length')
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        default=1.0e-8,
                        type=float,
                        help= 'epsilon')
    parser.add_argument('--num_targets',
                        dest='num_targets',
                        type=int,
                        default=50,
                        help= 'num_targets')
    parser.add_argument('--test_examples', dest = 'test_examples',
                        type=int, help='train_examples')
    parser.add_argument('--checkpoint_path', dest = 'checkpoint_path',
                        help='checkpoint_path',
                        default=None)
    
    args = parser.parse_args()
    return parser
    
    
    
def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out

def rev_comp_one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))
    input_characters = tf.reverse(input_characters,[0])
    
    vocabulary = tf.constant(['T', 'G', 'C', 'A'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out



def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


    
    
    

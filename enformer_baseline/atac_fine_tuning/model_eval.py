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
import time
import pandas as pd
from datetime import datetime
import random

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

## custom modules
import enformer_vanilla as enformer
import metrics as metrics
import eval_utils as eval_utils

import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr  
from scipy import stats

import eval_utils as eval_utils

import optimizers

def parse_bool_str(input_str):
    if input_str == 'False':
        return False
    else:
        return True

 ## reformat 
# ===========================================================================#

def main():
    # ============== arg parse ==============================================# 
    parser = argparse.ArgumentParser(
        description='process input for genformer training loop')
    parser = eval_utils.parse_args(parser)
    args = parser.parse_args()
    
    #================ init ==================================================# 
    
    ### make sure gcloud auth set to picard-testing-176520
        
    ### make sure TPU started

    # ============== define sweep options ==================== #
    sweep_config = {
            "name" : args.wandb_sweep_name,
            'method': "grid",
            'metric': {
                'name': 'hg_val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'checkpoint_path': {
                    'values':[args.checkpoint_path]
                }
                }

    }

    
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = eval_utils.tf_tpu_initialize(args.tpu_name)
        g = tf.random.Generator.from_seed(datetime.now().timestamp())
        ## rest must be w/in strategy scope
        with strategy.scope():
            config_defaults = {
                "lr_base": 0.01 ### will be overwritten
            }
            
            ### log training parameters
            wandb.init(config=config_defaults, 
                       project= args.wandb_project, 
                       entity=args.wandb_user)
            #wandb.init(mode="disabled")
            wandb.config.tpu=args.tpu_name
            wandb.config.gcs_path=args.gcs_path
            wandb.config.input_length=args.input_length
            
            wandb.config.test_examples=args.test_examples
            
            run_name = 'ENFORMER_test'
            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            
            wandb.run.name = run_name + "_" + date_string

            '''
            TPU init options
            '''
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.DATA
            options.deterministic=False

            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=1
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print(GLOBAL_BATCH_SIZE)
            num_test=wandb.config.test_examples

            wandb.config.update({"test_steps" : num_test // GLOBAL_BATCH_SIZE + 1},
                                allow_val_change=True)

            test_it, build_it =  \
                eval_utils.return_distributed_iterators(args.gcs_path,
                                                            GLOBAL_BATCH_SIZE,
                                                            196608,
                                                            args.num_targets,
                                                            1536,
                                                            args.num_parallel,
                                                            strategy,
                                                            options)
            

            print('made iterators')
            enformer_model = enformer.Enformer(output_heads_dict = {'human': args.num_targets})

            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            
            metric_dict = {}

            test_step, build_step,metric_dict = eval_utils.return_test_build_functions(enformer_model,
                                                                                        strategy,
                                                                                        metric_dict)
            
 
            print('building model...')
            build_step(build_it)
                    
            print('loading weights...')
            options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
            checkpoint = tf.train.Checkpoint(module=enformer_model)
            tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
            latest = tf.train.latest_checkpoint(wandb.config.checkpoint_path)
            checkpoint.restore(latest,options=options).assert_existing_objects_matched()

            print('computing quant metrics')
            pred_list = [] # list to store predictions
            true_list = [] # list to store true values
            gene_list = []
            cell_list = []
            counter = 0 
            for step in range(wandb.config.test_steps):
                pred, true, cell= strategy.run(test_step,args = (next(test_it),))
                for x in strategy.experimental_local_results(true): # flatten the true values
                    true_list.append(tf.reshape(x, [-1]))
                for x in strategy.experimental_local_results(pred): # flatten the pred values
                    pred_list.append(tf.reshape(x, [-1]))
                #for x in strategy.experimental_local_results(gene): # flatten the pred values
                #    gene_list.append(tf.reshape(x, [-1]))
                for x in strategy.experimental_local_results(cell): # flatten the pred values
                    cell_list.append(tf.reshape(x, [-1]))
                    gene_list.append(counter)
                    counter += 1
                if step % 5000 == 0:
                    results_df = pd.DataFrame()
                    results_df['true'] = tf.concat(true_list,0)
                    results_df['pred'] = tf.concat(pred_list,0)
                    results_df['cell_type_encoding'] = tf.concat(cell_list,0)
                    results_df['gene_encoding'] = gene_list

                    results_df.to_csv('test_set_results' + step + '.tsv',sep='\t',header=True,index=False)
                    pred_list = [] # list to store predictions
                    true_list = [] # list to store true values
                    gene_list = []
                    cell_list = []

            results_df = pd.DataFrame()
            results_df['true'] = tf.concat(true_list,0)
            results_df['pred'] = tf.concat(pred_list,0)
            results_df['cell_type_encoding'] = tf.concat(cell_list,0)
            results_df['gene_encoding'] = gene_list
        
            #print('cell_spec_mean_corrs: ' + str(cell_spec_mean_corrs))
            #print('gene_spec_mean_corrs: ' + str(gene_spec_mean_corrs))
                
            results_df.to_csv('test_set_results.tsv',sep='\t',header=True,index=False)

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()


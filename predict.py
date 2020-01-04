import sys
import numpy as np
import argparse
import os
import h5py
import json
import tensorflow as tf
from tensorflow import keras

import mytf.utils as mu
import mytf.validation as mv

# Need this for now..
#tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()

@profile
def bake_options():
    return [
            [['--verbose', '-v'],
                {'action': 'store_true',
                    'help': 'pass to to be verbose with commands'},
                ],
            [['--dry-run', '-d'],
                {'action': 'store_true',
                    'help': 'Dry run. Just print the command.  '},],

            [['--model-loc', '-m'],
                {
                    'help': 'file path of model.  '},],
            [['--test-loc', '-t'],
                {
                    'help': 'file path of test set.  '},],
            [['--batch-size', '-b'],
                {
                    'help': 'batch size. Needed not for training  '
                            'but to predict on data in chunks. '
                            'That way prediction requires less memory.'},],

            [['--work-dir', '-w'],
                {
                    'help': 'Directory to save new artifacts'},],

                #'required': False
                ]

@profile
def do_predict(kwargs):
    modelloc = kwargs['model_loc']
    workdir = kwargs['work_dir']
    test_loc = kwargs['test_loc']
    if kwargs['dry_run']:
        print(kwargs)
        sys.exit()

    steploss = mv.perf_wrapper(modelloc,
            dataloc=test_loc,
            eager=True,
            batch_size=kwargs['batch_size'])

    # Save this ...

    mv.json_save({'steploss': steploss}, 
                  f'{workdir}/{mu.quickts()}.json')
    
    print('Done.')

def do():
    parser = argparse.ArgumentParser()
    [parser.add_argument(*x[0], **x[1])
            for x in bake_options()]

    # Collect args from user.
    kwargs = vars(parser.parse_args())
    print(kwargs)
    do_predict(kwargs)

if __name__ == '__main__':
    do()


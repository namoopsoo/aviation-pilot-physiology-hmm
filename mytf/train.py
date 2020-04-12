import numpy as np
import argparse
import os
import h5py
import json
import tensorflow as tf
from tensorflow import keras

import mytf.utils as mu

# Need this for now..
tf.enable_eager_execution()

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
                {'action': 'store_true',
                    'help': 'file path of model.  '},],
            [['--train-loc', '-t'],
                {'action': 'store_true',
                    'help': 'file path of model.  '},],
            [['--batch-size', '-b'],
                {'action': 'store_true',
                    'help': 'batch size.  '},],
                #'required': False
                ]
    ##
    #             help='',
    #             default='',
    #             required='',
    #             choices='',
    #             action='',
    #             type='',


def do_train():

    # load model, 
    model = mu.load_model(model_loc)

    # load train data, 
    X, Ylabels = mu.read_h5_two(source_location, Xdataset, Ydataset)


    # do a fit.

    class_weights = {0: 1., 1: 1., 2: 1., 3: 1.}
    dataset_batches = mu.build_dataset_weighty_v3(
            {'x_train': X,
             'ylabels_train': Ylabels},
            list(range(size)), 
            class_weights,
            batch_size=BATCH_SIZE)
        
    mu.do_train(
        model,
        dataset_batches,
        k=size,
        saveloc=workdir)





def do():
    parser = argparse.ArgumentParser()

    [parser.add_argument(*x[0], **x[1])
            for x in bake_options()]

    # Collect args from user.
    args = vars(parser.parse_args())

    print(args)




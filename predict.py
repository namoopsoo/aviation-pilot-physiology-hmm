import sys
import numpy as np
import argparse
import os
import json
import tensorflow as tf

import mytf.utils as mu
import mytf.validation as mv

# Need this for now..

def bake_options():
    return [
            [['--verbose', '-v'],
                {'action': 'store_true',
                    'help': 'pass to to be verbose with commands'},
                ],
            [['--dry-run', '-d'],
                {'action': 'store_true',
                    'help': 'Dry run. Just print the command.  '},],
            [['--eager', '-e'],
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

def do_predict(kwargs):
    if kwargs['dry_run']:
        print(kwargs)
        sys.exit()

    if kwargs['eager']:
        tf.compat.v1.enable_eager_execution()
        steplosses = eager_predict(kwargs)
    else:
        steplosses = graph_predict(kwargs)

    # Save this ...
    workdir = kwargs['work_dir']
    mv.json_save({'steploss': steplosses}, 
                  f'{workdir}/{mu.quickts()}.json')
    
    print('Done.')

def graph_predict(kwargs):
    modelloc = kwargs['model_loc']
    test_loc = kwargs['test_loc']

    with tf.compat.v1.Session() as sess:
        steplosses = []
        model = mu.load_model(modelloc)
        for (Xdataset, Ydataset) in [['X_0', 'Ylabels_0'],
                                    ['X_1', 'Ylabels_1'],
                                    ['X_2', 'Ylabels_2'],
                                    ['X_3', 'Ylabels_3']]:
            tensor = mv.get_performance(
                    model, test_loc, Xdataset, Ydataset,
                    eager=False,
                    batch_size=int(kwargs['batch_size']))

            steplosses.append(sess.run(tensor))

    return steplosses


def eager_predict(kwargs):
    modelloc = kwargs['model_loc']
    test_loc = kwargs['test_loc']


    # tensor = foo()
    # Evaluate the tensor `c`.
    steplosses = mv.perf_wrapper(modelloc,
            dataloc=test_loc,
            eager=True,
            batch_size=int(kwargs['batch_size']))

    return steplosses


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


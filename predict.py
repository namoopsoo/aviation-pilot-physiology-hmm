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
            [['--labeled', '-l'],
                {'action': 'store_true',
                    'help': 'specify this dataset includes Y labels '
                            'so compute logloss. Otherwise, will '
                            'prepare the one-hotted predictions.'},],

            [['--parallel', '-p'],
                {'action': 'store_true',
                    'help': 'Parallel.'},],

                #'required': False
                ]

def do_predict(kwargs):
    if kwargs['dry_run']:
        print(kwargs)
        sys.exit()


    if kwargs['eager']:
        tf.compat.v1.enable_eager_execution()
        steplosses = eager_predict(kwargs)  #FIXME
    else:
        steplosses = graph_predict(kwargs)

    # TODO => depending on labeled or not, results are steplosses or preds..

    # Save this ...
    workdir = kwargs['work_dir']
    mu.to_json_local({'steploss': steplosses}, 
                  f'{workdir}/{mu.quickts()}.json')
    
    print('Done.')


def graph_predict(kwargs):
    modelloc = kwargs['model_loc']
    test_loc = kwargs['test_loc']
    batch_size = int(kwargs['batch_size']) or 100

    with tf.compat.v1.Session() as sess:
        steplosses = []
        model = mu.load_model(modelloc)
        for (Xdataset, Ydataset) in [['X_0', 'Ylabels_0'],
                                    ['X_1', 'Ylabels_1'],
                                    ['X_2', 'Ylabels_2'],
                                    ['X_3', 'Ylabels_3']]:
            #
            X, Ylabels = mu.read_h5_two(test_loc, Xdataset, Ydataset) 
            parts = mu.get_partitions(range(X.shape[0]), batch_size, keep_remainder=False)
            batchlosses = []
            for part in parts:
                tensor = mv.get_performance_noteager(
                        model, X=X, Ylabels=Ylabels,
                        part=part)
                batchlosses.append(sess.run(tensor))

            steplosses.append(np.mean(batchlosses))

    return steplosses


def eager_predict(kwargs):
    modelloc = kwargs['model_loc']
    test_loc = kwargs['test_loc']
    labeled = kwargs['labeled']

    # tensor = foo()
    # Evaluate the tensor `c`.
    steplosses = mv.perf_wrapper(modelloc,
            dataloc=test_loc,
            eager=True,
            batch_size=int(kwargs['batch_size']),
            labeled=labeled,
            parallel=kwargs['parallel'],
            workdir=kwargs['work_dir'])

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


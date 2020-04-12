import sys
import numpy as np
import pandas as pd
import argparse
import os
import json
import tensorflow as tf
import joblib

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
            [['--test-loc-h5', '-t'],
                {
                    'help': 'file path of h5 test set.'
                            ' (Already scaled data.)'},],
            [['--raw-test-loc', '-T'],
                {
                    'help': 'file path of raw test csv.'
                            ' (Not scaled data.)'},],
            [['--scalers-loc', '-s'],
                {
                    'help': 'file path of scalers joblib file.'
                            },],
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

            [['--preprocess', '-P'],
                {'action': 'store_true',
                    'help': 'do preprocess.'},],

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

    if kwargs['labeled']:
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
    test_loc = kwargs.get('test_loc_h5')
    raw_test_loc = kwargs.get('raw_test_loc')
    labeled = kwargs['labeled']
    preprocess = kwargs['preprocess']
    scalers_loc = kwargs.get('scalers_loc')
    workdir = kwargs['work_dir']


    if preprocess:
        assert raw_test_loc and not test_loc and scalers_loc
        do_preprocess(raw_test_loc, workdir=workdir,
                                    scalers_loc=scalers_loc)
        # This file is created in the preprocessing
        test_loc = f'{workdir}/finaltest_scaled.h5'
    else:
        assert test_loc and not raw_test_loc and not scalers_loc


    steplosses = mv.perf_wrapper(modelloc,
            dataloc=test_loc,
            eager=True,
            batch_size=int(kwargs['batch_size']),
            labeled=labeled,
            parallel=kwargs['parallel'],
            workdir=workdir)

    combine_preds(workdir,
            suffix=raw_test_loc.split('/')[-1].split('.')[0])
    final_delete(workdir)

    print(f'Finished working on {raw_test_loc}')

    return steplosses


def combine_preds(workdir, suffix=None):
    files = [f'{workdir}/{x}' for x in os.listdir(workdir)
            if x.startswith('preds')] 

    predsdf = pd.concat([pd.read_csv(x, index_col=None)
                        for x in files])
    predsdf.id = predsdf.id.map(lambda x: int(x))
    predsdf.to_csv(f'{workdir}/{mu.quickts()}-crewseat-preds-{suffix}.csv',
                    index=False)

    [os.remove(x) for x in files]


def do_preprocess(raw_test_loc, workdir, scalers_loc):
    df = pd.read_csv(raw_test_loc).sort_values(by='time')

    scalers = joblib.load(scalers_loc)

    featurecols = ['r', 'ecg', 'gsr',
              'eeg_fp1','eeg_f7', 'eeg_f8', 'eeg_t4', 'eeg_t6', ]
    mu.make_test_data(df,
            window_size=64,
            row_batch_size=10000,
            feature_cols=featurecols,
            save_dir=workdir)

    finalloc = f'{workdir}/finaltest.h5'
    finalscaledloc = f'{workdir}/finaltest_scaled.h5'
    mu.apply_scalers(finalloc,
               datasets=[x for x in mu.h5_keys(finalloc)
                           if '_X' in x],
               scaler=scalers,
               outloc=finalscaledloc)

    mu.transfer(source_location=finalloc,
            source_datasets=[x for x in mu.h5_keys(finalloc)
                               if '_IX' in x],
            save_location=finalscaledloc)


def final_delete(workdir):
    # Delete the finaltest.h5 and finaltest_scled.h5 files
    # because they take up alot of space.
    finalloc = f'{workdir}/finaltest.h5'
    finalscaledloc = f'{workdir}/finaltest_scaled.h5'
    os.remove(finalloc)
    os.remove(finalscaledloc)


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


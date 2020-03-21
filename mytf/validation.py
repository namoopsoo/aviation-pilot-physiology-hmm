import json
import traceback
import numpy as np
from multiprocessing import Pool
import tensorflow as tf

from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy

import mytf.utils as mu
import mytf.parallel as mp

def get_performance_noteager(model, X, Ylabels, part):
    # Fork of get_performance_parts, since I realized graph execution
    # needed more frequent evaluation b/c Memory filling up.
    # dataloc contains the test data..
    #
    preds = model(X[part].astype('float32'))
    loss = sparse_softmax_cross_entropy(
            labels=Ylabels[part].astype('int64'),
            logits=preds)
    return loss

def get_performance_parts(model, dataloc, dataset_names, eager, batch_size=None):
    # dataloc contains the test data..
    if batch_size is None:
        batch_size = 100
    lossvec = []
    for Xdataset, Ydataset in dataset_names:

        X, Ylabels = mu.read_h5_two(dataloc, Xdataset, Ydataset) 
        parts = mu.get_partitions(range(X.shape[0]), batch_size, keep_remainder=False)
        batchlosses = []
        for part in parts:
            preds = model(X[part].astype('float32'))
            
            if eager:
                tensor = sparse_softmax_cross_entropy(
                        labels=Ylabels[part].astype('int64'),
                        logits=preds.numpy())
                loss = tensor.numpy()
            else:
                loss = sparse_softmax_cross_entropy(
                        labels=Ylabels[part].astype('int64'),
                        logits=preds)
                #loss = tensor.eval()
            batchlosses.append(loss)
        if eager:
            lossvec.append(np.mean(batchlosses))
        else:
            lossvec.append(tf.math.reduce_mean(batchlosses))
    return lossvec


def unlabeled_prediction(model, dataloc, dataset_names, eager, batch_size=None):
    predsvec = []
    for Xdataset in dataset_names:

        X = mu.read_h5_raw(dataloc, Xdataset) 
        parts = mu.get_partitions(range(X.shape[0]), batch_size, keep_remainder=False)

        for part in parts:
            preds = model(X[part].astype('float32'))
            predsvec.append(preds)


    return np.concatenate(predsvec)


def perf_wrapper(modelloc, dataloc, eager, batch_size=None,
                                           labeled=None):
    # dataloc: h5 location for test data
    if batch_size is None:
        batch_size = 100

    # This was for my balanced and labeled dataset before
    dataset_names = [['X_0', 'Ylabels_0'],
                    ['X_1', 'Ylabels_1'],
                    ['X_2', 'Ylabels_2'],
                    ['X_3', 'Ylabels_3']]
    # FIXME
    # But for now hard coding , for the un-labeled case...
    dataset_names = [['dataset_0_X'],
                    ['dataset_1_X'],
                    ['dataset_2_X'],
                    ['dataset_3_X']]

    payloads = [{
        'modelloc': modelloc,
        'dataloc': dataloc,
        'dataset_names': [x],
        'eager': eager,
        'labeled': labeled,
        'batch_size': batch_size}
        for x in dataset_names]
    lossvec = mp.parallel_async_invoke(payloads, the_job)

    return lossvec


def _job_inner(payload):
    modelloc = payload['modelloc']
    dataloc = payload['dataloc']
    dataset_names = payload['dataset_names']
    eager = payload['eager']
    batch_size = payload['batch_size']
    labeled = payload['labeled']

    model = mu.load_model(modelloc)

    if labeled:
        return get_performance_parts(
                        model=model,
                        dataloc=dataloc,
                        dataset_names=dataset_names,
                        eager=eager,
                        batch_size=batch_size)
    else:
        return unlabeled_prediction(
                        model=model,
                        dataloc=dataloc,
                        dataset_names=dataset_names,
                        eager=eager,
                        batch_size=batch_size)


def the_job(input_payload, conn):
    try:
        out = _job_inner(input_payload)
    except Exception as e:
        out = {'error': 'error',
                'error_detail': repr(e),
                'stack_trace': traceback.format_exc().split('\n'),
                }

    conn.send([out])
    conn.close()


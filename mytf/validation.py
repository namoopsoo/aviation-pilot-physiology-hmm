import json
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy

import mytf.utils as mu



def get_performance_parts(model, dataloc, dataset_names, eager, batch_size=None):
    # 
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
                tensor = sparse_softmax_cross_entropy(
                        labels=Ylabels[part].astype('int64'),
                        logits=preds)
                loss = tensor.eval()
            batchlosses.append(loss)

        lossvec.append(np.mean(batchlosses))
    return lossvec


def perf_wrapper(modelloc, dataloc, eager, batch_size=None):
    # dataloc: h5 location for test data

    if batch_size is None:
        batch_size = 100

    model = mu.load_model(modelloc)

    return get_performance_parts(
                    model=model,
                    dataloc=dataloc,
                    dataset_names=[['X_0', 'Ylabels_0'],
                                  ['X_1', 'Ylabels_1'],
                                  ['X_2', 'Ylabels_2'],
                                  ['X_3', 'Ylabels_3']],
                    eager=eager,
                    batch_size=batch_size)

def json_save(x, loc):
    with open(loc, 'w') as fd:
        json.dump(x, fd, cls=JSONCustomEncoder)

class JSONCustomEncoder(json.JSONEncoder):

    def default(self, object):

        if isinstance(object, np.float32):

            return float(object)

        else:

            # call base class implementation which takes care of

            # raising exceptions for unsupported types

            return json.JSONEncoder.default(self, object)

 



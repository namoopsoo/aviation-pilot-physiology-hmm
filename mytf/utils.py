import tensorflow as tf
import datetime
import itertools
import math
import pytz
import h5py
from copy import deepcopy
import numpy as np
from functools import reduce
from tensorflow import keras


from collections import Counter

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

ALL_FEATURE_COLS = ['eeg_fp1', 'eeg_f7', 'eeg_f8', 'eeg_t4', 'eeg_t6', 'eeg_t5', 'eeg_t3', 'eeg_fp2', 'eeg_o1', 'eeg_p3', 'eeg_pz', 'eeg_f3', 'eeg_fz', 'eeg_f4', 'eeg_c4', 'eeg_p4', 'eeg_poz', 'eeg_c3', 'eeg_cz', 'eeg_o2', 'ecg', 'r', 'gsr',]


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')

def quickts():
    return datetime.datetime.utcnow().replace(tzinfo=pytz.UTC).strftime('%Y-%m-%dT%H%M%SZ')


def convert_nans(y):
    return np.vectorize(lambda x: 0 if np.isnan(x) else x)(y)

def tf_f1_score(y_true, y_pred, method=None):
    """Computes 3 different f1 scores, micro macro
    weighted.
    micro: f1 score accross the classes, as 1
    macro: mean of f1 scores per class
    weighted: weighted average of f1 scores per class,
            weighted from the support of each class


    Args:
        y_true (Tensor): labels, with shape (batch, num_classes)
        y_pred (Tensor): model's predictions, same shape as y_true

    Returns:
        tuple(Tensor): (micro, macro, weighted)
                    tuple of the computed f1 scores

    This code is from :
        https://stackoverflow.com/a/50251763/472876
        https://stackoverflow.com/users/3867406/ted
    """
    assert y_true.shape[0] > 0
    assert y_pred.shape[0] > 0

    assert set(y_true[0]) & set(y_pred[0]) != set()

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        precision = convert_nans(precision)
        recall = TP / (TP + FN)
        recall = convert_nans(recall)
        f1 = 2 * precision * recall / (precision + recall)
        f1 = convert_nans(f1)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted, f1


def fetch_some_examples(arrays, which_col, dict_key, n=10):
    '''
    Lazily find the row indices of the training data,
    for the given class (which_col).

    n: how many indices to fetch
    '''
    rows = arrays[dict_key].shape[0]
    gg = (i for i in np.arange(1, rows, 1)
            if arrays[dict_key][i][which_col] == 1)

    return [gg.__next__() for i in range(n)]


def choose_training_indices(arrays, counts, dict_key):
    return {
        i: fetch_some_examples(arrays, i, dict_key=dict_key, n=n)
        for (i, n) in enumerate(counts)}


def build_dataset(arrays, target_indices, batch_size):
    traindata = tf.convert_to_tensor(
            arrays['x_train'][target_indices, :, :], dtype=tf.float32)

    labeldata = tf.convert_to_tensor(
        np.argmax(arrays['y_train'][target_indices, :], axis=1))


    dataset = tf.data.Dataset.from_tensor_slices(
        (traindata, labeldata))

    dataset_batches = dataset.batch(batch_size)
    return dataset_batches


def build_dataset_weighty(arrays, target_indices, class_weights,
        batch_size):
    train_tensor = tf.convert_to_tensor(
            arrays['x_train'][target_indices, :, :], dtype=tf.float32)

    y_train = arrays['y_train'][target_indices, :]
    class_counts = tf.reduce_sum(y_train, axis=0)

    labels = np.argmax(y_train, axis=1)
    print(Counter(labels))
    label_tensor = tf.convert_to_tensor(
        labels)

    weights_per_class = np.array([class_weights[x] for x in range(4)]
            )/class_counts
    assert(abs(1.0 - tf.reduce_sum(class_counts*weights_per_class))
            < 0.0001)

    print('weights_per_class, ', weights_per_class)
    weights = [weights_per_class[x] for x in labels]
    print(sum(weights))
    assert(1.0 - sum(weights) < 0.001)

    weights_tensor = tf.convert_to_tensor(np.array(weights))

    dataset = tf.data.Dataset.from_tensor_slices(
        (train_tensor, label_tensor, weights_tensor))

    dataset_batches = dataset.batch(batch_size)
    return dataset_batches


def helper_build_dataset_weighty_v3(arrays, target_indices, class_weights,
        batch_size):
    # Fork of build_dataset_weighty , weights should add up to 1.0 per batch i think.
    #print('Start build v3: .. doesnt add up to 1.0')
    indices = deepcopy(target_indices)
    np.random.shuffle(indices)

    partitions = get_partitions(indices, batch_size)

    # for each  batch...
    weights_vec = []
    train_vec = []
    label_vec = []

    for part in partitions:
        train_vec.append(
                arrays['x_train'][part, :, :])

        if 'y_train' in arrays:
            y_train = arrays['y_train'][part, :]
            class_counts = tf.reduce_sum(y_train, axis=0)
            labels = np.argmax(y_train, axis=1)
        elif 'ylabels_train' in arrays:
            labels = arrays['ylabels_train'][part]
            adict = dict(Counter(labels))
            class_counts = [adict.get(i, 0) for i in [0, 1, 2, 3]]
            
        #print(Counter(labels))
        label_vec.append(labels)

        weights_per_class = np.array([class_weights[x] for x in range(4)]
                )/class_counts
        #assert(abs(1.0 - tf.reduce_sum(class_counts*weights_per_class))
        #        < 0.0001)

        #print('weights_per_class, ', weights_per_class)
        weights = [class_weights[x] for x in labels]   #
        #print(sum(weights))
        # assert(1.0 - sum(weights) < 0.001)

        weights_vec.append(weights)

    return train_vec, label_vec, weights_vec

def build_dataset_weighty_v3(arrays, target_indices, class_weights,
        batch_size):

    train_vec, label_vec, weights_vec = \
            helper_build_dataset_weighty_v3(arrays, target_indices, class_weights,
                    batch_size)

    weights_tensor = tf.convert_to_tensor(
            np.concatenate(weights_vec))

    train_tensor = tf.convert_to_tensor(
            np.concatenate(train_vec), dtype=tf.float32)
    #
    label_tensor = tf.convert_to_tensor(
        np.concatenate(label_vec))

    dataset = tf.data.Dataset.from_tensor_slices(
        (train_tensor, label_tensor, weights_tensor))

    dataset_batches = dataset.batch(batch_size)
    return dataset_batches



def shrink_dataset_subset(arrays, train_target_indices,
        test_target_indices):
    # Dataset is tremendous so lets use memory just for what i'm using...
    return {
            'x_train': arrays['x_train'][train_target_indices, :, :],
            'x_test': arrays['x_test'][test_target_indices, :, :],

            'y_train': arrays['y_train'][train_target_indices, :],
            'y_test': arrays['y_test'][test_target_indices, :],

            #'y_train_original': arrays['y_train_original'][train_target_indices],
            #'y_test_original': arrays['y_test_original'][test_target_indices],
            }


def do_train(model, dataset_batches, k, saveloc):
    optimizer = tf.train.AdamOptimizer()

    loss_history = []

    for (batch, (invec, labels, weights)) in enumerate(dataset_batches.take(k)):

        with tf.GradientTape() as tape:
            logits = model(invec, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights)

        loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                global_step=tf.compat.v1.train.get_or_create_global_step())

        save_model(model, f'{saveloc}/{str(batch).zfill(5)}_model.h5')
        to_json_local(loss_history, f'{saveloc}/{str(batch).zfill(5)}_train_loss_history.json')

    return loss_history

def to_json_local(data, loc):
    with open(loc, 'w') as fd:
        json.dump(data, fd)

def do_train_noweights(model, dataset_batches):
    optimizer = tf.train.AdamOptimizer()

    loss_history = []

    for (batch, (invec, labels)) in enumerate(dataset_batches.take(1000)):

        with tf.GradientTape() as tape:
            logits = model(invec, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                global_step=tf.compat.v1.train.get_or_create_global_step())

    return loss_history




def do_train_f1_loss(model, dataset_batches):
    optimizer = tf.train.AdamOptimizer()

    loss_history = []

    for (batch, (invec, labels, weights)) in enumerate(dataset_batches.take(1000)):

        with tf.GradientTape() as tape:
            logits = model(invec, training=True)

            original_loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights)

            micro, macro, weighted, f1 = tf_f1_score(
                    one_hot(labels, convert=True),
                    one_hot(np.argmax(logits, axis=1), convert=False),

                    )
            loss_value = macro


        loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                global_step=tf.compat.v1.train.get_or_create_global_step())

    return loss_history

def one_hot(vec, convert=False):
    foo = {0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
            3: [0, 0, 0, 1],
            }
    if convert:
        return np.array([
                deepcopy(foo[x.numpy()])
                for x in vec])
    else:
        return np.array([
                deepcopy(foo[x])
                for x in vec])

encode_class = np.vectorize(lambda x: {'A': 0,
                                      'B': 1,
                                      'C': 2,
                                      'D': 3}.get(x))

decode_class = np.vectorize(lambda x: {0: 'A',
                                      1: 'B',
                                      2: 'C',
                                      3: 'D'}.get(x))

def do_scaling(arrays):
    '''
    Expecting typical data I've had. apply minmax scaling.
    '''
    pass

    scaler = MinMaxScaler(feature_range=(0, 1))


def scale_this_thing(x, scaler):
    length = reduce(lambda y, z: y*z, x.shape)
    llist = np.reshape(x, newshape=(length, 1))

    return np.reshape(scaler.transform(llist),
            newshape=x.shape)



def get_partitions(vec, slice_size):
    assert slice_size > 0
    #assert isinstance(vec, list)
    num_slices = int(math.floor(len(vec)/slice_size))
    print('num slices', num_slices)
    size_remainder = len(vec) - num_slices*slice_size
    assert size_remainder >= 0
    print('size_remainder, ', size_remainder)
    slices = [vec[k*slice_size:k*slice_size+slice_size] for k in range(num_slices)]
    if size_remainder:
        slices.append(vec[-(size_remainder):])

    return slices


# Earlier data utils...

encode_class = np.vectorize(lambda x: {'A': 0,
                                      'B': 1,
                                      'C': 2,
                                      'D': 3}.get(x))

decode_class = np.vectorize(lambda x: {0: 'A',
                                      1: 'B',
                                      2: 'C',
                                      3: 'D'}.get(x))

simple_scaler = lambda x, a: x*a


def do_standard_scaling(df, cols, scalar_dict=None):
    if scalar_dict is None:
        scalar_dict = {col: StandardScaler().fit(df[[col]]) for col in cols}

    for col in cols:
        df[col + '_scaled'] = np.resize(
            scalar_dict[col].transform(df[[col]]),
            (df.shape[0],))

    return scalar_dict, df


def chomp_crews(df, crews, feature_cols):
    # Sort it and just bite off selected crews
    sort_cols = ['crew', 'seat', 'experiment', 'time']
    target_col = 'event'
    what_cols = sort_cols + list(feature_cols) + [target_col]

    return df[df.crew.isin(crews)][what_cols].sort_values(
        by=sort_cols).copy()


def make_data(df, crews={'training': [1],
                        'test': [2]},
                        window_size=256,
                        row_batch_size=None,
                        feature_cols=['r'],
                        save_dir=None):

    # current sorted as ['crew', 'seat', 'experiment', 'time']
    # [0, 1] # each seat
    # ['CA', 'DA', 'SS'] # experiment
    sort_cols = ['crew', 'seat', 'experiment', 'time']
    target_col = 'event'

    feat_cols_scaled = [
            x + '_scaled' for x in feature_cols]

    what_cols = sort_cols + feature_cols + [target_col]

    # Training
    print('Start building training set', quickts())
    traindf = df[df.crew.isin(crews['training'])][what_cols].copy()
    scalar_dict, _ = do_standard_scaling(traindf, feature_cols)
    train_datasets = get_windows_h5(traindf,
                                    cols=feat_cols_scaled + ['event'],
                                    window_size=window_size,
                                    row_batch_size=row_batch_size,
                                    save_location=f'{save_dir}/train.h5')

    # Testing
    print('Start building testing set', quickts())
    testdf = df[df.crew.isin(crews['test'])][what_cols].copy()
    _, _ = do_standard_scaling(testdf, feature_cols, scalar_dict)
    test_datasets = get_windows_h5(testdf,
                                    cols=feat_cols_scaled + ['event'],
                                    window_size=window_size,
                                    row_batch_size=row_batch_size,
                                    save_location=f'{save_dir}/test.h5')
    return

'''
    outdata = {
        "x_train": x_train,
        "y_train": reshape_y(encode_class(y_train), 4), # y_train,
        "x_test": x_test,
        "y_test": reshape_y(encode_class(y_test), 4), # y_test
        "y_train_original": y_train,
        "y_test_original": y_test,
        "traindf": traindf,
        "testdf": testdf,}
    metadata = {
        "metadata": {
            "output": {
                "shapes": {k: outdata[k].shape for k in list(outdata)},
                "Counter(outdata['y_train_original'])":
                dict(Counter(y_train)),
                "Counter(outdata['y_test_original'])":
                dict(Counter(y_test)),},
            "input": {"kwargs": {
                "crews": crews,
                #"percent_of_data": percent_of_data,
                "window_size": window_size,
                "feature_cols": list(feature_cols)}},
            "data_ts": quickts()
        }}

    return {**outdata, **metadata}
'''


#def runner():
#    print('Start make_data', timestamp())
#    outdata = make_data(df, crews={'training': [1],
#                        'test': [2]},
#              sequence_window=256, percent_of_data=1,
#             feature_cols={'r': simple_scaler})
#
#    validate_data(outdata)
#
#    print('Start bake_model', timestamp())
#    model = bake_model(**outdata, epochs=2)
#    return outdata, model

def validate_data(data):
    assert len(Counter(data['y_train_original'])) > 1
    assert len(Counter(data['y_test_original'])) > 1



def get_windows(df, cols, window_size, percent_of_data=100):
                                        # FIXME ^^ percent_of_data, perhaps too limiting.
    # Assumes last col is one and only label col
    whats_proportion_index = lambda x, y: round(x*y)

    X = []
    Y = []
    choices = (df.crew.unique().tolist(), [0, 1], ['CA', 'DA', 'SS'])
    for crew, seat, experiment in itertools.product(*choices):
        query = (df.crew == crew)&(df.seat == seat)&(df.experiment == experiment)
        thisdf = df[query][cols]
        X_i, Y_i = to_sequences(thisdf.values, window_size,
                                incols=range(len(cols) - 1),
                                outcol=-1)
        X.append(X_i[:
                     whats_proportion_index(
                         X_i.shape[0],
                         percent_of_data)])
        Y.append(Y_i[:
                     whats_proportion_index(
                        Y_i.shape[0],
                        percent_of_data)])

    return np.concatenate(X), np.concatenate(Y)


def get_windows_h5(df, cols, window_size, row_batch_size, save_location):
    # for every <row_batch_size> rows, save to disk, to <save_location>.
    parts = get_partitions(range(df.shape[0]), row_batch_size)
    datasets = []
    for i, part in enumerate(parts):
        X, Y = _inner_get_windows(df.iloc[part], cols, window_size)
        # Save to disk...
        with h5py.File(save_location, "a") as f:
            X_name, Y_name = f'dataset_{i}_X', f'dataset_{i}_Y'
            datasets.append({'X_name': X_name, 'Y_name': Y_name})
            f.create_dataset(X_name, data=np.array(X, dtype=float))
            f.create_dataset(Y_name,
                            data=np.array(
                                reshape_y(encode_class(Y), 4),
                                dtype=float))
    return datasets


def _inner_get_windows(df, cols, window_size):
    X = []
    Y = []
    choices = (df.crew.unique().tolist(), [0, 1], ['CA', 'DA', 'SS'])
    for crew, seat, experiment in itertools.product(*choices):
        query = (df.crew == crew)&(df.seat == seat)&(df.experiment == experiment)
        thisdf = df[query][cols]
        if thisdf.empty:
            continue

        X_i, Y_i = to_sequences(thisdf.values, window_size,
                                incols=range(len(cols) - 1),
                                outcol=-1)
        X.append(X_i)
        Y.append(Y_i)

    return np.concatenate(X), np.concatenate(Y)


# Borrowing parts of this func from
# https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class10_lstm.ipynb
def to_sequences(obs, seq_size, incols, outcol):
    x = []
    y = []

    for i in range(len(obs)-seq_size):
        window = obs[i:(i+seq_size)][:, incols]
        after_window = obs[i+seq_size - 1, outcol]

        x.append(window)
        y.append(after_window)

    xarr = np.array(x)
    yarr = np.array(y)
    return (xarr,
            yarr)



def reshape_y(y, num_cols):

    # y = np.array([1,2,3,2,3,1],dtype=np.int32)

    # Convert y2 to dummy variables
    y2 = np.zeros((y.shape[0], num_cols), dtype=np.float32)
    y2[np.arange(y.shape[0]), y] = 1.0
    return y2


def is_it_sorted_by_time(df):
    choices = (df.crew.unique(),
               df.seat.unique(),
               df.experiment.unique())
    meta = {}
    for crew, seat, experiment in itertools.product(*choices):
        query = ((df.crew == crew)
                & (df.seat == seat)
                & (df.experiment == experiment))
        times = df[query].time.tolist()
        meta[(crew, seat, experiment)] = times == sorted(list(set(times)))
    return meta


def split_data_by_crew(df, outdir):
    meta = {}
    feature_cols = ALL_FEATURE_COLS

    for crew in df.crew.unique():
        filename = f'{outdir}/crew_{crew}-train.pkl'
        # write ...
        outdf = chomp_crews(df, [crew], feature_cols)
        outdf.to_pickle(filename)
        meta[crew] = filename
    return meta




def read_h5(source_location, Xdataset, Ydataset):
    with h5py.File(source_location, 'r+') as fd:
        X = fd[Xdataset].__array__()
        Y = fd[Ydataset].__array__()
        Ylabels = np.argmax(Y, axis=1)
    return X, Ylabels

def read_h5_two(source_location, Xdataset, Ydataset):
    with h5py.File(source_location, 'r+') as fd:
        X = fd[Xdataset].__array__()
        Y = fd[Ydataset].__array__()
        #Ylabels = np.argmax(Y, axis=1)
        #counters_index[i] = dict(Counter(labels))
    return X, Y
        
    
def transfer_data(source_location,
                  source_datasets,
                 save_location,
                 label,
                 howmany):
    Xvec = []
    Ylabelvec = []
    # look for a certain amount of examples and transfer them to thew new location.
    sofar = 0
    for Xdataset, Ydataset in source_datasets:

        X, Ylabels = read_h5(source_location, Xdataset, Ydataset)
        indices = [i for i in range(Ylabels.shape[0])
                  if Ylabels[i] == label]
        if indices:
            X_a = X[indices, :]
            Ylabels_a = Ylabels[indices]
            Xvec.append(X_a)
            Ylabelvec.append(Ylabels_a)
            sofar += len(indices)
        
        if sofar >= howmany:
            print('ok breaking')
            break
    Xfull = np.concatenate(Xvec)
    Yfull = np.concatenate(Ylabelvec)
    save_that(save_location,
             f'X_{label}',
             Xfull)
    
    save_that(save_location,
             f'Ylabels_{label}',
             Yfull)
    print('Saved with ', Counter(Yfull))


def save_that(save_location, name, X):
    with h5py.File(save_location, "a") as f:
        f.create_dataset(name, data=np.array(X, dtype=float))


def get_performance(model, dataloc, dataset_names):
    # dataloc contains the test data..
    lossvec = []
    for Xdataset, Ydataset in dataset_names:

        X, Ylabels = read_h5_two(dataloc, Xdataset, Ydataset) 
    
        preds = model(X.astype('float32'))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=Ylabels.astype('int64'),
                                               logits=preds.numpy()).numpy()

        lossvec.append(loss)
    return lossvec



def save_model(model, loc):
    model.save(loc)
    #with open('2019-05-17T1914UTC-model-3.h5', 'rb') as fd: dumpedmodel = fd.read()

def load_model(loc):
    return keras.models.load_model(loc)


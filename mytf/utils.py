import tensorflow as tf
from copy import deepcopy
import numpy as np
from collections import Counter

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


def fetch_some_examples(arrays, which_col, n=10):
    '''
    Lazily find the row indices of the training data, 
    for the given class (which_col).

    n: how many indices to fetch
    '''
    gg = (i for i in np.arange(1, 446110, 1) 
            if arrays['y_train'][i][which_col] == 1)
    
    return [gg.__next__() for i in range(n)]


def choose_training_indices(arrays, counts):
    return {
        i: fetch_some_examples(arrays, i, n=n) for (i, n) in enumerate(counts)
    }


def build_dataset(arrays, target_indices):
    traindata = tf.convert_to_tensor(
            arrays['x_train'][target_indices, :, :],  dtype=tf.float32)

    
    labeldata = tf.convert_to_tensor(
        np.argmax(arrays['y_train'][target_indices, :], axis=1))
    
    # Just try equal weights for now
    equal_weight = 1/(len(target_indices))
    weights = tf.convert_to_tensor(np.array(
        [equal_weight for i in range(len(target_indices))]
    
    ))
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (traindata, labeldata, weights))

    dataset_batches = dataset.batch(100)
    return dataset_batches

def build_dataset_weighty(arrays, target_indices, class_weights,
        batch_size):
    train_tensor = tf.convert_to_tensor(
            arrays['x_train'][target_indices, :, :],  dtype=tf.float32)

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


def shrink_dataset_subset(arrays, train_target_indices,
        test_target_indices):
    # Dataset is tremendous so lets use memory just for what i'm using...
    return {
            'x_train': arrays['x_train'][train_target_indices, :, :],
            'x_test': arrays['x_test'][test_target_indices, :, :],

            'y_train': arrays['y_train'][train_target_indices, :],
            'y_test': arrays['y_test'][test_target_indices, :],

            'y_train_original': arrays['y_train_original'][train_target_indices],
            'y_test_original': arrays['y_test_original'][test_target_indices],
            }


def do_train(model, dataset_batches):
    optimizer = tf.train.AdamOptimizer()

    loss_history = []

    for (batch, (invec, labels, weights)) in enumerate(dataset_batches.take(1000)):

        with tf.GradientTape() as tape:
            logits = model(invec, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights)

        loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                global_step=tf.train.get_or_create_global_step())

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
                                global_step=tf.train.get_or_create_global_step())

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

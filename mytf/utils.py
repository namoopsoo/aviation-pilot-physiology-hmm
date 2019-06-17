import tensorflow as tf
import numpy as np
from collections import Counter


def tf_f1_score(y_true, y_pred):
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

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

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

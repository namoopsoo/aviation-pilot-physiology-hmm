

#### Okay next I tried stratifying a bit
* Took `1000` from each class, to get uniform representation, 
* Using equal weight in the `weights` vector passed to the loss function.
```python
def fetch_some_examples(arrays, which_col, n=10):
    gg = (i for i in np.arange(1, 446110, 1) if arrays['y_train'][i][which_col] == 1)
    
    return [gg.__next__() for i in range(n)]

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
```
```python
%time examples = {j: fetch_some_examples(outdata, j, n=1000) for j in range(4)}

target_indices = examples[0] + examples[1] + examples[2] + examples[3]
print(len(target_indices))

%time dataset_batches = build_dataset(outdata, target_indices)

print(dataset_batches)

```
* =>
```python
4000
CPU times: user 1.39 s, sys: 7.09 ms, total: 1.4 s
Wall time: 1.4 s
CPU times: user 30.9 ms, sys: 8.18 ms, total: 39.1 ms
Wall time: 53 ms
<DatasetV1Adapter shapes: ((?, 256, 1), (?,), (?,)), types: (tf.float32, tf.int64, tf.float64)>
```
```python
def do_train(model, dataset_batches):
    optimizer = tf.train.AdamOptimizer()

    loss_history = []

    for (batch, (invec, labels, weights)) in enumerate(dataset_batches.take(100)):

        with tf.GradientTape() as tape:
            logits = model(invec, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights)

        loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                global_step=tf.train.get_or_create_global_step())

    return loss_history

```
* Same model 
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64,  dropout=0.2, recurrent_dropout=0.2,
                input_shape=(None, 1)
              ),
    # 4 because 'A', 'B', 'C', 'D'.
    tf.keras.layers.Dense(4)
])
```
```python
%time loss_history = do_train(model, dataset_batches)

# =>
CPU times: user 1min 51s, sys: 8.48 s, total: 1min 59s
Wall time: 2min 2s
```
```python
plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
```

<img src="https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/assets/Screen%20Shot%202019-06-17%20at%209.16.47%20AM.png" width="648" height="408">

```
# Very interesting. So the sawtooth first of all is probably because there are 4 different labels
# back to back.
# Going to run a confusion matrix, just out of curiosity... on the test data...
```
* Testing... with `1000` elements randomly selected... 
```python
test_subset = np.random.choice(np.arange(0, outdata['x_test'].shape[0], 1), 1000, replace=False)
expected_onehot = tf.convert_to_tensor(outdata['y_test'][test_subset])
preds = model(tf.convert_to_tensor(outdata['x_test'][test_subset], dtype=tf.float32))
expected = tf.convert_to_tensor(np.argmax(outdata['y_test'][test_subset], axis=1))
```
```python
np.count_nonzero (expected_onehot, axis=0)
# =>
array([586,  27, 343,  44])

```
```python
tf.confusion_matrix( 
    expected,# labels
    np.argmax(preds, axis=1), # predictions
    num_classes=4
)

# =>
<tf.Tensor: id=11998076, shape=(4, 4), dtype=int32, numpy=
array([[ 82, 213, 291,   0],
       [  8,   7,  12,   0],
       [ 90,  75, 178,   0],
       [  0,  23,  21,   0]], dtype=int32)>

# Hmm okay this is very interesting. I had never gotten predictions that looked this balanced
# before. Even though many mistakes heh. An interesting result nonetheless.
```

#### Ok, actually use non equal weights vector this time...
```python
# 2019-06-16 00:58UTC 
# 
# try higher weights on classes now ...  
# .. Counter({0: 234352, 1: 7689, 2: 180851, 3: 23218, })


def build_dataset_weighty(arrays, target_indices, class_weights):
    train_tensor = tf.convert_to_tensor(
            arrays['x_train'][target_indices, :, :],  dtype=tf.float32)

    labels = np.argmax(outdata['y_train'][target_indices, :], axis=1)

    label_tensor = tf.convert_to_tensor(
        labels)
    weights = [class_weights[x]/1000 for x in labels]
    print(sum(weights))
    assert(1.0 - sum(weights) < 0.001)
    
    # Just try equal weights for now
    equal_weight = 1/(len(target_indices))
    weights_tensor = tf.convert_to_tensor(np.array(weights))
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_tensor, label_tensor, weights_tensor))

    dataset_batches = dataset.batch(100)
    return dataset_batches
```
```python
class_weights = {0: 0.1, 1: 0.5, 2: 0.1, 3: 0.3}
%time dataset_batches = build_dataset_weighty(outdata, target_indices, class_weights)
# =>
0.999999999999947
CPU times: user 33 ms, sys: 27.7 ms, total: 60.7 ms
Wall time: 150 ms
```
* Same model..
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64,  dropout=0.2, recurrent_dropout=0.2,
                input_shape=(None, 1)
              ),
    # 4 because 'A', 'B', 'C', 'D'.
    tf.keras.layers.Dense(4)
])
```
```python
%time loss_history = do_train(model, dataset_batches)
# 
CPU times: user 1min 44s, sys: 7.29 s, total: 1min 52s
Wall time: 1min 53s

```
```python
plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
```

<img src="https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/assets/Screen%20Shot%202019-06-17%20at%209.27.30%20AM.png" width="616" height="392">

```python
# oh my looks quite different indeed now. Following the unevenness of the weights i think
# Going to stick to the same random sampled test subset ...
%time preds = model(tf.convert_to_tensor(outdata['x_test'][test_subset], dtype=tf.float32))

print(Counter(np.argmax(preds, axis=1)))

# =>
CPU times: user 2.08 s, sys: 577 ms, total: 2.66 s
Wall time: 1.84 s
Counter({3: 502, 1: 498})
```
```python
tf.confusion_matrix( 
    expected,# labels
    np.argmax(preds, axis=1), # predictions
    num_classes=4
)

# =>
<tf.Tensor: id=15384435, shape=(4, 4), dtype=int32, numpy=
array([[  0, 295,   0, 291],
       [  0,  15,   0,  12],
       [  0, 165,   0, 178],
       [  0,  23,   0,  21]], dtype=int32)>
```
```python
import mytf.utils as mu
micro, macro, weighted, f1 = mu.tf_f1_score(expected_onehot, preds)
# from importlib import reload
# reload(mu)

micro, macro, weighted, f1, tf.reduce_mean(f1)
# =>

(<tf.Tensor: id=15384348, shape=(), dtype=float64, numpy=0.3333333333333333>,
 <tf.Tensor: id=15384385, shape=(), dtype=float64, numpy=0.2696476956965507>,
 <tf.Tensor: id=15384393, shape=(), dtype=float64, numpy=0.4607046086068987>,
 <tf.Tensor: id=15384383, shape=(4,), dtype=float64, numpy=array([0.53959484, 0.0512334 , 0.40688019, 0.08088235])>,
 <tf.Tensor: id=15384399, shape=(), dtype=float64, numpy=0.2696476956965507>)
```
```python
# Oh wow. that took things in the complete opposite direction! 
# I have a feeling that since i both 
# (a) stratified the input data 
# (b) and applied weights, i basically overcompensated  and now only the minority classes
#have been predicted. !!
# 
# But also even though high favoritism was given to the minority classes, there were still
# an overwhelming amount of mistakes as seen by the f1 scores for them, 
# , 0.0512334 and 0.08088235 
# So either there is not enough information in the r predictor or also perhaps there is not 
# enough data.
```
```python
Counter(np.argmax(outdata['y_train'], axis=1))
# 
Counter({0: 234352, 2: 180851, 3: 23218, 1: 7689})

```

#### Next one , 
* _Sticking to a new notebook this time..._ [here](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-06-16--today.md.ipynb)
```python
# 2019-06-16 21:33 UTC 
# try again now but 
# - use more than just 4000 training rows
# - Don't stratify the data, just apply a weights vector when training.
# - proportionally larger batches.

# There are Counter({0: 234352, 1: 7689, 2: 180851, 3: 23218, }) class counts
#    in my train data... use more of that ...
```

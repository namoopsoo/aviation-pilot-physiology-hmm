
#### Oh man I killed a few hours trying to figure this out..
* I was trying to adapt my code thus far using the [eager guide](https://www.tensorflow.org/guide/eager) , 
```python

import tensorflow as tf

tf.enable_eager_execution()
import numpy as np
# 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


model = Sequential([
    LSTM(64, # dropout=0.2, recurrent_dropout=0.2,
                input_shape=(256, 1)
              ),
    # 4 because 'A', 'B', 'C', 'D'.
    Dense(4)
])

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-175-468c1edf9459> in <module>
     11               ),
     12     # 4 because 'A', 'B', 'C', 'D'.
---> 13     Dense(4)
     14 ])
     15 
...
...
RuntimeError: tf.placeholder() is not compatible with eager execution.
```
* Then I changed took out the `input_shape` part, thinking this was related, 
```python
model = Sequential([
    LSTM(64,  ),
    # 4 because 'A', 'B', 'C', 'D'.
    Dense(4)
])
```
* But then I was getting the `RuntimeError: tf.placeholder()` error when trying to run , 
```python
# outdata['x_train'] is a numpy array of shape: (446110, 256, 1)  
trainsmall = tf.convert_to_tensor(outdata['x_train'][:100, :, :],  dtype=tf.float32)
model(trainsmall[:1])
...
RuntimeError: tf.placeholder()
```
* I was basically going crazy for a few hours trying to find why this was not matching anything in the [guide](https://www.tensorflow.org/guide/eager) , until I did this...
```python
import keras.models 
print(keras.models.__file__)
print(tf.keras.models.__file__)

/usr/local/miniconda3/envs/pandars3/lib/python3.7/site-packages/keras/models.py
/usr/local/miniconda3/envs/pandars3/lib/python3.7/site-packages/tensorflow/_api/v1/keras/models/__init__.py
```
* I don't even know how I ended up in this situation... 
* Anyway basically this started working now:
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64,  dropout=0.2, recurrent_dropout=0.2,
                input_shape=(256, 1)
              ),
    # 4 because 'A', 'B', 'C', 'D'.
    tf.keras.layers.Dense(4)
])
```

#### Okay, next I tried to run the eager style manual train loop,...
* Using the `model` above ^^ ... 
* Make some data using the tensor flow variables. Actually I didn't do this before. I did not write it down,
but when I was trying this [explanation of a manual train loop](https://www.tensorflow.org/guide/eager#train_a_model) , I was getting errors that using vanilla numpy data was throwing an error.
```python
outdata['x_train'].shape
# ==> (446110, 256, 1)


trainsmall = tf.convert_to_tensor(outdata['x_train'][:100, :, :],  dtype=tf.float32)
labelsmall = tf.convert_to_tensor(outdata['y_train'][:100, :])
dataset = tf.data.Dataset.from_tensor_slices(
(trainsmall, labelsmall))

dataset
# =>
<DatasetV1Adapter shapes: ((256, 1), (4,)), types: (tf.float32, tf.float32)>

dataset_batches = dataset.batch(2)

```
```python
for images,labels in dataset.take(1):
    print("Logits: ", model(trainsmall[:1]).numpy())
    
# ==>
Logits:  [[-0.0121733  -0.0003184   0.13217376 -0.1131559 ]]
```
```python
optimizer = tf.train.AdamOptimizer()

loss_history = []

for (batch, (invec, labels)) in enumerate(dataset_batches.take(100)):

    with tf.GradientTape() as tape:
        logits = model(invec, training=True)
        loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())


# ==>
InvalidArgumentError                      Traceback (most recent call last)
InvalidArgumentError: Can not squeeze dim[1], expected a dimension of 1, got 4 [Op:Squeeze]

```
* Okay darn, so based on the `InvalidArgumentError` , looks to me like this shape difference is causing me a problem.
I thought it was okay to use `tf.keras.layers.Dense(4)` as the second layer to observe a _per-class_ output. 
* I'm not sure why ` tf.losses.sparse_softmax_cross_entropy` wants this shape.

### Side notes.. 

#### try to make the output data pure numpy array not an array of arrays!
```python


def make_data(df, crews={'training': [1],
                        'test': [2]},
              sequence_window=256, percent_of_data=100,
             feature_cols={'r': 'standard_scaler'}):

    # current sorted as ['crew', 'experiment', 'time']
    [0, 1] # each seat
    ['CA', 'DA', 'SS'] # experiment
    
    sort_cols = ['crew', 'seat', 'experiment', 'time']
    target_col = 'event'
    
    what_cols = sort_cols + list(feature_cols) + [target_col]

    # Training
    traindf = df[df.crew.isin(crews['training'])][what_cols].sort_values(
        by=sort_cols).copy()
    
    scalar_dict, _ = do_standard_scaling(traindf, ['r'])
    
    print('Start building training set', timestamp())
    x_train, y_train = get_windows(traindf, ['r_scaled', 'event'],
                                   sequence_window,
                                  percent_of_data=percent_of_data)
    
    # Testing
    testdf = df[df.crew.isin(crews['test'])][what_cols].sort_values(
        by=sort_cols).copy()

    _, _ = do_standard_scaling(testdf, ['r'], scalar_dict)
    
    
    print('Start building testing set', timestamp())
    x_test, y_test = get_windows(testdf, ['r_scaled', 'event'],
                                 sequence_window,
                                 percent_of_data=percent_of_data)


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
                "percent_of_data": percent_of_data,
                "sequence_window": sequence_window,
                "feature_cols": list(feature_cols)}},
            "data_ts": timestamp()
        }}
            
    return {**outdata, **metadata}
    
    
def validate_data(data):
    assert len(Counter(data['y_train_original'])) > 1
    assert len(Counter(data['y_test_original'])) > 1
  
    
def get_windows(df, cols, window_size, percent_of_data=100):
    
    whats_proportion_index = lambda x, y: round(x*y)
    
    X = []
    Y = []
    choices = (df.crew.unique().tolist(), [0, 1], ['CA', 'DA', 'SS'])
    for crew, seat, experiment in itertools.product(*choices):
        query = (df.crew == crew)&(df.seat == seat)&(df.experiment == experiment)
        thisdf = df[query][cols]
        X_i, Y_i = to_sequences(thisdf.values, window_size)
        X.append(X_i[:
                     whats_proportion_index(
                         X_i.shape[0],
                         percent_of_data)])
        Y.append(Y_i[:
                     whats_proportion_index(
                        Y_i.shape[0],
                        percent_of_data)])
        
    return np.concatenate(X), np.concatenate(Y)

# Borrowing parts of this func from 
# https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class10_lstm.ipynb
def to_sequences(obs, seq_size, incols=[0], outcols=[1]):
    x = []
    y = []

    for i in range(len(obs)-seq_size-1):
        #print(i)
        window = obs[i:(i+seq_size)][..., 0]
        after_window = obs[i+seq_size, 1] # FIXME :off by 1 error here?
        # window = [[x] for x in window]

        x.append(window)
        y.append(after_window)
        
    xarr = np.array(x)
    yarr = np.array(y)
    return (np.resize(xarr, xarr.shape + (1,)),
            yarr)

```


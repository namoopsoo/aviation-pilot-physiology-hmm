_Draft.. still need to copy from notebook..._

#### At one point I did this... 
```python
target_indices = examples[0] + examples[1] + examples[2] + examples[3]
print(len(target_indices))

%time examples = {j: fetch_some_examples(outdata, j, n=1000) for j in range(4)}

%time dataset_batches = build_dataset(outdata, target_indices)

print(dataset_batches)
```

##### ..
```python
4000
CPU times: user 1.39 s, sys: 7.09 ms, total: 1.4 s
Wall time: 1.4 s
CPU times: user 30.9 ms, sys: 8.18 ms, total: 39.1 ms
Wall time: 53 ms
<DatasetV1Adapter shapes: ((?, 256, 1), (?,), (?,)), types: (tf.float32, tf.int64, tf.float64)>
```

#### Still using this model from yesterday
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64,  dropout=0.2, recurrent_dropout=0.2,
                input_shape=(None, 1)
              ),
    # 4 because 'A', 'B', 'C', 'D'.
    tf.keras.layers.Dense(4)
])
```
* And took about `2min` with those `4000` inputs.
```python
%time loss_history = do_train(model, dataset_batches)

# =>
CPU times: user 1min 51s, sys: 8.48 s, total: 1min 59s
Wall time: 2min 2s
```

#### Evaluation
```python

test_subset = np.random.choice(np.arange(0, outdata['x_test'].shape[0], 1), 100000, replace=False)
expected_onehot = tf.convert_to_tensor(outdata['y_test'][test_subset])
 
%time preds = model(tf.convert_to_tensor(outdata['x_test'][test_subset], dtype=tf.float32))

print(Counter(np.argmax(preds, axis=1)))
expected = tf.convert_to_tensor(np.argmax(outdata['y_test'][test_subset], axis=1))

tf.confusion_matrix( 
    expected,# labels
    np.argmax(preds, axis=1), # predictions
    num_classes=4
)


```

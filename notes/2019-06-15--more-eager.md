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

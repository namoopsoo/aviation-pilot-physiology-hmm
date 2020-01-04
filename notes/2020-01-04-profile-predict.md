#### Summary
* My predict , which I've been using for plotting batch validation loss now, is really slow, so I figure I would just run `kernprofile` on it...
* So using my commit `f09236c` , I ran the following on my sagemaker notebook instance terminal 
* This was on an arbitrarily one of the last models I had..
```python
(tensorflow_p36) $ kernprof -v -l predict.py --test-loc history/2019-12-22T174803Z/test_balanced.h5 --batch-size 32 --model-loc history/2019-12-29T000509Z/epoch_000_batch_00030_model.h5   --work-dir history/2020-01-04T1945Z
{'verbose': False, 'dry_run': False, 'model_loc': 'history/2019-12-29T000509Z/epoch_000_batch_00030_model.h5', 'test_loc': 'history/2019-12-22T174803Z/test_balanced.h5', 'batch_size': '32', 'work_dir': 'history/2020-01-04T1945Z'}
2020-01-04 20:27:41.378986: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  AVX512F
To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
2020-01-04 20:27:41.386826: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2500000000 Hz
2020-01-04 20:27:41.387318: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563f90e1a940 executing computations on platform Host. Devices:
2020-01-04 20:27:41.387339: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2020-01-04 20:27:41.387631: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
WARNING: Logging before flag parsing goes to stderr.
W0104 20:27:41.490362 139981662951232 nn_ops.py:4224] Large dropout rate: 0.6 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.
W0104 20:27:41.550024 139981662951232 nn_ops.py:4224] Large dropout rate: 0.6 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.
W0104 20:27:41.601838 139981662951232 nn_ops.py:4224] Large dropout rate: 0.6 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.
W0104 20:27:41.653169 139981662951232 nn_ops.py:4224] Large dropout rate: 0.6 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.
W0104 20:27:41.711273 139981662951232 nn_ops.py:4224] Large dropout rate: 0.6 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.
W0104 20:27:42.493826 139981662951232 hdf5_format.py:221] No training configurationfound in save file: the model was *not* compiled. Compile it manually.


Done.
Wrote profile results to predict.py.lprof
Timer unit: 1e-06 s

Total time: 996.896 s
File: /home/ec2-user/SageMaker/aviation-pilot-physiology-hmm/mytf/validation.py
Function: get_performance_parts at line 10

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    10                                           @profile
    11                                           def get_performance_parts(model, dataloc, dataset_names, eager, batch_size=None):
    12                                               #
    13                                               # dataloc contains the test data..
    14         1          2.0      2.0      0.0      if batch_size is None:
    15                                                   batch_size = 100
    16         1          1.0      1.0      0.0      lossvec = []
    17         5          8.0      1.6      0.0      for Xdataset, Ydataset in dataset_names:
    18
    19         4     177532.0  44383.0      0.0          X, Ylabels = mu.read_h5_two(dataloc, Xdataset, Ydataset)
    20         4       1806.0    451.5      0.0          parts = mu.get_partitions(range(X.shape[0]), batch_size, keep_remainder=False)
    21         4        177.0     44.2      0.0          batchlosses = []
    22      2094       3403.0      1.6      0.0          for part in parts:
    23      2090  992588227.0 474922.6     99.6              preds = model(X[part].astype('float32'))
    24
    25      2090       5504.0      2.6      0.0              if eager:
    26      2090       4296.0      2.1      0.0                  tensor = sparse_softmax_cross_entropy(
    27      2090      84506.0     40.4      0.0                          labels=Ylabels[part].astype('int64'),
    28      2090    3971184.0   1900.1      0.4                          logits=preds.numpy())
    29      2090      54487.0     26.1      0.0                  loss = tensor.numpy()
    30                                                       else:
    31                                                           tensor = sparse_softmax_cross_entropy(
    32                                                                   labels=Ylabels[part].astype('int64'),
    33                                                                   logits=preds)
    34                                                           loss = tensor.eval()
    35      2090       4100.0      2.0      0.0              batchlosses.append(loss)
    36
    37         4        706.0    176.5      0.0          lossvec.append(np.mean(batchlosses))
    38         1          2.0      2.0      0.0      return lossvec

Total time: 998.111 s
File: /home/ec2-user/SageMaker/aviation-pilot-physiology-hmm/mytf/validation.py
Function: perf_wrapper at line 41

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    41                                           @profile
    42                                           def perf_wrapper(modelloc, dataloc, eager, batch_size=None):
    43                                               # dataloc: h5 location for test data
    44
    45         1          3.0      3.0      0.0      if batch_size is None:
    46                                                   batch_size = 100
    47
    48         1    1181041.0 1181041.0      0.1      model = mu.load_model(modelloc)
    49
    50         1          3.0      3.0      0.0      return get_performance_parts(
    51         1          1.0      1.0      0.0                      model=model,
    52         1          1.0      1.0      0.0                      dataloc=dataloc,
    53         1          0.0      0.0      0.0                      dataset_names=[['X_0', 'Ylabels_0'],
    54         1          1.0      1.0      0.0                                    ['X_1', 'Ylabels_1'],
    55         1          1.0      1.0      0.0                                    ['X_2', 'Ylabels_2'],
    56         1          1.0      1.0      0.0                                    ['X_3', 'Ylabels_3']],
    57         1          1.0      1.0      0.0                      eager=eager,
    58         1  996930019.0 996930019.0     99.9                      batch_size=batch_size)

```


### Summary
- So the idea I'm trying here is that so far my train/test split has not been a neat random split, but instead out of the `10` or so *Crews* in the full data, I have used `crew=1` as train and `crew=2` as test. But I had observed earlier that `crew=2` has values in features that far exceed that observed in `crew=1` and this has been messing with scaling. 
- So here I'm combining `crew=1` and `crew=2` and generating a random train/test split.

```python
from importlib import reload
import os
import pandas as pd
from io import StringIO
import itertools
import ipdb
import datetime
from collections import Counter

import h5py
import json
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
print(tf.__version__)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

from keras.callbacks import EarlyStopping
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import mytf.s3utils as msu
import mytf.utils as mu
import mytf.validation as mv
import mytf.plot as mp
```

    1.14.0


    Using TensorFlow backend.



```python
tf.enable_eager_execution()

```


```python
workdir = 'history/2020-02-03T000055Z'
balanced_one_loc = f'{workdir}/balanced_one.h5'
balanced_two_loc = f'{workdir}/balanced_two.h5'
print(mu.h5_keys(balanced_one_loc), mu.h5_keys(balanced_two_loc))
```

    ['X_0', 'X_1', 'X_2', 'X_3', 'Ylabels_0', 'Ylabels_1', 'Ylabels_2', 'Ylabels_3'] ['X_0', 'X_1', 'X_2', 'X_3', 'Ylabels_0', 'Ylabels_1', 'Ylabels_2', 'Ylabels_3']



```python
!ls -lah history/2020-02-03T000055Z
```

So as per [earlier notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-02-01.md#shuffle-also)  , going to shuffle these together and make new train/test split datasets..,


```python
vecs1 = [mu.read_h5_two(
                source_location=balanced_one_loc, 
                Xdataset=f'X_{i}',
                Ydataset=f'Ylabels_{i}')
                 for i in [0, 1, 2, 3]]
vecs2 = [mu.read_h5_two(
                source_location=balanced_two_loc, 
                Xdataset=f'X_{i}',
                Ydataset=f'Ylabels_{i}')
                 for i in [0, 1, 2, 3]]
```


```python
vecs = list(zip(vecs1, vecs2))
```


```python
[a.shape for a in vecs1[0]]
```




    [(15713, 64, 8), (15713,)]




```python
# np.concatenate()
print([[x[0].shape, x[1].shape] for x in vecs[0]])
print(np.concatenate([x[0] for x in vecs[0]]).shape)
print(np.concatenate([x[1] for x in vecs[0]]).shape)
```

    [[(15713, 64, 8), (15713,)], [(20896, 64, 8), (20896,)]]
    (36609, 64, 8)
    (36609,)



```python
X = [np.concatenate([x[0] for x in vecs[i]])
      for i in range(4)]
Y = [np.concatenate([x[1] for x in vecs[i]])
      for i in range(4)]
```


```python
X[0].shape, Y[0].shape

```




    ((36609, 64, 8), (36609,))




```python
# Randomly assign half to train... and the rest to test
fullsize = X[0].shape[0]

print([a.shape for a in X])
print([a.shape for a in Y])

def split_indices(A):
    fullsize = A.shape[0]
    train_size = fullsize//2
    train_indices = np.random.choice(range(fullsize), size=train_size, replace=False)
    #np.array(list(set(range(5)) - set(np.array([1,2])))), set(train_indices[:4])
    test_indices = np.array(list(set(range(fullsize)) - set(train_indices)))
    print(train_indices.shape, test_indices.shape)
    assert fullsize == train_indices.shape[0] + test_indices.shape[0]
    assert fullsize == len(set(train_indices) | set(test_indices))
    return train_indices, test_indices

indices = [split_indices(A) for A in X]

Xtrain = [X[i][indices[i][0]] for i in range(4)]
Ytrain = [Y[i][indices[i][0]] for i in range(4)]
print('Xtrain:', [A.shape for A in Xtrain])
print('Ytrain:', [A.shape for A in Ytrain])

Xtest = [X[i][indices[i][1]] for i in range(4)]
Ytest = [Y[i][indices[i][1]] for i in range(4)]
print('Xtest:', [A.shape for A in Xtest])
print('Ytest:', [A.shape for A in Ytest])

# Shuffle the X though
```

    [(36609, 64, 8), (22996, 64, 8), (29808, 64, 8), (23761, 64, 8)]
    [(36609,), (22996,), (29808,), (23761,)]
    (18304,) (18305,)
    (11498,) (11498,)
    (14904,) (14904,)
    (11880,) (11881,)
    Xtrain: [(18304, 64, 8), (11498, 64, 8), (14904, 64, 8), (11880, 64, 8)]
    Ytrain: [(18304,), (11498,), (14904,), (11880,)]
    Xtest: [(18305, 64, 8), (11498, 64, 8), (14904, 64, 8), (11881, 64, 8)]
    Ytest: [(18305,), (11498,), (14904,), (11881,)]



```python
X_trainall = np.concatenate([a for a in Xtrain])
Y_trainall = np.concatenate([a for a in Ytrain])

# Shuffle...
size = X_trainall.shape[0]
indices = np.random.choice(range(size), size=size, replace=False)
X_train_shfl = X_trainall[indices]
Ylabels_train_shfl = Y_trainall[indices].astype('int64')


# SAVE ...
mu.save_that(save_location=f'{workdir}/train_scaled_balanced_shuffled.h5', 
             name='X', X=X_train_shfl)

mu.save_that(save_location=f'{workdir}/train_scaled_balanced_shuffled.h5', 
             name='Ylabels', X=Ylabels_train_shfl)
```


```python
# Save test, unshuffled
#                Xdataset=f'X_{i}',
#                Ydataset=f'Ylabels_{i}')
[mu.save_that(save_location=f'{workdir}/train_balanced.h5',
                name=f'X_{i}', X=Xtrain[i])
 for i in range(4)]
[mu.save_that(save_location=f'{workdir}/train_balanced.h5',
                name=f'Ylabels_{i}', X=Ytrain[i])
 for i in range(4)]


# And test too..
[mu.save_that(save_location=f'{workdir}/test_balanced.h5',
                name=f'X_{i}', X=Xtest[i])
 for i in range(4)]
[mu.save_that(save_location=f'{workdir}/test_balanced.h5',
                name=f'Ylabels_{i}', X=Ytest[i])
 for i in range(4)]


```




    [None, None, None, None]




```python
mu.h5_keys('history/2020-02-02T044441Z/test_balanced.h5')
```




    ['X_0',
     'X_1',
     'X_2',
     'X_3',
     'Ylabels_0',
     'Ylabels_1',
     'Ylabels_2',
     'Ylabels_3']



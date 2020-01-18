

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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import mytf.s3utils as msu
import mytf.utils as mu
import mytf.validation as mv
import mytf.plot as mp
```

    1.14.0



```python
tf.enable_eager_execution()

```


```python
# So for a model thats roughly just starting out... what do its logits look like...
detail = {
 'model_loc': 'history/2019-12-29T000509Z/epoch_000_batch_00030_model.h5', 
    'test_loc': 'history/2019-12-22T174803Z/test_balanced.h5', 
 'batch_size': '32', 
 #'work_dir': 'history/2020-01-04T1945Z'
}

```


```python
# Borrowing from earlier notebook...
# https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-01-loglossbook--update.md

def get_raw_preds(model, dataloc, dataset_names):
    # dataloc contains the test data..
    lossvec = []
    predsvec = []
    for Xdataset, Ydataset in tqdm(dataset_names):

        X, Ylabels = mu.read_h5_two(dataloc, Xdataset, Ydataset) 
        parts = mu.get_partitions(range(X.shape[0]), 100)
        batchlosses = []
        for part in parts:
            preds = model(X[part].astype('float32'))
            loss = tf.losses.sparse_softmax_cross_entropy(
                        labels=Ylabels[part].astype('int64'),
                        logits=preds.numpy()).numpy()
            batchlosses.append(loss)
            predsvec.extend(preds)

        lossvec.append(np.mean(batchlosses))
    return lossvec, predsvec

def plot_logits(vec, title=''):
    plt.plot([x[0] for x in vec], color='blue', label='0')
    plt.plot([x[1] for x in vec], color='green', label='1')
    plt.plot([x[2] for x in vec], color='red', label='2')
    plt.plot([x[3] for x in vec], color='orange', label='3')
    plt.title(title)
    plt.legend()
    
# modelloc = f'history/2019-12-08T220612Z/{modeln}'
model = mu.load_model(detail['model_loc'])

lossvec, predsvec = get_raw_preds(
                    model=model,
                    dataloc=detail['test_loc'],
                    dataset_names=[['X_0', 'Ylabels_0'],
                                  ])
predarr = np.vstack([x.numpy() for x in predsvec])
```


```python
plot_logits(predarr, f'{detail["model_loc"]} logits \n on label=0 \n (loss={lossvec[0]}) ')
```


![png](2020-01-10-confidence--update_files/2020-01-10-confidence--update_4_0.png)



```python
%%time
label_logits_vec = {}

for label in tqdm(range(4)):
    print(f'label {label}')
    lossvec, predsvec = get_raw_preds(
                        model=model,
                        dataloc=detail['test_loc'],
                        dataset_names=[[f'X_{label}', f'Ylabels_{label}'],
                                      ])
    predarr = np.vstack([x.numpy() for x in predsvec])
    label_logits_vec[label] = [lossvec, predarr]
```

      0%|          | 0/4 [00:00<?, ?it/s]
      0%|          | 0/1 [00:00<?, ?it/s][A

    label 0


    
    100%|██████████| 1/1 [00:29<00:00, 29.78s/it][A
     25%|██▌       | 1/4 [00:29<01:29, 29.93s/it]
      0%|          | 0/1 [00:00<?, ?it/s][A

    label 1


    
    100%|██████████| 1/1 [00:27<00:00, 27.99s/it][A
     50%|█████     | 2/4 [00:58<00:58, 29.40s/it]
      0%|          | 0/1 [00:00<?, ?it/s][A

    label 2


    
    100%|██████████| 1/1 [00:36<00:00, 36.43s/it][A
     75%|███████▌  | 3/4 [01:34<00:31, 31.57s/it]
      0%|          | 0/1 [00:00<?, ?it/s][A

    label 3


    
    100%|██████████| 1/1 [00:29<00:00, 29.69s/it][A
    100%|██████████| 4/4 [02:04<00:00, 31.18s/it]

    CPU times: user 2min 3s, sys: 389 ms, total: 2min 4s
    Wall time: 2min 4s


    



```python

def plot_logits_ax(vec, ax, title=''):
    ax.plot([x[0] for x in vec], color='blue', label='0')
    ax.plot([x[1] for x in vec], color='green', label='1')
    ax.plot([x[2] for x in vec], color='red', label='2')
    ax.plot([x[3] for x in vec], color='orange', label='3')
    ax.set(title=title)
    #ylabel=col,
    #xlabel='time')

    ax.legend()

fig = plt.figure(figsize=(20,12))


for i in range(4):
    ax = fig.add_subplot(int('22' + str(i+1)))
    plot_logits_ax(vec=label_logits_vec[i][1], 
                   ax=ax, 
                   title=(f'  logits '
                          f' on label={i}  \n '
                          f' (loss={label_logits_vec[i][0][0]})'))

                   

print(f'using detail["model_loc"]')

```

    using detail["model_loc"]



![png](2020-01-10-confidence--update_files/2020-01-10-confidence--update_6_1.png)


### 2020-01-11

#### what the heck
* These jumps above are gnarly for `label=1` and `label=3` .
* What is it about this validation data jumping?
* Also the logloss hmm , for `label=0` . i feel like it would be way less than `1.0`. 


```python
# Lets review this logloss calc more precisely...
import ipdb
label = 0
lossvec, predsvec =  ipdb.runcall(get_raw_preds,
                        model=model,
                        dataloc=detail['test_loc'],
                        dataset_names=[[f'X_{label}', f'Ylabels_{label}'],
                                      ])

```

    > [0;32m<ipython-input-5-7bbcc0674959>[0m(6)[0;36mget_raw_preds[0;34m()[0m
    [0;32m      5 [0;31m    [0;31m# dataloc contains the test data..[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m----> 6 [0;31m    [0mlossvec[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0m
    [0m[0;32m      7 [0;31m    [0mpredsvec[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0m
    [0m
    ipdb> n
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(7)[0;36mget_raw_preds[0;34m()[0m
    [0;32m      6 [0;31m    [0mlossvec[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0m
    [0m[0;32m----> 7 [0;31m    [0mpredsvec[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0m
    [0m[0;32m      8 [0;31m    [0;32mfor[0m [0mXdataset[0m[0;34m,[0m [0mYdataset[0m [0;32min[0m [0mtqdm[0m[0;34m([0m[0mdataset_names[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0m
    [0m
    ipdb> 
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(8)[0;36mget_raw_preds[0;34m()[0m
    [0;32m      7 [0;31m    [0mpredsvec[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0m
    [0m[0;32m----> 8 [0;31m    [0;32mfor[0m [0mXdataset[0m[0;34m,[0m [0mYdataset[0m [0;32min[0m [0mtqdm[0m[0;34m([0m[0mdataset_names[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0m
    [0m[0;32m      9 [0;31m[0;34m[0m[0m
    [0m
    ipdb> 


      0%|          | 0/1 [00:00<?, ?it/s]

    > [0;32m<ipython-input-5-7bbcc0674959>[0m(10)[0;36mget_raw_preds[0;34m()[0m
    [0;32m      9 [0;31m[0;34m[0m[0m
    [0m[0;32m---> 10 [0;31m        [0mX[0m[0;34m,[0m [0mYlabels[0m [0;34m=[0m [0mmu[0m[0;34m.[0m[0mread_h5_two[0m[0;34m([0m[0mdataloc[0m[0;34m,[0m [0mXdataset[0m[0;34m,[0m [0mYdataset[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m     11 [0;31m        [0mparts[0m [0;34m=[0m [0mmu[0m[0;34m.[0m[0mget_partitions[0m[0;34m([0m[0mrange[0m[0;34m([0m[0mX[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0;36m100[0m[0;34m)[0m[0;34m[0m[0m
    [0m
    ipdb> p Xdataset, Ydataset
    ('X_0', 'Ylabels_0')
    ipdb> n
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(11)[0;36mget_raw_preds[0;34m()[0m
    [0;32m     10 [0;31m        [0mX[0m[0;34m,[0m [0mYlabels[0m [0;34m=[0m [0mmu[0m[0;34m.[0m[0mread_h5_two[0m[0;34m([0m[0mdataloc[0m[0;34m,[0m [0mXdataset[0m[0;34m,[0m [0mYdataset[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m---> 11 [0;31m        [0mparts[0m [0;34m=[0m [0mmu[0m[0;34m.[0m[0mget_partitions[0m[0;34m([0m[0mrange[0m[0;34m([0m[0mX[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0;36m100[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m     12 [0;31m        [0mbatchlosses[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0m
    [0m
    ipdb> 
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(12)[0;36mget_raw_preds[0;34m()[0m
    [0;32m     11 [0;31m        [0mparts[0m [0;34m=[0m [0mmu[0m[0;34m.[0m[0mget_partitions[0m[0;34m([0m[0mrange[0m[0;34m([0m[0mX[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0;36m100[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m---> 12 [0;31m        [0mbatchlosses[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0m
    [0m[0;32m     13 [0;31m        [0;32mfor[0m [0mpart[0m [0;32min[0m [0mparts[0m[0;34m:[0m[0;34m[0m[0m
    [0m
    ipdb> 
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(13)[0;36mget_raw_preds[0;34m()[0m
    [0;32m     12 [0;31m        [0mbatchlosses[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0m
    [0m[0;32m---> 13 [0;31m        [0;32mfor[0m [0mpart[0m [0;32min[0m [0mparts[0m[0;34m:[0m[0;34m[0m[0m
    [0m[0;32m     14 [0;31m            [0mpreds[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mX[0m[0;34m[[0m[0mpart[0m[0;34m][0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m'float32'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0m
    [0m
    ipdb> 
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(14)[0;36mget_raw_preds[0;34m()[0m
    [0;32m     13 [0;31m        [0;32mfor[0m [0mpart[0m [0;32min[0m [0mparts[0m[0;34m:[0m[0;34m[0m[0m
    [0m[0;32m---> 14 [0;31m            [0mpreds[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mX[0m[0;34m[[0m[0mpart[0m[0;34m][0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m'float32'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m     15 [0;31m            loss = tf.losses.sparse_softmax_cross_entropy(
    [0m
    ipdb> 
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(15)[0;36mget_raw_preds[0;34m()[0m
    [0;32m     14 [0;31m            [0mpreds[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mX[0m[0;34m[[0m[0mpart[0m[0;34m][0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m'float32'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m---> 15 [0;31m            loss = tf.losses.sparse_softmax_cross_entropy(
    [0m[0;32m     16 [0;31m                        [0mlabels[0m[0;34m=[0m[0mYlabels[0m[0;34m[[0m[0mpart[0m[0;34m][0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m'int64'[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0m
    [0m
    ipdb> p len(part)
    100
    ipdb> p X.shape
    (16263, 64, 8)
    ipdb> p len(parts)
    163
    ipdb> n
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(16)[0;36mget_raw_preds[0;34m()[0m
    [0;32m     15 [0;31m            loss = tf.losses.sparse_softmax_cross_entropy(
    [0m[0;32m---> 16 [0;31m                        [0mlabels[0m[0;34m=[0m[0mYlabels[0m[0;34m[[0m[0mpart[0m[0;34m][0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m'int64'[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0m
    [0m[0;32m     17 [0;31m                        logits=preds.numpy()).numpy()
    [0m
    ipdb> 
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(17)[0;36mget_raw_preds[0;34m()[0m
    [0;32m     16 [0;31m                        [0mlabels[0m[0;34m=[0m[0mYlabels[0m[0;34m[[0m[0mpart[0m[0;34m][0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m'int64'[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0m
    [0m[0;32m---> 17 [0;31m                        logits=preds.numpy()).numpy()
    [0m[0;32m     18 [0;31m            [0mbatchlosses[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mloss[0m[0;34m)[0m[0;34m[0m[0m
    [0m
    ipdb> 
    > [0;32m<ipython-input-5-7bbcc0674959>[0m(18)[0;36mget_raw_preds[0;34m()[0m
    [0;32m     17 [0;31m                        logits=preds.numpy()).numpy()
    [0m[0;32m---> 18 [0;31m            [0mbatchlosses[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mloss[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m     19 [0;31m            [0mpredsvec[0m[0;34m.[0m[0mextend[0m[0;34m([0m[0mpreds[0m[0;34m)[0m[0;34m[0m[0m
    [0m
    ipdb> p loss
    1.2797719
    ipdb> p pred.shape
    *** NameError: name 'pred' is not defined
    ipdb> p preds.shape
    TensorShape([Dimension(100), Dimension(4)])
    ipdb> preds
    <tf.Tensor: id=6851865, shape=(100, 4), dtype=float32, numpy=
    array([[ 1.83809385e-01,  1.00108096e-02, -1.75332371e-03,
             1.12673618e-01],
           [ 1.83426142e-01,  9.55305714e-03, -1.38717750e-03,
             1.12821095e-01],
           [ 1.83396950e-01,  9.47616715e-03, -1.38786808e-03,
             1.12689503e-01],
           [ 1.83594793e-01,  9.67238564e-03, -1.55105582e-03,
             1.12347908e-01],
           [ 1.83497131e-01,  9.49037168e-03, -1.35979336e-03,
             1.12225391e-01],
           [ 1.83367223e-01,  9.46970563e-03, -1.26552163e-03,
             1.12246551e-01],
           [ 1.83384418e-01,  9.54668876e-03, -1.24029070e-03,
             1.12149298e-01],
           [ 1.83410868e-01,  9.52172000e-03, -1.27099594e-03,
             1.12001091e-01],
           [ 1.83190092e-01,  9.51937307e-03, -1.20195514e-03,
             1.12195089e-01],
           [ 1.83049321e-01,  9.59867146e-03, -1.15521858e-03,
             1.12552956e-01],
           [ 1.83526218e-01,  9.89974383e-03, -1.48606906e-03,
             1.12403102e-01],
           [ 1.83662623e-01,  9.76842176e-03, -1.44290272e-03,
             1.12581454e-01],
           [ 1.83464974e-01,  9.48761869e-03, -1.29150040e-03,
             1.12878956e-01],
           [ 1.83388665e-01,  9.26300790e-03, -1.26104616e-03,
             1.12844333e-01],
           [ 1.83071762e-01,  8.87161586e-03, -1.08504808e-03,
             1.12915330e-01],
           [ 1.83204845e-01,  9.10987984e-03, -1.28194178e-03,
             1.12842888e-01],
           [ 1.83768079e-01,  9.62716248e-03, -1.68323331e-03,
             1.12340093e-01],
           [ 1.83950871e-01,  9.74959228e-03, -1.74526079e-03,
             1.12373255e-01],
           [ 1.84185684e-01,  9.94520914e-03, -1.83196180e-03,
             1.12559684e-01],
           [ 1.84032306e-01,  9.75103024e-03, -1.66189065e-03,
             1.12765394e-01],
           [ 1.83505714e-01,  9.15043149e-03, -1.28929736e-03,
             1.13014624e-01],
           [ 1.83513254e-01,  9.19665117e-03, -1.35500915e-03,
             1.13043800e-01],
           [ 1.83542073e-01,  9.12838895e-03, -1.27445674e-03,
             1.12699777e-01],
           [ 1.83344766e-01,  9.00703762e-03, -1.16608525e-03,
             1.12673469e-01],
           [ 1.83427945e-01,  9.24207736e-03, -1.22686755e-03,
             1.12664104e-01],
           [ 1.83159739e-01,  9.01816506e-03, -1.03682932e-03,
             1.12618946e-01],
           [ 1.82930246e-01,  8.93077161e-03, -9.67461150e-04,
             1.12632811e-01],
           [ 1.83023825e-01,  9.12382174e-03, -1.00686122e-03,
             1.12580307e-01],
           [ 1.82964757e-01,  9.17755719e-03, -9.10394359e-04,
             1.12620704e-01],
           [ 1.82980195e-01,  9.30781942e-03, -9.50718299e-04,
             1.12403855e-01],
           [ 1.83441624e-01,  9.63270385e-03, -1.17576122e-03,
             1.12089247e-01],
           [ 1.83103278e-01,  9.28756315e-03, -8.57419800e-04,
             1.12410307e-01],
           [ 1.82891473e-01,  9.13425442e-03, -7.51798740e-04,
             1.12719774e-01],
           [ 1.83101043e-01,  9.08723939e-03, -8.47567338e-04,
             1.12976186e-01],
           [ 1.83253065e-01,  8.98871850e-03, -8.39160755e-04,
             1.13044865e-01],
           [ 1.83206886e-01,  8.87447130e-03, -7.95911998e-04,
             1.13322876e-01],
           [ 1.83402464e-01,  8.87255277e-03, -9.21640545e-04,
             1.13231473e-01],
           [ 1.83582336e-01,  8.99768900e-03, -1.04969461e-03,
             1.13163278e-01],
           [ 1.84070632e-01,  9.23054013e-03, -1.31386938e-03,
             1.13168038e-01],
           [ 1.84246451e-01,  9.19828471e-03, -1.34700537e-03,
             1.13330685e-01],
           [ 1.84383675e-01,  9.07266419e-03, -1.36724580e-03,
             1.13401122e-01],
           [ 1.84810117e-01,  9.23624169e-03, -1.57452142e-03,
             1.13304839e-01],
           [ 1.84814006e-01,  9.18975938e-03, -1.53015228e-03,
             1.13497749e-01],
           [ 1.84642032e-01,  8.86514317e-03, -1.39958365e-03,
             1.13571525e-01],
           [ 1.84768245e-01,  8.88718385e-03, -1.48984697e-03,
             1.13631733e-01],
           [ 1.85020611e-01,  8.99135415e-03, -1.62447523e-03,
             1.13405868e-01],
           [ 1.85266495e-01,  8.90528876e-03, -1.71769504e-03,
             1.13197856e-01],
           [ 1.85288563e-01,  8.82173423e-03, -1.73399597e-03,
             1.13453314e-01],
           [ 1.85119241e-01,  8.89696460e-03, -1.70976901e-03,
             1.13770053e-01],
           [ 1.85269848e-01,  9.01340786e-03, -1.98364491e-03,
             1.13906555e-01],
           [ 1.84869528e-01,  8.51168577e-03, -1.81037094e-03,
             1.14449874e-01],
           [ 1.85351372e-01,  8.79755709e-03, -2.30810232e-03,
             1.14559039e-01],
           [ 1.85970560e-01,  9.01589636e-03, -2.70801736e-03,
             1.14267744e-01],
           [ 1.86237052e-01,  8.79690517e-03, -2.85202218e-03,
             1.14199117e-01],
           [ 1.86247319e-01,  8.56915396e-03, -2.86089862e-03,
             1.14544846e-01],
           [ 1.86079010e-01,  8.34266189e-03, -2.86879158e-03,
             1.14714220e-01],
           [ 1.86013177e-01,  8.35716072e-03, -3.00837914e-03,
             1.14821725e-01],
           [ 1.86274201e-01,  8.70853942e-03, -3.24393716e-03,
             1.14912927e-01],
           [ 1.86520934e-01,  9.09026805e-03, -3.43263941e-03,
             1.14912301e-01],
           [ 1.86590746e-01,  9.15667508e-03, -3.45148006e-03,
             1.14859261e-01],
           [ 1.86991230e-01,  9.21133999e-03, -3.53030954e-03,
             1.14804611e-01],
           [ 1.86981574e-01,  8.95407517e-03, -3.34534515e-03,
             1.15029022e-01],
           [ 1.86970845e-01,  8.78692698e-03, -3.27889295e-03,
             1.15134165e-01],
           [ 1.87227443e-01,  8.76174029e-03, -3.27700702e-03,
             1.14868604e-01],
           [ 1.87213257e-01,  8.47804639e-03, -3.12770624e-03,
             1.14689343e-01],
           [ 1.86865553e-01,  8.27426370e-03, -2.96807988e-03,
             1.14734583e-01],
           [ 1.86787456e-01,  8.26285873e-03, -2.76897242e-03,
             1.14631757e-01],
           [ 1.86537698e-01,  8.14121496e-03, -2.49261595e-03,
             1.14705995e-01],
           [ 1.86006770e-01,  7.83473905e-03, -2.11561145e-03,
             1.14747711e-01],
           [ 1.86064541e-01,  7.94193987e-03, -1.90028036e-03,
             1.14310816e-01],
           [ 1.85945988e-01,  7.87886698e-03, -1.56034157e-03,
             1.14293888e-01],
           [ 1.85618609e-01,  7.74658192e-03, -1.18492823e-03,
             1.14341840e-01],
           [ 1.85319915e-01,  7.55306054e-03, -8.66284128e-04,
             1.14234388e-01],
           [ 1.85336679e-01,  7.75365811e-03, -8.58305488e-04,
             1.14130341e-01],
           [ 1.85365587e-01,  7.96361919e-03, -8.35366547e-04,
             1.14038162e-01],
           [ 1.85650662e-01,  8.13303236e-03, -8.59306660e-04,
             1.13665923e-01],
           [ 1.85592294e-01,  8.00848100e-03, -6.77116914e-04,
             1.13660559e-01],
           [ 1.85226813e-01,  7.54116196e-03, -3.27722635e-04,
             1.13884166e-01],
           [ 1.85002148e-01,  7.09145050e-03, -1.18391821e-04,
             1.13909572e-01],
           [ 1.84447810e-01,  6.71072956e-03,  1.33648980e-04,
             1.14105090e-01],
           [ 1.84550658e-01,  6.71200734e-03, -9.93914437e-05,
             1.13992110e-01],
           [ 1.84959173e-01,  7.19560403e-03, -4.48482810e-04,
             1.13966994e-01],
           [ 1.85048163e-01,  7.33019505e-03, -4.98979585e-04,
             1.13929704e-01],
           [ 1.85314596e-01,  7.21434411e-03, -4.24314290e-04,
             1.13733225e-01],
           [ 1.85431346e-01,  7.25012925e-03, -3.95299401e-04,
             1.13744713e-01],
           [ 1.85214922e-01,  6.96921255e-03, -1.17514050e-04,
             1.13950163e-01],
           [ 1.87568024e-01,  8.05983599e-03, -3.77909001e-03,
             1.14564478e-01],
           [ 1.89544991e-01,  8.88864603e-03, -6.28790120e-03,
             1.15484431e-01],
           [ 1.90953612e-01,  9.25218221e-03, -8.05721991e-03,
             1.16737694e-01],
           [ 1.92968845e-01,  9.89973452e-03, -1.05823576e-02,
             1.18148208e-01],
           [ 1.94396734e-01,  1.00413961e-02, -1.21619552e-02,
             1.19290322e-01],
           [ 1.95768982e-01,  1.01512270e-02, -1.35654528e-02,
             1.20166399e-01],
           [ 1.96944922e-01,  1.02125863e-02, -1.46759097e-02,
             1.20847732e-01],
           [ 1.97604984e-01,  1.04152216e-02, -1.55473631e-02,
             1.21534884e-01],
           [ 1.98240653e-01,  1.07544726e-02, -1.63659938e-02,
             1.21881254e-01],
           [ 1.98817775e-01,  1.10079376e-02, -1.68887451e-02,
             1.22321546e-01],
           [ 1.99332401e-01,  1.13642076e-02, -1.73578896e-02,
             1.22776493e-01],
           [ 1.99537978e-01,  1.13205267e-02, -1.76140014e-02,
             1.23060055e-01],
           [ 1.99857056e-01,  1.11855557e-02, -1.78099517e-02,
             1.23147562e-01],
           [ 1.99935094e-01,  1.11069726e-02, -1.79213472e-02,
             1.23467848e-01]], dtype=float32)>
    ipdb> p tf.losses.sparse_softmax_cross_entropy(                          labels=Ylabels[part].astype('int64'),                          logits=preds.numpy()).numpy()
    1.2797719
    ipdb> p part
    range(0, 100)
    ipdb> p len(parts)
    163
    ipdb> p parts[0]
    range(0, 100)
    ipdb> p parts[2]
    range(200, 300)
    ipdb> p list(parts[2])[:5]
    [200, 201, 202, 203, 204]
    ipdb> Ylabels[part].astype('int64')
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ipdb> p Counter
    <class 'collections.Counter'>
    ipdb> p Ylabels.shape
    (16263,)
    ipdb> p Counter(Ylabels)
    Counter({0.0: 16263})
    ipdb> q



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-26-53ab0ce9167c> in <module>()
          5                         model=model,
          6                         dataloc=detail['test_loc'],
    ----> 7                         dataset_names=[[f'X_{label}', f'Ylabels_{label}'],
          8                                       ])


    TypeError: 'NoneType' object is not iterable

#### Play with sparse_softmax_cross_entropy a little here

```python
# print(lossvec)
# [1.2610213]
# ipdb> p X.shape
# (16263, 64, 8)
preds = np.array([[ 1.83809385e-01,  1.00108096e-02, -1.75332371e-03,
         1.12673618e-01],
       [ 1.83426142e-01,  9.55305714e-03, -1.38717750e-03,
         1.12821095e-01],
       [ 1.83396950e-01,  9.47616715e-03, -1.38786808e-03,
         1.12689503e-01],])

for preds, ylabels in [[np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],]), np.array([0, 0, 0, 0, 0])],
                      [np.array([[1, 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [1, 0, 0, 0],]), np.array([0, 0, 0, 0, 0])],
                      [np.array([[1, 0, 0, 1],
                                 [1, 0, 0, 1],
                                 [1, 0, 0, 1],
                                 [1, 0, 0, 1],
                                 [1, 0, 0, 1],]), np.array([0, 0, 0, 0, 0])],
                      [np.array([[2, 0, 0, 0],
                                 [2, 0, 0, 0],
                                 [2, 0, 0, 0],
                                 [2, 0, 0, 0],
                                 [2, 0, 0, 0],]), np.array([0, 0, 0, 0, 0])],
                      [np.array([[9, 0, 0, 0],
                                 [9, 0, 0, 0],
                                 [9, 0, 0, 0],
                                 [9, 0, 0, 0],
                                 [9, 0, 0, 0],]), np.array([0, 0, 0, 0, 0])],
                      [np.array([[ 1.83809385e-01,  1.00108096e-02, -1.75332371e-03,
                         1.12673618e-01],
                       [ 1.83426142e-01,  9.55305714e-03, -1.38717750e-03,
                         1.12821095e-01],
                       [ 1.83396950e-01,  9.47616715e-03, -1.38786808e-03,
                         1.12689503e-01],]), np.array([0, 0, 0])],

                       
                      ]:
    loss = tf.losses.sparse_softmax_cross_entropy(
                         labels=ylabels.astype('int64'),
                         logits=preds.astype('float64')).numpy()
    print({'ylabels': ylabels, 'preds': preds, 'loss': loss})

```

    {'ylabels': array([0, 0, 0, 0, 0]), 'preds': array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]), 'loss': 1.3862943649291992}
    {'ylabels': array([0, 0, 0, 0, 0]), 'preds': array([[1, 0, 0, 0],
           [1, 0, 0, 0],
           [1, 0, 0, 0],
           [1, 0, 0, 0],
           [1, 0, 0, 0]]), 'loss': 0.7436683773994446}
    {'ylabels': array([0, 0, 0, 0, 0]), 'preds': array([[1, 0, 0, 1],
           [1, 0, 0, 1],
           [1, 0, 0, 1],
           [1, 0, 0, 1],
           [1, 0, 0, 1]]), 'loss': 1.0064088106155396}
    {'ylabels': array([0, 0, 0, 0, 0]), 'preds': array([[2, 0, 0, 0],
           [2, 0, 0, 0],
           [2, 0, 0, 0],
           [2, 0, 0, 0],
           [2, 0, 0, 0]]), 'loss': 0.3407529592514038}
    {'ylabels': array([0, 0, 0, 0, 0]), 'preds': array([[9, 0, 0, 0],
           [9, 0, 0, 0],
           [9, 0, 0, 0],
           [9, 0, 0, 0],
           [9, 0, 0, 0]]), 'loss': 0.00037016088026575744}
    {'ylabels': array([0, 0, 0]), 'preds': array([[ 0.18380938,  0.01001081, -0.00175332,  0.11267362],
           [ 0.18342614,  0.00955306, -0.00138718,  0.11282109],
           [ 0.18339695,  0.00947617, -0.00138787,  0.1126895 ]]), 'loss': 1.281795859336853}



```python
# So looking at the above, the tf.losses.sparse_softmax_cross_entropy appears to want,
# Not a softmax kind of output where each logit adds to 1.0. 
# Since that clearly yields around 0.74
#
# Instead a high [9, 0, 0, 0] here produces a nice low logloss like 0.000 
#
# But definitely nice to see that [1, 0, 0, 1] logloss is more than that for [1, 0, 0, 0]
# meaning it definitely penalizes the wrong logits. So thats good.
```


```python
# Ok ... lookat the flattened data again... 
# So for label=1 , after dataset 7500, 
X, Ylabels = mu.read_h5_two(source_location=detail['test_loc'], 
                            Xdataset='X_1', 
                            Ydataset='Ylabels_1') 

```


```python
#X[:, :, 0].ravel().shape
# (983744,)

X.shape

```




    (15371, 64, 8)




```python
fig = plt.figure(figsize=(20,12))
colors = [
    'blue', 'orange', 
    'green', 'red', 'purple', 
    'brown', 'pink', 'gray', 'olive', 'cyan']
ax = fig.add_subplot(int('111'))
for j in range(8):
    ax.plot(X[:, 0, j].ravel(), color=colors[j], label=f'{j}')
    ax.set(title='test data , label=1')
    #ylabel=col,
    #xlabel='time'

    ax.legend()



```


![png](2020-01-10-confidence--update_files/2020-01-10-confidence--update_13_0.png)



```python
# I guess from looking at the above, this may be related to the fact that the
# MinMax Scaler I used for training data has some data that , 
# does not bound the test data. 
# 
# So what does one do about that? Should the test data be clipped ? Hmm
# Or maybe the minMaxScaler should be built from more data.
# So basically I can take another stap perhaps on 
# this , https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md

```




    983744




```python
%%time
# ok... and just one more time, look at train data again..
moredetail = {'train_loc': 'history/2019-12-22T174803Z/train_balanced.h5', }

label_logits_vec = {}

for label in tqdm(range(4)):
    print(f'label {label}')
    lossvec, predsvec = get_raw_preds(
                        model=model,
                        dataloc=moredetail['train_loc'],
                        dataset_names=[[f'X_{label}', f'Ylabels_{label}'],
                                      ])
    predarr = np.vstack([x.numpy() for x in predsvec])
    label_logits_vec[label] = [lossvec, predarr]
    
fig = plt.figure(figsize=(20,12))


for i in range(4):
    ax = fig.add_subplot(int('22' + str(i+1)))
    plot_logits_ax(vec=label_logits_vec[i][1], 
                   ax=ax, 
                   title=(f'  logits '
                          f' on label={i}  \n '
                          f' (loss={label_logits_vec[i][0][0]})'))

                   


```

    
      0%|          | 0/4 [00:00<?, ?it/s][A
    
      0%|          | 0/1 [00:00<?, ?it/s][A[A

    label 0


    
    
    100%|██████████| 1/1 [00:18<00:00, 18.56s/it][A[A
    
     25%|██▌       | 1/4 [00:18<00:56, 18.68s/it][A
    
      0%|          | 0/1 [00:00<?, ?it/s][A[A

    label 1


    
    
    100%|██████████| 1/1 [00:13<00:00, 13.59s/it][A[A
    
     50%|█████     | 2/4 [00:32<00:34, 17.20s/it][A
    
      0%|          | 0/1 [00:00<?, ?it/s][A[A

    label 2


    
    
    100%|██████████| 1/1 [00:15<00:00, 15.49s/it][A[A
    
     75%|███████▌  | 3/4 [00:48<00:16, 16.72s/it][A
    
      0%|          | 0/1 [00:00<?, ?it/s][A[A

    label 3


    
    
    100%|██████████| 1/1 [00:16<00:00, 16.14s/it][A[A
    
    100%|██████████| 4/4 [01:04<00:00, 16.08s/it][A


    CPU times: user 1min 3s, sys: 248 ms, total: 1min 4s
    Wall time: 1min 4s



![png](2020-01-10-confidence--update_files/2020-01-10-confidence--update_15_10.png)



```python
# Hmm, dang so eventhough this is training data ^^ umm, which yes it has been all 
# nicely scaled into -1.0, 1.0, 
# it is still showing these crazy jumps. But I thought I shuffled this data niecly. Hmm.
# I mean it should be ok to have different kind of data. But yea I thought I had shuffled it.
```


```python
%%time
# and ... on a model after many batches...
modelloc = 'history/2019-12-29T000509Z/epoch_002_batch_01090_model.h5'
model = mu.load_model(modelloc)


# ok... and just one more time, look at train data again..
moredetail = {'train_loc': 'history/2019-12-22T174803Z/train_balanced.h5', }

label_logits_vec = {}

for label in tqdm(range(4)):
    print(f'label {label}')
    lossvec, predsvec = get_raw_preds(
                        model=model,
                        dataloc=moredetail['train_loc'],
                        dataset_names=[[f'X_{label}', f'Ylabels_{label}'],
                                      ])
    predarr = np.vstack([x.numpy() for x in predsvec])
    label_logits_vec[label] = [lossvec, predarr]
    
fig = plt.figure(figsize=(20,12))


for i in range(4):
    ax = fig.add_subplot(int('22' + str(i+1)))
    plot_logits_ax(vec=label_logits_vec[i][1], 
                   ax=ax, 
                   title=(f'  logits '
                          f' on label={i}  \n '
                          f' (loss={label_logits_vec[i][0][0]})'))

                   

```

    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.


    
      0%|          | 0/4 [00:00<?, ?it/s][A
    
      0%|          | 0/1 [00:00<?, ?it/s][A[A

    label 0


    
    
    100%|██████████| 1/1 [00:18<00:00, 18.62s/it][A[A
    
     25%|██▌       | 1/4 [00:18<00:56, 18.73s/it][A
    
      0%|          | 0/1 [00:00<?, ?it/s][A[A

    label 1


    
    
    100%|██████████| 1/1 [00:13<00:00, 13.63s/it][A[A
    
     50%|█████     | 2/4 [00:32<00:34, 17.25s/it][A
    
      0%|          | 0/1 [00:00<?, ?it/s][A[A

    label 2


    
    
    100%|██████████| 1/1 [00:14<00:00, 14.96s/it][A[A
    
     75%|███████▌  | 3/4 [00:47<00:16, 16.61s/it][A
    
      0%|          | 0/1 [00:00<?, ?it/s][A[A

    label 3


    
    
    100%|██████████| 1/1 [00:16<00:00, 16.60s/it][A[A
    
    100%|██████████| 4/4 [01:04<00:00, 16.08s/it][A


    CPU times: user 1min 4s, sys: 275 ms, total: 1min 5s
    Wall time: 1min 4s



![png](2020-01-10-confidence--update_files/2020-01-10-confidence--update_17_11.png)



```python
# hmm ok and the above is looking at the model 
# `history/2019-12-29T000509Z/epoch_002_batch_01090_model.h5` , 
# which although the train log loss, 
# seen here https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-28-two-plot.md 
# is getting better, it is the loss of the batches and ok looking above, 
# The full training set logloss only looks good for 2 out of 4 of the labels.

```

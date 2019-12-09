

#### Validation
* Here I took the outcomes from [my latest model training](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-08-.ipynb) ( [markdownified](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-08-.md) ) , and for each 10th batch, out of about 1000 batches, I ran the model on my validation set.
* Here's [the plot](#plot-validation)
* The model was only about `88k` so saving it to a file for each of `~1000` batches didn't kill much disk space.
* However Training took only 7 minutes, and the validation was taking above `3 hours` so making the validation async was not exactly meaningful in this case heh. 
* But I think it may have taken so long because I was running `model(X)` in eager execution. I wonder how much time this would have taken in *not eager mode*.


```python

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import mytf.utils as mu

```


```python
tf.enable_eager_execution()

```


```python
historydir = 'history'
with open(f'{historydir}/2019-12-01T223537Z.json') as fd:
    losshistory = json.load(fd)
    
plt.plot(losshistory) 
```




    [<matplotlib.lines.Line2D at 0x7f446828ce80>]




![png](2019-12-01-loglossbook_files/2019-12-01-loglossbook_2_1.png)



```python
'2019-12-08T220612Z'

historydir = 'history'
with open(f'{historydir}/2019-12-08T220612Z/01068_train_loss_history.json') as fd:
    losshistory = json.load(fd)
    
plt.plot(losshistory) 
```




    [<matplotlib.lines.Line2D at 0x7f8a94071358>]




![png](2019-12-01-loglossbook_files/2019-12-01-loglossbook_3_1.png)



```python
from tensorflow import keras
import mytf.utils as mu
import ipdb
from tqdm import tqdm
```


```python
def get_performance_parts(model, dataloc, dataset_names):
    # dataloc contains the test data..
    lossvec = []
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

        lossvec.append(np.mean(batchlosses))
    return lossvec


```


```python
# mu.get_performance(model, dataloc, dataset_names)
def perf_wrapper(modelloc):
    model = mu.load_model(modelloc)
    #mu.get_performance(model=model, 
    return get_performance_parts(
                    model=model,
                    dataloc='data/2019-12-07-test-balanced.h5',
                    dataset_names=[['X_0', 'Ylabels_0'],
                                  ['X_1', 'Ylabels_1'],
                                  ['X_2', 'Ylabels_2'],
                                  ['X_3', 'Ylabels_3']])
```


```python
import os; os.getpid()
```




    10936




```python
batch_losses_vec = []
for step in np.arange(0, 1068, 10):
    print(step)
    modelname = f'history/2019-12-08T220612Z/{str(step).zfill(5)}_model.h5'
    print(modelname)
    steploss = perf_wrapper(modelname)
    print('steploss, ', steploss)
    batch_losses_vec.append(steploss)
```

    0
    history/2019-12-08T220612Z/00000_model.h5


    W1208 22:36:50.513340 140235498792768 hdf5_format.py:221] No training configuration found in save file: the model was *not* compiled. Compile it manually.
      0%|          | 0/4 [00:00<?, ?it/s]


```python
len(batch_losses_vec)
```




    107




```python
lossesarr = np.array(batch_losses_vec)
meanlossesarr = np.mean(lossesarr, axis=1)
```

#### plot validation

```python
batch_losses_vec[:5]
#batch_losses_vec = []
#for step in np.arange(0, 1068, 10):
# [2.8359528, 0.45356295, 1.7049086, 4.099845]

plt.plot([x[0] for x in batch_losses_vec], color='blue', label='0')
plt.plot([x[1] for x in batch_losses_vec], color='green', label='1')
plt.plot([x[2] for x in batch_losses_vec], color='red', label='2')
plt.plot([x[3] for x in batch_losses_vec], color='orange', label='3')
plt.plot(meanlossesarr, color='black', label='mean')

plt.legend()
#plt.plot
```




    <matplotlib.legend.Legend at 0x7f8a26378e10>




![png](2019-12-01-loglossbook_files/2019-12-01-loglossbook_11_1.png)



```python
lossesarr
```




    array([[1.3351634 , 1.4731871 , 1.3140502 , 1.4657389 ],
           [1.4036251 , 1.5530692 , 1.0446882 , 1.645331  ],
           [1.4810811 , 1.6259581 , 0.91911376, 1.7881192 ],
           [1.6964207 , 1.6927042 , 0.8015094 , 2.0324314 ],
           [1.7781167 , 1.5143406 , 0.8248832 , 2.1740363 ],
           [1.935156  , 1.2892603 , 0.84526   , 2.3428662 ],
           [2.1638696 , 1.0792673 , 0.70872974, 2.6655743 ],
           [1.9534022 , 0.73756737, 0.8852098 , 2.622182  ],
           [1.9406806 , 0.5933818 , 1.0058273 , 2.7354705 ],
           [2.0017438 , 0.5292517 , 1.0626508 , 2.9350176 ],
           [2.196939  , 0.45657447, 1.1189158 , 3.1512318 ],
           [2.329297  , 0.51421535, 1.352263  , 3.1518145 ],
           [2.1047034 , 0.5037825 , 1.5859323 , 3.08032   ],
           [1.8594807 , 0.5587238 , 1.5429379 , 2.8631775 ],
           [1.9347439 , 0.5394595 , 1.6954019 , 3.0993047 ],
           [2.1181533 , 0.5336305 , 1.5180852 , 3.3368828 ],
           [2.0038323 , 0.6204544 , 1.2530756 , 2.9970613 ],
           [2.1981256 , 0.58792925, 1.1931396 , 3.2589035 ],
           [2.1430426 , 0.7190803 , 1.0682174 , 2.4001927 ],
           [2.0913672 , 0.8401063 , 1.0228812 , 2.1404743 ],
           [2.2090833 , 0.78182775, 1.0093541 , 2.2566795 ],
           [2.1692474 , 0.811452  , 1.2212783 , 2.3062282 ],
           [2.3541267 , 0.65815514, 1.3489636 , 3.077089  ],
           [2.1464877 , 0.737907  , 1.2662387 , 2.9448156 ],
           [2.0213137 , 0.8562525 , 1.2609545 , 2.7348871 ],
           [2.0428443 , 0.77263075, 1.2463275 , 2.8663592 ],
           [1.9546723 , 0.7461912 , 1.1496452 , 2.8901618 ],
           [1.9966451 , 0.7508285 , 1.1748587 , 2.9331622 ],
           [2.2410986 , 0.69919235, 1.2346983 , 3.122286  ],
           [2.4885015 , 0.57462883, 1.2301722 , 3.4027977 ],
           [2.4516363 , 0.57067007, 1.2223443 , 3.2781599 ],
           [2.2824984 , 0.65961546, 1.0776057 , 3.0167935 ],
           [2.2210763 , 0.69872713, 1.1177806 , 2.8805058 ],
           [2.2931957 , 0.69109434, 1.032774  , 2.9761772 ],
           [2.4493954 , 0.622795  , 1.0957028 , 3.0888433 ],
           [2.4876742 , 0.6601831 , 1.1140472 , 3.013493  ],
           [2.4601307 , 0.6562334 , 1.1353438 , 3.0694702 ],
           [2.4411385 , 0.71929234, 1.1507264 , 3.159165  ],
           [2.6047325 , 0.73656625, 1.1984404 , 3.3015747 ],
           [2.9491456 , 0.7011701 , 1.3073893 , 3.6267223 ],
           [3.1835597 , 0.67035425, 1.3346659 , 3.7555268 ],
           [2.8140779 , 0.7479792 , 1.0791199 , 3.442301  ],
           [2.600772  , 0.8597047 , 0.8856437 , 3.3031945 ],
           [2.403481  , 0.87139636, 0.8985988 , 3.2903469 ],
           [2.4799223 , 0.7870504 , 0.9057817 , 3.4419188 ],
           [2.4583771 , 0.7497664 , 0.88477594, 3.5578077 ],
           [2.493268  , 0.72802633, 0.98754203, 3.488412  ],
           [2.9508946 , 0.5640698 , 1.1308994 , 3.627878  ],
           [3.0199916 , 0.5521208 , 1.1339356 , 3.6998925 ],
           [3.0168402 , 0.5772391 , 1.1556966 , 3.746559  ],
           [2.9846725 , 0.5850268 , 1.1861988 , 3.6672456 ],
           [3.0022569 , 0.60611886, 1.2475421 , 3.5765157 ],
           [3.225216  , 0.5954642 , 1.3899362 , 3.599627  ],
           [3.0158265 , 0.7053217 , 1.2542483 , 3.4227312 ],
           [3.0595798 , 0.6240693 , 1.1281008 , 3.5235925 ],
           [2.7621658 , 0.76327676, 1.0095214 , 3.5958629 ],
           [2.6548889 , 0.7609274 , 0.9439439 , 3.4703522 ],
           [2.5334435 , 0.78953874, 0.8454408 , 3.463432  ],
           [2.4720204 , 0.73448735, 0.9984613 , 3.550102  ],
           [2.524865  , 0.71166635, 1.1470479 , 3.5369616 ],
           [2.5988526 , 0.71541595, 1.1652452 , 3.5189233 ],
           [2.9248216 , 0.5428488 , 1.2349063 , 3.8142357 ],
           [3.1952226 , 0.4435198 , 1.3888544 , 3.993797  ],
           [3.3085244 , 0.38709944, 1.4710102 , 4.080298  ],
           [3.1669972 , 0.40973452, 1.5098591 , 4.0666866 ],
           [2.7107153 , 0.5117123 , 1.3973982 , 3.7479405 ],
           [2.587512  , 0.5786207 , 1.5054486 , 3.563124  ],
           [2.6245546 , 0.59590423, 1.6130393 , 3.5515974 ],
           [2.6286    , 0.5222682 , 1.5027198 , 3.7699971 ],
           [2.6817899 , 0.5177797 , 1.5869193 , 3.7244189 ],
           [2.904556  , 0.50106466, 1.8164269 , 3.6922882 ],
           [2.9071503 , 0.5518159 , 1.8914468 , 3.5280676 ],
           [2.73826   , 0.59329253, 1.9030589 , 3.5137975 ],
           [2.4965215 , 0.62963957, 1.9080484 , 3.460288  ],
           [2.435967  , 0.6121716 , 1.9032393 , 3.5985804 ],
           [2.8340874 , 0.58127844, 2.1304395 , 3.703755  ],
           [2.7536929 , 0.56476605, 2.078189  , 3.7767286 ],
           [2.5790102 , 0.5921851 , 1.8708863 , 3.7291815 ],
           [2.752435  , 0.5093397 , 1.8106233 , 3.9004035 ],
           [2.8359528 , 0.45356295, 1.7049086 , 4.099845  ],
           [2.8773637 , 0.44053876, 1.7727754 , 4.0866933 ],
           [2.6562023 , 0.536994  , 1.9040228 , 3.7602668 ],
           [2.5591996 , 0.62757915, 2.0161712 , 3.658457  ],
           [2.4294486 , 0.6389067 , 1.8855482 , 3.585786  ],
           [2.389747  , 0.6090067 , 1.7404081 , 3.628272  ],
           [2.395344  , 0.59588754, 1.6116827 , 3.6454554 ],
           [2.440686  , 0.57233673, 1.5966128 , 3.6856556 ],
           [2.4107077 , 0.61616904, 1.7184944 , 3.6492982 ],
           [2.4589574 , 0.6520386 , 1.7200731 , 3.6777391 ],
           [2.494923  , 0.6659917 , 1.7606882 , 3.7461236 ],
           [2.585011  , 0.60325044, 1.9338262 , 3.8209126 ],
           [2.5457416 , 0.5890586 , 1.8292966 , 3.7708669 ],
           [2.8144364 , 0.53364944, 1.974421  , 3.7635906 ],
           [2.9019258 , 0.5020002 , 1.9413275 , 3.8502471 ],
           [2.8643317 , 0.50512874, 1.9206105 , 3.8255084 ],
           [2.7968724 , 0.5137779 , 1.9171289 , 3.7540233 ],
           [2.7858882 , 0.51953316, 1.9333706 , 3.7037604 ],
           [2.9250474 , 0.4980415 , 2.0115767 , 3.7844214 ],
           [2.8365712 , 0.5144848 , 1.9771044 , 3.76555   ],
           [2.7632754 , 0.49625787, 1.9523457 , 3.8485744 ],
           [2.7280297 , 0.5030039 , 1.972389  , 3.9366643 ],
           [2.9572167 , 0.5071157 , 2.1727612 , 4.0679636 ],
           [2.8911443 , 0.535912  , 2.2557335 , 4.120013  ],
           [2.545441  , 0.824353  , 2.267557  , 3.8870494 ],
           [2.783403  , 0.6001091 , 2.2852643 , 4.0161557 ],
           [3.237786  , 0.4521004 , 2.3615253 , 4.294332  ],
           [3.483766  , 0.4215114 , 2.3849225 , 4.452968  ]], dtype=float32)



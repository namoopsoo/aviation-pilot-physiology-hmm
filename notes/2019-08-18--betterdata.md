#### better data
* Last data set is just randomly created and so it's really I think not enough data for the rare classes, since it looks like this
```python
with open('models/2019-07-21T1815-UTC-outdata-SUBSET50k.pkl', 'rb') as fd:
    julydata = pickle.load(fd)

print(Counter(np.argmax(julydata['y_train'], axis=1)))
#  Counter({0: 26267, 2: 20239, 3: 2597, 1: 897})

print(Counter(np.argmax(julydata['y_test'], axis=1)))
# Counter({0: 25903, 2: 20542, 3: 2676, 1: 879})
```
* Building this out again..

```python
import pandas as pd 
import mytf.utils as mu 
df = pd.read_csv('data/2019-07-14-crews1-2.csv')   

outdata = ipdb.runcall(mu.make_data, df, crews={'training': [1], 
                             'test': [2]}, 
                   sequence_window=256, percent_of_data=1, 
                  feature_cols=['r', 'gsr', 'ecg', ])

In [17]: outdata['y_train'].shape                                                                                                    
Out[17]: (446110, 4)

In [22]: Counter(np.argmax(outdata['y_train'], axis=1))                                                                              
Out[22]: Counter({0: 234352, 2: 180851, 3: 23218, 1: 7689})

In [23]: Counter(np.argmax(outdata['y_test'], axis=1))                                                                               
Out[23]: Counter({2: 183718, 0: 325198, 3: 27039, 1: 15371})
```
* So yea, just should take all 25k of each basically..
```python

indices = mu.choose_training_indices(outdata, {0:10, 1:10, 2:10, 3:10})

```
```python
In [40]: %%time 
    ...: indices = mu.choose_training_indices(outdata, [10, 10, 10, 10], dict_key='y_test') 
    ...:  
    ...:                                                                                                                             
CPU times: user 1.67 s, sys: 11.8 ms, total: 1.68 s
Wall time: 1.7 s

In [41]: Counter(np.argmax(outdata['y_test'][reduce(lambda x,y:x+y,indices.values()), :], axis=1))                                   
Out[41]: Counter({0: 10, 1: 10, 2: 10, 3: 10})

```
* Okay all of it now...
```python
%%time  
# {2: 183718, 0: 325198, 3: 27039, 1: 15371}
indices = mu.choose_training_indices(outdata, [15000, 15000, 15000, 15000], dict_key='y_test') 

In [53]: Counter(np.argmax(outdata['y_test'][reduce(lambda x,y:x+y,indices.values()), :], axis=1))                                   
Out[53]: Counter({0: 15000, 1: 15000, 2: 15000, 3: 15000})

```

##### Save it.  but use this as train data  (flipping it basically) because somehow test has more examples.
```python
train_subset = reduce(lambda x,y:x+y,indices.values())
aug18data = {
    'x_train': outdata['x_test'][train_subset],
    'y_train': outdata['y_test'][train_subset],
    }
    
    
with open('models/2019-08-18-outdata-SUBSET60k.pkl', 'wb') as fd:
    pickle.dump(aug18data, fd)
```





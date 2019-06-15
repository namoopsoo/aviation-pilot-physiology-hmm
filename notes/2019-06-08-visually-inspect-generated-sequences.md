
#### Took a detour to plot the sequences I'm currently building, as a sanity check...
* I had saved my `"outdata"` earlier as `"models/2019-05-19T001217-UTC-outdata.pkl"`, 
* And as seen in this `metadata` below, I had only built sequences using one variable, `'r'` which is a _Respiration_ amplitude value.
```python
import pickle
import matplotlib.pyplot as plt

with open('models/2019-05-19T001217-UTC-outdata.pkl', 'rb') as fd: outdata = pickle.load(fd)

outdata.keys()
# =>
dict_keys(['x_train', 'y_train', 'x_test', 'y_test', 'y_train_original', 'y_test_original', 'traindf', 'testdf', 'metadata'])

#...
outdata['metadata']
{'output': {'shapes': {'x_train': (446110, 256, 1),
   'y_train': (446110, 4),
   'x_test': (551326, 256, 1),
   'y_test': (551326, 4),
   'y_train_original': (446110,),
   'y_test_original': (551326,),
   'traindf': (447652, 7),
   'testdf': (552868, 7)},
  "Counter(outdata['y_train_original'])": {'A': 234352,
   'C': 180851,
   'D': 23218,
   'B': 7689},
  "Counter(outdata['y_test_original'])": {'C': 183718,
   'A': 325198,
   'D': 27039,
   'B': 15371}},
 'input': {'kwargs': {'crews': {'training': [1], 'test': [2]},
   'percent_of_data': 1,
   'sequence_window': 256,
   'feature_cols': ['r']}},
 'data_ts': '2019-05-18 20:18:33 UTC'}
 
```
```python
# Look at my output data here real quick..
print(outdata['y_train'][:5,:])

# so, I already checked this prior to the demultiplexing, but lets compare the sums ,
# using this demultiplexed form..
ytrain_sums = [[j, np.sum(outdata['y_train'][:,j])] for j in [0,1,2,3]]
print(ytrain_sums)

# ==>
[[1. 0. 0. 0.]
 [1. 0. 0. 0.]
 [1. 0. 0. 0.]
 [1. 0. 0. 0.]
 [1. 0. 0. 0.]]
[[0, 234352.0], [1, 7689.0], [2, 180851.0], [3, 23218.0]]

# Ok cool. so per the above, we do have representation in the de-multiplexed data as well.
print ( outdata['y_train'].shape, outdata['x_train'].shape)
# =>
(446110, 4) (446110, 256, 1)

```

#### Hone in on some data for a plot..
* There are `4` output classes, `0, 1, 2, 3`, so I want to fetch a few  sequences for each class, lazily,
```python
def fetch_some_examples(array, which_col):
    gg = (i for i in np.arange(1, 446110, 1) if array['y_train'][i][which_col] == 1)
    
    return [gg.__next__() for i in range(4)]

examples = {j: fetch_some_examples(outdata, j)
               for j in range(4)}
```

```python
print(examples)
print([[j, outdata['y_train'][examples[j][0]]] 
                             for j in range(4)])
                             
# =>
{0: [1, 2, 3, 4], 1: [204490, 204491, 204492, 204493], 2: [1457, 1458, 1459, 1460], 3: [99781, 99782, 99783, 99784]}
[[0, array([1., 0., 0., 0.], dtype=float32)], [1, array([0., 1., 0., 0.], dtype=float32)], [2, array([0., 0., 1., 0.], dtype=float32)], [3, array([0., 0., 0., 1.], dtype=float32)]]
```

##### And plot
```python
# Umm, and I want to look at a couple of these..
# Going to print some random plots of my sequence data here, for each category.
#
# row 1: "A" or "Baseline"; row2: "B" or "Startle/Surprise"; 
# row3: "C" or "Channelized Attention" ; 
# row4: "D"  or "Diverted Attention"

x = np.arange(0, 256, 1)
print(x.shape, y.shape)
fig = plt.figure() # x, y, )

for i,j in itertools.product(range(4), range(4)):
    ix = 1 + i*4 + j
    
    sample_row = examples[i][j]

    ax = fig.add_subplot(4, 4, ix)
    y = outdata['x_train'][sample_row][:,0]
    ax.scatter(x, y, color='lightblue', marker='.')
```

<img src="https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/assets/Screen%20Shot%202019-06-15%20at%2011.29.22%20AM.png"
width="607" height="383">

```
# I dont know if there is enough in here to visually discern
# What does a respiration look like for each category, 
# I guess it would really come down to the slope.
# 
# So above, I do see that the last row's slope "Diverted Attention" does happen to look
#   steeper than the first row's slope "Baseline"
# 
# In any case, I think this visual inspection is enough to tell me at least on a high level there is some interesting signal in even
#  just this one variable r.
# 
# But indeed the classes are not balanced and I still think that is the next challenge to deal with.
```



#### Sort it
* In [here](2019-05-14-wrangling-time-data.md), I created a sorted version, 

```python
df = pd.read_csv('data/train.csv')
df.sort_values(by=['crew', 'experiment', 'time'], inplace=True)

df.to_csv('data/sorted_train.csv', index=False)
```

####  I saved a smaller reduced version 
* [here](2019-07-13-Five-more-data.md)
```python
import pandas as pd
%time df = pd.read_csv('data/sorted_train.csv')
import mytf.utils as mu

%time smalldf = mu.chomp_crews(df, crews=[1, 2], feature_cols=['r', 'gsr', 'ecg', 'eeg_t3'])

smalldf.to_csv('data/2019-07-14-crews1-2.csv') 

In [6]: %time smalldf = mu.chomp_crews(df, crews=[1, 2], feature_cols=['r', 'gsr', 'ecg', 'eeg_t3'])                                                                                    
CPU times: user 646 ms, sys: 1.1 s, total: 1.75 s
Wall time: 3.65 s
```


### 2019-10-27

#### Sort again
* I think my initial sort was not the best. Going to split this up again...
* Starting from scratch..
```python
import pandas as pd
from decimal import Decimal
import numpy as np
import mytf.utils as mu
df = pd.read_csv('data/train.csv')    

# ok so indeed not sorted as I recall..
In [7]: %time mu.is_it_sorted_by_time(df)                                                                                            
CPU times: user 1.29 s, sys: 2.91 s, total: 4.2 s
Wall time: 9.57 s
Out[7]: 
{(1, 1): False,
 (1, 0): False,
 (2, 1): False,
 (2, 0): False,
 (3, 1): False,
 (3, 0): False,
 (4, 1): False,
 (4, 0): False,
 (5, 1): False,
 (5, 0): False,
 (6, 1): False,
 (6, 0): False,
 (7, 1): False,
 (7, 0): False,
 (8, 1): False,
 (8, 0): False,
 (13, 1): False,
 (13, 0): False}
 
# before sorting, build an integrity hash to try validate stuff didnt get messed up... 
sampleindices =  np.random.choice(df.index.tolist(), size=1000, replace=False)

def make_integrity_hash(df, sampleindices):
    return {(x['crew'], 
                   x['seat'],
                   Decimal(x['time']).quantize(Decimal('0.12345678'))
                   ): (x['event'], x['r']) for x in df.loc[sampleindices].to_dict(orient='records')}

integrity_hash = make_integrity_hash(df, sampleindices)

# Ok now sort... 
# Not by experiment but sort by seat this time...
df.sort_values(by=['crew', 'seat', 'time'], inplace=True)

```
* and checking integrity seems good.. And seem to be sorted by time now..
```python
In [30]: df.sort_values(by=['crew', 'seat', 'time'], inplace=True) 
    ...:                                                                                                                             

In [31]: integrity_hash == make_integrity_hash(df, sampleindices)                                                                    
Out[31]: True

In [33]: mu.is_it_sorted_by_time(df)                                                                                                 
Out[33]: 
{(1, 0): True,
 (1, 1): True,
 (2, 0): True,
 (2, 1): True,
 (3, 0): True,
 (3, 1): True,
 (4, 0): True,
 (4, 1): True,
 (5, 0): True,
 (5, 1): True,
 (6, 0): True,
 (6, 1): True,
 (7, 0): True,
 (7, 1): True,
 (8, 0): True,
 (8, 1): True,
 (13, 0): True,
 (13, 1): True}
```
* Actually since indices dont matter let me redo validation 
```python
def validate_integrity(df, integrity_hash):
    out = {x:
    df[(df.crew == x[0]) 
            & (df.seat == x[1])
            & (df.time == x[2])][['event', 'r']].to_dict(orient='records')
            for x in integrity_hash.keys()}
    return out

# wait huh... what the heck, so time is not unique? at all?

In [54]: df[(df.crew == x[0])  
    ...:             & (df.seat == x[1]) 
    ...:             & (df.time == x[2])][['event', 'r']].to_dict(orient='records')                                                  
Out[54]: 
[{'event': 'C', 'r': 805.463989},
 {'event': 'A', 'r': 819.320007},
 {'event': 'A', 'r': 827.888}]

In [55]: df[(df.crew == x[0])  
    ...:             & (df.seat == x[1]) 
    ...:             & (df.time == x[2])]#[['event', 'r']].to_dict(orient='records')                                                 
Out[55]: 
         crew experiment        time  seat  eeg_fp1    eeg_f7  ...     eeg_cz    eeg_o2           ecg           r         gsr  event
1074449     3         CA  228.449219     1 -8.80895   4.02958  ...   -4.21730 -11.59520   8972.959961  805.463989  423.260986      C
1258698     3         DA  228.449219     1  9.62710  10.30540  ...   -3.16866 -15.02740  16933.599609  819.320007  596.104980      A
1442971     3         SS  228.449219     1 -5.47289  -6.09790  ...   -3.51994  -2.83535   4274.399902  827.888000  452.666992      A

[3 rows x 28 columns]

In [58]:  df.shape[0]/df.time.unique().shape[0]                                                                                      
Out[58]: 10.066346937340498

In [59]: df[df.time == Decimal('228.44921875')]                                                                                      
Out[59]: 
         crew experiment        time  seat    eeg_fp1  ...       eeg_o2           ecg           r          gsr  event
258209      1         DA  228.449219     0   5.111540  ...    -0.881334      0.000000  661.611023   719.155029      A
73908       1         CA  228.449219     1 -12.268600  ...     1.778420  -4969.060059  817.096985   526.585022      C
258210      1         DA  228.449219     1   6.572690  ...     3.351270  -3176.709961  815.763977   457.071991      A
705811      2         DA  228.449219     0  20.724300  ...     5.312250  -4351.200195  736.330017  1276.130005      A
521580      2         CA  228.449219     1   8.874040  ...    12.135100   9616.280273  815.117981     0.000000      C
705812      2         DA  228.449219     1 -24.031900  ...    -5.994270   7549.959961  811.080994     0.000000      A
890062      2         SS  228.449219     1  19.597300  ...     6.744730   8017.299805  816.783020     0.000000      A
1074448     3         CA  228.449219     0 -24.493500  ...     7.104500   8610.440430  577.809998   731.348999      C
1258697     3         DA  228.449219     0  -9.189250  ...    -8.007170  10518.900391  559.109009   727.111023      A
1442970     3         SS  228.449219     0  14.258100  ...    -5.533900   4829.640137  642.481018   650.156006      A
1074449     3         CA  228.449219     1  -8.808950  ...   -11.595200   8972.959961  805.463989   423.260986      C
1258698     3         DA  228.449219     1   9.627100  ...   -15.027400  16933.599609  819.320007   596.104980      A
1442971     3         SS  228.449219     1  -5.472890  ...    -2.835350   4274.399902  827.888000   452.666992      A
1811604     4         DA  228.449219     0   3.275210  ...    -0.349200   -812.010986  596.268982   900.401001      A
1811605     4         DA  228.449219     1  -7.652430  ...    -3.529380   6090.390137  827.346008   690.976013      A
2364362     5         DA  228.449219     0  -4.742620  ...    -7.651720  -2121.540039  648.614990  1749.819946      A
2364363     5         DA  228.449219     1  -9.918460  ...    10.810500   6477.689941  813.301025     0.000000      A
2732930     6         CA  228.449219     0   0.850056  ...    -8.089880  31555.699219  656.135986  1167.969971      C
2917204     6         DA  228.449219     0  -0.568591  ...     3.012870  33225.500000  649.932983  1037.209961      A
3101597     6         SS  228.449219     0 -10.262200  ...    -5.680480  27051.300781  652.750000   958.145020      A
2732931     6         CA  228.449219     1  -3.829330  ...   -16.162500  10577.299805  796.263977  1761.770020      C
2917205     6         DA  228.449219     1  14.050400  ...    -2.311960  11821.400391  793.242004  1687.010010      A
3101598     6         SS  228.449219     1 -63.099499  ...   -11.041000  15425.599609  801.044983  1810.310059      A
3470154     7         DA  228.449219     0 -19.291800  ...     1.647440   3397.870117  717.320007     0.000000      A
3470155     7         DA  228.449219     1  -8.123900  ...    -4.570940  19862.599609  830.302002   753.466980      A
4022881     8         DA  228.449219     0   5.508320  ...    -2.657950  26939.199219  679.333984  1605.760010      A
3838690     8         CA  228.449219     1  -4.353200  ...   -12.114100   3718.290039  812.460022   764.593994      C
4022882     8         DA  228.449219     1  14.915700  ...     5.052380    607.473022  801.893005   665.099976      A
4207081     8         SS  228.449219     1   9.665910  ...    -8.998850   5761.310059  799.244995   683.098022      A
4572908    13         DA  228.449219     0  10.525200  ...    -1.287820 -16573.699219  682.773987   382.571014      A
4572909    13         DA  228.449219     1   0.385544  ...   -15.748300  -8168.060059  807.070007   791.914001      A
```

#### Is time at least unique for `crew,seat,experiment` then?
* I redid this func to check for uniqueness of time too..  `commit 70a5dac`
```python
# mytf/utils.py ...
def is_it_sorted_by_time(df):
    choices = (df.crew.unique(),
               df.seat.unique(),
               df.experiment.unique())
    meta = {}
    for crew, seat, experiment in itertools.product(*choices):
        query = ((df.crew == crew)
                & (df.seat == seat)
                & (df.experiment == experiment))
        times = df[query].time.tolist()
        meta[(crew, seat, experiment)] = times == sorted(list(set(times)))
    return meta
```
* So now 
```python
In [62]: reload(mu)                                                                                                                  
Out[62]: <module 'mytf.utils' from '/Users/michal/LeDropbox/Dropbox/Code/Kaggle/reducing-commercial-aviation-fatalities/mytf/utils.py'>

In [63]:                                                                                                                             

In [63]: mu.is_it_sorted_by_time(df)                                                                                                 
Out[63]: 
{(1, 0, 'SS'): True,
 (1, 0, 'DA'): True,
 (1, 0, 'CA'): True,
 (1, 1, 'SS'): True,
 (1, 1, 'DA'): True,
 (1, 1, 'CA'): True,
 (2, 0, 'SS'): True,
 (2, 0, 'DA'): True,
 (2, 0, 'CA'): True,
 (2, 1, 'SS'): True,
 (2, 1, 'DA'): True,
 (2, 1, 'CA'): True,
 (3, 0, 'SS'): True,
 (3, 0, 'DA'): True,
 (3, 0, 'CA'): True,
 (3, 1, 'SS'): True,
 (3, 1, 'DA'): True,
 (3, 1, 'CA'): True,
 (4, 0, 'SS'): True,
 (4, 0, 'DA'): True,
 (4, 0, 'CA'): True,
 (4, 1, 'SS'): True,
 (4, 1, 'DA'): True,
 (4, 1, 'CA'): True,
 (5, 0, 'SS'): True,
 (5, 0, 'DA'): True,
 (5, 0, 'CA'): True,
 (5, 1, 'SS'): True,
 (5, 1, 'DA'): True,
 (5, 1, 'CA'): True,
 (6, 0, 'SS'): True,
 (6, 0, 'DA'): True,
 (6, 0, 'CA'): True,
 (6, 1, 'SS'): True,
 (6, 1, 'DA'): True,
 (6, 1, 'CA'): True,
 (7, 0, 'SS'): True,
 (7, 0, 'DA'): True,
 (7, 0, 'CA'): True,
 (7, 1, 'SS'): True,
 (7, 1, 'DA'): True,
 (7, 1, 'CA'): True,
 (8, 0, 'SS'): True,
 (8, 0, 'DA'): True,
 (8, 0, 'CA'): True,
 (8, 1, 'SS'): True,
 (8, 1, 'DA'): True,
 (8, 1, 'CA'): True,
 (13, 0, 'SS'): True,
 (13, 0, 'DA'): True,
 (13, 0, 'CA'): True,
 (13, 1, 'SS'): True,
 (13, 1, 'DA'): True,
 (13, 1, 'CA'): True}

```
* Okay so since time is unique by `crew,seat,experiment`, as long as any dataset holds that still then time series can be formed safely.

#### Do this raw one more time..
```python
df = pd.read_csv('data/train.csv')

df.sort_values(by=['crew', 'seat', 'experiment', 'time'], inplace=True) 

# Looking at this function I had from a few months ago I already knew this hah but I forgot.. 
mu.chomp_crews(df, crews, feature_cols)

In [73]: %%time 
    ...: mu.split_data_by_crew(df, outdir) 
    ...:  
    ...:                                                                                                                             
CPU times: user 4.27 s, sys: 5.57 s, total: 9.84 s
Wall time: 23.3 s
Out[73]: 
{1: 'data/2019-10-27/crew_1-train.pkl',
 2: 'data/2019-10-27/crew_2-train.pkl',
 3: 'data/2019-10-27/crew_3-train.pkl',
 4: 'data/2019-10-27/crew_4-train.pkl',
 5: 'data/2019-10-27/crew_5-train.pkl',
 6: 'data/2019-10-27/crew_6-train.pkl',
 7: 'data/2019-10-27/crew_7-train.pkl',
 8: 'data/2019-10-27/crew_8-train.pkl',
 13: 'data/2019-10-27/crew_13-train.pkl'}


```


#### Ok, save by crew now.. instead of other way i had ..
* And going to simply save w/o the index, so hopefully that will help
```python

df.to_csv('data/2019-10-27-sorted_train.csv', index=False)
```

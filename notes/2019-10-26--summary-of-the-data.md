

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

```python

df.to_csv('data/2019-10-27-sorted_train.csv', index=False)
```

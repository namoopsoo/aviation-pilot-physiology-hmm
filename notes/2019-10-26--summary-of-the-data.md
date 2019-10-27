

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


